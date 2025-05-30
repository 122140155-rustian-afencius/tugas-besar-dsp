"""
Signal processing utilities for vital signs monitoring.

This module contains common signal processing functions used across
different processors for filtering, analysis, and feature extraction.
"""

import numpy as np
import scipy.signal as signal
import pywt
from typing import Tuple, List, Optional


def apply_bandpass_filter(data: List[float], lowcut: float, highcut: float, 
                         fs: float, order: int = 4) -> np.ndarray:
    """
    Apply bandpass filter to signal data.
    
    Args:
        data: Input signal data
        lowcut: Low frequency cutoff in Hz
        highcut: High frequency cutoff in Hz  
        fs: Sampling frequency in Hz
        order: Filter order (default: 4)
        
    Returns:
        Filtered signal as numpy array
        
    Raises:
        ValueError: If cutoff frequencies are invalid
    """
    if lowcut >= highcut:
        raise ValueError("Low cutoff must be less than high cutoff")
    
    if highcut >= fs / 2:
        raise ValueError("High cutoff must be less than Nyquist frequency")
    
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    
    try:
        b, a = signal.butter(order, [low, high], btype='band')
        return signal.filtfilt(b, a, data)
    except Exception as e:
        raise ValueError(f"Filter design failed: {e}")


def apply_savgol_filter(data: np.ndarray, window_length: int = 15, 
                       polyorder: int = 2) -> np.ndarray:
    """
    Apply Savitzky-Golay smoothing filter.
    
    Args:
        data: Input signal data
        window_length: Length of the smoothing window (must be odd)
        polyorder: Polynomial order for fitting
        
    Returns:
        Smoothed signal
        
    Raises:
        ValueError: If window_length is invalid
    """
    if len(data) < window_length:
        return data
        
    # Ensure window length is odd and valid
    if window_length % 2 == 0:
        window_length -= 1
    window_length = max(3, min(window_length, len(data) - 2))
    
    return signal.savgol_filter(data, window_length, polyorder)


def wavelet_denoise(data: np.ndarray, wavelet: str = 'sym4', 
                   levels: int = 3) -> np.ndarray:
    """
    Apply wavelet denoising to signal.
    
    Args:
        data: Input signal data
        wavelet: Wavelet type (default: 'sym4')
        levels: Decomposition levels (default: 3)
        
    Returns:
        Denoised signal
        
    Raises:
        ValueError: If data is too short for decomposition
    """
    if len(data) <= 2**levels:
        return data
    
    try:
        # Wavelet decomposition
        coeffs = pywt.wavedec(data, wavelet, level=levels)
        
        # Calculate threshold using median absolute deviation
        sigma = np.median(np.abs(coeffs[-1])) / 0.6745
        threshold = sigma * np.sqrt(2 * np.log(len(data)))
        
        # Apply soft thresholding to detail coefficients
        for i in range(1, len(coeffs)):
            coeffs[i] = pywt.threshold(coeffs[i], threshold, mode='soft')
        
        # Reconstruct signal
        return pywt.waverec(coeffs, wavelet)
        
    except Exception as e:
        print(f"Wavelet denoising failed: {e}")
        return data


def normalize_signal(signal_data: List[float]) -> np.ndarray:
    """
    Normalize signal to zero mean and unit variance.
    
    Args:
        signal_data: Input signal data
        
    Returns:
        Normalized signal array
        
    Raises:
        ValueError: If signal has zero variance
    """
    signal_array = np.array(signal_data)
    mean = np.mean(signal_array)
    std = np.std(signal_array)
    
    if std < 1e-9:
        raise ValueError("Signal has zero variance - cannot normalize")
    
    return (signal_array - mean) / std


def find_dominant_frequency(signal_data: np.ndarray, fs: float, 
                          freq_range: Optional[Tuple[float, float]] = None) -> float:
    """
    Find dominant frequency in signal using FFT.
    
    Args:
        signal_data: Input signal data
        fs: Sampling frequency in Hz
        freq_range: Optional frequency range to search (min_freq, max_freq)
        
    Returns:
        Dominant frequency in Hz
        
    Raises:
        ValueError: If signal is too short or invalid
    """
    if len(signal_data) < 4:
        raise ValueError("Signal too short for frequency analysis")
    
    # Compute FFT
    n = len(signal_data)
    freqs = np.fft.rfftfreq(n, d=1/fs)
    fft_vals = np.abs(np.fft.rfft(signal_data))
    
    # Apply frequency range filter if specified
    if freq_range:
        min_freq, max_freq = freq_range
        valid_indices = (freqs >= min_freq) & (freqs <= max_freq)
        if np.any(valid_indices):
            freqs = freqs[valid_indices]
            fft_vals = fft_vals[valid_indices]
    
    # Find peak frequency
    if len(fft_vals) == 0 or np.max(fft_vals) == 0:
        return 0.0
    
    peak_idx = np.argmax(fft_vals)
    return freqs[peak_idx]


def detect_peaks_with_validation(signal_data: np.ndarray, fs: float,
                                min_distance_sec: float = 0.5,
                                prominence: float = 0.2,
                                min_width_sec: float = 0.08) -> Tuple[np.ndarray, List[float]]:
    """
    Detect peaks in signal with validation for physiological plausibility.
    
    Args:
        signal_data: Input signal data
        fs: Sampling frequency in Hz
        min_distance_sec: Minimum time between peaks in seconds
        prominence: Minimum peak prominence
        min_width_sec: Minimum peak width in seconds
        
    Returns:
        Tuple of (peak_indices, validated_rates)
        
    Raises:
        ValueError: If parameters are invalid
    """
    if len(signal_data) < 10:
        raise ValueError("Signal too short for peak detection")
    
    # Convert time parameters to samples
    min_distance = int(min_distance_sec * fs)
    min_width = int(min_width_sec * fs)
    
    # Find peaks
    peaks, properties = signal.find_peaks(
        signal_data,
        prominence=prominence,
        distance=min_distance,
        width=min_width
    )
    
    # Validate peak intervals for physiological plausibility
    validated_rates = []
    if len(peaks) >= 2:
        for i in range(1, len(peaks)):
            interval_samples = peaks[i] - peaks[i-1]
            if interval_samples > 0:
                # Convert to rate (beats/breaths per minute)
                rate = 60.0 * fs / interval_samples
                # Accept rates in physiological range (40-180 BPM)
                if 40 <= rate <= 180:
                    validated_rates.append(rate)
    
    return peaks, validated_rates


def calculate_signal_quality(signal_data: np.ndarray) -> float:
    """
    Calculate signal quality metric based on amplitude and variability.
    
    Args:
        signal_data: Input signal data
        
    Returns:
        Signal quality score (higher is better)
        
    Raises:
        ValueError: If signal is empty
    """
    if len(signal_data) == 0:
        raise ValueError("Empty signal data")
    
    # Calculate amplitude range
    amplitude_range = np.max(signal_data) - np.min(signal_data)
    
    # Calculate signal-to-noise ratio estimate
    signal_power = np.var(signal_data)
    if signal_power < 1e-9:
        return 0.0
    
    # Estimate noise as high-frequency component
    if len(signal_data) > 10:
        # Simple high-pass filter to estimate noise
        diff_signal = np.diff(signal_data)
        noise_power = np.var(diff_signal)
        snr = signal_power / (noise_power + 1e-9)
    else:
        snr = 1.0
    
    # Combine amplitude and SNR for quality score
    quality_score = amplitude_range * np.log10(snr + 1)
    return max(0.0, quality_score)


def smooth_signal_exponential(current_value: float, previous_value: float, 
                            alpha: float = 0.3) -> float:
    """
    Apply exponential smoothing to signal values.
    
    Args:
        current_value: Current signal value
        previous_value: Previous smoothed value
        alpha: Smoothing factor (0 < alpha < 1)
        
    Returns:
        Smoothed value
        
    Raises:
        ValueError: If alpha is out of range
    """
    if not 0 < alpha < 1:
        raise ValueError("Alpha must be between 0 and 1")
    
    return alpha * current_value + (1 - alpha) * previous_value
