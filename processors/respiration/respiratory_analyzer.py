"""
Respiratory signal analyzer for breathing rate calculation.

This module provides focused analysis of respiratory signals, including
frequency domain analysis, filtering, and breathing rate calculation.
"""

import numpy as np
import logging
from typing import List, Optional, Tuple
from dataclasses import dataclass

from utils.signal_utils import (
    apply_bandpass_filter,
    find_dominant_frequency
)


@dataclass
class FilterConfig:
    """
    Configuration for respiratory signal filtering.
    
    Attributes:
        lowcut: Low frequency cutoff in Hz (minimum breathing rate)
        highcut: High frequency cutoff in Hz (maximum breathing rate)
        fs: Sampling frequency in Hz
    """
    lowcut: float = 0.1   # 6 breaths per minute
    highcut: float = 0.5  # 30 breaths per minute
    fs: float = 30.0      # 30 fps


@dataclass
class AnalysisResult:
    """
    Result of respiratory signal analysis.
    
    Attributes:
        respiration_rate: Calculated respiration rate in breaths per minute
        dominant_frequency: Dominant frequency in Hz
        signal_quality: Quality score of the analysis (0-1)
        confidence: Confidence in the result (0-1)
        filtered_signal: Filtered signal used for analysis
    """
    respiration_rate: float
    dominant_frequency: float
    signal_quality: float
    confidence: float
    filtered_signal: List[float]


class RespiratorySignalAnalyzer:
    """
    Analyzes respiratory signals to calculate breathing rate.
    
    This class focuses specifically on the signal processing and analysis
    aspects of respiration rate detection, providing clean separation
    from data collection and visualization components.
    """
    
    def __init__(self, filter_config: Optional[FilterConfig] = None):
        """
        Initialize the respiratory signal analyzer.
        
        Args:
            filter_config: Configuration for signal filtering (uses default if None)
        """
        self.config = filter_config or FilterConfig()
        
        # Validate filter configuration
        self._validate_filter_config()
        
        # Analysis parameters
        self.min_analysis_length = 30  # Minimum samples for reliable analysis
        self.confidence_threshold = 0.3  # Minimum confidence for valid results
        
        logging.info(f"Respiratory analyzer initialized: {self.config.lowcut:.2f}-{self.config.highcut:.2f} Hz")
    
    def _validate_filter_config(self) -> None:
        """Validate filter configuration parameters."""
        if self.config.lowcut >= self.config.highcut:
            raise ValueError("Low cutoff must be less than high cutoff")
        
        if self.config.lowcut < 0.05 or self.config.highcut > 1.0:
            raise ValueError("Cutoff frequencies must be in range [0.05, 1.0] Hz")
        
        if self.config.fs <= 0:
            raise ValueError("Sampling frequency must be positive")
    
    def analyze_signal(self, signal: List[float]) -> Optional[AnalysisResult]:
        """
        Analyze respiratory signal to calculate breathing rate.
        
        Args:
            signal: Raw respiratory signal data
            
        Returns:
            AnalysisResult object if analysis successful, None otherwise
        """
        if len(signal) < self.min_analysis_length:
            logging.debug(f"Insufficient data for analysis: {len(signal)} < {self.min_analysis_length}")
            return None
        
        try:
            # Apply bandpass filtering to isolate respiratory frequencies
            filtered_signal = self._filter_signal(signal)
            
            if len(filtered_signal) == 0:
                logging.warning("Filtering produced empty signal")
                return None
            
            # Calculate signal quality before frequency analysis
            signal_quality = self._calculate_signal_quality(filtered_signal)
            
            # Find dominant frequency in respiratory range
            dominant_freq = self._find_respiratory_frequency(filtered_signal)
            
            # Convert frequency to breaths per minute
            respiration_rate = dominant_freq * 60
            
            # Calculate confidence based on signal characteristics
            confidence = self._calculate_confidence(filtered_signal, dominant_freq, signal_quality)
            
            # Validate result
            if confidence < self.confidence_threshold:
                logging.debug(f"Low confidence result: {confidence:.3f} < {self.confidence_threshold}")
                return None
            
            result = AnalysisResult(
                respiration_rate=respiration_rate,
                dominant_frequency=dominant_freq,
                signal_quality=signal_quality,
                confidence=confidence,
                filtered_signal=filtered_signal
            )
            
            logging.info(f"Respiration analysis: {respiration_rate:.1f} BPM (conf: {confidence:.3f})")
            return result
            
        except Exception as e:
            logging.error(f"Respiratory analysis error: {e}")
            return None
    
    def _filter_signal(self, signal: List[float]) -> List[float]:
        """
        Apply bandpass filter to isolate respiratory frequencies.
        
        Args:
            signal: Raw signal data
            
        Returns:
            Filtered signal data
        """
        try:
            filtered = apply_bandpass_filter(
                signal, 
                self.config.lowcut, 
                self.config.highcut, 
                fs=self.config.fs
            )
            return filtered
            
        except Exception as e:
            logging.error(f"Signal filtering error: {e}")
            return []
    
    def _find_respiratory_frequency(self, signal: List[float]) -> float:
        """
        Find dominant frequency in respiratory range.
        
        Args:
            signal: Filtered signal data
            
        Returns:
            Dominant frequency in Hz
        """
        try:
            freq = find_dominant_frequency(
                signal, 
                self.config.fs,
                freq_range=(self.config.lowcut, self.config.highcut)
            )
            return freq
            
        except Exception as e:
            logging.error(f"Frequency analysis error: {e}")
            return 0.0
    
    def _calculate_signal_quality(self, signal: List[float]) -> float:
        """
        Calculate signal quality metric.
        
        Args:
            signal: Signal data for quality assessment
            
        Returns:
            Quality score (0-1, higher is better)
        """
        if len(signal) < 10:
            return 0.0
        
        # Calculate signal-to-noise ratio approximation
        signal_array = np.array(signal)
        
        # Signal power (variance of the signal)
        signal_power = np.var(signal_array)
        
        # Estimate noise as high-frequency components
        # Use difference between consecutive samples as noise estimate
        noise_estimate = np.var(np.diff(signal_array))
        
        # Calculate SNR
        if noise_estimate > 0:
            snr = signal_power / noise_estimate
            # Normalize SNR to 0-1 range
            quality = min(1.0, snr / 10.0)  # SNR of 10 = quality of 1.0
        else:
            quality = 0.5  # Default if noise can't be estimated
        
        return quality
    
    def _calculate_confidence(self, signal: List[float], dominant_freq: float, 
                            signal_quality: float) -> float:
        """
        Calculate confidence in the analysis result.
        
        Args:
            signal: Filtered signal data
            dominant_freq: Calculated dominant frequency
            signal_quality: Signal quality score
            
        Returns:
            Confidence score (0-1)
        """
        # Start with signal quality as base confidence
        confidence = signal_quality
        
        # Penalize frequencies outside reasonable respiratory range
        if dominant_freq < 0.15 or dominant_freq > 0.4:  # 9-24 BPM range
            confidence *= 0.5
        
        # Reward frequencies in typical respiratory range
        if 0.2 <= dominant_freq <= 0.33:  # 12-20 BPM (normal range)
            confidence *= 1.2
        
        # Consider signal length (longer signals are more reliable)
        length_factor = min(1.0, len(signal) / 100.0)  # Full confidence at 100+ samples
        confidence *= length_factor
        
        # Ensure confidence is in valid range
        return np.clip(confidence, 0.0, 1.0)
    
    def update_filter_config(self, lowcut: float, highcut: float) -> None:
        """
        Update filter configuration parameters.
        
        Args:
            lowcut: New low frequency cutoff in Hz
            highcut: New high frequency cutoff in Hz
            
        Raises:
            ValueError: If parameters are invalid
        """
        # Validate new parameters
        if lowcut >= highcut:
            raise ValueError("Low cutoff must be less than high cutoff")
        
        if lowcut < 0.05 or highcut > 1.0:
            raise ValueError("Cutoff frequencies must be in range [0.05, 1.0] Hz")
        
        # Update configuration
        self.config.lowcut = lowcut
        self.config.highcut = highcut
        
        logging.info(f"Updated filter parameters: {lowcut:.2f} - {highcut:.2f} Hz")
    
    def get_filter_config(self) -> FilterConfig:
        """
        Get current filter configuration.
        
        Returns:
            Current FilterConfig object
        """
        return self.config
    
    def validate_signal(self, signal: List[float]) -> Tuple[bool, str]:
        """
        Validate signal data for analysis.
        
        Args:
            signal: Signal data to validate
            
        Returns:
            Tuple of (is_valid, validation_message)
        """
        if len(signal) == 0:
            return False, "Empty signal"
        
        if len(signal) < self.min_analysis_length:
            return False, f"Insufficient data: {len(signal)} < {self.min_analysis_length}"
        
        # Check for valid numeric data
        try:
            signal_array = np.array(signal)
            if np.any(np.isnan(signal_array)) or np.any(np.isinf(signal_array)):
                return False, "Signal contains invalid values (NaN or Inf)"
        except Exception:
            return False, "Signal contains non-numeric data"
        
        # Check signal variance (too flat signals are problematic)
        if np.var(signal_array) < 1e-6:
            return False, "Signal has insufficient variation"
        
        return True, "Signal is valid for analysis"
    
    def get_analysis_parameters(self) -> dict:
        """
        Get current analysis parameters.
        
        Returns:
            Dictionary containing analysis configuration
        """
        return {
            'filter_lowcut': self.config.lowcut,
            'filter_highcut': self.config.highcut,
            'sampling_frequency': self.config.fs,
            'min_analysis_length': self.min_analysis_length,
            'confidence_threshold': self.confidence_threshold
        }
