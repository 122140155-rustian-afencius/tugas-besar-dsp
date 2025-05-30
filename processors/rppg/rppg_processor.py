"""
Remote Photoplethysmography (rPPG) processor for heart rate detection.

This module implements the Plane-Orthogonal-to-Skin (POS) method for
extracting heart rate signals from facial video using color variations
caused by blood volume changes.
"""

import numpy as np
import mediapipe as mp
import cv2
import time
from collections import deque
from typing import Optional, Tuple, List

from utils.signal_utils import (
    apply_bandpass_filter, 
    apply_savgol_filter,
    wavelet_denoise,
    normalize_signal,
    detect_peaks_with_validation,
    calculate_signal_quality,
    smooth_signal_exponential
)


class RPPGProcessor:
    """
    Real-time rPPG processor using POS method for heart rate detection.
    
    This class extracts heart rate signals from facial video by analyzing
    subtle color changes in the forehead region caused by blood flow variations.
    """
    
    def __init__(self, fps: int = 30, window_length: float = 1.6):
        """
        Initialize the rPPG processor.
        
        Args:
            fps: Video frame rate in frames per second
            window_length: Analysis window length in seconds
        """
        self.fps = fps
        self.window_length = window_length
        self.window_size = int(window_length * fps)
        
        # Initialize MediaPipe face detection
        self._setup_face_detection()
        
        # Initialize signal buffers with maximum capacity for memory efficiency
        self.max_buffer_size = fps * 30  # 30 seconds of data
        self._initialize_signal_buffers()
        
        # Configure filter parameters for heart rate frequency range
        self.lowcut = 0.7   # Hz (42 BPM)
        self.highcut = 2.5  # Hz (150 BPM)
        self.filter_order = 3
        
        # Initialize heart rate calculation parameters
        self._initialize_hr_parameters()
        
        # Configure signal processing parameters
        self._initialize_processing_parameters()
    
    def _setup_face_detection(self) -> None:
        """Initialize MediaPipe face detection with optimized settings."""
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=1,  # Full-range model for better accuracy
            min_detection_confidence=0.5
        )
    
    def _initialize_signal_buffers(self) -> None:
        """Initialize all signal storage buffers."""
        self.r_signal = deque(maxlen=self.max_buffer_size)
        self.g_signal = deque(maxlen=self.max_buffer_size)
        self.b_signal = deque(maxlen=self.max_buffer_size)
        self.rppg_signal = deque(maxlen=self.max_buffer_size)
        self.filtered_rppg = deque(maxlen=self.max_buffer_size)
        self.timestamps = deque(maxlen=self.max_buffer_size)
    
    def _initialize_hr_parameters(self) -> None:
        """Initialize heart rate calculation parameters."""
        self.current_hr = 0
        self.hr_history = deque(maxlen=15)  # Store recent HR estimates
        self.hr_timestamps = deque(maxlen=15)
        self.last_valid_hr_time = 0
    
    def _initialize_processing_parameters(self) -> None:
        """Initialize signal processing and filtering parameters."""
        # Smoothing parameters
        self.smooth_window_size = 9
        self.wavelet_name = 'sym4'
        self.wavelet_level = 3
        
        # Peak detection parameters
        self.min_peak_distance = int(self.fps * 0.5)  # Minimum 0.5s between peaks
        self.peak_prominence = 0.3
        self.min_signal_quality = 1.2
        
        # Timing control
        self.force_recalc_interval = 1.0  # Force recalculation every second
        self.last_calculation_time = 0
        self.prev_filtered_value = 0
        self.last_filter_time = 0
    
    def cpu_pos(self, signal_array: np.ndarray) -> np.ndarray:
        """
        Implement the Plane-Orthogonal-to-Skin (POS) method for rPPG extraction.
        
        The POS method projects RGB signals onto a plane orthogonal to the skin-tone
        direction to minimize motion artifacts and enhance pulse signal.
        
        Args:
            signal_array: RGB signal array of shape (batch_size, 3, time_samples)
            
        Returns:
            Extracted rPPG signal array
            
        Reference:
            Wang, W., et al. "Algorithmic principles of remote PPG." 
            IEEE Trans. Biomed. Eng. 64.7 (2017): 1479-1491.
        """
        eps = 1e-9  # Small constant to prevent division by zero
        X = signal_array
        e, c, f = X.shape  # batch_size, channels, frames
        w = self.window_size
        
        # Return zeros if insufficient data
        if f < w:
            return np.zeros(f)
        
        # POS projection matrix: projects RGB to two orthogonal directions
        P = np.array([[0, 1, -1], [-2, 1, 1]])
        Q = np.stack([P for _ in range(e)], axis=0)
        H = np.zeros((e, f))
        
        # Sliding window processing
        for n in range(w, f):
            m = n - w + 1
            # Extract current window
            Cn = X[:, :, m:(n + 1)]
            
            # Normalize by temporal mean to reduce illumination effects
            M = 1.0 / (np.mean(Cn, axis=2) + eps)
            M = np.expand_dims(M, axis=2)
            Cn = np.multiply(M, Cn)
            
            # Apply POS projection
            S = np.dot(Q, Cn)
            S = S[0, :, :, :]
            S = np.swapaxes(S, 0, 1)
            
            # Extract two projected signals
            S1 = S[:, 0, :]
            S2 = S[:, 1, :]
            
            # Calculate adaptive weights based on signal variance
            alpha = np.std(S1, axis=1) / (eps + np.std(S2, axis=1))
            alpha = np.expand_dims(alpha, axis=1)
            
            # Combine signals with adaptive weighting
            Hn = np.add(S1, alpha * S2)
            
            # Remove temporal mean
            Hnm = Hn - np.expand_dims(np.mean(Hn, axis=1), axis=1)
            
            # Add to output with overlap-add
            H[:, m:(n + 1)] = np.add(H[:, m:(n + 1)], Hnm)
        
        return H[0, :]
    
    def extract_face_roi(self, frame: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[Tuple[int, int, int, int]]]:
        """
        Extract forehead region of interest using MediaPipe face detection.
        
        The forehead region is selected as it typically has good blood perfusion
        and is less affected by facial expressions and movements.
        
        Args:
            frame: Input video frame in BGR format
            
        Returns:
            Tuple of (roi_image, bbox_coordinates) or (None, None) if no face detected
        """
        # Convert to RGB for MediaPipe processing
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_detection.process(frame_rgb)
        
        if not results.detections:
            return None, None
        
        # Use the most confident detection
        detection = results.detections[0]
        bbox = detection.location_data.relative_bounding_box
        h, w, _ = frame.shape
        
        # Convert relative coordinates to absolute pixels
        x = int(bbox.xmin * w)
        y = int(bbox.ymin * h)
        width = int(bbox.width * w)
        height = int(bbox.height * h)
        
        # Define forehead region (upper 20% of face, center 80% width)
        forehead_height = int(height * 0.2)
        forehead_width = int(width * 0.8)
        forehead_x = x + (width - forehead_width) // 2
        forehead_y = y
        
        # Ensure coordinates are within frame boundaries
        forehead_x = max(0, min(forehead_x, w - forehead_width))
        forehead_y = max(0, min(forehead_y, h - forehead_height))
        forehead_width = min(forehead_width, w - forehead_x)
        forehead_height = min(forehead_height, h - forehead_y)
        
        # Extract ROI
        roi = frame[forehead_y:forehead_y+forehead_height, 
                   forehead_x:forehead_x+forehead_width]
        bbox_coords = (forehead_x, forehead_y, forehead_width, forehead_height)
        
        return roi, bbox_coords
    
    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, bool, float]:
        """
        Process a single video frame for rPPG signal extraction.
        
        Args:
            frame: Input video frame in BGR format
            
        Returns:
            Tuple of (processed_frame, face_detected, current_heart_rate)
        """
        processed_frame = frame.copy()
        face_detected = False
        
        # Extract face region of interest
        roi, bbox_coords = self.extract_face_roi(frame)
        
        if roi is not None and roi.size > 0:
            face_detected = True
            x, y, width, height = bbox_coords
            
            # Draw ROI visualization on frame
            cv2.rectangle(processed_frame, (x, y), (x + width, y + height), (0, 255, 0), 2)
            cv2.putText(processed_frame, "Face ROI", (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            # Extract mean RGB values from ROI
            r_mean = np.mean(roi[:, :, 2])  # Red channel
            g_mean = np.mean(roi[:, :, 1])  # Green channel  
            b_mean = np.mean(roi[:, :, 0])  # Blue channel
            
            # Store signals with timestamp
            self.r_signal.append(r_mean)
            self.g_signal.append(g_mean)
            self.b_signal.append(b_mean)
            self.timestamps.append(time.time())
            
            # Process signals if sufficient data available
            if len(self.r_signal) >= self.window_size:
                self._update_rppg_signal()
                self._calculate_heart_rate()
        
        return processed_frame, face_detected, self.current_hr
    
    def _update_rppg_signal(self) -> None:
        """Update rPPG signal using the POS method."""
        if len(self.r_signal) < self.window_size:
            return
        
        # Prepare RGB array for POS processing
        rgb_array = np.array([
            list(self.r_signal),
            list(self.g_signal),
            list(self.b_signal)
        ])
        
        # Reshape for POS method: (batch_size=1, channels=3, time_samples)
        rgb_array = rgb_array.reshape(1, 3, -1)
        
        # Extract rPPG signal using POS method
        rppg = self.cpu_pos(rgb_array)
        
        # Store the latest rPPG value
        if len(rppg) > 0:
            self.rppg_signal.append(rppg[-1])
            
            # Apply filtering if sufficient data available
            if len(self.rppg_signal) >= self.window_size:
                self._apply_filter()
    
    def _apply_filter(self) -> None:
        """
        Apply comprehensive filtering pipeline to rPPG signal.
        
        The filtering pipeline includes:
        1. Bandpass filtering for heart rate frequencies
        2. Wavelet denoising for artifact removal
        3. Savitzky-Golay smoothing for noise reduction
        4. Exponential smoothing for temporal consistency
        """
        if len(self.rppg_signal) < self.window_size:
            return
        
        # Rate limiting: don't filter too frequently
        current_time = time.time()
        if current_time - self.last_filter_time < 0.033 and len(self.filtered_rppg) > 0:
            return
        
        self.last_filter_time = current_time
        
        # Use recent data for analysis (up to 4 seconds)
        analysis_window = min(int(self.fps * 4), len(self.rppg_signal))
        
        if analysis_window < 10:
            return
            
        recent_samples = list(self.rppg_signal)[-analysis_window:]
        
        try:
            # Apply bandpass filter for heart rate frequencies
            filtered_samples = apply_bandpass_filter(
                recent_samples, self.lowcut, self.highcut, self.fps, self.filter_order
            )
            
            # Apply wavelet denoising if sufficient data
            if len(filtered_samples) > 2**self.wavelet_level + 2:
                filtered_samples = wavelet_denoise(
                    filtered_samples, self.wavelet_name, self.wavelet_level
                )
            
            # Apply Savitzky-Golay smoothing
            window_length = min(15, len(filtered_samples) - 2)
            if window_length > 3:
                filtered_samples = apply_savgol_filter(filtered_samples, window_length, 2)
            
            # Calculate adaptive smoothing factor based on signal quality
            signal_var = np.var(filtered_samples)
            alpha = min(0.5, max(0.1, 0.3 / (1 + signal_var)))
            
            # Apply exponential smoothing
            latest_value = filtered_samples[-1]
            
            if len(self.filtered_rppg) > 0:
                smoothed_value = smooth_signal_exponential(
                    latest_value, self.prev_filtered_value, alpha
                )
            else:
                smoothed_value = latest_value
            
            self.prev_filtered_value = smoothed_value
            self.filtered_rppg.append(smoothed_value)
            
        except Exception as e:
            print(f"Filtering error: {e}")
            # Fallback: use raw signal if filtering fails
            if len(self.rppg_signal) > 0:
                self.filtered_rppg.append(self.rppg_signal[-1])
    
    def _calculate_heart_rate(self) -> None:
        """
        Calculate heart rate from filtered rPPG signal using peak detection.
        
        The method uses adaptive peak detection with validation to ensure
        physiologically plausible heart rate estimates.
        """
        if len(self.filtered_rppg) < self.window_size:
            return
        
        # Rate limiting for heart rate calculation
        current_time = time.time()
        time_since_last = current_time - self.last_calculation_time
        if time_since_last < 0.2 and time_since_last < self.force_recalc_interval:
            return
        
        self.last_calculation_time = current_time
        
        # Use medium-term window for stable HR estimation (up to 5 seconds)
        medium_window = min(int(self.fps * 5), len(self.filtered_rppg))
        
        if medium_window < self.fps * 1.5:  # Need at least 1.5 seconds
            return
        
        medium_segment = list(self.filtered_rppg)[-medium_window:]
        
        try:
            # Normalize signal for consistent peak detection
            norm_signal = normalize_signal(medium_segment)
            
            # Adaptive prominence based on signal quality
            signal_quality = calculate_signal_quality(norm_signal)
            
            if signal_quality > 2.5:
                prominence = 0.35
            elif signal_quality < 1.5:
                prominence = 0.15
            else:
                prominence = 0.2
            
            # Detect peaks with validation
            peaks, validated_rates = detect_peaks_with_validation(
                norm_signal, self.fps,
                min_distance_sec=0.5,
                prominence=prominence,
                min_width_sec=0.08
            )
            
            # Calculate heart rate from validated peak intervals
            if validated_rates:
                final_hr = np.median(validated_rates)
                self.hr_history.append(final_hr)
                self.hr_timestamps.append(current_time)
                self.last_valid_hr_time = current_time
                
                # Weighted average of recent HR estimates
                if len(self.hr_history) >= 3:
                    weights = np.linspace(0.5, 1.0, len(self.hr_history))
                    self.current_hr = np.average(self.hr_history, weights=weights)
                else:
                    self.current_hr = final_hr
        
        except Exception as e:
            print(f"Heart rate calculation error: {str(e)}")
    
    def reset_signals(self) -> None:
        """Reset all signal buffers and heart rate estimates."""
        self.r_signal.clear()
        self.g_signal.clear()
        self.b_signal.clear()
        self.rppg_signal.clear()
        self.filtered_rppg.clear()
        self.timestamps.clear()
        self.hr_history.clear()
        self.hr_timestamps.clear()
        self.current_hr = 0
        self.last_valid_hr_time = 0
        self.prev_filtered_value = 0
