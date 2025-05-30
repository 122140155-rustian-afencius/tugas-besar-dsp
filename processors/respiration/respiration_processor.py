"""
Respiration rate processor using pose landmark analysis.

This module implements respiration rate detection by analyzing the vertical
movement of shoulder landmarks, which reflect chest expansion and contraction
during breathing cycles.
"""

import numpy as np
import mediapipe as mp
import cv2
import logging
from collections import deque
from typing import Optional, Tuple

from utils.signal_utils import (
    apply_bandpass_filter,
    find_dominant_frequency
)


class RespirationProcessor:
    """
    Real-time respiration processor using pose landmarks for breathing analysis.
    
    This class detects respiration rate by tracking the vertical movement of
    shoulder landmarks, which correlates with chest movement during breathing.
    """
    
    def __init__(self, fps: int = 30):
        """
        Initialize the respiration processor.
        
        Args:
            fps: Video frame rate in frames per second
        """
        self.fps = fps
        
        # Initialize pose detection system
        self._setup_pose_detection()
        
        # Initialize signal buffers for respiratory signal storage
        self._initialize_signal_buffers()
        
        # Configure filter parameters for respiration frequency range
        self._initialize_filter_parameters()
        
        # Initialize respiration rate calculation
        self.current_rr = 0
        self.frame_idx = 0
    
    def _setup_pose_detection(self) -> None:
        """
        Initialize MediaPipe pose detection with fallback options.
        
        Attempts to use the advanced pose landmarker if available,
        falls back to basic pose estimation for compatibility.
        """
        try:
            # Try to use advanced MediaPipe Tasks pose landmarker
            model_path = "./models/pose_landmarker_full.task"
            from mediapipe.tasks.python import BaseOptions
            from mediapipe.tasks.python.vision import (
                PoseLandmarker, PoseLandmarkerOptions, RunningMode
            )
            
            options = PoseLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=model_path),
                running_mode=RunningMode.VIDEO,
                num_poses=1
            )
            self.landmarker = PoseLandmarker.create_from_options(options)
            self.pose_available = True
            logging.info("Using advanced pose landmarker")
            
        except Exception as e:
            # Fallback to basic pose estimation
            logging.info(f"Advanced pose landmarker not available ({e}), using basic pose")
            self.mp_pose = mp.solutions.pose
            self.pose = self.mp_pose.Pose(
                static_image_mode=False,
                model_complexity=1,
                smooth_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            self.pose_available = True
            self.landmarker = None
    
    def _initialize_signal_buffers(self) -> None:
        """Initialize signal storage buffers for respiratory data."""
        # Store up to 1 minute of data for analysis
        self.max_buffer_size = self.fps * 60
        self.raw_y_buffer = deque(maxlen=self.max_buffer_size)
        self.filtered_y_buffer = deque(maxlen=self.max_buffer_size)
        self.time_buffer = deque(maxlen=self.max_buffer_size)
    
    def _initialize_filter_parameters(self) -> None:
        """Initialize bandpass filter parameters for respiration frequencies."""
        # Normal respiration rate: 12-20 breaths per minute (0.2-0.33 Hz)
        # Allow wider range to capture individual variations
        self.lowcut = 0.1   # Hz (6 breaths per minute)
        self.highcut = 0.5  # Hz (30 breaths per minute)
    
    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, bool, float]:
        """
        Process a single video frame for respiration signal extraction.
        
        Args:
            frame: Input video frame in BGR format
            
        Returns:
            Tuple of (processed_frame, pose_detected, current_respiration_rate)
        """
        processed_frame = frame.copy()
        pose_detected = False
        
        h, w = frame.shape[:2]
        
        if self.landmarker:
            # Use advanced MediaPipe Tasks landmarker
            pose_detected, processed_frame = self._process_with_landmarker(
                frame, processed_frame, h, w
            )
        else:
            # Use basic pose estimation
            pose_detected, processed_frame = self._process_with_basic_pose(
                frame, processed_frame, h, w
            )
        
        self.frame_idx += 1
        return processed_frame, pose_detected, self.current_rr
    
    def _process_with_landmarker(self, frame: np.ndarray, processed_frame: np.ndarray,
                               h: int, w: int) -> Tuple[bool, np.ndarray]:
        """
        Process frame using advanced MediaPipe Tasks pose landmarker.
        
        Args:
            frame: Original input frame
            processed_frame: Frame copy for drawing
            h: Frame height
            w: Frame width
            
        Returns:
            Tuple of (pose_detected, processed_frame_with_annotations)
        """
        pose_detected = False
        
        # Convert to RGB and create MediaPipe image
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        
        # Calculate timestamp for video processing
        timestamp_ms = int((self.frame_idx / self.fps) * 1000)
        
        try:
            result = self.landmarker.detect_for_video(mp_image, timestamp_ms)
            
            if result.pose_landmarks and len(result.pose_landmarks) > 0:
                landmarks = result.pose_landmarks[0]
                
                # Extract shoulder landmarks (indices 11 and 12)
                l_sh = landmarks[11]  # Left shoulder
                r_sh = landmarks[12]  # Right shoulder
                
                # Calculate shoulder positions and average Y coordinate
                ly = int(l_sh.y * h)
                ry = int(r_sh.y * h)
                avg_y = (ly + ry) / 2
                
                # Draw visualization on frame
                lx, rx = int(l_sh.x * w), int(r_sh.x * w)
                self._draw_pose_landmarks(processed_frame, lx, ly, rx, ry)
                
                pose_detected = True
                
                # Store respiratory signal data
                self._store_respiratory_data(avg_y)
                
        except Exception as e:
            logging.error(f"Pose landmarker processing error: {e}")
        
        return pose_detected, processed_frame
    
    def _process_with_basic_pose(self, frame: np.ndarray, processed_frame: np.ndarray,
                               h: int, w: int) -> Tuple[bool, np.ndarray]:
        """
        Process frame using basic MediaPipe pose estimation.
        
        Args:
            frame: Original input frame
            processed_frame: Frame copy for drawing
            h: Frame height
            w: Frame width
            
        Returns:
            Tuple of (pose_detected, processed_frame_with_annotations)
        """
        pose_detected = False
        
        # Convert to RGB for MediaPipe processing
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(frame_rgb)
        
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            
            # Extract shoulder landmarks
            l_sh = landmarks[11]  # Left shoulder
            r_sh = landmarks[12]  # Right shoulder
            
            # Calculate shoulder positions and average Y coordinate
            ly = int(l_sh.y * h)
            ry = int(r_sh.y * h)
            avg_y = (ly + ry) / 2
            
            # Draw visualization on frame
            lx, rx = int(l_sh.x * w), int(r_sh.x * w)
            self._draw_pose_landmarks(processed_frame, lx, ly, rx, ry)
            
            pose_detected = True
            
            # Store respiratory signal data
            self._store_respiratory_data(avg_y)
        
        return pose_detected, processed_frame
    
    def _draw_pose_landmarks(self, frame: np.ndarray, lx: int, ly: int, 
                           rx: int, ry: int) -> None:
        """
        Draw pose landmarks and ROI on the frame for visualization.
        
        Args:
            frame: Frame to draw on
            lx, ly: Left shoulder coordinates
            rx, ry: Right shoulder coordinates
        """
        # Draw shoulder points
        cv2.circle(frame, (lx, ly), 4, (255, 0, 0), -1)
        cv2.circle(frame, (rx, ry), 4, (255, 0, 0), -1)
        
        # Draw line connecting shoulders
        cv2.line(frame, (lx, ly), (rx, ry), (0, 255, 255), 2)
        
        # Add label
        cv2.putText(frame, "Pose ROI", (lx, ly-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    
    def _store_respiratory_data(self, avg_y: float) -> None:
        """
        Store respiratory signal data and trigger analysis if sufficient data available.
        
        Args:
            avg_y: Average Y coordinate of shoulder landmarks
        """
        # Add to signal buffers
        self.raw_y_buffer.append(avg_y)
        self.time_buffer.append(self.frame_idx / self.fps)
        
        # Perform analysis if sufficient data is available
        if len(self.raw_y_buffer) >= 30:
            self._calculate_respiration_rate()
    
    def _calculate_respiration_rate(self) -> None:
        """
        Calculate respiration rate from shoulder movement using frequency analysis.
        
        The method applies bandpass filtering to isolate respiratory frequencies
        and uses FFT to find the dominant breathing frequency.
        """
        # Validate filter parameters
        if self.lowcut >= self.highcut:
            logging.warning("Invalid filter parameters: lowcut >= highcut")
            return
        
        try:
            # Apply bandpass filter to isolate respiratory frequencies
            filtered_signal = apply_bandpass_filter(
                list(self.raw_y_buffer), 
                self.lowcut, 
                self.highcut, 
                fs=self.fps
            )
            
            # Store filtered signal for visualization
            if len(filtered_signal) > 0:
                self.filtered_y_buffer.append(filtered_signal[-1])
            
            # Calculate dominant frequency and convert to breaths per minute
            dominant_freq = find_dominant_frequency(
                filtered_signal, 
                self.fps,
                freq_range=(self.lowcut, self.highcut)
            )
            
            # Convert frequency to breaths per minute
            self.current_rr = dominant_freq * 60
            
            # Log the result for monitoring
            if self.current_rr > 0:
                logging.info(f"Respiration Rate: {self.current_rr:.1f} BPM")
                
        except Exception as e:
            logging.error(f"Respiration calculation error: {e}")
    
    def update_filter_params(self, lowcut: float, highcut: float) -> None:
        """
        Update bandpass filter parameters for respiration analysis.
        
        Args:
            lowcut: Low frequency cutoff in Hz
            highcut: High frequency cutoff in Hz
            
        Raises:
            ValueError: If parameters are invalid
        """
        if lowcut >= highcut:
            raise ValueError("Low cutoff must be less than high cutoff")
        
        if lowcut < 0.05 or highcut > 1.0:
            raise ValueError("Cutoff frequencies must be in range [0.05, 1.0] Hz")
        
        self.lowcut = lowcut
        self.highcut = highcut
        logging.info(f"Updated filter parameters: {lowcut:.2f} - {highcut:.2f} Hz")
    
    def reset_signals(self) -> None:
        """Reset all signal buffers and respiration rate estimates."""
        self.raw_y_buffer.clear()
        self.filtered_y_buffer.clear()
        self.time_buffer.clear()
        self.current_rr = 0
        self.frame_idx = 0
        logging.info("Respiration signals reset")
    
    def get_signal_quality(self) -> float:
        """
        Calculate and return current signal quality metric.
        
        Returns:
            Signal quality score (0-1, higher is better)
        """
        if len(self.raw_y_buffer) < 10:
            return 0.0
        
        # Calculate signal variability as quality indicator
        recent_data = list(self.raw_y_buffer)[-30:]  # Last 30 samples
        signal_std = np.std(recent_data)
        signal_range = np.max(recent_data) - np.min(recent_data)
        
        # Normalize quality score (higher movement = better signal)
        quality = min(1.0, signal_range / 50.0)  # Assuming 50 pixels max movement
        
        return quality
