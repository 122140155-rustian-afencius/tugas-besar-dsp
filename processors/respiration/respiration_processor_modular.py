"""
Refactored respiration processor using modular components.

This module provides a simplified RespirationProcessor that coordinates
smaller, focused components for pose detection, signal management,
analysis, and visualization.
"""

import numpy as np
import logging
from typing import Optional, Tuple

from utils.pose_detector import PoseDetectionHandler, PoseLandmarks
from utils.signal_buffer import SignalBufferManager, BufferConfig
from processors.respiration.respiratory_analyzer import RespiratorySignalAnalyzer, FilterConfig, AnalysisResult
from utils.visualization_helper import VisualizationHelper, VisualizationConfig


class RespirationProcessorModular:
    """
    Modular respiration processor using coordinated components.
    
    This refactored version delegates specific responsibilities to focused
    components, making the code more maintainable and testable.
    """
    
    def __init__(self, fps: int = 30, model_path: Optional[str] = None):
        """
        Initialize the modular respiration processor.
        
        Args:
            fps: Video frame rate in frames per second
            model_path: Optional path to pose detection model
        """
        self.fps = fps
        
        # Initialize all component modules
        self._initialize_components(model_path)
        
        # Track current state
        self.current_rr = 0
        self.frame_idx = 0
        self.last_analysis_result: Optional[AnalysisResult] = None
        
        logging.info("Modular respiration processor initialized")
    
    def _initialize_components(self, model_path: Optional[str]) -> None:
        """
        Initialize all component modules.
        
        Args:
            model_path: Path to pose detection model
        """
        # Configure components
        buffer_config = BufferConfig(
            max_size=self.fps * 60,  # 1 minute of data
            min_analysis_size=30,    # Minimum for analysis
            fps=float(self.fps)
        )
        
        filter_config = FilterConfig(
            lowcut=0.1,   # 6 breaths per minute
            highcut=0.5,  # 30 breaths per minute
            fs=float(self.fps)
        )
        
        visualization_config = VisualizationConfig()
        
        # Initialize components
        self.pose_detector = PoseDetectionHandler(model_path)
        self.signal_buffer = SignalBufferManager(buffer_config)
        self.signal_analyzer = RespiratorySignalAnalyzer(filter_config)
        self.visualizer = VisualizationHelper(visualization_config)
        
        logging.info("All components initialized successfully")
    
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
        
        # Detect pose landmarks
        landmarks = self.pose_detector.detect_pose(frame, self.frame_idx, self.fps)
        
        if landmarks:
            pose_detected = True
            
            # Draw pose visualization
            processed_frame = self.visualizer.draw_pose_landmarks(processed_frame, landmarks)
            
            # Store respiratory signal data
            quality = self._calculate_landmark_quality(landmarks)
            self.signal_buffer.add_sample(landmarks.average_y, quality)
            
            # Perform analysis if sufficient data available
            if self.signal_buffer.has_sufficient_data():
                self._analyze_respiratory_signal()
        
        # Add respiration rate display
        signal_quality = self.signal_buffer.get_overall_quality()
        processed_frame = self.visualizer.draw_respiration_rate(
            processed_frame, self.current_rr, signal_quality
        )
        
        # Add signal overlay if available
        if self.last_analysis_result and len(self.last_analysis_result.filtered_signal) > 0:
            processed_frame = self.visualizer.create_signal_overlay(
                processed_frame, self.last_analysis_result.filtered_signal
            )
        
        self.frame_idx += 1
        return processed_frame, pose_detected, self.current_rr
    
    def _calculate_landmark_quality(self, landmarks: PoseLandmarks) -> float:
        """
        Calculate quality score for detected landmarks.
        
        Args:
            landmarks: Detected pose landmarks
            
        Returns:
            Quality score (0-1)
        """
        # Base quality from landmark confidence
        base_quality = landmarks.confidence
        
        # Check shoulder distance (should be reasonable)
        shoulder_distance = abs(landmarks.left_shoulder_x - landmarks.right_shoulder_x)
        if 50 <= shoulder_distance <= 300:  # Reasonable shoulder width in pixels
            distance_quality = 1.0
        else:
            distance_quality = 0.5
        
        # Combine quality factors
        overall_quality = (base_quality + distance_quality) / 2
        
        return np.clip(overall_quality, 0.0, 1.0)
    
    def _analyze_respiratory_signal(self) -> None:
        """Analyze respiratory signal using the signal analyzer component."""
        # Get recent signal data
        signal_data = self.signal_buffer.get_raw_signal()
        
        # Perform analysis
        result = self.signal_analyzer.analyze_signal(signal_data)
        
        if result:
            self.last_analysis_result = result
            self.current_rr = result.respiration_rate
            
            # Store filtered signal for visualization
            if result.filtered_signal:
                self.signal_buffer.add_filtered_sample(result.filtered_signal[-1])
            
            logging.info(f"Respiration Rate: {self.current_rr:.1f} BPM (confidence: {result.confidence:.3f})")
        else:
            logging.debug("Respiratory analysis failed or low confidence")
    
    def update_filter_params(self, lowcut: float, highcut: float) -> None:
        """
        Update bandpass filter parameters.
        
        Args:
            lowcut: Low frequency cutoff in Hz
            highcut: High frequency cutoff in Hz
        """
        self.signal_analyzer.update_filter_config(lowcut, highcut)
        logging.info(f"Filter parameters updated: {lowcut:.2f} - {highcut:.2f} Hz")
    
    def reset_signals(self) -> None:
        """Reset all signal buffers and analysis state."""
        self.signal_buffer.clear_buffers()
        self.current_rr = 0
        self.frame_idx = 0
        self.last_analysis_result = None
        logging.info("Respiration signals reset")
    
    def get_signal_quality(self) -> float:
        """
        Get current signal quality metric.
        
        Returns:
            Signal quality score (0-1, higher is better)
        """
        return self.signal_buffer.get_overall_quality()
    
    def get_buffer_statistics(self) -> dict:
        """
        Get comprehensive buffer and analysis statistics.
        
        Returns:
            Dictionary containing statistics
        """
        buffer_stats = self.signal_buffer.get_buffer_statistics()
        analysis_params = self.signal_analyzer.get_analysis_parameters()
        
        stats = {
            **buffer_stats,
            'analysis_parameters': analysis_params,
            'current_respiration_rate': self.current_rr,
            'pose_detection_available': self.pose_detector.is_available(),
            'frame_count': self.frame_idx
        }
        
        if self.last_analysis_result:
            stats['last_analysis'] = {
                'respiration_rate': self.last_analysis_result.respiration_rate,
                'dominant_frequency': self.last_analysis_result.dominant_frequency,
                'confidence': self.last_analysis_result.confidence,
                'signal_quality': self.last_analysis_result.signal_quality
            }
        
        return stats
    
    def is_pose_detection_available(self) -> bool:
        """
        Check if pose detection is available.
        
        Returns:
            True if pose detection is working, False otherwise
        """
        return self.pose_detector.is_available()
    
    def get_signal_data(self, num_samples: Optional[int] = None) -> dict:
        """
        Get current signal data for external analysis or visualization.
        
        Args:
            num_samples: Number of recent samples to retrieve
            
        Returns:
            Dictionary containing signal arrays
        """
        signal_data = self.signal_buffer.get_signal_data(num_samples)
        
        return {
            'raw_signal': signal_data.raw_signal,
            'filtered_signal': signal_data.filtered_signal,
            'timestamps': signal_data.timestamps,
            'quality_scores': signal_data.quality_scores
        }
    
    def configure_visualization(self, **kwargs) -> None:
        """
        Configure visualization parameters.
        
        Args:
            **kwargs: Visualization configuration parameters
        """
        self.visualizer.update_config(**kwargs)
        logging.info("Visualization configuration updated")
    
    def validate_current_signal(self) -> Tuple[bool, str]:
        """
        Validate current signal data.
        
        Returns:
            Tuple of (is_valid, validation_message)
        """
        signal_data = self.signal_buffer.get_raw_signal()
        return self.signal_analyzer.validate_signal(signal_data)
