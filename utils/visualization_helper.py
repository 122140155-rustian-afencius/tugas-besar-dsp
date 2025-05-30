"""
Visualization helper for pose landmarks and respiratory data.

This module provides specialized visualization functions for drawing
pose landmarks, signal plots, and respiratory monitoring indicators.
"""

import cv2
import numpy as np
from typing import Tuple, List, Optional
from dataclasses import dataclass

from utils.pose_detector import PoseLandmarks


@dataclass
class VisualizationConfig:
    """
    Configuration for visualization elements.
    
    Attributes:
        landmark_color: Color for pose landmarks (BGR format)
        line_color: Color for connecting lines (BGR format)
        text_color: Color for text labels (BGR format)
        landmark_radius: Radius of landmark circles
        line_thickness: Thickness of connecting lines
        text_font: OpenCV font type
        text_scale: Font scale factor
        text_thickness: Text line thickness
    """
    landmark_color: Tuple[int, int, int] = (255, 0, 0)  # Blue
    line_color: Tuple[int, int, int] = (0, 255, 255)    # Yellow
    text_color: Tuple[int, int, int] = (255, 255, 255)  # White
    landmark_radius: int = 4
    line_thickness: int = 2
    text_font: int = cv2.FONT_HERSHEY_SIMPLEX
    text_scale: float = 0.5
    text_thickness: int = 1


class VisualizationHelper:
    """
    Provides visualization functions for respiratory monitoring.
    
    This class handles all drawing and visualization operations,
    keeping them separate from the main processing logic.
    """
    
    def __init__(self, config: Optional[VisualizationConfig] = None):
        """
        Initialize the visualization helper.
        
        Args:
            config: Visualization configuration (uses default if None)
        """
        self.config = config or VisualizationConfig()
    
    def draw_pose_landmarks(self, frame: np.ndarray, landmarks: PoseLandmarks) -> np.ndarray:
        """
        Draw pose landmarks on the frame.
        
        Args:
            frame: Input frame to draw on
            landmarks: Pose landmarks to visualize
            
        Returns:
            Frame with landmarks drawn
        """
        # Create a copy to avoid modifying the original frame
        annotated_frame = frame.copy()
        
        # Draw shoulder landmarks
        self._draw_landmark(annotated_frame, 
                          int(landmarks.left_shoulder_x), int(landmarks.left_shoulder_y),
                          "L_SHOULDER")
        
        self._draw_landmark(annotated_frame,
                          int(landmarks.right_shoulder_x), int(landmarks.right_shoulder_y), 
                          "R_SHOULDER")
        
        # Draw connecting line between shoulders
        cv2.line(annotated_frame,
                (int(landmarks.left_shoulder_x), int(landmarks.left_shoulder_y)),
                (int(landmarks.right_shoulder_x), int(landmarks.right_shoulder_y)),
                self.config.line_color, self.config.line_thickness)
        
        # Draw ROI indicator
        self._draw_roi_indicator(annotated_frame, landmarks)
        
        # Draw confidence indicator if available
        if landmarks.confidence < 1.0:
            self._draw_confidence_indicator(annotated_frame, landmarks.confidence)
        
        return annotated_frame
    
    def _draw_landmark(self, frame: np.ndarray, x: int, y: int, label: str) -> None:
        """
        Draw a single landmark point with label.
        
        Args:
            frame: Frame to draw on
            x: X coordinate
            y: Y coordinate
            label: Text label for the landmark
        """
        # Draw landmark circle
        cv2.circle(frame, (x, y), self.config.landmark_radius, 
                  self.config.landmark_color, -1)
        
        # Draw label above the landmark
        label_y = max(y - 10, 20)  # Ensure label is visible
        cv2.putText(frame, label, (x - 20, label_y),
                   self.config.text_font, self.config.text_scale,
                   self.config.text_color, self.config.text_thickness)
    
    def _draw_roi_indicator(self, frame: np.ndarray, landmarks: PoseLandmarks) -> None:
        """
        Draw region of interest indicator.
        
        Args:
            frame: Frame to draw on
            landmarks: Pose landmarks for ROI calculation
        """
        # Calculate ROI center
        center_x = int((landmarks.left_shoulder_x + landmarks.right_shoulder_x) / 2)
        center_y = int(landmarks.average_y)
        
        # Draw ROI label
        cv2.putText(frame, "Respiration ROI", (center_x - 50, center_y - 30),
                   self.config.text_font, self.config.text_scale,
                   self.config.text_color, self.config.text_thickness)
    
    def _draw_confidence_indicator(self, frame: np.ndarray, confidence: float) -> None:
        """
        Draw confidence indicator on the frame.
        
        Args:
            frame: Frame to draw on
            confidence: Confidence score (0-1)
        """
        h, w = frame.shape[:2]
        
        # Draw confidence bar in top-right corner
        bar_width = 100
        bar_height = 10
        bar_x = w - bar_width - 10
        bar_y = 10
        
        # Background rectangle
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height),
                     (64, 64, 64), -1)
        
        # Confidence fill
        fill_width = int(confidence * bar_width)
        color = self._get_confidence_color(confidence)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_width, bar_y + bar_height),
                     color, -1)
        
        # Confidence text
        cv2.putText(frame, f"Conf: {confidence:.2f}", (bar_x, bar_y - 5),
                   self.config.text_font, self.config.text_scale,
                   self.config.text_color, self.config.text_thickness)
    
    def _get_confidence_color(self, confidence: float) -> Tuple[int, int, int]:
        """
        Get color based on confidence level.
        
        Args:
            confidence: Confidence score (0-1)
            
        Returns:
            BGR color tuple
        """
        if confidence > 0.8:
            return (0, 255, 0)    # Green
        elif confidence > 0.5:
            return (0, 255, 255)  # Yellow
        else:
            return (0, 0, 255)    # Red
    
    def draw_respiration_rate(self, frame: np.ndarray, respiration_rate: float,
                            signal_quality: float = 1.0) -> np.ndarray:
        """
        Draw respiration rate information on the frame.
        
        Args:
            frame: Input frame to draw on
            respiration_rate: Current respiration rate in BPM
            signal_quality: Signal quality score (0-1)
            
        Returns:
            Frame with respiration rate displayed
        """
        annotated_frame = frame.copy()
        h, w = frame.shape[:2]
        
        # Draw respiration rate
        if respiration_rate > 0:
            rate_text = f"Respiration: {respiration_rate:.1f} BPM"
            quality_color = self._get_quality_color(signal_quality)
        else:
            rate_text = "Respiration: -- BPM"
            quality_color = (128, 128, 128)  # Gray
        
        # Position in bottom-left corner
        text_x = 10
        text_y = h - 30
        
        # Draw background rectangle for better visibility
        text_size = cv2.getTextSize(rate_text, self.config.text_font, 
                                   self.config.text_scale * 1.2, self.config.text_thickness)[0]
        cv2.rectangle(annotated_frame, 
                     (text_x - 5, text_y - text_size[1] - 5),
                     (text_x + text_size[0] + 5, text_y + 5),
                     (0, 0, 0), -1)
        
        # Draw text
        cv2.putText(annotated_frame, rate_text, (text_x, text_y),
                   self.config.text_font, self.config.text_scale * 1.2,
                   quality_color, self.config.text_thickness + 1)
        
        # Draw quality indicator
        self._draw_quality_indicator(annotated_frame, signal_quality, text_x, text_y - 25)
        
        return annotated_frame
    
    def _get_quality_color(self, quality: float) -> Tuple[int, int, int]:
        """
        Get color based on signal quality.
        
        Args:
            quality: Quality score (0-1)
            
        Returns:
            BGR color tuple
        """
        if quality > 0.7:
            return (0, 255, 0)    # Green
        elif quality > 0.4:
            return (0, 255, 255)  # Yellow
        else:
            return (0, 128, 255)  # Orange
    
    def _draw_quality_indicator(self, frame: np.ndarray, quality: float, 
                              x: int, y: int) -> None:
        """
        Draw signal quality indicator.
        
        Args:
            frame: Frame to draw on
            quality: Quality score (0-1)
            x: X position for indicator
            y: Y position for indicator
        """
        quality_text = f"Quality: {quality:.2f}"
        color = self._get_quality_color(quality)
        
        cv2.putText(frame, quality_text, (x, y),
                   self.config.text_font, self.config.text_scale,
                   color, self.config.text_thickness)
    
    def draw_breathing_indicator(self, frame: np.ndarray, breathing_phase: str,
                               intensity: float = 0.5) -> np.ndarray:
        """
        Draw breathing phase indicator (inhale/exhale).
        
        Args:
            frame: Input frame to draw on
            breathing_phase: "INHALE", "EXHALE", or "UNKNOWN"
            intensity: Intensity of the breathing (0-1)
            
        Returns:
            Frame with breathing indicator
        """
        annotated_frame = frame.copy()
        h, w = frame.shape[:2]
        
        # Position in top-left corner
        indicator_x = 10
        indicator_y = 40
        
        # Choose color and text based on phase
        if breathing_phase == "INHALE":
            color = (255, 0, 0)  # Blue
            text = "INHALING"
        elif breathing_phase == "EXHALE":
            color = (0, 0, 255)  # Red
            text = "EXHALING"
        else:
            color = (128, 128, 128)  # Gray
            text = "DETECTING..."
        
        # Adjust color intensity
        color = tuple(int(c * intensity) for c in color)
        
        # Draw text
        cv2.putText(annotated_frame, text, (indicator_x, indicator_y),
                   self.config.text_font, self.config.text_scale * 1.2,
                   color, self.config.text_thickness + 1)
        
        return annotated_frame
    
    def create_signal_overlay(self, frame: np.ndarray, signal_data: List[float],
                            max_points: int = 100) -> np.ndarray:
        """
        Create a signal overlay on the frame.
        
        Args:
            frame: Input frame
            signal_data: Signal values to plot
            max_points: Maximum number of points to display
            
        Returns:
            Frame with signal overlay
        """
        if len(signal_data) < 2:
            return frame
        
        annotated_frame = frame.copy()
        h, w = frame.shape[:2]
        
        # Use recent data for overlay
        recent_data = signal_data[-max_points:] if len(signal_data) > max_points else signal_data
        
        # Normalize signal data to fit in overlay area
        overlay_height = h // 4  # Use bottom quarter of frame
        overlay_y_start = h - overlay_height - 10
        
        signal_array = np.array(recent_data)
        if np.max(signal_array) != np.min(signal_array):
            normalized_signal = ((signal_array - np.min(signal_array)) / 
                               (np.max(signal_array) - np.min(signal_array)))
            y_values = overlay_y_start + normalized_signal * overlay_height
        else:
            y_values = np.full_like(signal_array, overlay_y_start + overlay_height // 2)
        
        # Create x coordinates
        x_values = np.linspace(w - len(recent_data) * 2, w - 10, len(recent_data))
        
        # Draw signal line
        points = list(zip(x_values.astype(int), y_values.astype(int)))
        for i in range(len(points) - 1):
            cv2.line(annotated_frame, points[i], points[i + 1], (0, 255, 0), 1)
        
        # Draw overlay border
        cv2.rectangle(annotated_frame, 
                     (w - max_points * 2 - 20, overlay_y_start - 10),
                     (w - 5, h - 5),
                     (255, 255, 255), 1)
        
        # Add overlay label
        cv2.putText(annotated_frame, "Signal", 
                   (w - max_points * 2 - 15, overlay_y_start - 15),
                   self.config.text_font, self.config.text_scale,
                   self.config.text_color, self.config.text_thickness)
        
        return annotated_frame
    
    def update_config(self, **kwargs) -> None:
        """
        Update visualization configuration.
        
        Args:
            **kwargs: Configuration parameters to update
        """
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
            else:
                raise ValueError(f"Invalid configuration parameter: {key}")
    
    def get_config(self) -> VisualizationConfig:
        """
        Get current visualization configuration.
        
        Returns:
            Current VisualizationConfig object
        """
        return self.config
