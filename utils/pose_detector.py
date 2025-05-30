"""
Pose detection handler for respiration monitoring.

This module provides a dedicated handler for pose detection setup and processing,
supporting both advanced MediaPipe Tasks and basic pose estimation with fallback.
"""

import cv2
import numpy as np
import mediapipe as mp
import logging
from typing import Optional, Tuple, Union
from dataclasses import dataclass


@dataclass
class PoseLandmarks:
    """
    Container for pose landmark data.
    
    Attributes:
        left_shoulder_x: X coordinate of left shoulder
        left_shoulder_y: Y coordinate of left shoulder  
        right_shoulder_x: X coordinate of right shoulder
        right_shoulder_y: Y coordinate of right shoulder
        average_y: Average Y coordinate of both shoulders
        confidence: Detection confidence score
    """
    left_shoulder_x: float
    left_shoulder_y: float
    right_shoulder_x: float
    right_shoulder_y: float
    average_y: float
    confidence: float = 1.0


class PoseDetectionHandler:
    """
    Handles pose detection setup and landmark extraction for respiration monitoring.
    
    This class encapsulates all pose detection logic, providing a clean interface
    for extracting shoulder landmarks used in respiration rate calculation.
    """
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize the pose detection handler.
        
        Args:
            model_path: Path to advanced pose landmarker model (optional)
        """
        self.model_path = model_path or "./models/pose_landmarker_full.task"
        self.pose_available = False
        self.landmarker = None
        self.mp_pose = None
        self.pose = None
        
        self._setup_pose_detection()
    
    def _setup_pose_detection(self) -> None:
        """
        Initialize MediaPipe pose detection with fallback options.
        
        Attempts to use the advanced pose landmarker if available,
        falls back to basic pose estimation for compatibility.
        """
        try:
            # Try to use advanced MediaPipe Tasks pose landmarker
            self._setup_advanced_landmarker()
            logging.info("Using advanced pose landmarker")
            
        except Exception as e:
            # Fallback to basic pose estimation
            logging.info(f"Advanced pose landmarker not available ({e}), using basic pose")
            self._setup_basic_pose()
    
    def _setup_advanced_landmarker(self) -> None:
        """Setup advanced MediaPipe Tasks pose landmarker."""
        from mediapipe.tasks.python import BaseOptions
        from mediapipe.tasks.python.vision import (
            PoseLandmarker, PoseLandmarkerOptions, RunningMode
        )
        
        options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=self.model_path),
            running_mode=RunningMode.VIDEO,
            num_poses=1
        )
        self.landmarker = PoseLandmarker.create_from_options(options)
        self.pose_available = True
    
    def _setup_basic_pose(self) -> None:
        """Setup basic MediaPipe pose estimation."""
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
    
    def detect_pose(self, frame: np.ndarray, frame_idx: int, fps: float) -> Optional[PoseLandmarks]:
        """
        Detect pose landmarks in a video frame.
        
        Args:
            frame: Input video frame in BGR format
            frame_idx: Current frame index for timestamp calculation
            fps: Video frame rate
            
        Returns:
            PoseLandmarks object if pose detected, None otherwise
        """
        if not self.pose_available:
            return None
        
        h, w = frame.shape[:2]
        
        if self.landmarker:
            return self._detect_with_landmarker(frame, frame_idx, fps, h, w)
        else:
            return self._detect_with_basic_pose(frame, h, w)
    
    def _detect_with_landmarker(self, frame: np.ndarray, frame_idx: int, 
                               fps: float, h: int, w: int) -> Optional[PoseLandmarks]:
        """
        Detect pose using advanced MediaPipe Tasks landmarker.
        
        Args:
            frame: Input video frame
            frame_idx: Current frame index
            fps: Video frame rate
            h: Frame height
            w: Frame width
            
        Returns:
            PoseLandmarks object if successful, None otherwise
        """
        try:
            # Convert to RGB and create MediaPipe image
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            
            # Calculate timestamp for video processing
            timestamp_ms = int((frame_idx / fps) * 1000)
            
            result = self.landmarker.detect_for_video(mp_image, timestamp_ms)
            
            if result.pose_landmarks and len(result.pose_landmarks) > 0:
                landmarks = result.pose_landmarks[0]
                
                # Extract shoulder landmarks (indices 11 and 12)
                l_sh = landmarks[11]  # Left shoulder
                r_sh = landmarks[12]  # Right shoulder
                
                return PoseLandmarks(
                    left_shoulder_x=l_sh.x * w,
                    left_shoulder_y=l_sh.y * h,
                    right_shoulder_x=r_sh.x * w,
                    right_shoulder_y=r_sh.y * h,
                    average_y=(l_sh.y * h + r_sh.y * h) / 2,
                    confidence=min(l_sh.visibility, r_sh.visibility) if hasattr(l_sh, 'visibility') else 1.0
                )
                
        except Exception as e:
            logging.error(f"Pose landmarker processing error: {e}")
        
        return None
    
    def _detect_with_basic_pose(self, frame: np.ndarray, h: int, w: int) -> Optional[PoseLandmarks]:
        """
        Detect pose using basic MediaPipe pose estimation.
        
        Args:
            frame: Input video frame
            h: Frame height
            w: Frame width
            
        Returns:
            PoseLandmarks object if successful, None otherwise
        """
        try:
            # Convert to RGB for MediaPipe processing
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(frame_rgb)
            
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                
                # Extract shoulder landmarks
                l_sh = landmarks[11]  # Left shoulder
                r_sh = landmarks[12]  # Right shoulder
                
                return PoseLandmarks(
                    left_shoulder_x=l_sh.x * w,
                    left_shoulder_y=l_sh.y * h,
                    right_shoulder_x=r_sh.x * w,
                    right_shoulder_y=r_sh.y * h,
                    average_y=(l_sh.y * h + r_sh.y * h) / 2,
                    confidence=min(l_sh.visibility, r_sh.visibility) if hasattr(l_sh, 'visibility') else 1.0
                )
                
        except Exception as e:
            logging.error(f"Basic pose processing error: {e}")
        
        return None
    
    def is_available(self) -> bool:
        """
        Check if pose detection is available and ready.
        
        Returns:
            True if pose detection is available, False otherwise
        """
        return self.pose_available
