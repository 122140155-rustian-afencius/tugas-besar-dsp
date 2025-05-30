"""
Signal buffer manager for respiratory signal processing.

This module provides efficient management of signal buffers used in respiration
rate analysis, including data storage, filtering, and quality assessment.
"""

import numpy as np
import logging
from collections import deque
from typing import List, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class BufferConfig:
    """
    Configuration for signal buffers.
    
    Attributes:
        max_size: Maximum buffer size (number of samples)
        min_analysis_size: Minimum samples required for analysis
        fps: Video frame rate for time calculations
    """
    max_size: int = 1800  # 1 minute at 30 fps
    min_analysis_size: int = 30  # Minimum samples for analysis
    fps: float = 30.0


@dataclass 
class SignalData:
    """
    Container for signal data arrays.
    
    Attributes:
        raw_signal: Raw signal values
        filtered_signal: Filtered signal values
        timestamps: Time values for each sample
        quality_scores: Signal quality scores over time
    """
    raw_signal: List[float] = field(default_factory=list)
    filtered_signal: List[float] = field(default_factory=list)
    timestamps: List[float] = field(default_factory=list)
    quality_scores: List[float] = field(default_factory=list)


class SignalBufferManager:
    """
    Manages signal buffers for respiratory signal processing.
    
    This class provides efficient storage and retrieval of respiratory signals,
    with automatic buffer management and quality assessment capabilities.
    """
    
    def __init__(self, config: Optional[BufferConfig] = None):
        """
        Initialize the signal buffer manager.
        
        Args:
            config: Buffer configuration (uses default if None)
        """
        self.config = config or BufferConfig()
        
        # Initialize signal buffers with maximum size limits
        self.raw_buffer = deque(maxlen=self.config.max_size)
        self.filtered_buffer = deque(maxlen=self.config.max_size)
        self.time_buffer = deque(maxlen=self.config.max_size)
        self.quality_buffer = deque(maxlen=self.config.max_size)
        
        # Track current frame for timestamp calculation
        self.frame_count = 0
        
        logging.info(f"Signal buffer manager initialized with max size: {self.config.max_size}")
    
    def add_sample(self, value: float, quality_score: Optional[float] = None) -> None:
        """
        Add a new sample to the signal buffers.
        
        Args:
            value: Raw signal value to add
            quality_score: Optional quality score for this sample
        """
        # Add raw signal value
        self.raw_buffer.append(value)
        
        # Calculate and store timestamp
        timestamp = self.frame_count / self.config.fps
        self.time_buffer.append(timestamp)
        
        # Store quality score if provided, otherwise calculate basic quality
        if quality_score is not None:
            self.quality_buffer.append(quality_score)
        else:
            quality = self._calculate_sample_quality(value)
            self.quality_buffer.append(quality)
        
        self.frame_count += 1
        
        logging.debug(f"Added sample: value={value:.2f}, timestamp={timestamp:.2f}s")
    
    def add_filtered_sample(self, value: float) -> None:
        """
        Add a filtered signal sample.
        
        Args:
            value: Filtered signal value to add
        """
        self.filtered_buffer.append(value)
    
    def get_raw_signal(self, num_samples: Optional[int] = None) -> List[float]:
        """
        Get raw signal data.
        
        Args:
            num_samples: Number of recent samples to retrieve (all if None)
            
        Returns:
            List of raw signal values
        """
        if num_samples is None:
            return list(self.raw_buffer)
        else:
            return list(self.raw_buffer)[-num_samples:] if len(self.raw_buffer) >= num_samples else list(self.raw_buffer)
    
    def get_filtered_signal(self, num_samples: Optional[int] = None) -> List[float]:
        """
        Get filtered signal data.
        
        Args:
            num_samples: Number of recent samples to retrieve (all if None)
            
        Returns:
            List of filtered signal values
        """
        if num_samples is None:
            return list(self.filtered_buffer)
        else:
            return list(self.filtered_buffer)[-num_samples:] if len(self.filtered_buffer) >= num_samples else list(self.filtered_buffer)
    
    def get_timestamps(self, num_samples: Optional[int] = None) -> List[float]:
        """
        Get timestamp data.
        
        Args:
            num_samples: Number of recent timestamps to retrieve (all if None)
            
        Returns:
            List of timestamp values
        """
        if num_samples is None:
            return list(self.time_buffer)
        else:
            return list(self.time_buffer)[-num_samples:] if len(self.time_buffer) >= num_samples else list(self.time_buffer)
    
    def get_signal_data(self, num_samples: Optional[int] = None) -> SignalData:
        """
        Get complete signal data package.
        
        Args:
            num_samples: Number of recent samples to retrieve (all if None)
            
        Returns:
            SignalData object containing all signal arrays
        """
        return SignalData(
            raw_signal=self.get_raw_signal(num_samples),
            filtered_signal=self.get_filtered_signal(num_samples),
            timestamps=self.get_timestamps(num_samples),
            quality_scores=self.get_quality_scores(num_samples)
        )
    
    def get_quality_scores(self, num_samples: Optional[int] = None) -> List[float]:
        """
        Get signal quality scores.
        
        Args:
            num_samples: Number of recent scores to retrieve (all if None)
            
        Returns:
            List of quality scores
        """
        if num_samples is None:
            return list(self.quality_buffer)
        else:
            return list(self.quality_buffer)[-num_samples:] if len(self.quality_buffer) >= num_samples else list(self.quality_buffer)
    
    def has_sufficient_data(self) -> bool:
        """
        Check if buffer contains sufficient data for analysis.
        
        Returns:
            True if sufficient data is available, False otherwise
        """
        return len(self.raw_buffer) >= self.config.min_analysis_size
    
    def get_buffer_size(self) -> int:
        """
        Get current buffer size.
        
        Returns:
            Number of samples currently in buffer
        """
        return len(self.raw_buffer)
    
    def get_buffer_duration(self) -> float:
        """
        Get current buffer duration in seconds.
        
        Returns:
            Duration of buffered data in seconds
        """
        return len(self.raw_buffer) / self.config.fps
    
    def clear_buffers(self) -> None:
        """Clear all signal buffers and reset frame count."""
        self.raw_buffer.clear()
        self.filtered_buffer.clear()
        self.time_buffer.clear()
        self.quality_buffer.clear()
        self.frame_count = 0
        
        logging.info("Signal buffers cleared")
    
    def get_overall_quality(self) -> float:
        """
        Calculate overall signal quality from recent samples.
        
        Returns:
            Overall quality score (0-1, higher is better)
        """
        if len(self.quality_buffer) == 0:
            return 0.0
        
        # Use recent samples for quality assessment
        recent_quality = list(self.quality_buffer)[-30:] if len(self.quality_buffer) >= 30 else list(self.quality_buffer)
        return np.mean(recent_quality)
    
    def _calculate_sample_quality(self, value: float) -> float:
        """
        Calculate quality score for a single sample based on signal characteristics.
        
        Args:
            value: Signal value to assess
            
        Returns:
            Quality score (0-1)
        """
        if len(self.raw_buffer) < 2:
            return 0.5  # Default quality for first samples
        
        # Calculate quality based on signal variability and consistency
        recent_samples = list(self.raw_buffer)[-10:] if len(self.raw_buffer) >= 10 else list(self.raw_buffer)
        
        # Check for reasonable signal range (movement detection)
        signal_range = max(recent_samples) - min(recent_samples)
        range_quality = min(1.0, signal_range / 50.0)  # Normalize to 50 pixel range
        
        # Check for signal consistency (not too noisy)
        if len(recent_samples) >= 3:
            signal_diff = abs(value - recent_samples[-1])
            max_expected_diff = np.std(recent_samples) * 2  # 2 standard deviations
            consistency_quality = max(0.0, 1.0 - (signal_diff / max(max_expected_diff, 1.0)))
        else:
            consistency_quality = 0.5
        
        # Combine quality metrics
        overall_quality = (range_quality + consistency_quality) / 2
        
        return np.clip(overall_quality, 0.0, 1.0)
    
    def get_buffer_statistics(self) -> dict:
        """
        Get comprehensive buffer statistics.
        
        Returns:
            Dictionary containing buffer statistics
        """
        if len(self.raw_buffer) == 0:
            return {
                'size': 0,
                'duration': 0.0,
                'mean': 0.0,
                'std': 0.0,
                'range': 0.0,
                'quality': 0.0
            }
        
        raw_data = list(self.raw_buffer)
        
        return {
            'size': len(raw_data),
            'duration': self.get_buffer_duration(),
            'mean': np.mean(raw_data),
            'std': np.std(raw_data),
            'range': np.max(raw_data) - np.min(raw_data),
            'quality': self.get_overall_quality()
        }
