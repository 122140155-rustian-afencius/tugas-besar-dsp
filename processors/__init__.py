"""
Processors package for vital signs monitoring application.

This package contains signal processing modules for:
- Heart rate (rPPG) detection
- Respiration rate detection
- Signal filtering and analysis utilities
- Pose detection and landmark extraction
- Signal buffer management
- Respiratory signal analysis
- Visualization helpers
"""

from processors.rppg.rppg_processor import RPPGProcessor
from processors.respiration.respiration_processor import RespirationProcessor
from processors.respiration.respiration_processor_modular import RespirationProcessorModular
from utils.pose_detector import PoseDetectionHandler, PoseLandmarks
from utils.signal_buffer import SignalBufferManager, BufferConfig, SignalData
from processors.respiration.respiratory_analyzer import RespiratorySignalAnalyzer, FilterConfig, AnalysisResult
from utils.visualization_helper import VisualizationHelper, VisualizationConfig

__all__ = [
    'RPPGProcessor', 
    'RespirationProcessor',
    'RespirationProcessorModular',
    'PoseDetectionHandler',
    'PoseLandmarks',
    'SignalBufferManager',
    'BufferConfig',
    'SignalData',
    'RespiratorySignalAnalyzer',
    'FilterConfig',
    'AnalysisResult',
    'VisualizationHelper',
    'VisualizationConfig'
]
