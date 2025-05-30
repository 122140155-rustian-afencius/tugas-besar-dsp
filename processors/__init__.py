"""
Processors package for vital signs monitoring application.

This package contains signal processing modules for:
- Heart rate (rPPG) detection
- Respiration rate detection
- Signal filtering and analysis utilities
"""

from processors.rppg_processor import RPPGProcessor
from processors.respiration_processor import RespirationProcessor

__all__ = ['RPPGProcessor', 'RespirationProcessor']
