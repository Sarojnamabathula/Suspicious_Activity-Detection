"""
SentinelAI — Base Detector Abstraction.

Defines the interface that all ML detectors must implement to ensure
consistent lifecycle management and inference pipelines.
"""

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class BaseDetector(ABC):
    """Abstract base class for all vision-based detectors."""

    @abstractmethod
    def detect(self, frame: np.ndarray) -> Any:
        """Run inference on a single frame.
        
        Args:
            frame: BGR image as an ``np.ndarray`` (H × W × 3).
            
        Returns:
            A detector-specific result object (e.g., FaceDetectionResult).
        """
        pass

    @abstractmethod
    def close(self) -> None:
        """Release any hardware/framework resources held by the detector."""
        pass
