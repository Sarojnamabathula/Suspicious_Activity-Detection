"""
SentinelAI — Motion Detector.
"""

from __future__ import annotations

import logging
import cv2
import numpy as np

from app.api.schemas import MotionDetectionResult
from app.config.thresholds import get_thresholds
from app.detectors.base_detector import BaseDetector

logger = logging.getLogger(__name__)

class MotionDetector(BaseDetector):
    """OpenCV background subtraction for motion bursts."""

    def __init__(self) -> None:
        try:
            motion_config = get_thresholds().motion
            self._bg_subtractor = cv2.createBackgroundSubtractorMOG2(
                history=motion_config.history,
                varThreshold=motion_config.var_threshold,
                detectShadows=motion_config.detect_shadows
            )
            self._kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        except Exception as e:
            logger.error(f"Failed to initialize background subtractor: {e}")
            self._bg_subtractor = None

    def detect(self, frame: np.ndarray) -> MotionDetectionResult:
        if self._bg_subtractor is None:
            return MotionDetectionResult()

        try:
            # Apply background subtraction
            fg_mask = self._bg_subtractor.apply(frame)
            
            # Remove noise
            fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, self._kernel)
            
            # Calculate motion area
            total_pixels = fg_mask.shape[0] * fg_mask.shape[1]
            if total_pixels == 0:
                return MotionDetectionResult()
                
            motion_pixels = cv2.countNonZero(fg_mask)
            motion_level = motion_pixels / total_pixels
            
            thresholds = get_thresholds()
            high_motion = motion_level > thresholds.motion.motion_area_threshold
            
            return MotionDetectionResult(
                motion_level=float(motion_level),
                high_motion=high_motion
            )
            
        except Exception as e:
            logger.error(f"Motion detection failed: {e}")
            return MotionDetectionResult()

    def close(self) -> None:
        pass
