"""
SentinelAI — Face Detector.

Wraps MediaPipe Tasks Vision FaceDetector to locate and score faces in a single frame.
Returns a :class:`FaceDetectionResult` containing the highest-confidence
face bounding box (pixel coordinates) or an empty result when no face is
found or an error occurs.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import numpy as np

try:
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
except (ImportError, AttributeError):
    mp = None
    python = None
    vision = None

from app.api.schemas import BoundingBox, FaceDetectionResult
from app.config.thresholds import get_thresholds
from app.detectors.base_detector import BaseDetector

logger = logging.getLogger(__name__)


class FaceDetector(BaseDetector):
    """MediaPipe short-range face detector.

    Parameters are sourced from ``get_thresholds().detection`` so the
    confidence floor is never hard-coded.
    """

    def __init__(self) -> None:
        """Initialise the MediaPipe FaceDetector model."""
        threshold = get_thresholds().detection.face_min_confidence
        model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "models", "blaze_face.tflite")
        
        if mp is not None and os.path.exists(model_path):
            base_options = python.BaseOptions(model_asset_path=model_path)
            options = vision.FaceDetectorOptions(
                base_options=base_options,
                min_detection_confidence=threshold,
            )
            self._face_detection = vision.FaceDetector.create_from_options(options)
        else:
            self._face_detection = None
            if not os.path.exists(model_path):
                logger.warning("Face model not found at %s", model_path)
                
        logger.info(
            "FaceDetector initialised (min_confidence=%.2f, active=%s)", 
            threshold, self._face_detection is not None
        )

    # ── public API ───────────────────────────────────────────────────

    def detect(self, frame: np.ndarray) -> FaceDetectionResult:
        """Run face detection on *frame* and return the best detection.

        Args:
            frame: BGR image as an ``np.ndarray`` (H × W × 3).

        Returns:
            A :class:`FaceDetectionResult` populated with the highest-
            confidence face.  If no face passes the threshold or an error
            occurs, an empty default result is returned.
        """
        if self._face_detection is None:
            return FaceDetectionResult(face_present=False)

        try:
            h, w = frame.shape[:2]
            rgb = frame[:, :, ::-1]  # BGR → RGB (no copy needed for MP)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            
            detection_result = self._face_detection.detect(mp_image)

            if not detection_result.detections:
                return FaceDetectionResult(face_present=False)

            # Pick the detection with the highest confidence score.
            best = max(
                detection_result.detections,
                key=lambda d: d.categories[0].score,
            )
            confidence: float = float(best.categories[0].score)

            # Re-check against threshold
            threshold = get_thresholds().detection.face_min_confidence
            if confidence < threshold:
                return FaceDetectionResult(face_present=False)

            # Convert relative bounding box → pixel coordinates.
            bbox_mp = best.bounding_box
            x1 = max(0, int(bbox_mp.origin_x))
            y1 = max(0, int(bbox_mp.origin_y))
            x2 = min(w, int(bbox_mp.origin_x + bbox_mp.width))
            y2 = min(h, int(bbox_mp.origin_y + bbox_mp.height))

            bbox = BoundingBox(
                x1=x1, y1=y1, x2=x2, y2=y2,
                confidence=confidence,
                label="face",
            )

            return FaceDetectionResult(
                face_present=True,
                confidence=confidence,
                bounding_box=bbox,
            )

        except Exception as exc:
            logger.error("FaceDetector.detect failed: %s", exc, exc_info=True)
            return FaceDetectionResult(face_present=False)

    # ── lifecycle ────────────────────────────────────────────────────

    def close(self) -> None:
        """Release MediaPipe resources."""
        try:
            if self._face_detection:
                self._face_detection.close()
            logger.info("FaceDetector resources released.")
        except Exception as exc:
            logger.warning("FaceDetector.close error: %s", exc)
