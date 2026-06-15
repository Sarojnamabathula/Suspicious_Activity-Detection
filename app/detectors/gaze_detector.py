"""
SentinelAI — Gaze and Head Pose Detector.
"""

from __future__ import annotations

import logging
import os
import numpy as np

try:
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
except (ImportError, AttributeError):
    mp = None
    python = None
    vision = None

from app.api.schemas import GazeDetectionResult, GazeStatus
from app.config.thresholds import get_thresholds
from app.detectors.base_detector import BaseDetector

logger = logging.getLogger(__name__)

class GazeDetector(BaseDetector):
    """Estimates gaze direction and head pose using MediaPipe FaceLandmarker Tasks."""

    def __init__(self) -> None:
        model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "models", "face_landmarker.task")
        
        if mp is None or not os.path.exists(model_path):
            logger.warning("mediapipe not installed or model missing. GazeDetector will run in mock mode.")
            self._face_mesh = None
            return

        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._face_mesh = vision.FaceLandmarker.create_from_options(options)

    def detect(self, frame: np.ndarray) -> GazeDetectionResult:
        if self._face_mesh is None:
            return GazeDetectionResult(gaze_status=GazeStatus.UNKNOWN)

        try:
            import cv2
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            
            results = self._face_mesh.detect(mp_image)

            if not results.face_landmarks:
                return GazeDetectionResult(gaze_status=GazeStatus.UNKNOWN)

            landmarks = results.face_landmarks[0]
            
            # Left eye indices
            left_iris_indices = [468, 469, 470, 471, 472]
            left_inner = 133
            left_outer = 33
            left_top = 159
            left_bottom = 145

            # Right eye indices
            right_iris_indices = [473, 474, 475, 476, 477]
            right_inner = 362
            right_outer = 263
            right_top = 386
            right_bottom = 374

            # Compute horizontal ratios
            def compute_ratio(iris_indices, inner_idx, outer_idx, top_idx, bottom_idx):
                iris_x = np.mean([landmarks[i].x for i in iris_indices])
                iris_y = np.mean([landmarks[i].y for i in iris_indices])
                
                inner_x = landmarks[inner_idx].x
                outer_x = landmarks[outer_idx].x
                
                top_y = landmarks[top_idx].y
                bottom_y = landmarks[bottom_idx].y
                
                # Protect against division by zero
                width = abs(outer_x - inner_x)
                height = abs(bottom_y - top_y)
                
                if width < 1e-6 or height < 1e-6:
                    return 0.5, 0.5
                
                h_ratio = (iris_x - min(inner_x, outer_x)) / width
                v_ratio = (iris_y - min(top_y, bottom_y)) / height
                return h_ratio, v_ratio

            l_h, l_v = compute_ratio(left_iris_indices, left_inner, left_outer, left_top, left_bottom)
            r_h, r_v = compute_ratio(right_iris_indices, right_inner, right_outer, right_top, right_bottom)
            
            # Average and normalize to [-1, 1] centered at 0.5
            h_offset = float(((l_h + r_h) / 2.0 - 0.5) * 2.0)
            v_offset = float(((l_v + r_v) / 2.0 - 0.5) * 2.0)

            thresholds = get_thresholds()
            max_dev = thresholds.detection.gaze_deviation_threshold
            
            gaze_status = GazeStatus.NORMAL
            if abs(h_offset) > max_dev or abs(v_offset) > max_dev:
                gaze_status = GazeStatus.AWAY

            # Very rough head pose proxy using nose tip (1) vs frame center
            nose_x = landmarks[1].x
            nose_y = landmarks[1].y
            head_yaw = float((nose_x - 0.5) * 180.0)  # approximate
            head_pitch = float((nose_y - 0.5) * 180.0)

            return GazeDetectionResult(
                gaze_status=gaze_status,
                horizontal_offset=h_offset,
                vertical_offset=v_offset,
                head_yaw=head_yaw,
                head_pitch=head_pitch
            )

        except Exception as e:
            logger.error(f"Gaze detection failed: {e}")
            return GazeDetectionResult(gaze_status=GazeStatus.UNKNOWN)

    def close(self) -> None:
        if self._face_mesh is not None:
            self._face_mesh.close()
