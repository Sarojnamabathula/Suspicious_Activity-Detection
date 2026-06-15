"""
SentinelAI — YOLOv8 Object Detector.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

from app.api.schemas import BoundingBox, ObjectDetectionResult
from app.config.settings import get_settings
from app.config.thresholds import get_thresholds
from app.detectors.base_detector import BaseDetector

logger = logging.getLogger(__name__)

class ObjectDetector(BaseDetector):
    """YOLOv8 object detector for persons, phones, and suspicious objects."""

    def __init__(self) -> None:
        if YOLO is None:
            logger.warning("ultralytics not installed. ObjectDetector will run in mock mode.")
            self._model = None
            return

        settings = get_settings()
        model_path = settings.resolve_yolo_path()
        device = settings.inference_device.value if settings.inference_device.value != "auto" else ""
        
        try:
            self._model = YOLO(str(model_path))
            if device:
                self._model.to(device)
            logger.info(f"Loaded YOLO model from {model_path} on device '{device or 'auto'}'")
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")
            self._model = None

    def detect(self, frame: np.ndarray) -> ObjectDetectionResult:
        if self._model is None:
            return ObjectDetectionResult()

        try:
            thresholds = get_thresholds()
            min_conf = thresholds.detection.yolo_min_confidence
            phone_min_conf = thresholds.detection.phone_min_confidence
            
            # Run inference
            results = self._model(frame, verbose=False)
            
            if not results or len(results) == 0:
                return ObjectDetectionResult()

            result = results[0]
            boxes = result.boxes
            
            person_count = 0
            phone_detected = False
            phone_confidence = 0.0
            person_boxes = []
            phone_box = None
            suspicious_objects = []

            yolo_classes = thresholds.yolo_classes
            
            if boxes is not None:
                for box in boxes:
                    conf = float(box.conf[0])
                    cls_id = int(box.cls[0])
                    xyxy = box.xyxy[0].cpu().numpy()
                    
                    bbox = BoundingBox(
                        x1=int(xyxy[0]), y1=int(xyxy[1]),
                        x2=int(xyxy[2]), y2=int(xyxy[3]),
                        confidence=conf, label=self._model.names.get(cls_id, str(cls_id))
                    )

                    if cls_id == yolo_classes.person_class_id and conf >= min_conf:
                        person_count += 1
                        person_boxes.append(bbox)
                    
                    elif cls_id == yolo_classes.phone_class_id and conf >= phone_min_conf:
                        phone_detected = True
                        if conf > phone_confidence:
                            phone_confidence = conf
                            phone_box = bbox
                            
                    elif cls_id in yolo_classes.suspicious_class_ids and conf >= min_conf:
                        suspicious_objects.append(bbox)

            return ObjectDetectionResult(
                person_count=person_count,
                phone_detected=phone_detected,
                phone_confidence=phone_confidence,
                person_boxes=person_boxes,
                phone_box=phone_box,
                suspicious_objects=suspicious_objects
            )

        except Exception as e:
            logger.error(f"Object detection failed: {e}")
            return ObjectDetectionResult()

    def close(self) -> None:
        """Release resources."""
        pass
