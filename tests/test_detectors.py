import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from app.detectors.face_detector import FaceDetector
from app.detectors.gaze_detector import GazeDetector
from app.detectors.object_detector import ObjectDetector
from app.detectors.motion_detector import MotionDetector
from app.api.schemas import GazeStatus

@patch('app.detectors.face_detector.vision.FaceDetector')
def test_face_detector(mock_face_detector):
    mock_instance = MagicMock()
    mock_face_detector.create_from_options.return_value = mock_instance
    
    mock_result = MagicMock()
    mock_detection = MagicMock()
    mock_cat = MagicMock()
    mock_cat.score = 0.95
    mock_detection.categories = [mock_cat]
    mock_detection.bounding_box.origin_x = 64
    mock_detection.bounding_box.origin_y = 48
    mock_detection.bounding_box.width = 128
    mock_detection.bounding_box.height = 128
    mock_result.detections = [mock_detection]
    
    mock_instance.detect.return_value = mock_result
    
    detector = FaceDetector()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    res = detector.detect(frame)
    assert res.face_present
    assert res.confidence == 0.95
    assert res.bounding_box.x1 == 64

@patch('app.detectors.gaze_detector.vision.FaceLandmarker')
def test_gaze_detector(mock_face_landmarker):
    mock_instance = MagicMock()
    mock_face_landmarker.create_from_options.return_value = mock_instance
    
    detector = GazeDetector()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Empty result
    mock_result = MagicMock()
    mock_result.face_landmarks = []
    mock_instance.detect.return_value = mock_result
    
    res = detector.detect(frame)
    assert res.gaze_status == GazeStatus.UNKNOWN

@patch('app.detectors.object_detector.YOLO')
def test_object_detector(mock_yolo):
    # If ultralytics is not installed, YOLO is None and falls back to mock mode gracefully
    # We'll just test the fallback behavior if YOLO is mocked to be a function
    detector = ObjectDetector()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    res = detector.detect(frame)
    
    # Regardless of YOLO being present or not, calling detect should return a valid result
    assert hasattr(res, 'person_count')
    assert res.person_count >= 0

def test_motion_detector():
    detector = MotionDetector()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    # Give it an initial frame to populate the history
    detector.detect(frame)
    # The second identical frame should have no motion
    res = detector.detect(frame)
    
    assert hasattr(res, 'motion_level')
    assert res.motion_level >= 0.0
    assert not res.high_motion
