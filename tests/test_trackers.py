import pytest
from app.trackers.face_tracker import FaceTracker
from app.trackers.person_tracker import PersonTracker
from app.trackers.phone_tracker import PhoneTracker
from app.trackers.gaze_tracker import GazeTracker
from app.trackers.suspicious_object_tracker import SuspiciousObjectTracker
from app.trackers.motion_tracker import MotionTracker
from app.api.schemas import GazeStatus, RuleCode
from app.config.thresholds import get_thresholds

def test_face_tracker():
    tracker = FaceTracker()
    thresholds = get_thresholds().temporal
    dt = 0.1
    
    # 1. Normal state - face is present
    state = tracker.update(face_present=True, confidence=0.9, dt=dt)
    assert not state.active
    assert not state.violated
    assert state.accumulated_time_s == 0.0

    # 2. Face missing starts accumulating
    state = tracker.update(face_present=False, confidence=0.0, dt=dt)
    assert state.active
    assert not state.violated
    assert state.accumulated_time_s == dt
    
    # 3. Exceed duration
    exceed_dt = thresholds.face_missing_duration_s
    state = tracker.update(face_present=False, confidence=0.0, dt=exceed_dt)
    assert state.violated
    
    # 4. Hysteresis (clear)
    state = tracker.update(face_present=True, confidence=0.9, dt=dt)
    assert state.violated  # Still violated until clear duration
    state = tracker.update(face_present=True, confidence=0.9, dt=thresholds.hysteresis_clear_duration_s)
    assert not state.violated
    assert not state.active

def test_person_tracker():
    tracker = PersonTracker()
    thresholds = get_thresholds().temporal
    dt = 0.1
    
    state = tracker.update(person_count=1, dt=dt)
    assert not state.active
    
    # Needs multiple frames to get rolling average > 1.0 depending on window size
    window = get_thresholds().temporal.person_count_rolling_window
    for _ in range(window):
        state = tracker.update(person_count=2, dt=dt)
        
    assert state.active
    assert not state.violated
    
    # Exceed duration
    state = tracker.update(person_count=2, dt=thresholds.multiple_persons_duration_s)
    assert state.violated

def test_phone_tracker():
    tracker = PhoneTracker()
    thresholds = get_thresholds().temporal
    dt = 0.1
    
    state = tracker.update(phone_detected=True, confidence=0.8, dt=dt)
    assert state.active
    assert not state.violated
    
    for _ in range(thresholds.phone_consecutive_frames - 1):
        state = tracker.update(phone_detected=True, confidence=0.8, dt=dt)
        
    assert state.violated
    
def test_gaze_tracker():
    tracker = GazeTracker()
    thresholds = get_thresholds().temporal
    dt = 0.1
    
    state = tracker.update(gaze_status=GazeStatus.NORMAL, dt=dt)
    assert not state.active
    
    state = tracker.update(gaze_status=GazeStatus.AWAY, dt=dt)
    assert state.active
    assert not state.violated
    
    state = tracker.update(gaze_status=GazeStatus.UNKNOWN, dt=dt)
    # Unknown shouldn't change time
    assert state.active
    assert state.accumulated_time_s == dt
    
    state = tracker.update(gaze_status=GazeStatus.AWAY, dt=thresholds.gaze_away_duration_s)
    assert state.violated

def test_suspicious_object_tracker():
    tracker = SuspiciousObjectTracker()
    thresholds = get_thresholds().temporal
    dt = 0.1
    
    state = tracker.update(objects_detected=True, confidence=0.8, dt=dt)
    assert state.active
    assert not state.violated
    
    for _ in range(thresholds.suspicious_object_consecutive_frames - 1):
        state = tracker.update(objects_detected=True, confidence=0.8, dt=dt)
        
    assert state.violated

def test_motion_tracker():
    tracker = MotionTracker()
    thresholds = get_thresholds().temporal
    dt = 0.1
    
    state = tracker.update(high_motion=True, dt=dt)
    assert state.active
    assert not state.violated
    
    state = tracker.update(high_motion=True, dt=thresholds.motion_burst_duration_s)
    assert state.violated
