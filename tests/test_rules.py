import pytest
from app.api.schemas import FrameDetections, FaceDetectionResult, ObjectDetectionResult, GazeDetectionResult, MotionDetectionResult, GazeStatus, Severity, TrackerState, RuleCode
from app.engine.rule_engine import RuleEngine
from app.engine.risk_scorer import RiskScorer
from app.engine.decision_engine import DecisionEngine
from app.config.thresholds import get_thresholds

def create_mock_detections(
    face_present=True,
    person_count=1,
    phone_detected=False,
    gaze_status=GazeStatus.NORMAL,
    high_motion=False,
    suspicious_objects=0
):
    return FrameDetections(
        frame_id=1,
        face=FaceDetectionResult(face_present=face_present, confidence=0.9),
        objects=ObjectDetectionResult(
            person_count=person_count, 
            phone_detected=phone_detected, 
            suspicious_objects=[None] * suspicious_objects
        ),
        gaze=GazeDetectionResult(gaze_status=gaze_status),
        motion=MotionDetectionResult(high_motion=high_motion)
    )

def test_rule_engine():
    engine = RuleEngine()
    dt = 0.1
    
    # Normal frame
    detections = create_mock_detections()
    states = engine.evaluate(detections, dt)
    assert all(not state.active for state in states)
    
    # Trigger FACE_MISSING
    detections = create_mock_detections(face_present=False)
    states = engine.evaluate(detections, dt)
    face_state = next(s for s in states if s.rule_code == RuleCode.FACE_MISSING)
    assert face_state.active
    
    # Exceed threshold
    thresholds = get_thresholds().temporal
    states = engine.evaluate(detections, thresholds.face_missing_duration_s)
    violated = engine.get_violated_rules(states)
    assert len(violated) == 1
    assert violated[0].rule_code == RuleCode.FACE_MISSING
    
    engine.reset_all()
    assert all(not state.violated and not state.active for state in engine.evaluate(create_mock_detections(), dt))

def test_risk_scorer():
    scorer = RiskScorer()
    
    # No violations
    assert scorer.compute([]) == 0
    
    # One violation
    state1 = TrackerState(rule_code=RuleCode.FACE_MISSING, violated=True)
    weights = get_thresholds().risk_weights
    expected_score = int(weights.face_missing)
    assert scorer.compute([state1]) == expected_score
    
    # Multiple violations
    state2 = TrackerState(rule_code=RuleCode.PHONE_DETECTED, violated=True)
    expected_score = min(100, int(weights.face_missing + weights.phone_detected))
    assert scorer.compute([state1, state2]) == expected_score

def test_decision_engine():
    engine = DecisionEngine()
    
    # Safe frame
    decision = engine.process_frame(create_mock_detections(), dt=0.1)
    assert decision.severity == Severity.SAFE
    assert not decision.suspicious
    assert len(decision.reasons) == 0
    
    # Violated frame (exceed threshold for phone)
    thresholds = get_thresholds().temporal
    dt_trigger = 0.1
    for _ in range(thresholds.phone_consecutive_frames + 1):
        decision = engine.process_frame(create_mock_detections(phone_detected=True), dt=dt_trigger)
        
    assert decision.suspicious
    assert RuleCode.PHONE_DETECTED.value in decision.reasons
    assert decision.risk_score > 0
    # Depending on the weight, severity might be low, medium, etc.
