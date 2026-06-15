"""
SentinelAI — Decision Engine.
Assembles the final structured detection payload.
"""

from __future__ import annotations

from app.api.schemas import DecisionOutput, FrameDetections, Severity
from app.config.thresholds import get_thresholds
from app.engine.risk_scorer import RiskScorer
from app.engine.rule_engine import RuleEngine
from app.utils.helpers import iso_now

class DecisionEngine:
    """Evaluates rules and scores to produce final frame decision."""

    def __init__(self) -> None:
        self.rule_engine = RuleEngine()
        self.risk_scorer = RiskScorer()

    def process_frame(self, detections: FrameDetections, dt: float) -> DecisionOutput:
        """Process one frame and return final decision."""
        # 1. Update trackers
        states = self.rule_engine.evaluate(detections, dt)
        
        # 2. Get violated rules
        violated = self.rule_engine.get_violated_rules(states)
        
        # 3. Compute risk score
        risk_score = self.risk_scorer.compute(violated)
        
        # 4. Determine severity
        bands = get_thresholds().severity
        if risk_score <= bands.safe_max:
            severity = Severity.SAFE
        elif risk_score <= bands.low_max:
            severity = Severity.LOW
        elif risk_score <= bands.medium_max:
            severity = Severity.MEDIUM
        elif risk_score <= bands.high_max:
            severity = Severity.HIGH
        else:
            severity = Severity.CRITICAL
            
        # 5. Build reason codes
        reasons = [state.rule_code.value for state in violated]
        suspicious = len(reasons) > 0
        
        # 6. Aggregate confidence
        confidence = max((state.confidence for state in violated), default=0.0)
        
        # 7. Build DecisionOutput
        return DecisionOutput(
            timestamp=iso_now(),
            frame_id=detections.frame_id,
            suspicious=suspicious,
            severity=severity,
            reasons=reasons,
            confidence=confidence,
            face_present=detections.face.face_present,
            person_count=detections.objects.person_count,
            phone_detected=detections.objects.phone_detected,
            gaze_status=detections.gaze.gaze_status,
            risk_score=risk_score
        )
