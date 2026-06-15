"""
SentinelAI — Rule Engine.
Evaluates all tracker states based on detection results.
"""

from __future__ import annotations

from app.api.schemas import FrameDetections, TrackerState
from app.trackers.face_tracker import FaceTracker
from app.trackers.person_tracker import PersonTracker
from app.trackers.phone_tracker import PhoneTracker
from app.trackers.gaze_tracker import GazeTracker
from app.trackers.suspicious_object_tracker import SuspiciousObjectTracker
from app.trackers.motion_tracker import MotionTracker


class RuleEngine:
    """Manages and evaluates all temporal trackers."""

    def __init__(self) -> None:
        self.trackers = [
            FaceTracker(),
            PersonTracker(),
            PhoneTracker(),
            GazeTracker(),
            SuspiciousObjectTracker(),
            MotionTracker(),
        ]

    def evaluate(self, detections: FrameDetections, dt: float) -> list[TrackerState]:
        """Update all trackers with current frame detections and return states."""
        states = []
        for tracker in self.trackers:
            if isinstance(tracker, FaceTracker):
                state = tracker.update(detections.face.face_present, detections.face.confidence, dt)
            elif isinstance(tracker, PersonTracker):
                state = tracker.update(detections.objects.person_count, dt)
            elif isinstance(tracker, PhoneTracker):
                state = tracker.update(detections.objects.phone_detected, detections.objects.phone_confidence, dt)
            elif isinstance(tracker, GazeTracker):
                state = tracker.update(detections.gaze.gaze_status, dt)
            elif isinstance(tracker, SuspiciousObjectTracker):
                # We'll use max confidence of suspicious objects, or 0.0 if none
                conf = max((obj.confidence for obj in detections.objects.suspicious_objects), default=0.0)
                state = tracker.update(len(detections.objects.suspicious_objects) > 0, conf, dt)
            elif isinstance(tracker, MotionTracker):
                state = tracker.update(detections.motion.high_motion, dt)
            else:
                continue
            states.append(state)
        return states

    def get_violated_rules(self, states: list[TrackerState]) -> list[TrackerState]:
        """Filter states to only those that are currently violated."""
        return [state for state in states if state.violated]

    def reset_all(self) -> None:
        """Reset all trackers to their initial state."""
        for tracker in self.trackers:
            tracker.reset()
