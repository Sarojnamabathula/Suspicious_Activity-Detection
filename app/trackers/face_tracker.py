"""
SentinelAI — Face-Missing Tracker.

Monitors continuous face absence and raises a FACE_MISSING violation
once the accumulated absence exceeds the configured temporal threshold.
Uses hysteresis to avoid resetting the accumulator on brief flickers.
"""

from __future__ import annotations

from app.api.schemas import RuleCode, TrackerState
from app.config.thresholds import get_thresholds
from app.utils.helpers import clamp


class FaceTracker:
    """Temporal tracker for the FACE_MISSING rule.

    Accumulates time while the face is absent.  The accumulator only
    resets when the face has been *continuously present* for at least
    ``hysteresis_clear_duration_s`` seconds, preventing rapid
    flicker-induced resets.
    """

    def __init__(self) -> None:
        self._active: bool = False
        self._violated: bool = False
        self._accumulated_time_s: float = 0.0
        self._accumulated_frames: int = 0
        self._clear_time_s: float = 0.0
        self._last_confidence: float = 0.0

    # ── public API ────────────────────────────────────────────────────

    def update(
        self,
        face_present: bool,
        confidence: float,
        dt: float,
    ) -> TrackerState:
        """Process one frame and return the updated tracker state.

        Args:
            face_present: Whether the face detector found a face.
            confidence: Detection confidence (0-1), kept for reporting.
            dt: Seconds elapsed since the previous frame.

        Returns:
            Current :class:`TrackerState` snapshot.
        """
        thresholds = get_thresholds().temporal
        trigger = not face_present

        if trigger:
            self._active = True
            self._accumulated_time_s += dt
            self._accumulated_frames += 1
            self._clear_time_s = 0.0
            self._last_confidence = clamp(1.0 - confidence)

            if self._accumulated_time_s >= thresholds.face_missing_duration_s:
                self._violated = True
        else:
            self._active = False
            self._clear_time_s += dt
            self._last_confidence = clamp(confidence)

            if self._clear_time_s >= thresholds.hysteresis_clear_duration_s:
                self._accumulated_time_s = 0.0
                self._accumulated_frames = 0
                self._violated = False

        return self.get_state()

    def reset(self) -> None:
        """Reset all internal state to defaults."""
        self._active = False
        self._violated = False
        self._accumulated_time_s = 0.0
        self._accumulated_frames = 0
        self._clear_time_s = 0.0
        self._last_confidence = 0.0

    def get_state(self) -> TrackerState:
        """Return a snapshot of the current tracker state."""
        return TrackerState(
            rule_code=RuleCode.FACE_MISSING,
            active=self._active,
            violated=self._violated,
            accumulated_time_s=self._accumulated_time_s,
            accumulated_frames=self._accumulated_frames,
            confidence=self._last_confidence,
        )
