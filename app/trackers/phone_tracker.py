"""
SentinelAI — Phone-Detected Tracker.

Counts consecutive frames in which a phone is detected and raises a
PHONE_DETECTED violation once the count reaches the configured
``phone_consecutive_frames`` threshold.
"""

from __future__ import annotations

from app.api.schemas import RuleCode, TrackerState
from app.config.thresholds import get_thresholds
from app.utils.helpers import clamp


class PhoneTracker:
    """Temporal tracker for the PHONE_DETECTED rule.

    Uses a **frame-count** gate rather than a time-based gate.  The
    accumulator only resets after the phone has been *continuously
    absent* for ``hysteresis_clear_duration_s`` seconds.
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
        phone_detected: bool,
        confidence: float,
        dt: float,
    ) -> TrackerState:
        """Process one frame and return the updated tracker state.

        Args:
            phone_detected: Whether a phone was detected this frame.
            confidence: Detection confidence (0-1).
            dt: Seconds elapsed since the previous frame.

        Returns:
            Current :class:`TrackerState` snapshot.
        """
        thresholds = get_thresholds().temporal
        trigger: bool = phone_detected

        if trigger:
            self._active = True
            self._accumulated_time_s += dt
            self._accumulated_frames += 1
            self._clear_time_s = 0.0
            self._last_confidence = clamp(confidence)

            if self._accumulated_frames >= thresholds.phone_consecutive_frames:
                self._violated = True
        else:
            self._active = False
            self._clear_time_s += dt

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
            rule_code=RuleCode.PHONE_DETECTED,
            active=self._active,
            violated=self._violated,
            accumulated_time_s=self._accumulated_time_s,
            accumulated_frames=self._accumulated_frames,
            confidence=self._last_confidence,
        )
