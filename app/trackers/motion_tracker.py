"""
SentinelAI — Motion-Burst Tracker.

Accumulates duration while background motion is classified as "high"
and raises a MOTION_BURST violation once ``motion_burst_duration_s``
is exceeded.  Hysteresis prevents premature reset on brief calm
patches within an otherwise turbulent sequence.
"""

from __future__ import annotations

from app.api.schemas import RuleCode, TrackerState
from app.config.thresholds import get_thresholds


class MotionTracker:
    """Temporal tracker for the MOTION_BURST rule.

    Uses a pure time-based gate: the violation latches once
    accumulated high-motion time reaches the configured threshold.
    """

    def __init__(self) -> None:
        self._active: bool = False
        self._violated: bool = False
        self._accumulated_time_s: float = 0.0
        self._accumulated_frames: int = 0
        self._clear_time_s: float = 0.0
        self._last_confidence: float = 0.0

    # ── public API ────────────────────────────────────────────────────

    def update(self, high_motion: bool, dt: float) -> TrackerState:
        """Process one frame and return the updated tracker state.

        Args:
            high_motion: Whether the motion detector flagged this frame.
            dt: Seconds elapsed since the previous frame.

        Returns:
            Current :class:`TrackerState` snapshot.
        """
        thresholds = get_thresholds().temporal
        trigger: bool = high_motion

        if trigger:
            self._active = True
            self._accumulated_time_s += dt
            self._accumulated_frames += 1
            self._clear_time_s = 0.0
            self._last_confidence = 1.0

            if self._accumulated_time_s >= thresholds.motion_burst_duration_s:
                self._violated = True
        else:
            self._active = False
            self._clear_time_s += dt
            self._last_confidence = 0.0

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
            rule_code=RuleCode.MOTION_BURST,
            active=self._active,
            violated=self._violated,
            accumulated_time_s=self._accumulated_time_s,
            accumulated_frames=self._accumulated_frames,
            confidence=self._last_confidence,
        )
