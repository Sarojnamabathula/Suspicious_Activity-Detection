"""
SentinelAI — Gaze-Away Tracker.

Accumulates duration while gaze is classified as AWAY and raises a
GAZE_AWAY violation once the configured ``gaze_away_duration_s``
threshold is exceeded.  Frames with ``GazeStatus.UNKNOWN`` are
silently ignored — they neither accumulate nor clear the timer.
"""

from __future__ import annotations

from app.api.schemas import GazeStatus, RuleCode, TrackerState
from app.config.thresholds import get_thresholds


class GazeTracker:
    """Temporal tracker for the GAZE_AWAY rule.

    ``UNKNOWN`` frames are treated as a no-op: the tracker neither
    accumulates nor resets on them.  This prevents noisy gaze estimates
    (e.g. when the face is partially occluded) from polluting the
    temporal signal.
    """

    def __init__(self) -> None:
        self._active: bool = False
        self._violated: bool = False
        self._accumulated_time_s: float = 0.0
        self._accumulated_frames: int = 0
        self._clear_time_s: float = 0.0
        self._last_confidence: float = 0.0

    # ── public API ────────────────────────────────────────────────────

    def update(self, gaze_status: GazeStatus, dt: float) -> TrackerState:
        """Process one frame and return the updated tracker state.

        Args:
            gaze_status: Gaze classification for the current frame.
            dt: Seconds elapsed since the previous frame.

        Returns:
            Current :class:`TrackerState` snapshot.
        """
        if gaze_status == GazeStatus.UNKNOWN:
            return self.get_state()

        thresholds = get_thresholds().temporal
        trigger: bool = gaze_status == GazeStatus.AWAY

        if trigger:
            self._active = True
            self._accumulated_time_s += dt
            self._accumulated_frames += 1
            self._clear_time_s = 0.0
            self._last_confidence = 1.0

            if self._accumulated_time_s >= thresholds.gaze_away_duration_s:
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
            rule_code=RuleCode.GAZE_AWAY,
            active=self._active,
            violated=self._violated,
            accumulated_time_s=self._accumulated_time_s,
            accumulated_frames=self._accumulated_frames,
            confidence=self._last_confidence,
        )
