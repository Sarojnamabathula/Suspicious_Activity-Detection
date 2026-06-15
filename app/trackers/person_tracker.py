"""
SentinelAI — Multiple-Persons Tracker.

Uses a rolling-window average of person counts to suppress transient
mis-detections and raises a MULTIPLE_PERSONS violation when the
smoothed count stays above 1.0 for the configured duration.
"""

from __future__ import annotations

import collections

from app.api.schemas import RuleCode, TrackerState
from app.config.thresholds import get_thresholds
from app.utils.helpers import clamp


class PersonTracker:
    """Temporal tracker for the MULTIPLE_PERSONS rule.

    Maintains a rolling window (``collections.deque``) of recent person
    counts.  The trigger fires when the rolling average exceeds 1.0,
    and the violation flag latches once the accumulated duration crosses
    ``multiple_persons_duration_s``.  Hysteresis prevents premature
    reset when the count briefly drops.
    """

    def __init__(self) -> None:
        window_size: int = get_thresholds().temporal.person_count_rolling_window
        self._window: collections.deque[int] = collections.deque(maxlen=window_size)
        self._active: bool = False
        self._violated: bool = False
        self._accumulated_time_s: float = 0.0
        self._accumulated_frames: int = 0
        self._clear_time_s: float = 0.0
        self._last_confidence: float = 0.0

    # ── public API ────────────────────────────────────────────────────

    def update(self, person_count: int, dt: float) -> TrackerState:
        """Process one frame and return the updated tracker state.

        Args:
            person_count: Number of persons detected in the current frame.
            dt: Seconds elapsed since the previous frame.

        Returns:
            Current :class:`TrackerState` snapshot.
        """
        thresholds = get_thresholds().temporal

        self._window.append(person_count)
        rolling_avg: float = sum(self._window) / len(self._window)
        trigger: bool = rolling_avg > 1.0

        if trigger:
            self._active = True
            self._accumulated_time_s += dt
            self._accumulated_frames += 1
            self._clear_time_s = 0.0
            # Confidence proportional to how far the average exceeds 1.
            self._last_confidence = clamp((rolling_avg - 1.0) / max(rolling_avg, 1.0))

            if self._accumulated_time_s >= thresholds.multiple_persons_duration_s:
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
        self._window.clear()
        self._active = False
        self._violated = False
        self._accumulated_time_s = 0.0
        self._accumulated_frames = 0
        self._clear_time_s = 0.0
        self._last_confidence = 0.0

    def get_state(self) -> TrackerState:
        """Return a snapshot of the current tracker state."""
        return TrackerState(
            rule_code=RuleCode.MULTIPLE_PERSONS,
            active=self._active,
            violated=self._violated,
            accumulated_time_s=self._accumulated_time_s,
            accumulated_frames=self._accumulated_frames,
            confidence=self._last_confidence,
        )
