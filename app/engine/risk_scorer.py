"""
SentinelAI — Composite Risk Scorer.
Computes a sophisticated risk score using weights, time-decay, and repeat multipliers.
"""

from __future__ import annotations

import time
from collections import defaultdict

from app.api.schemas import TrackerState
from app.config.thresholds import get_thresholds
from app.utils.helpers import clamp_int
from app.services.event_bus import bus, Event, EventType

class RiskScorer:
    """Enterprise risk scoring engine."""

    def __init__(self):
        # Tracking history for decay and multipliers
        self._offense_counts = defaultdict(int)
        self._last_violation_times = {}
        
        # Subscribe to rule violations to increment offense counts
        bus.subscribe(EventType.RULE_VIOLATION, self._on_violation)

    def _on_violation(self, event: Event) -> None:
        alert_item = event.payload
        self._offense_counts[alert_item.rule_code] += 1
        self._last_violation_times[alert_item.rule_code] = time.time()

    def compute(self, violated_states: list[TrackerState]) -> int:
        """Compute integer risk score from 0-100."""
        if not violated_states:
            # Optionally implement a global decay if no violations are present
            # For now, if perfectly clean, risk is 0.
            return 0
            
        weights = get_thresholds().risk_weights
        total_score = 0.0
        now = time.time()
        
        for state in violated_states:
            rule_code = state.rule_code.value
            field_name = rule_code.lower()
            base_weight = getattr(weights, field_name, 0.0)
            
            # 1. Repeat Offense Multiplier
            # Each prior offense adds 10% to the weight, capped at 2.0x
            past_offenses = self._offense_counts.get(rule_code, 0)
            multiplier = min(2.0, 1.0 + (past_offenses * 0.1))
            
            # 2. Time-based Decay Function
            # If the violation was active recently, weight remains high.
            # If it hasn't triggered an EVENT recently, we apply slight decay.
            # (Though if it's in violated_states, it's currently active. 
            # We use decay for historical lingering risk if we were to return >0 when empty).
            # Here, we ensure active violations hit at least base_weight * multiplier.
            
            rule_score = base_weight * multiplier
            total_score += rule_score
            
        # Add lingering historical risk (decay function)
        for rule, last_time in self._last_violation_times.items():
            # If it's not currently active, add a decayed historical penalty
            if not any(s.rule_code.value == rule for s in violated_states):
                age = now - last_time
                if age < 60.0:  # linger for 60 seconds max
                    field_name = rule.lower()
                    base_weight = getattr(weights, field_name, 0.0)
                    decay_factor = max(0.0, 1.0 - (age / 60.0))
                    # Add max 30% of original weight as lingering risk
                    total_score += (base_weight * 0.3 * decay_factor)
            
        return clamp_int(int(total_score), 0, 100)
