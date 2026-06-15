"""
SentinelAI — Alert Service.
Manages alert lifecycle, fingerprint deduplication, and prioritisation.
"""

from __future__ import annotations

import logging
import time

from app.api.schemas import AlertHistoryItem, DecisionOutput, Priority
from app.config.thresholds import get_thresholds
from app.services.logger_service import get_logger
from app.services.event_bus import bus, Event, EventType

logger = get_logger("alerts")

class AlertService:
    """Manages alerts, prioritisation, and deduplication via fingerprinting."""

    def __init__(self) -> None:
        self._fingerprints: dict[str, float] = {}
        self._alert_history: list[AlertHistoryItem] = []
        self._alert_counter: int = 0
        self._cooldown_seconds = get_thresholds().alert.cooldown_seconds

    def _get_priority(self, rule_code: str) -> Priority:
        if rule_code in ["PHONE_DETECTED", "MULTIPLE_PERSONS"]:
            return Priority.P1
        if rule_code in ["FACE_MISSING", "SUSPICIOUS_OBJECT"]:
            return Priority.P2
        return Priority.P3

    def should_alert(self, rule_code: str) -> bool:
        """Fingerprint-based deduplication window."""
        now = time.time()
        # Fingerprint is the rule_code (in a real app, might include candidate ID or object ID)
        fingerprint = rule_code
        last_time = self._fingerprints.get(fingerprint, 0.0)
        return (now - last_time) >= self._cooldown_seconds

    def record_alert(self, decision: DecisionOutput, evidence_file: str | None = None) -> AlertHistoryItem | None:
        """Record an alert if it passes the fingerprint deduplication check."""
        if not decision.reasons:
            return None

        alert_triggered = False
        now = time.time()
        
        for reason in decision.reasons:
            if self.should_alert(reason):
                alert_triggered = True
                self._fingerprints[reason] = now
                
        if not alert_triggered:
            return None

        self._alert_counter += 1
        
        # Primary reason is the first one
        primary_rule = decision.reasons[0]
        priority = self._get_priority(primary_rule)
        
        item = AlertHistoryItem(
            id=self._alert_counter,
            rule_code=primary_rule,
            severity=decision.severity.value,
            priority=priority.value,
            risk_score=decision.risk_score,
            timestamp=decision.timestamp,
            reasons=decision.reasons,
            evidence_file=evidence_file
        )
        
        self._alert_history.append(item)
        
        # Publish to Enterprise Event Bus
        bus.publish(Event(EventType.RULE_VIOLATION, payload=item))
        
        msg = f"ALERT: {primary_rule} | Priority: {priority.value} | Severity: {decision.severity.value} | Risk: {decision.risk_score}"
        if hasattr(logger, "alert"):
            logger.alert(msg)
        else:
            logger.warning(msg)
        
        return item

    def get_history(self) -> list[AlertHistoryItem]:
        return self._alert_history

    def get_session_alert_count(self) -> int:
        return self._alert_counter

    @property
    def total_alerts(self) -> int:
        return self._alert_counter
