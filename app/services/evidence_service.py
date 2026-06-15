"""
SentinelAI — Evidence Capture Service.
Saves annotated frames for auditing.
"""

from __future__ import annotations

import json
import logging
import cv2
import numpy as np

from app.api.schemas import DecisionOutput, EvidenceEntry, FrameDetections
from app.config.settings import get_settings
from app.utils.helpers import timestamp_filename

logger = logging.getLogger(__name__)

class EvidenceService:
    """Manages evidence snapshots and indexing."""

    def __init__(self) -> None:
        self.settings = get_settings()
        self.evidence_dir = self.settings.evidence_dir
        self.index_path = self.evidence_dir / "index.json"
        self._index: list[EvidenceEntry] = self._load_index()

    def _load_index(self) -> list[EvidenceEntry]:
        if not self.index_path.exists():
            return []
        try:
            with open(self.index_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                return [EvidenceEntry(**item) for item in data]
        except Exception as e:
            logger.error(f"Failed to load evidence index: {e}")
            return []

    def _save_index(self) -> None:
        try:
            with open(self.index_path, "w", encoding="utf-8") as f:
                json.dump([item.model_dump() for item in self._index], f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save evidence index: {e}")

    def capture(self, frame: np.ndarray, decision: DecisionOutput, detections: FrameDetections) -> str | None:
        """Save an annotated frame as evidence."""
        if not self.settings.enable_evidence_capture:
            return None

        rule_code = decision.reasons[0] if decision.reasons else "VIOLATION"
        filename = f"{rule_code}_{timestamp_filename()}.jpg"
        filepath = self.evidence_dir / filename

        try:
            # Create a simple annotated copy
            annotated = frame.copy()
            
            # Simple text overlay
            text = f"[{decision.severity.value}] Risk: {decision.risk_score} - {rule_code}"
            cv2.putText(annotated, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.putText(annotated, decision.timestamp, (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
            
            # Save it
            cv2.imwrite(str(filepath), annotated)
            logger.info(f"Evidence captured: {filename}")

            entry = EvidenceEntry(
                filename=filename,
                rule_code=rule_code,
                severity=decision.severity.value,
                risk_score=decision.risk_score,
                timestamp=decision.timestamp,
                reasons=decision.reasons
            )
            self._index.append(entry)
            self._save_index()
            
            return filename
            
        except Exception as e:
            logger.error(f"Failed to capture evidence: {e}")
            return None

    def get_index(self) -> list[EvidenceEntry]:
        """Return the evidence index."""
        return self._index

    def get_entries(self) -> list[EvidenceEntry]:
        """Return all evidence entries."""
        return self._index
