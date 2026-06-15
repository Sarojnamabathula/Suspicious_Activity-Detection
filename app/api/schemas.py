"""
SentinelAI — Shared Pydantic Schemas.

Canonical data models exchanged between layers.  Every detector, tracker,
engine, and API endpoint references these — never ad-hoc dicts.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


# ──────────────────────────────────────────────────────────────────────
# Enums
# ──────────────────────────────────────────────────────────────────────

class Severity(str, Enum):
    """Violation severity levels."""
    SAFE = "SAFE"
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"

class Priority(str, Enum):
    """Alert priority for enterprise categorization."""
    P1 = "P1"
    P2 = "P2"
    P3 = "P3"

class GazeStatus(str, Enum):
    """Eye-gaze classification."""
    NORMAL = "NORMAL"
    AWAY = "AWAY"
    UNKNOWN = "UNKNOWN"


class RuleCode(str, Enum):
    """Identifiers for every temporal-validation rule."""
    FACE_MISSING = "FACE_MISSING"
    MULTIPLE_PERSONS = "MULTIPLE_PERSONS"
    PHONE_DETECTED = "PHONE_DETECTED"
    GAZE_AWAY = "GAZE_AWAY"
    SUSPICIOUS_OBJECT = "SUSPICIOUS_OBJECT"
    MOTION_BURST = "MOTION_BURST"


# ──────────────────────────────────────────────────────────────────────
# Bounding box
# ──────────────────────────────────────────────────────────────────────

class BoundingBox(BaseModel):
    """Axis-aligned bounding box in pixel coordinates."""
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    label: str = ""


# ──────────────────────────────────────────────────────────────────────
# Detection results (output of detector layer)
# ──────────────────────────────────────────────────────────────────────

class FaceDetectionResult(BaseModel):
    """Output of the face detector for a single frame."""
    face_present: bool = False
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    bounding_box: Optional[BoundingBox] = None


class ObjectDetectionResult(BaseModel):
    """Output of the YOLO object detector for a single frame."""
    person_count: int = Field(default=0, ge=0)
    phone_detected: bool = False
    phone_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    suspicious_objects: list[BoundingBox] = Field(default_factory=list)
    person_boxes: list[BoundingBox] = Field(default_factory=list)
    phone_box: Optional[BoundingBox] = None


class GazeDetectionResult(BaseModel):
    """Output of the gaze/head-pose estimator for a single frame."""
    gaze_status: GazeStatus = GazeStatus.UNKNOWN
    horizontal_offset: float = Field(default=0.0)
    vertical_offset: float = Field(default=0.0)
    head_yaw: float = Field(default=0.0)
    head_pitch: float = Field(default=0.0)


class MotionDetectionResult(BaseModel):
    """Output of the motion/background-subtraction detector."""
    motion_level: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Fraction of frame area with detected motion.",
    )
    high_motion: bool = False


# ──────────────────────────────────────────────────────────────────────
# Composite per-frame detection payload
# ──────────────────────────────────────────────────────────────────────

class FrameDetections(BaseModel):
    """Aggregated detector outputs for a single frame."""
    frame_id: int = 0
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    face: FaceDetectionResult = Field(default_factory=FaceDetectionResult)
    objects: ObjectDetectionResult = Field(default_factory=ObjectDetectionResult)
    gaze: GazeDetectionResult = Field(default_factory=GazeDetectionResult)
    motion: MotionDetectionResult = Field(default_factory=MotionDetectionResult)


# ──────────────────────────────────────────────────────────────────────
# Tracker state (output of tracking layer)
# ──────────────────────────────────────────────────────────────────────

class TrackerState(BaseModel):
    """State of a single temporal tracker."""
    rule_code: RuleCode
    active: bool = False
    violated: bool = False
    accumulated_time_s: float = 0.0
    accumulated_frames: int = 0
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


# ──────────────────────────────────────────────────────────────────────
# Decision engine output
# ──────────────────────────────────────────────────────────────────────

class DecisionOutput(BaseModel):
    """
    Final structured payload emitted by the decision engine each frame.

    This is the canonical wire format consumed by the dashboard, API,
    logging pipeline, and evidence capture system.
    """
    timestamp: str = ""
    frame_id: int = 0
    suspicious: bool = False
    severity: Severity = Severity.SAFE
    reasons: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    face_present: bool = True
    person_count: int = 0
    phone_detected: bool = False
    gaze_status: GazeStatus = GazeStatus.UNKNOWN
    risk_score: int = Field(default=0, ge=0, le=100)


# ──────────────────────────────────────────────────────────────────────
# Evidence index entry
# ──────────────────────────────────────────────────────────────────────

class EvidenceEntry(BaseModel):
    """Single row in evidence/index.json."""
    filename: str
    rule_code: str
    severity: str
    risk_score: int
    timestamp: str
    reasons: list[str] = Field(default_factory=list)


# ──────────────────────────────────────────────────────────────────────
# API telemetry response
# ──────────────────────────────────────────────────────────────────────

class TelemetryResponse(BaseModel):
    """Shape of the /api/status JSON response."""
    status: str = "ok"
    uptime_seconds: float = 0.0
    total_frames_processed: int = 0
    total_alerts: int = 0
    current_decision: Optional[DecisionOutput] = None


class AlertHistoryItem(BaseModel):
    """Single alert record for the /api/alerts endpoint."""
    id: int
    rule_code: str
    severity: str
    priority: str = "P3"
    risk_score: int
    timestamp: str
    reasons: list[str] = Field(default_factory=list)
    evidence_file: Optional[str] = None
