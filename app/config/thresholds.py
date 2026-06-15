"""
SentinelAI — Detection & Violation Thresholds.

Every tuneable numeric constant lives here.  No magic numbers in logic code.
Values are grouped by subsystem and documented with units.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


# ──────────────────────────────────────────────────────────────────────
# Detection confidence floors
# ──────────────────────────────────────────────────────────────────────

class DetectionConfidence(BaseModel):
    """Minimum confidence scores to accept a detection as valid."""

    face_min_confidence: float = Field(
        default=0.6,
        ge=0.0, le=1.0,
        description="MediaPipe face detection confidence floor.",
    )
    yolo_min_confidence: float = Field(
        default=0.6,
        ge=0.0, le=1.0,
        description="YOLOv8 general detection confidence floor.",
    )
    phone_min_confidence: float = Field(
        default=0.30,
        ge=0.0, le=1.0,
        description="Phone-specific detection confidence floor.",
    )
    gaze_deviation_threshold: float = Field(
        default=0.25,
        ge=0.0, le=1.0,
        description="Normalised iris offset beyond which gaze is 'AWAY'.",
    )


# ──────────────────────────────────────────────────────────────────────
# Temporal rule thresholds
# ──────────────────────────────────────────────────────────────────────

class TemporalThresholds(BaseModel):
    """
    Duration / frame-count gates that prevent instant violation triggers.

    All durations are in **seconds** unless noted otherwise.
    Frame counts assume the configured target FPS.
    """

    # Rule 1 — FACE_MISSING
    face_missing_duration_s: float = Field(
        default=10.0,
        description="Seconds of continuous face absence before violation.",
    )

    # Rule 2 — MULTIPLE_PERSONS
    multiple_persons_duration_s: float = Field(
        default=3.0,
        description="Seconds of >1 person before violation.",
    )

    # Rule 3 — PHONE_DETECTED
    phone_consecutive_frames: int = Field(
        default=6,
        description="Consecutive frames with phone before violation.",
    )

    # Rule 4 — GAZE_AWAY
    gaze_away_duration_s: float = Field(
        default=5.0,
        description="Seconds of off-screen gaze before violation.",
    )

    # Rule 5 — SUSPICIOUS_OBJECT
    suspicious_object_consecutive_frames: int = Field(
        default=15,
        description="Consecutive frames with book/cheatsheet before violation.",
    )

    # Rule 6 — MOTION_BURST
    motion_burst_duration_s: float = Field(
        default=2.0,
        description="Seconds of high background motion before violation.",
    )

    # ── False-positive suppression ───────────────────────────────────
    hysteresis_clear_duration_s: float = Field(
        default=3.0,
        description="Seconds of 'clear' signal required to reset a tracker.",
    )
    person_count_rolling_window: int = Field(
        default=5,
        description="Frame window for rolling-average person count.",
    )


# ──────────────────────────────────────────────────────────────────────
# Alert / cooldown thresholds
# ──────────────────────────────────────────────────────────────────────

class AlertThresholds(BaseModel):
    """Cooldown and deduplication settings for alert management."""

    cooldown_seconds: float = Field(
        default=30.0,
        description="Min seconds between repeated alerts for the same rule.",
    )


# ──────────────────────────────────────────────────────────────────────
# Risk scoring weights
# ──────────────────────────────────────────────────────────────────────

class RiskWeights(BaseModel):
    """Per-rule contribution to the composite risk score (0–100)."""

    face_missing: float = Field(default=30.0)
    multiple_persons: float = Field(default=25.0)
    phone_detected: float = Field(default=25.0)
    gaze_away: float = Field(default=10.0)
    suspicious_object: float = Field(default=20.0)
    motion_burst: float = Field(default=5.0)


# ──────────────────────────────────────────────────────────────────────
# Severity boundaries
# ──────────────────────────────────────────────────────────────────────

class SeverityBands(BaseModel):
    """Risk-score ranges that map to severity labels."""

    safe_max: int = Field(default=10)
    low_max: int = Field(default=30)
    medium_max: int = Field(default=55)
    high_max: int = Field(default=80)
    # Anything above high_max is CRITICAL


# ──────────────────────────────────────────────────────────────────────
# Motion detector tuning
# ──────────────────────────────────────────────────────────────────────

class MotionSettings(BaseModel):
    """Background-subtraction tunables."""

    history: int = Field(default=500, description="MOG2 history length.")
    var_threshold: float = Field(default=40.0, description="MOG2 variance threshold.")
    detect_shadows: bool = Field(default=False)
    motion_area_threshold: float = Field(
        default=0.05,
        ge=0.0, le=1.0,
        description="Fraction of frame area that counts as 'high motion'.",
    )


# ──────────────────────────────────────────────────────────────────────
# YOLO class IDs of interest (COCO dataset)
# ──────────────────────────────────────────────────────────────────────

class YoloClassConfig(BaseModel):
    """COCO class IDs used by the object detector."""

    person_class_id: int = Field(default=0)
    phone_class_id: int = Field(default=67)  # "cell phone"
    book_class_id: int = Field(default=73)   # "book"
    laptop_class_id: int = Field(default=63)

    suspicious_class_ids: list[int] = Field(
        default=[73, 63],
        description="COCO class IDs treated as suspicious objects.",
    )


# ──────────────────────────────────────────────────────────────────────
# Aggregate container
# ──────────────────────────────────────────────────────────────────────

class Thresholds(BaseModel):
    """Top-level container that bundles every threshold group."""

    detection: DetectionConfidence = Field(default_factory=DetectionConfidence)
    temporal: TemporalThresholds = Field(default_factory=TemporalThresholds)
    alert: AlertThresholds = Field(default_factory=AlertThresholds)
    risk_weights: RiskWeights = Field(default_factory=RiskWeights)
    severity: SeverityBands = Field(default_factory=SeverityBands)
    motion: MotionSettings = Field(default_factory=MotionSettings)
    yolo_classes: YoloClassConfig = Field(default_factory=YoloClassConfig)


# ──────────────────────────────────────────────────────────────────────
# Singleton accessor
# ──────────────────────────────────────────────────────────────────────

_thresholds: Thresholds | None = None


def get_thresholds() -> Thresholds:
    """Return the cached thresholds singleton."""
    global _thresholds
    if _thresholds is None:
        _thresholds = Thresholds()
    return _thresholds
