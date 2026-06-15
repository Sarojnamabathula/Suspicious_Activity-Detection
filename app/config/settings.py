"""
SentinelAI — Global Application Settings.

Centralises all non-threshold configuration: paths, model locations,
video source parameters, API networking, and runtime flags.
"""

from __future__ import annotations

import os
from pathlib import Path
from enum import Enum

from pydantic import Field
from pydantic_settings import BaseSettings


# ──────────────────────────────────────────────────────────────────────
# Enums
# ──────────────────────────────────────────────────────────────────────

class StreamMode(str, Enum):
    """Frame acquisition source mode."""
    WEBCAM = "webcam"
    VIDEO_FILE = "video_file"
    SIMULATION = "simulation"


class InferenceDevice(str, Enum):
    """Hardware target for model inference."""
    CPU = "cpu"
    CUDA = "cuda"
    MPS = "mps"
    AUTO = "auto"


# ──────────────────────────────────────────────────────────────────────
# Path helpers
# ──────────────────────────────────────────────────────────────────────

_PROJECT_ROOT = Path(__file__).resolve().parents[2]  # SentinelAI/


# ──────────────────────────────────────────────────────────────────────
# Settings model
# ──────────────────────────────────────────────────────────────────────

class Settings(BaseSettings):
    """Immutable application-wide configuration loaded from env / .env."""

    # ── Project paths ────────────────────────────────────────────────
    project_root: Path = Field(default=_PROJECT_ROOT)
    log_dir: Path = Field(default=_PROJECT_ROOT / "logs")
    evidence_dir: Path = Field(default=_PROJECT_ROOT / "evidence")
    models_dir: Path = Field(default=_PROJECT_ROOT / "models")

    # ── Stream acquisition ───────────────────────────────────────────
    stream_mode: StreamMode = Field(
        default=StreamMode.WEBCAM,
        description="Frame source: webcam, video_file, or simulation.",
    )
    webcam_index: int = Field(default=0, ge=0)
    video_file_path: str = Field(default="")
    target_fps: int = Field(default=30, ge=1, le=120)
    frame_width: int = Field(default=640, ge=320)
    frame_height: int = Field(default=480, ge=240)

    # ── Model paths ──────────────────────────────────────────────────
    yolo_model_path: str = Field(
        default="yolov8n.pt",
        description="YOLOv8 weights file (relative to models_dir or absolute).",
    )
    inference_device: InferenceDevice = Field(default=InferenceDevice.AUTO)

    # ── API server ───────────────────────────────────────────────────
    api_host: str = Field(default="127.0.0.1")
    api_port: int = Field(default=8000, ge=1024, le=65535)

    # ── Logging ──────────────────────────────────────────────────────
    log_level: str = Field(default="DEBUG")
    log_max_bytes: int = Field(default=10 * 1024 * 1024)  # 10 MB
    log_backup_count: int = Field(default=5)

    # ── Dashboard ────────────────────────────────────────────────────
    dashboard_refresh_hz: int = Field(
        default=4,
        ge=1,
        le=30,
        description="Terminal dashboard refresh rate (Hz).",
    )

    # ── Runtime flags ────────────────────────────────────────────────
    enable_evidence_capture: bool = Field(default=True)
    enable_api_server: bool = Field(default=True)
    enable_dashboard: bool = Field(default=True)
    headless: bool = Field(
        default=False,
        description="If True, suppress OpenCV GUI windows.",
    )

    class Config:
        env_prefix = "SENTINEL_"
        env_file = ".env"
        env_file_encoding = "utf-8"

    # ── Derived helpers ──────────────────────────────────────────────

    def ensure_directories(self) -> None:
        """Create required output directories if they do not exist."""
        for d in (self.log_dir, self.evidence_dir, self.models_dir):
            d.mkdir(parents=True, exist_ok=True)

    def resolve_yolo_path(self) -> Path:
        """Return the fully-resolved path to the YOLO weights file."""
        p = Path(self.yolo_model_path)
        if p.is_absolute():
            return p
        return self.models_dir / p


# ──────────────────────────────────────────────────────────────────────
# Singleton accessor
# ──────────────────────────────────────────────────────────────────────

_settings: Settings | None = None


def get_settings() -> Settings:
    """Return the cached application settings singleton."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
