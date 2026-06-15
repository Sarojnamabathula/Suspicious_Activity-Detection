"""
SentinelAI — FastAPI API Routes.

Defines the ``/api`` router with telemetry, alert history, evidence
listing, and health-check endpoints.  Runtime state is injected via
:func:`init_api_state` and updated each frame through :func:`update_decision`.
"""

from __future__ import annotations

import time
from typing import Any
import asyncio
import cv2
import numpy as np

from fastapi import APIRouter, FastAPI
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import os

from app.api.schemas import (
    AlertHistoryItem,
    DecisionOutput,
    EvidenceEntry,
    TelemetryResponse,
)

# ──────────────────────────────────────────────────────────────────────
# Router
# ──────────────────────────────────────────────────────────────────────

router = APIRouter(prefix="/api")

# ──────────────────────────────────────────────────────────────────────
# Shared mutable runtime state
# ──────────────────────────────────────────────────────────────────────

_state: dict[str, Any] = {
    "start_time": None,
    "frames_processed": 0,
    "current_decision": None,
    "alert_service": None,
    "evidence_service": None,
    "perf_monitor": None,
    "session_service": None,
    "latest_frame": None,
}


# ──────────────────────────────────────────────────────────────────────
# State helpers
# ──────────────────────────────────────────────────────────────────────

def init_api_state(
    alert_service: Any,
    evidence_service: Any,
    perf_monitor: Any = None,
    session_service: Any = None,
    start_time: float | None = None,
) -> None:
    """Inject shared services and record the session start time.

    Parameters
    ----------
    alert_service:
        Object exposing ``.get_history() -> list[AlertHistoryItem]``
        and ``.total_alerts -> int``.
    evidence_service:
        Object exposing ``.get_entries() -> list[EvidenceEntry]``.
    start_time:
        Monotonic epoch (``time.time()``).  Defaults to *now* when
        omitted.
    """
    _state["alert_service"] = alert_service
    _state["evidence_service"] = evidence_service
    _state["perf_monitor"] = perf_monitor
    _state["session_service"] = session_service
    _state["start_time"] = start_time if start_time is not None else time.time()


def update_decision(decision: DecisionOutput, frame_count: int) -> None:
    """Update the globally readable state with the newest frame's results."""
    _state["current_decision"] = decision
    _state["frames_processed"] = frame_count

def update_stream_frame(frame: np.ndarray) -> None:
    """Update the globally readable state with the latest annotated frame."""
    _state["latest_frame"] = frame


# ──────────────────────────────────────────────────────────────────────
# Endpoints
# ──────────────────────────────────────────────────────────────────────

@router.get("/status", response_model=TelemetryResponse)
async def get_status() -> TelemetryResponse:
    """Return live telemetry: uptime, frame count, alerts, and current decision."""
    start: float | None = _state["start_time"]
    uptime = (time.time() - start) if start is not None else 0.0

    alert_svc = _state["alert_service"]
    total_alerts: int = alert_svc.total_alerts if alert_svc is not None else 0

    return TelemetryResponse(
        status="ok",
        uptime_seconds=round(uptime, 2),
        total_frames_processed=_state["frames_processed"],
        total_alerts=total_alerts,
        current_decision=_state["current_decision"],
    )


@router.get("/alerts", response_model=list[AlertHistoryItem])
async def get_alerts() -> list[AlertHistoryItem]:
    """Return the full alert history from the alert service."""
    alert_svc = _state["alert_service"]
    if alert_svc is None:
        return []
    return alert_svc.get_history()


@router.get("/evidence", response_model=list[EvidenceEntry])
def get_evidence() -> Any:
    """Return an index of all captured evidence."""
    svc = _state["evidence_service"]
    if not svc:
        return []
    return svc.get_entries()

@router.get("/performance")
def get_performance() -> Any:
    """Return live performance metrics."""
    perf = _state["perf_monitor"]
    if not perf:
        return {"error": "Performance monitor not initialized"}
    return perf.get_stats().model_dump()

@router.get("/session/report")
def get_session_report() -> Any:
    """End the current session and return the final report."""
    session_svc = _state["session_service"]
    if not session_svc:
        return {"error": "Session service not initialized"}
    return session_svc.end_session()

async def generate_video_stream():
    """Generator yielding MJPEG frame boundaries continuously."""
    while True:
        frame = _state.get("latest_frame")
        if frame is None:
            await asyncio.sleep(0.1)
            continue
            
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            await asyncio.sleep(0.1)
            continue
            
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        
        # 30fps cap roughly
        await asyncio.sleep(0.033)

@router.get("/stream")
def video_stream():
    """Stream annotated frames to clients via MJPEG."""
    return StreamingResponse(generate_video_stream(), media_type="multipart/x-mixed-replace; boundary=frame")


@router.get("/health")
async def health_check() -> dict[str, str]:
    """Lightweight liveness probe."""
    return {"status": "healthy"}


# ──────────────────────────────────────────────────────────────────────
# Application factory
# ──────────────────────────────────────────────────────────────────────

def create_app() -> FastAPI:
    """Build and return the fully-configured FastAPI application.

    The factory:
    * sets application metadata (title, description, version),
    * registers the ``/api`` router,
    * enables CORS middleware with permissive dev-mode origins.
    """
    app = FastAPI(
        title="SentinelAI API",
        description=(
            "Real-time AI proctoring telemetry, alert history, "
            "and evidence retrieval."
        ),
        version="1.0.0",
    )

    # CORS — wide-open for local development; tighten in production.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(router)
    
    # Mount the frontend directory if it exists
    frontend_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "frontend")
    if os.path.exists(frontend_dir):
        app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")
        
    return app
