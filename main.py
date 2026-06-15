"""
SentinelAI — Main Entry Point.
Starts the processing loop, terminal dashboard, and FastAPI server.
"""

import sys
import threading
import time
import concurrent.futures
import cv2
import numpy as np
import uvicorn

from app.config.settings import get_settings, StreamMode
from app.services.logger_service import setup_logging
from app.services.evidence_service import EvidenceService
from app.services.alert_service import AlertService
from app.services.perf_monitor import PerformanceMonitor
from app.services.session_service import SessionService
from app.database.repository import DatabaseRepository
from app.detectors.face_detector import FaceDetector
from app.detectors.object_detector import ObjectDetector
from app.detectors.gaze_detector import GazeDetector
from app.detectors.motion_detector import MotionDetector
from app.engine.decision_engine import DecisionEngine
from app.api.schemas import FrameDetections
from app.api.routes import create_app, init_api_state, update_decision, update_stream_frame

try:
    from app.utils.annotator import FrameAnnotator
except ImportError:
    FrameAnnotator = None

logger = setup_logging()

def video_capture_loop(buffer, stop_event):
    """Continuously fetches frames into the thread-safe buffer."""
    settings = get_settings()
    
    if settings.stream_mode == StreamMode.WEBCAM:
        cap = cv2.VideoCapture(settings.webcam_index)
    elif settings.stream_mode == StreamMode.VIDEO_FILE:
        cap = cv2.VideoCapture(settings.video_file_path)
    else:
        cap = None

    if cap:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, settings.frame_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, settings.frame_height)
        cap.set(cv2.CAP_PROP_FPS, settings.target_fps)

    frame_delay = 1.0 / settings.target_fps
    
    while not stop_event.is_set():
        start_time = time.time()
        
        if settings.stream_mode == StreamMode.SIMULATION:
            # Generate synthetic frame
            frame = np.zeros((settings.frame_height, settings.frame_width, 3), dtype=np.uint8)
            cv2.putText(frame, "SIMULATION MODE", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
            buffer.put(frame)
        else:
            if cap and cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    buffer.put(frame)
                else:
                    if settings.stream_mode == StreamMode.VIDEO_FILE:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0) # Loop video
            
        elapsed = time.time() - start_time
        time.sleep(max(0, frame_delay - elapsed))

    if cap:
        cap.release()

def process_loop(buffer, stop_event, alert_service, evidence_service, perf_monitor, session_service):
    """Consumes frames, runs inference, evaluates rules, and records performance."""
    face_detector = FaceDetector()
    obj_detector = ObjectDetector()
    gaze_detector = GazeDetector()
    motion_detector = MotionDetector()
    
    decision_engine = DecisionEngine()
    if FrameAnnotator:
        annotator = FrameAnnotator()
    else:
        annotator = None
        
    settings = get_settings()
    frame_count = 0
    last_time = time.time()
    
    while not stop_event.is_set():
        frame = buffer.get(timeout=0.5)
        if frame is None:
            continue
            
        now = time.time()
        dt = now - last_time
        last_time = now
        frame_count += 1
        
        # Run all detectors IN PARALLEL using a thread pool
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            f_face   = pool.submit(face_detector.detect,   frame)
            f_obj    = pool.submit(obj_detector.detect,    frame)
            f_gaze   = pool.submit(gaze_detector.detect,   frame)
            f_motion = pool.submit(motion_detector.detect, frame)

        face_res   = f_face.result()
        obj_res    = f_obj.result()
        gaze_res   = f_gaze.result()
        motion_res = f_motion.result()
        
        detections = FrameDetections(
            frame_id=frame_count,
            face=face_res,
            objects=obj_res,
            gaze=gaze_res,
            motion=motion_res
        )
        
        # Evaluate
        decision = decision_engine.process_frame(detections, dt)
        
        # Alert & Evidence
        if decision.suspicious:
            alert_item = alert_service.record_alert(decision)
            if alert_item:
                evidence_file = evidence_service.capture(frame, decision, detections)
                alert_item.evidence_file = evidence_file
                
        # Update Session & Performance
        session_service.update_session(decision.risk_score, alert_service.total_alerts)
        perf_monitor.record_frame(dt)
                
        # Update API State
        update_decision(decision, frame_count)
        
        # Always annotate the frame so it can be streamed to the web dashboard
        if annotator:
            annotated = annotator.annotate(frame, decision, detections)
            update_stream_frame(annotated)
            
            # Optional Window
            if not settings.headless:
                cv2.imshow("SentinelAI Monitor", annotated)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    stop_event.set()

    face_detector.close()
    obj_detector.close()
    gaze_detector.close()
    motion_detector.close()
    cv2.destroyAllWindows()

def run_dashboard(stop_event):
    """Terminal dashboard using Rich."""
    try:
        from rich.live import Live
        from rich.layout import Layout
        from rich.panel import Panel
        from rich.table import Table
        from app.api.routes import _state
    except ImportError:
        logger.warning("Rich not installed, disabling dashboard.")
        return

    settings = get_settings()
    
    def generate_ui():
        decision = _state.get("current_decision")
        if not decision:
            return Panel("Waiting for frames...", title="SentinelAI Live Monitor")
            
        grid = Table.grid(expand=True)
        grid.add_column(justify="left")
        grid.add_column(justify="right")
        
        grid.add_row("Face Present", "✓" if decision.face_present else "✗ (MISSING)")
        grid.add_row("Persons", str(decision.person_count))
        grid.add_row("Phone", "Detected!" if decision.phone_detected else "None")
        grid.add_row("Gaze", decision.gaze_status.value)
        grid.add_row("Risk Score", f"{decision.risk_score} / 100")
        
        color = "green" if decision.severity.value == "SAFE" else "red"
        grid.add_row("Severity", f"[{color}]{decision.severity.value}[/{color}]")
        
        panel = Panel(grid, title="[bold cyan]SentinelAI — Live Monitor[/bold cyan]", border_style="cyan")
        return panel

    with Live(generate_ui(), refresh_per_second=settings.dashboard_refresh_hz) as live:
        while not stop_event.is_set():
            live.update(generate_ui())
            time.sleep(1.0 / settings.dashboard_refresh_hz)

def main():
    logger.info("Starting SentinelAI...")
    settings = get_settings()
    
    from app.utils.frame_buffer import FrameBuffer
    buffer = FrameBuffer(maxlen=5)
    stop_event = threading.Event()
    
    alert_service = AlertService()
    evidence_service = EvidenceService()
    
    # Enterprise Services
    db_repo = DatabaseRepository()
    session_service = SessionService(db_repo)
    session_service.start_session("guest-user")
    perf_monitor = PerformanceMonitor()
    
    init_api_state(alert_service, evidence_service, perf_monitor, session_service, time.time())
    
    cap_thread = threading.Thread(target=video_capture_loop, args=(buffer, stop_event))
    proc_thread = threading.Thread(target=process_loop, args=(buffer, stop_event, alert_service, evidence_service, perf_monitor, session_service))
    dash_thread = threading.Thread(target=run_dashboard, args=(stop_event,))
    
    cap_thread.start()
    proc_thread.start()
    
    if settings.enable_dashboard:
        dash_thread.start()
        
    if settings.enable_api_server:
        app = create_app()
        uvicorn_config = uvicorn.Config(app, host=settings.api_host, port=settings.api_port, log_level="warning")
        server = uvicorn.Server(uvicorn_config)
        server_thread = threading.Thread(target=server.run)
        server_thread.start()

    try:
        while not stop_event.is_set():
            time.sleep(0.5)
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        stop_event.set()
        
    cap_thread.join()
    proc_thread.join()
    if settings.enable_dashboard:
        dash_thread.join()
    if settings.enable_api_server:
        server.should_exit = True
        server_thread.join()
        
    session_service.end_session()
    logger.info("SentinelAI shutdown complete.")

if __name__ == "__main__":
    main()
