"""
SentinelAI — Performance Monitoring
Tracks FPS, inference times, and hardware usage metrics.
"""

import time
import psutil
from collections import deque
from pydantic import BaseModel

class PerformanceStats(BaseModel):
    fps: float
    inference_time_ms: float
    cpu_percent: float
    memory_mb: float

class PerformanceMonitor:
    def __init__(self, window_size: int = 30):
        self._frame_times = deque(maxlen=window_size)
        self._inference_times = deque(maxlen=window_size)
        self._last_frame_time = time.time()
        
    def record_frame(self, inference_duration_s: float):
        now = time.time()
        dt = now - self._last_frame_time
        if dt > 0:
            self._frame_times.append(dt)
        self._last_frame_time = now
        self._inference_times.append(inference_duration_s)
        
    def get_stats(self) -> PerformanceStats:
        avg_dt = sum(self._frame_times) / max(len(self._frame_times), 1)
        fps = 1.0 / avg_dt if avg_dt > 0 else 0.0
        
        avg_inf = sum(self._inference_times) / max(len(self._inference_times), 1)
        inf_ms = avg_inf * 1000.0
        
        process = psutil.Process()
        cpu_percent = process.cpu_percent()
        memory_mb = process.memory_info().rss / (1024 * 1024)
        
        return PerformanceStats(
            fps=round(fps, 1),
            inference_time_ms=round(inf_ms, 1),
            cpu_percent=round(cpu_percent, 1),
            memory_mb=round(memory_mb, 1)
        )
