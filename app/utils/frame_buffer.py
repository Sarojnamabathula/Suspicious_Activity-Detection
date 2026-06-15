"""
SentinelAI — Thread-safe Frame Buffer.

Provides a bounded FIFO queue for decoupling the frame-acquisition thread
from the processing pipeline, with optional frame-skip logic.
"""

from __future__ import annotations

import threading
from collections import deque
from typing import Optional

import numpy as np


class FrameBuffer:
    """
    Thread-safe, bounded frame buffer backed by :class:`collections.deque`.

    Parameters
    ----------
    maxlen : int
        Maximum number of frames held.  Oldest frames are silently dropped
        when the buffer is full (back-pressure strategy).
    """

    def __init__(self, maxlen: int = 30) -> None:
        self._buffer: deque[np.ndarray] = deque(maxlen=maxlen)
        self._lock = threading.Lock()
        self._not_empty = threading.Condition(self._lock)

    # ── Producer API ─────────────────────────────────────────────────

    def put(self, frame: np.ndarray) -> None:
        """Enqueue a frame (drops oldest if buffer is full)."""
        with self._lock:
            self._buffer.append(frame)
            self._not_empty.notify()

    # ── Consumer API ─────────────────────────────────────────────────

    def get(self, timeout: float | None = None) -> Optional[np.ndarray]:
        """
        Dequeue the oldest frame, blocking until one is available.

        Returns ``None`` if *timeout* expires without a frame.
        """
        with self._not_empty:
            while len(self._buffer) == 0:
                if not self._not_empty.wait(timeout=timeout):
                    return None
            return self._buffer.popleft()

    def get_latest(self) -> Optional[np.ndarray]:
        """Return the newest frame, discarding older ones.  Non-blocking."""
        with self._lock:
            if not self._buffer:
                return None
            frame = self._buffer[-1]
            self._buffer.clear()
            return frame

    # ── Inspection ───────────────────────────────────────────────────

    def __len__(self) -> int:
        with self._lock:
            return len(self._buffer)

    @property
    def empty(self) -> bool:
        """Return ``True`` if the buffer contains no frames."""
        with self._lock:
            return len(self._buffer) == 0
