"""
SentinelAI — General-purpose helper utilities.

Timestamp formatting, directory bootstrapping, and small pure functions
used across multiple modules.
"""

from __future__ import annotations

from datetime import datetime, timezone


def iso_now() -> str:
    """Return the current UTC time as an ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def timestamp_filename() -> str:
    """Return a filesystem-safe timestamp string: YYYY-MM-DD_HH-MM-SS."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S")


def clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    """Clamp *value* to [lo, hi]."""
    return max(lo, min(hi, value))


def clamp_int(value: int, lo: int = 0, hi: int = 100) -> int:
    """Clamp an integer *value* to [lo, hi]."""
    return max(lo, min(hi, value))
