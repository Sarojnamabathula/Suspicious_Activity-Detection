"""
SentinelAI — Structured Logging Service.
"""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
import sys

from app.config.settings import get_settings

# Add custom ALERT level
ALERT_LEVEL_NUM = 25
logging.addLevelName(ALERT_LEVEL_NUM, "ALERT")

def alert(self, message, *args, **kws):
    if self.isEnabledFor(ALERT_LEVEL_NUM):
        self._log(ALERT_LEVEL_NUM, message, args, **kws)

logging.Logger.alert = alert

def setup_logging() -> logging.Logger:
    """Configure system-wide logging with RotatingFileHandlers."""
    settings = get_settings()
    settings.ensure_directories()
    
    logger = logging.getLogger("sentinel")
    logger.setLevel(getattr(logging, settings.log_level.upper(), logging.INFO))
    
    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s")
    
    # Main log file
    main_handler = RotatingFileHandler(
        filename=settings.log_dir / "sentinel.log",
        maxBytes=settings.log_max_bytes,
        backupCount=settings.log_backup_count,
        encoding="utf-8"
    )
    main_handler.setFormatter(formatter)
    logger.addHandler(main_handler)
    
    # Alerts-only log file
    alert_handler = RotatingFileHandler(
        filename=settings.log_dir / "alerts.log",
        maxBytes=settings.log_max_bytes,
        backupCount=settings.log_backup_count,
        encoding="utf-8"
    )
    alert_handler.setLevel(ALERT_LEVEL_NUM)
    alert_handler.setFormatter(formatter)
    logger.addHandler(alert_handler)
    
    # Console handler using Rich if available
    try:
        from rich.logging import RichHandler
        console_handler = RichHandler(rich_tracebacks=True, markup=True)
    except ImportError:
        console_handler = logging.StreamHandler(sys.stdout)
    
    console_handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(console_handler)
    
    return logger

def get_logger(name: str) -> logging.Logger:
    """Get a child logger."""
    return logging.getLogger(f"sentinel.{name}")
