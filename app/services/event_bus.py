"""
SentinelAI — Event Bus Architecture
Decouples detectors, rule engines, and loggers.
"""

import logging
from typing import Callable, Dict, List, Any
from enum import Enum
from dataclasses import dataclass

logger = logging.getLogger(__name__)

class EventType(Enum):
    DETECTION_READY = "DETECTION_READY"
    RULE_VIOLATION = "RULE_VIOLATION"
    SESSION_START = "SESSION_START"
    SESSION_END = "SESSION_END"
    APP_SHUTDOWN = "APP_SHUTDOWN"

@dataclass
class Event:
    type: EventType
    payload: Any

class EventBus:
    def __init__(self):
        self._subscribers: Dict[EventType, List[Callable[[Event], None]]] = {
            event_type: [] for event_type in EventType
        }

    def subscribe(self, event_type: EventType, callback: Callable[[Event], None]) -> None:
        if callback not in self._subscribers[event_type]:
            self._subscribers[event_type].append(callback)

    def unsubscribe(self, event_type: EventType, callback: Callable[[Event], None]) -> None:
        if callback in self._subscribers[event_type]:
            self._subscribers[event_type].remove(callback)

    def publish(self, event: Event) -> None:
        for callback in self._subscribers[event.type]:
            try:
                callback(event)
            except Exception as e:
                logger.error(f"Error in event subscriber for {event.type.name}: {e}", exc_info=True)

# Global event bus singleton
bus = EventBus()
