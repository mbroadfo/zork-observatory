"""Zork Observatory — an instrument for watching agents play interactive fiction."""

__version__ = "0.1.0"

from .events import Event, EventBus
from .session import Session, SessionConfig

__all__ = ["Event", "EventBus", "Session", "SessionConfig", "__version__"]
