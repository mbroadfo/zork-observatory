"""The event stream.

Everything the observatory shows is derived from a linear sequence of events.
Live play and trace replay both produce the same stream, so the front end never
knows (or cares) which one it is watching.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field, asdict
from typing import Any, Literal

EventType = Literal[
    "session.started",
    "turn.begin",
    "agent.thought",
    "command.issued",
    "observation",
    "state.snapshot",
    "map.update",
    "object.delta",
    "checkpoint.created",
    "discovery.made",
    "session.ended",
    "error",
]


@dataclass
class Event:
    type: EventType
    payload: dict[str, Any]
    seq: int = 0
    ts: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class EventBus:
    """Assigns sequence numbers, keeps a replayable backlog, fans out to sinks.

    The backlog is what lets a browser tab opened at turn 200 render the whole
    run instead of just the tail.
    """

    def __init__(self) -> None:
        self._seq = 0
        self.backlog: list[Event] = []
        self._sinks: list[Any] = []

    def subscribe(self, sink: Any) -> None:
        """`sink` is any callable taking an Event. May be async."""
        self._sinks.append(sink)

    def unsubscribe(self, sink: Any) -> None:
        if sink in self._sinks:
            self._sinks.remove(sink)

    def clear(self) -> None:
        self._seq = 0
        self.backlog.clear()

    def emit(self, type_: EventType, **payload: Any) -> Event:
        self._seq += 1
        event = Event(type=type_, payload=payload, seq=self._seq)
        self.backlog.append(event)
        for sink in list(self._sinks):
            sink(event)
        return event
