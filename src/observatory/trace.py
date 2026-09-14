"""Traces: runs as portable artifacts.

A trace is JSONL, one event per line, in the order the session emitted them.
Replaying a trace pushes the same events through the same bus the live session
uses, so the observatory renders a recorded run and a live run identically —
and a run from someone else's machine just as well.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Iterator

from .events import Event


class TraceWriter:
    """Append-only JSONL.

    A closed writer reopens itself on the next write. That is deliberate: a
    session ends, closing the trace, and then a rewind resurrects it to explore
    a branch. Those branch turns belong in the same file as the run they forked
    from — a trace holds a trajectory *tree*, not a single line, and the rewind
    events in the stream are what mark the fork points.
    """

    def __init__(self, path: str | Path, meta: dict[str, Any] | None = None) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("w", encoding="utf-8")
        header = {"kind": "trace.header", "version": 1, "written_at": time.time()}
        header.update(meta or {})
        self._write(header)

    def _write(self, obj: dict[str, Any]) -> None:
        if self._fh.closed:
            self._fh = self.path.open("a", encoding="utf-8")
        self._fh.write(json.dumps(obj, default=str) + "\n")
        self._fh.flush()

    def write(self, event: Event) -> None:
        self._write({"kind": "event", **event.to_dict()})

    def close(self) -> None:
        if not self._fh.closed:
            self._fh.close()

    def __enter__(self) -> "TraceWriter":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


def read_trace(path: str | Path) -> Iterator[dict[str, Any]]:
    """Yield raw records. The header comes first, then events in order."""
    with Path(path).open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def read_events(path: str | Path) -> Iterator[Event]:
    for record in read_trace(path):
        if record.get("kind") == "event":
            yield Event(
                type=record["type"],
                payload=record["payload"],
                seq=record.get("seq", 0),
                ts=record.get("ts", 0.0),
            )


def trace_header(path: str | Path) -> dict[str, Any]:
    for record in read_trace(path):
        if record.get("kind") == "trace.header":
            return record
    return {}
