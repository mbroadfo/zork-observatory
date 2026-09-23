"""A live board for a sweep in flight.

The model server's log says a request took 2.5 seconds, four hundred times.
This says which arm is playing, how far in, what it has found, and whether it
is going anywhere — read from the traces as they are written.

    python tools/sweep_status.py            # refresh until the sweep ends
    python tools/sweep_status.py --once     # print once and exit

Only the tail of each trace is parsed, so it stays cheap as the files grow.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

TAIL_BYTES = 400_000        # enough for the last few hundred events
CLEAR = "\033[2J\033[H"
DIM, BOLD, GREEN, YELLOW, RED, RESET = "\033[2m", "\033[1m", "\033[32m", "\033[33m", "\033[31m", "\033[0m"


def tail_events(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Events from the end of a trace, and the header if it is still in reach."""
    size = path.stat().st_size
    with path.open("rb") as fh:
        if size > TAIL_BYTES:
            fh.seek(size - TAIL_BYTES)
            fh.readline()       # the line we landed inside of
        raw = fh.read().decode("utf-8", errors="replace")
    events, header = [], {}
    for line in raw.splitlines():
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if row.get("kind") == "event":
            events.append(row)
        elif row.get("kind") == "trace.header":
            header = row
    return events, header


def last(events: list[dict[str, Any]], type_: str) -> dict[str, Any]:
    for row in reversed(events):
        if row["type"] == type_:
            return row["payload"]
    return {}


def arm_state(path: Path) -> dict[str, Any]:
    events, _ = tail_events(path)
    if not events:
        return {"name": path.stem, "turn": 0, "status": "starting"}
    snap = last(events, "state.snapshot")
    obs = last(events, "observation")
    ended = last(events, "session.ended")
    quality = obs.get("quality") or {}
    issued = last(events, "command.issued")
    return {
        "name": path.stem,
        "turn": ended.get("turns") or issued.get("turn") or 0,
        "score": ended.get("final_score", snap.get("score", 0)),
        "rooms": (snap.get("coverage") or {}).get("rooms_seen", 0),
        "objects": (snap.get("coverage") or {}).get("objects_seen", 0),
        "futile": quality.get("futile_pct", 0.0),
        "wasted": quality.get("wasted_pct", 0.0),
        "distinct": quality.get("distinct_commands", 0),
        "command": issued.get("command", ""),
        "where": snap.get("location_name", ""),
        "status": "done" if ended else "running",
        "reason": ended.get("reason", ""),
        "mtime": path.stat().st_mtime,
    }


def render(root: Path) -> str:
    traces = sorted(root.glob("*.jsonl"))
    arms = [arm_state(p) for p in traces]
    now = time.time()
    out = [f"{BOLD}sweep{RESET}  {len(arms)} arm(s) so far  ·  {time.strftime('%H:%M:%S')}", ""]
    head = f"{'arm':<18}{'turn':>6}{'score':>6}{'rooms':>6}{'obj':>5}{'futile':>8}{'wasted':>8}{'distinct':>9}  where / last command"
    out.append(head)
    out.append("-" * (len(head) + 12))
    for a in arms:
        live = a["status"] == "running" and now - a["mtime"] < 120
        colour = GREEN if live else DIM if a["status"] == "done" else YELLOW
        futile = a.get("futile", 0.0)
        mark = RED if futile >= 20 else RESET
        tail = f"{a.get('where', '')[:22]:<22} {DIM}{a.get('command', '')[:28]}{RESET}"
        out.append(
            f"{colour}{a['name']:<18}{RESET}{a['turn']:>6}{a.get('score', 0):>6}"
            f"{a.get('rooms', 0):>6}{a.get('objects', 0):>5}"
            f"{mark}{futile:>7}%{RESET}{a.get('wasted', 0.0):>7}%{a.get('distinct', 0):>9}  {tail}"
        )
        if a["status"] == "done" and a["reason"]:
            out.append(f"{DIM}{'':<18}{a['reason']}{RESET}")
    log = root / "sweep.log"
    if log.exists():
        out += ["", f"{DIM}" + log.read_text(encoding="utf-8").strip().splitlines()[-1] + RESET]
    return "\n".join(out)


def main(*args: str) -> int:
    root = Path(next((a for a in args if not a.startswith("-")), "traces/sweep"))
    if not root.exists():
        print(f"no sweep at {root}")
        return 1
    if "--once" in args:
        print(render(root))
        return 0
    try:
        while True:
            print(CLEAR + render(root), flush=True)
            time.sleep(3)
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    raise SystemExit(main(*sys.argv[1:]))
