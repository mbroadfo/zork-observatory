"""Read the shape of the map back out of traces that were recorded without it.

The September 2026 sweep produced twelve runs and one table of scores, and the
thing that actually distinguished the arms — whether a player pushed outward or
paced — was in the traces the whole time, unmeasured. This rebuilds MapGraph
from the recorded events and runs world/frontier.py over it, so a run from
before those measures existed can be read with them.

It reconstructs rather than reads: every trace carries `command.issued` and the
`state.snapshot` that followed, which is exactly what the live session folds
into its map, so the graph built here is the graph that run had. Traces
recorded from now on also carry `structure` on each `map.update` and do not
need this — but the timeline does, because staleness drifts on turns when the
map does not change at all.

    python tools/structure.py                      # every arm, final shape
    python tools/structure.py traces/sweep         # ... from another directory
    python tools/structure.py --timeline traces/sweep/bare.jsonl
    python tools/structure.py --timeline traces/sweep/bare.jsonl --every 10
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterator

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from observatory.world.frontier import structure  # noqa: E402
from observatory.world.graph import MapGraph  # noqa: E402


def events(path: Path) -> Iterator[dict[str, Any]]:
    """Stream one trace. Traces run to tens of megabytes; do not slurp them."""
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except ValueError:      # a run killed mid-write leaves a half line
                continue
            if row.get("kind") == "event":
                yield row


def rebuild(path: Path, every: int = 0) -> tuple[MapGraph, list[dict[str, Any]], dict[str, Any]]:
    """Replay a trace into a map. Returns the map, samples, and the ending.

    `every` > 0 records a structure sample that many turns apart, which is the
    only way to see staleness: it climbs on precisely the turns the map does
    not change, so a sample taken per map update would never show it.
    """
    graph = MapGraph()
    samples: list[dict[str, Any]] = []
    ended: dict[str, Any] = {}
    command, text, done = "", "", False
    prev_room: str | None = None
    turn = 0
    next_sample = 0

    for row in events(path):
        kind, payload = row["type"], row["payload"]

        if kind == "command.issued":
            command = payload.get("command", "")
            turn = payload.get("turn", turn)
        elif kind == "observation":
            text = payload.get("text", "")
            done = bool(payload.get("done"))
        elif kind == "session.ended":
            ended = payload
        elif kind == "state.snapshot":
            room_id = payload.get("room_id") or MapGraph.room_id(
                payload.get("location_id", 0), payload.get("location_name", "")
            )
            graph.observe_transition(
                prev_id=prev_room,
                command=command,
                new_id=room_id,
                new_name=payload.get("location_name", ""),
                turn=turn,
                response=text,
                dark=bool(payload.get("dark")),
                ended=done,
            )
            prev_room = room_id
            command, text, done = "", "", False

            if every and turn >= next_sample:
                samples.append({
                    "turn": turn,
                    "score": payload.get("score", 0),
                    "rooms": len(graph.rooms),
                    **structure(graph, turn, room_id),
                })
                next_sample = turn + every

    return graph, samples, ended


COLUMNS = [
    ("arm", "{name:<18}", 18),
    ("turns", "{turns:>6}", 6),
    ("score", "{score:>6}", 6),
    ("rooms", "{rooms:>6}", 6),
    ("radius", "{radius:>7}", 7),
    ("frag", "{components:>5}", 5),
    ("reach", "{reachable:>6}", 6),
    ("revisit", "{revisit_ratio:>8.0%}", 8),
    ("dirs/rm", "{tried_per_room:>8}", 8),
    ("once", "{probed_once:>5}", 5),
    ("walls1", "{unretried_walls:>7}", 7),
    ("dead", "{since_new_room:>6}", 6),
]


def table(root: Path) -> int:
    rows = []
    for path in sorted(root.glob("*.jsonl")):
        graph, _, ended = rebuild(path)
        final_turn = ended.get("turns") or max(
            (r.last_seen_turn for r in graph.rooms.values()), default=0
        )
        rows.append({
            "name": path.stem,
            "turns": final_turn,
            "score": ended.get("final_score", 0),
            "rooms": len(graph.rooms),
            **structure(graph, final_turn, graph.path[-1] if graph.path else None),
        })
    if not rows:
        print(f"no traces in {root}")
        return 1

    head = "".join(f"{label:>{width}}" if label != "arm" else f"{label:<{width}}"
                   for label, _, width in COLUMNS)
    print(head)
    print("-" * len(head))
    for row in sorted(rows, key=lambda r: (-r["radius"], -r["rooms"])):
        print("".join(fmt.format(**row) for _, fmt, _ in COLUMNS))
    print()
    print("  radius  moves from the start room to the furthest room found")
    print("  frag    map fragments with no known route between them")
    print("  reach   rooms walkable from where it finished")
    print("  revisit share of room-to-room moves that landed somewhere known")
    print("  dirs/rm distinct directions tried per room, walls included")
    print("  once    rooms it entered and left by one direction, never probed")
    print("  walls1  refusals hit once and never retested")
    print("  dead    turns it ran on after finding its last new room")
    return 0


def timeline(path: Path, every: int) -> int:
    _, samples, _ = rebuild(path, every=every)
    if not samples:
        print(f"no turns in {path}")
        return 1
    head = (f"{'turn':>6}{'score':>6}{'rooms':>6}{'radius':>7}{'frag':>5}{'reach':>6}"
            f"{'revisit':>8}{'dirs/rm':>8}{'dead':>6}")
    print(f"{path.stem}\n")
    print(head)
    print("-" * len(head))
    for s in samples:
        print(
            f"{s['turn']:>6}{s['score']:>6}{s['rooms']:>6}{s['radius']:>7}"
            f"{s['components']:>5}{s['reachable']:>6}{s['revisit_ratio']:>7.0%}"
            f"{s['tried_per_room']:>8}{s['since_new_room']:>6}"
        )
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("root", nargs="?", type=Path, default=Path("traces/sweep"),
                    help="directory of traces to tabulate (default: traces/sweep)")
    ap.add_argument("--timeline", type=Path, help="one trace, sampled over its length")
    ap.add_argument("--every", type=int, default=25, help="turns between samples")
    args = ap.parse_args()

    if args.timeline:
        return timeline(args.timeline, args.every)
    return table(args.root)


if __name__ == "__main__":
    raise SystemExit(main())
