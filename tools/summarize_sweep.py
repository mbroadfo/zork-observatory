"""Read a sweep's traces back into one table.

What a run is worth is not its score alone: a run that scores nothing while
covering nine rooms without repeating itself has told you something, and a run
that scores ten by walking into the kitchen and then loops for three hundred
turns has told you something else. So every column here is a different question:

  score / rooms / objects   how far it got
  futile %                  commands it had already watched fail in that room
  wasted %                  turns that changed nothing
  distinct                  how many different things it thought to try
  errors                    turns the server, not the player, lost
  scaffold                  how often the harness intervened, and whether the
                            model took the hint

Usage:  python tools/summarize_sweep.py [traces/sweep]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


def read(path: Path) -> dict[str, Any] | None:
    events, header = [], {}
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if row.get("kind") == "trace.header":
                header = row
            elif row.get("kind") == "event":
                events.append(row)
    except (OSError, ValueError):
        return None
    if not events:
        return None

    started = next((e["payload"] for e in events if e["type"] == "session.started"), {})
    ended = next((e["payload"] for e in events if e["type"] == "session.ended"), {})
    thoughts = [e["payload"].get("meta") or {} for e in events if e["type"] == "agent.thought"]
    quality = [
        e["payload"]["quality"] for e in events
        if e["type"] == "observation" and e["payload"].get("quality")
    ]
    snaps = [e["payload"] for e in events if e["type"] == "state.snapshot"]
    last = quality[-1] if quality else {}
    coverage = (snaps[-1].get("coverage") if snaps else {}) or {}
    config = started.get("agent_config") or {}

    nudged = sum(1 for m in thoughts if m.get("nudged"))
    return {
        "name": path.stem,
        "agent": started.get("agent", header.get("agent", "?")),
        "turns": ended.get("turns", len(thoughts)),
        "score": ended.get("final_score", snaps[-1]["score"] if snaps else 0),
        "rooms": coverage.get("rooms_seen", 0),
        "objects": coverage.get("objects_seen", 0),
        "deaths": ended.get("deaths", 0),
        "futile_pct": last.get("futile_pct", 0.0),
        "wasted_pct": last.get("wasted_pct", 0.0),
        "distinct": last.get("distinct_commands", 0),
        "errors": sum(1 for m in thoughts if m.get("error")),
        "retries": sum(1 for m in thoughts if m.get("retried_after")),
        "nudged": nudged,
        "ignored": sum(1 for m in thoughts if m.get("nudge_ignored")),
        "skipped": sum(len(m.get("skipped") or []) for m in thoughts),
        "exhausted": sum(1 for m in thoughts if m.get("all_candidates_inert")),
        "in_tok": (ended.get("usage") or {}).get("input_tokens", 0),
        "out_tok": (ended.get("usage") or {}).get("output_tokens", 0),
        "ms_avg": (ended.get("usage") or {}).get("latency_ms_avg", 0),
        "censored": ended.get("censored", False),
        "reason": ended.get("reason", "(unfinished)"),
        "temperature": (config.get("options") or {}).get("temperature"),
        "recall": config.get("recall"),
        "level": config.get("info_level"),
    }


def main(root: str = "traces/sweep") -> int:
    rows = [r for r in (read(p) for p in sorted(Path(root).glob("*.jsonl"))) if r]
    if not rows:
        print(f"no traces in {root}")
        return 1

    head = (f"{'arm':<18}{'turns':>6}{'score':>6}{'rooms':>6}{'obj':>5}{'die':>4}"
            f"{'futile':>8}{'wasted':>8}{'distinct':>9}{'err':>5}{'nudge':>7}{'ign':>5}"
            f"{'skip':>6}{'stuck':>6}{'s/turn':>8}")
    print(head)
    print("-" * len(head))
    for r in sorted(rows, key=lambda r: (-r["score"], -r["rooms"])):
        print(
            f"{r['name']:<18}{r['turns']:>6}{r['score']:>6}{r['rooms']:>6}{r['objects']:>5}"
            f"{r['deaths']:>4}{r['futile_pct']:>7}%{r['wasted_pct']:>7}%{r['distinct']:>9}"
            f"{r['errors']:>5}{r['nudged']:>7}{r['ignored']:>5}{r['skipped']:>6}"
            f"{r['exhausted']:>6}{r['ms_avg'] / 1000:>8.1f}"
        )
    print()
    for r in sorted(rows, key=lambda r: r["name"]):
        print(f"  {r['name']:<18} {r['agent']}  ·  {r['reason']}"
              f"{'  [censored]' if r['censored'] else ''}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(*sys.argv[1:]))
