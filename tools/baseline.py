"""Read many runs of the same configuration back as a distribution.

The September sweep ran one run per arm and said so in its own caveat: a single
350-turn run is an anecdote with good instrumentation, and "40 versus 10" is not
a measurement to quote. Everything on the roadmap is gated on fixing that, which
needs no new harness — `cli.py` already takes `--runs` and `--seed` — only a way
to read several traces of the same configuration together and see the spread.

So this groups traces by *arm* rather than by file. An arm is the agent's
configuration with the seed taken out: same model, same context, same
temperature, same scaffolds, different roll. Runs that differ only by seed are
the same arm and belong in the same row; anything else is a different arm and
gets its own.

Which columns, and why these:

  acquired    distinct objects that entered the inventory. The sweep found that
              taking things tracked real play where score did not — the arm that
              won took 43 times and the worst arm twice. Examining is not
              taking, so this counts possession, not attention.

  scoring@    the step at which the discovery ledger first saw the score move.
              On Zork the points are indoors, so this doubles as the turn the
              player got inside — the sweep's single cleanest signal — without
              hardcoding a room name that only exists in one game.

  found       how many of the ledger's realisations the run reached at all. A
              run that scores nothing and establishes eight things about the
              universe has told you something a zero cannot.

  dead        the longest stretch of steps with no new room, and the stretch it
              ended on. The sweep's postscript names this the measure it should
              have been stopped on: nine hours of GPU bought roughly two
              thousand turns that came after the map stopped growing.

Score and rooms are here too, at the end, because they are what everyone asks
for first and neither one distinguished the sweep's arms.

Everything is in *steps*, never in-world turns. A rollback rewinds the turn
counter — after a death at turn 52 restores to turn 40, the next turn is 41
again — so a time-to-anything measured in turns silently under-reports every run
that died, and dying is the interesting case.

    python tools/baseline.py                          # traces/, grouped by arm
    python tools/baseline.py traces/sweep             # ... another directory
    python tools/baseline.py traces/sweep --runs      # one row per file too
    python tools/baseline.py traces/sweep --json      # for further analysis
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any, Iterator

# Configuration keys that do not make a run a different experiment. The seed is
# the whole point — runs that differ only by it are the repeats we are looking
# for. The rest are noise: a prompt is reproduced by its level, and a digest
# and host describe where the model ran, not what was asked of it.
NOT_THE_ARM = {"seed", "system_prompt", "digest", "host", "details"}

# What a row reports, in the order the header prints them.
METRICS = ("acquired", "scoring@", "found", "dead_max", "dead_end", "score", "rooms")


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
            yield row


def fingerprint(config: dict[str, Any]) -> str:
    """A stable name for the arm this run belongs to.

    Built from the configuration rather than the filename, because filenames in
    a sweep are chosen by whoever wrote the shell script and two of them may
    well describe the same experiment.
    """
    def prune(value: Any) -> Any:
        if isinstance(value, dict):
            return {k: prune(v) for k, v in sorted(value.items()) if k not in NOT_THE_ARM}
        return value

    return json.dumps(prune(config), sort_keys=True, default=str)


def label(config: dict[str, Any]) -> str:
    """Something short enough to print. Names the scaffolds that are on."""
    model = config.get("model") or config.get("name") or "?"
    bits = [model.split("/")[-1], config.get("info_level", "?")]
    options = config.get("options") or {}
    if "temperature" in options:
        bits.append(f"t{options['temperature']}")
    if config.get("think"):
        bits.append("think")
    for scaffold in ("nudge", "candidates", "agenda", "vocabulary", "journal"):
        if config.get(scaffold):
            bits.append(scaffold)
    if config.get("recall") and config["recall"] != "transcript":
        bits.append(str(config["recall"]))
    return "/".join(str(b) for b in bits)


def measure(path: Path) -> dict[str, Any] | None:
    """One run, read once, in a single pass.

    Returns None for a file that is not a trace or that recorded no turns —
    a sweep directory usually contains a few of both.
    """
    config: dict[str, Any] = {}
    step = 0
    score = 0
    held: set[str] = set()
    acquired: set[str] = set()
    rooms: set[str] = set()
    discoveries: dict[str, int] = {}
    last_new_room = 0
    dead_max = 0
    started = False

    for row in events(path):
        if row.get("kind") != "event":
            continue
        kind, payload = row.get("type"), row.get("payload") or {}

        if kind == "session.started":
            started = True
            config = payload.get("agent_config") or {}

        elif kind == "command.issued":
            step += 1

        elif kind == "state.snapshot":
            # Keyed on the object number, not the printed name. Zork prints
            # "Darkness" for every unlit room, so counting names merges them
            # into one and reports a run as more stuck than it was.
            room = payload.get("location_id") or payload.get("location_name")
            if room and room not in rooms:
                # A room first seen on the snapshot that *follows* the command
                # that reached it, so the gap closes on the step that earned it.
                rooms.add(room)
                dead_max = max(dead_max, step - last_new_room)
                last_new_room = step

            inventory = payload.get("inventory")
            if inventory is not None:
                now = set(inventory)
                # Only what was not held before and has never been held: an
                # item dropped and picked up again is not a second discovery,
                # and counting it would reward a run for fumbling.
                acquired |= (now - held) - acquired
                held = now

            if payload.get("score") is not None:
                score = payload["score"]

        elif kind == "discovery.made":
            key = payload.get("key")
            if key and key not in discoveries:
                discoveries[key] = payload.get("step", step)

        elif kind == "session.ended":
            if payload.get("final_score") is not None:
                score = payload["final_score"]

    if not started or not step:
        return None

    # The trailing gap is not closed by a new room, so it is never folded into
    # dead_max above. It is also the most diagnostic number in the row: a run
    # that ended a hundred steps after its map stopped growing was over long
    # before it stopped.
    dead_end = step - last_new_room
    return {
        "name": path.stem,
        "arm": fingerprint(config),
        "label": label(config),
        "seed": (config.get("options") or {}).get("seed"),
        "steps": step,
        "acquired": len(acquired),
        # A run that never scored has no number here, and zero would be a lie —
        # it would sort as the fastest.
        "scoring@": discoveries.get("scoring"),
        "found": len(discoveries),
        "dead_max": dead_max,
        "dead_end": dead_end,
        "score": score,
        "rooms": len(rooms),
    }


def spread(values: list[Any]) -> str:
    """Median, and the range when the runs disagree.

    Missing values are counted rather than dropped: three runs of which two
    never scored is a different fact from one run that did, and averaging the
    survivors would report the opposite of what happened.
    """
    present = [v for v in values if v is not None]
    if not present:
        return "-"
    median = statistics.median(present)
    shown = f"{median:g}"
    if len(present) > 1 and min(present) != max(present):
        shown += f" [{min(present):g}-{max(present):g}]"
    if len(present) < len(values):
        shown += f" ({len(present)}/{len(values)})"
    return shown


def table(rows: list[dict[str, Any]], headers: list[str]) -> str:
    def cell(value: Any) -> str:
        # A run that never scored has no number, and printing `None` invites
        # somebody to read it as a value.
        return "-" if value is None else str(value)

    body = [[cell(r.get(h, "")) for h in headers] for r in rows]
    widths = [max(len(h), *(len(b[i]) for b in body)) if body else len(h)
              for i, h in enumerate(headers)]
    out = ["  ".join(h.ljust(w) for h, w in zip(headers, widths)).rstrip()]
    for line in body:
        out.append("  ".join(c.ljust(w) for c, w in zip(line, widths)).rstrip())
    return "\n".join(out)


def group(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    arms: dict[str, list[dict[str, Any]]] = {}
    for run in runs:
        arms.setdefault(run["arm"], []).append(run)

    rows = []
    for members in arms.values():
        row: dict[str, Any] = {
            "arm": members[0]["label"],
            "n": len(members),
            "steps": spread([m["steps"] for m in members]),
        }
        for metric in METRICS:
            row[metric] = spread([m[metric] for m in members])
        rows.append(row)
    # Most acquired first would need the medians back as numbers; the arm name
    # is stable and comparing rows is what this table is for.
    return sorted(rows, key=lambda r: r["arm"])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("directory", nargs="?", default="traces",
                        help="a directory of .jsonl traces, or a single trace")
    parser.add_argument("--runs", action="store_true",
                        help="also print one row per trace, not just per arm")
    parser.add_argument("--json", action="store_true",
                        help="emit the per-run measurements instead of a table")
    args = parser.parse_args(argv)

    root = Path(args.directory)
    paths = [root] if root.is_file() else sorted(root.rglob("*.jsonl"))
    runs = [r for r in (measure(p) for p in paths) if r]

    if not runs:
        print(f"no traces with recorded turns under {root}")
        return 1

    if args.json:
        print(json.dumps(runs, indent=2))
        return 0

    if args.runs:
        named = [dict(run, run=run["name"]) for run in runs]
        print(table(sorted(named, key=lambda r: r["run"]),
                    ["run", "seed", "steps", *METRICS]))
        print()

    arms = group(runs)
    print(table(arms, ["arm", "n", "steps", *METRICS]))

    singles = sum(1 for row in arms if row["n"] == 1)
    if singles:
        print(f"\n{singles} of {len(arms)} arms have one run. A single run is an "
              f"anecdote; the roadmap gates reporting on three seeds.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
