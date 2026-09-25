"""A record the player keeps on purpose, while the thing it is about is still
on screen.

The lessons in memory.py are written at a death, from the last twelve
exchanges, capped at two sentences, in answer to "what do you believe you have
learned". That question produces theories, and a run asked it at turn 200
cannot write down what the mailbox did at turn 2 because it can no longer see
turn 2. What comes back is *"darkness is lethal, find a light source before
moving east"* — half fact, half superstition about the direction it happened to
die in — and four runs later the notebook holds four restatements of it.

A journal is the other half of the bargain. It is written by the agent, in the
same reply as the move, on the turn the thing happened, and it is meant for
what changed rather than what it means. Not the harness's record of the run:
that already exists in world/, and handing it over would be the observatory
telling the player what it did. This is the player choosing to write something
down, which is what a person playing with intent does, and choosing badly is
the interesting part.

Two things follow from writing it at the moment rather than at the end.

  It can be checked. Every turn is already classified by world/outcomes.py,
  and PROGRESS means the world actually changed — a different state hash,
  room, score or inventory. An entry written on a PROGRESS turn is
  corroborated by the engine; one written on an INERT or FUTILE turn is the
  model announcing an achievement that did not happen. The ratio is a direct
  measurement of whether a model knows when it has accomplished something,
  and it costs nothing to compute because the classification is already there.

  It compounds. Beliefs written down are read back with authority, and a run
  that writes forty of them per game compounds faster than one writing two.
  The notebook has already demonstrated this failure at the slow rate. The
  corroboration figure is what makes it visible in the first run instead of
  the fourth.

The journal is additive, always. It is rendered beside the transcript window
and never in place of it — the September 2026 sweep is unambiguous that a
summary substituted for the raw text is worse than no summary at all.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any

# One line, not a paragraph. The cap is the same guard as a lesson's: a model
# writing an essay every turn crowds its own transcript out of the window.
MAX_ENTRY_CHARS = 200

# How many entries are shown. Forty is roughly two hundred turns of a player
# writing only when something changed — the live runs classify about one turn
# in four as PROGRESS — so in practice a whole game fits and the cap only
# bites on a player that writes indiscriminately. Which is itself worth seeing.
DEFAULT_SHOWN = 40

# Outcomes that mean the world actually changed. Kept as strings rather than
# importing the enum: world/ is the observatory's side of the line and agents/
# does not import from it.
CORROBORATING = ("progress",)


@dataclass
class Entry:
    """One line the agent chose to keep, and what the engine made of the turn."""

    text: str
    turn: int
    room: str = ""
    command: str = ""
    # The engine's verdict on the turn this was written on, filled in by the
    # session once the command has run. "" when nothing classified it.
    outcome: str = ""
    run: int = 0
    ts: float = field(default_factory=time.time)

    @property
    def corroborated(self) -> bool:
        return self.outcome in CORROBORATING

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["corroborated"] = self.corroborated
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Entry":
        known = cls.__dataclass_fields__
        return cls(**{k: v for k, v in data.items() if k in known})


class Journal:
    """What the agent wrote down, in the order it wrote it."""

    def __init__(self, shown: int = DEFAULT_SHOWN) -> None:
        self.entries: list[Entry] = []
        self.shown = shown

    def __len__(self) -> int:
        return len(self.entries)

    def add(
        self, text: str, turn: int, room: str = "", command: str = "",
        outcome: str = "", run: int = 0,
    ) -> Entry | None:
        """Keep one line. Returns None when there was nothing to keep.

        Repeats are dropped. A model that writes the same sentence every turn
        would otherwise fill its own window with one fact, and the sweep
        watched exactly that happen to a notebook over four runs.
        """
        text = " ".join((text or "").split())[:MAX_ENTRY_CHARS].strip()
        if not text:
            return None
        if any(e.text.lower() == text.lower() for e in self.entries):
            return None
        entry = Entry(
            text=text, turn=turn, room=room, command=command,
            outcome=outcome, run=run,
        )
        self.entries.append(entry)
        return entry

    def render(self) -> str:
        """The journal as the agent will see it, or "" when it is empty.

        Entries are given back exactly as they were written, with where and
        when attached. Nothing is summarized, re-ordered or interpreted on the
        way out: the point of the record is that it is the record.
        """
        if not self.entries:
            return ""
        recent = self.entries[-self.shown:]
        lines = ["Your journal, in the order you wrote it:"]
        for entry in recent:
            where = f" ({entry.room})" if entry.room else ""
            run = f"run {entry.run}, " if entry.run else ""
            lines.append(f"- [{run}turn {entry.turn}{where}] {entry.text}")
        if len(self.entries) > len(recent):
            lines.append(f"({len(self.entries) - len(recent)} earlier entries not shown)")
        return "\n".join(lines)

    def to_dict(self) -> list[dict[str, Any]]:
        return [entry.to_dict() for entry in self.entries]

    def summary(self) -> dict[str, Any]:
        """Counts, and how much of it the engine agreed with.

        `corroborated` is the share of entries written on a turn that actually
        changed the world. It is the honest question to ask of a record the
        player wrote about itself.
        """
        total = len(self.entries)
        agreed = sum(1 for e in self.entries if e.corroborated)
        outcomes: dict[str, int] = {}
        for entry in self.entries:
            if entry.outcome:
                outcomes[entry.outcome] = outcomes.get(entry.outcome, 0) + 1
        return {
            "count": total,
            "corroborated": agreed,
            "corroborated_pct": round(100 * agreed / total, 1) if total else 0.0,
            "outcomes": outcomes,
        }
