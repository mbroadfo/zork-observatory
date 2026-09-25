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

Which is why only corroborated entries are kept. The first version of this
kept everything, and the run that measured it is the argument for the change:
over ninety-five turns it wrote twenty-two entries, eight of them true, and
the false ones were not scattered — they clustered precisely where it was
stuck. Six consecutive true entries while it opened the window, entered the
house, took the lamp and lit it; then five consecutive false ones while it
pushed at a door that does not open, including one recording that it had
moved east through a refused exit. It wrote "took the grating" on a turn the
engine classified futile, and then spent forty turns trying to take the
grating it had just written down taking.

So the record tracks progress, not attempts. The agent still chooses what to
write and the words are still its own; the world decides which of them last.
The refused ones are counted and traced rather than dropped, because what a
model claims on a turn when nothing happened is the interesting half.

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
    """What the agent wrote down, in the order it wrote it.

    By default only entries the engine corroborated are kept. The agent still
    chooses what to write; the world decides what survives. The rest are held
    aside — counted, traced, never rendered — because what a model claims on a
    turn where nothing happened is the measurement, not noise to discard.

    `corroborated_only=False` keeps everything, which is how this was first
    built and measured. That arm is on record: over 95 turns a qwen3:14b run
    wrote 22 entries, 8 corroborated, and the false ones clustered exactly
    where it was stuck — including "took the grating" on a turn the engine
    classified futile, after which it spent forty more turns trying to take
    the grating it had just recorded taking.
    """

    def __init__(self, shown: int = DEFAULT_SHOWN, corroborated_only: bool = True) -> None:
        self.entries: list[Entry] = []
        # Written, then refused. Kept for the count and the trace; never shown
        # to the agent and never written to the notebook.
        self.rejected: list[Entry] = []
        self.shown = shown
        self.corroborated_only = corroborated_only

    def __len__(self) -> int:
        return len(self.entries)

    @property
    def attempted(self) -> int:
        return len(self.entries) + len(self.rejected)

    def add(
        self, text: str, turn: int, room: str = "", command: str = "",
        outcome: str = "", run: int = 0,
    ) -> Entry | None:
        """Offer one line. Returns the entry if it was kept, None if not.

        Three ways to be refused, in order:

          Nothing was written, so there is nothing to keep.

          The world did not change on this turn. A record of what a player
          did is only worth reading back if the things in it happened, and a
          model that is stuck writes its most confident fiction — so the turn
          that produced the line is what decides, not the line.

          It says something already written. A model that repeats one fact
          every turn would fill its own window with it, which is the shape the
          notebook's light-source spiral took over four runs. Matched on the
          room and command as well as the words, because the same failed
          attempt gets described four different ways.
        """
        text = " ".join((text or "").split())[:MAX_ENTRY_CHARS].strip()
        if not text:
            return None
        entry = Entry(
            text=text, turn=turn, room=room, command=command,
            outcome=outcome, run=run,
        )
        if self.corroborated_only and not entry.corroborated:
            self.rejected.append(entry)
            return None
        if self._already_said(entry):
            self.rejected.append(entry)
            return None
        self.entries.append(entry)
        return entry

    def _already_said(self, entry: Entry) -> bool:
        act = (entry.room.lower(), entry.command.lower())
        for kept in self.entries:
            if kept.text.lower() == entry.text.lower():
                return True
            if act != ("", "") and (kept.room.lower(), kept.command.lower()) == act:
                return True
        return False

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
        """What was kept, what was offered, and what the refused ones claimed.

        `corroborated_pct` is of everything the agent tried to write, not of
        what survived — filtering must not be allowed to flatter itself. With
        the filter on it is the share of the agent's claims that were true,
        which is the figure worth watching: in the runs measured so far it
        sits near 100% while a run is getting somewhere and collapses to zero
        the moment it starts pushing at something that will not move.
        """
        kept = len(self.entries)
        attempted = self.attempted
        agreed = sum(1 for e in self.entries if e.corroborated)
        refused: dict[str, int] = {}
        for entry in self.rejected:
            refused[entry.outcome or "unclassified"] = refused.get(entry.outcome or "unclassified", 0) + 1
        outcomes: dict[str, int] = {}
        for entry in self.entries:
            if entry.outcome:
                outcomes[entry.outcome] = outcomes.get(entry.outcome, 0) + 1
        return {
            "count": kept,
            "attempted": attempted,
            "rejected": len(self.rejected),
            "corroborated": agreed,
            "corroborated_pct": round(100 * agreed / attempted, 1) if attempted else 0.0,
            "outcomes": outcomes,
            "rejected_outcomes": refused,
            "filtered": self.corroborated_only,
        }
