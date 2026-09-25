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

import re
import time
from dataclasses import asdict, dataclass, field
from typing import Any

# Models like to stamp the turn on the front of the line. Harmless in itself,
# and fatal to a check for repetition: qwen3:14b wrote "Turn 82: Took the sack
# from the table." and then, on the turn it walked west, "Turn 83: Took the
# sack from the table." — the same sentence, kept because the prefix differed.
_TURN_PREFIX = re.compile(r"^\s*turn\s+\d+\s*[:.–—-]\s*", re.IGNORECASE)


def normalized(text: str) -> str:
    """A line reduced to what it actually claims, for comparison only.

    The stored entry stays exactly as the agent wrote it. What a model chose
    to say is the datum and is never edited; this is just the key it is
    compared under.
    """
    return _TURN_PREFIX.sub("", text or "").strip().lower()

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


# Why an offered line was not kept. Only the first is a claim about the world
# being wrong; the other two are true things not worth carrying, and lumping
# them together would make a model look like it was inventing when it was
# repeating itself.
UNCORROBORATED = "uncorroborated"   # the turn changed nothing
MOVEMENT = "movement"               # true, but the map already records it
REDUNDANT = "redundant"             # true, and already written down


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
    # Empty when the entry was kept; otherwise why it was not.
    reason: str = ""
    ts: float = field(default_factory=time.time)

    @property
    def corroborated(self) -> bool:
        return self.outcome in CORROBORATING

    @property
    def truthful(self) -> bool:
        """The world bore this turn out, whether or not the line was kept.

        A line dropped for repeating itself was still true, and counting it as
        fiction would understate the model badly: of the first hundred and
        twenty-one lines one run offered, ten were refused for saying again
        what it had already said.
        """
        return self.reason != UNCORROBORATED and self.corroborated

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

    def __init__(
        self,
        shown: int = DEFAULT_SHOWN,
        corroborated_only: bool = True,
        keep_movement: bool = False,
    ) -> None:
        self.entries: list[Entry] = []
        # Written, then refused. Kept for the count and the trace; never shown
        # to the agent and never written to the notebook.
        self.rejected: list[Entry] = []
        self.shown = shown
        self.corroborated_only = corroborated_only
        self.keep_movement = keep_movement

    def __len__(self) -> int:
        return len(self.entries)

    @property
    def attempted(self) -> int:
        return len(self.entries) + len(self.rejected)

    def add(
        self, text: str, turn: int, room: str = "", command: str = "",
        outcome: str = "", run: int = 0, movement: bool = False,
    ) -> Entry | None:
        """Offer one line. Returns the entry if it was kept, None if not.

        Four ways to be refused, in order:

          Nothing was written, so there is nothing to keep. Not counted as an
          attempt either — the field was simply left out.

          The world did not change on this turn. A record is only worth
          reading back if the things in it happened, and a model that is stuck
          writes its most confident fiction — so the turn that produced the
          line decides, not the line.

          It is about going somewhere. Movement is real change and it passes
          the test above, but the map already holds every room and every
          passage, in more detail and without the mistakes: of the first
          twenty-eight entries one run kept, sixteen were "Moved north from
          the forest path" and its variants, crowding out the eight that said
          anything the map does not.

          It says something already written. Matched on the room and command
          as well as the words, because the same act gets described several
          ways — and matched on the words with any "Turn 82:" stamp removed,
          because otherwise the same sentence on the next turn is a new one.
        """
        text = " ".join((text or "").split())[:MAX_ENTRY_CHARS].strip()
        if not text:
            return None
        entry = Entry(
            text=text, turn=turn, room=room, command=command,
            outcome=outcome, run=run,
        )
        if self.corroborated_only and not entry.corroborated:
            return self._refuse(entry, UNCORROBORATED)
        if movement and not self.keep_movement:
            return self._refuse(entry, MOVEMENT)
        if self._already_said(entry):
            return self._refuse(entry, REDUNDANT)
        self.entries.append(entry)
        return entry

    def _refuse(self, entry: Entry, reason: str) -> None:
        entry.reason = reason
        self.rejected.append(entry)
        return None

    def _already_said(self, entry: Entry) -> bool:
        said = normalized(entry.text)
        act = (entry.room.lower(), entry.command.lower())
        for kept in self.entries:
            if normalized(kept.text) == said:
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
        """What was kept, what was offered, and why the rest was not.

        Three numbers that used to be one, because they answer different
        questions and the single number answered none of them honestly:

          truthful_pct  of everything offered, how much the world bore out.
                        This is the measure of the model, and it counts lines
                        refused for repetition or for being about movement —
                        those were true. Reported over what was offered and
                        never over what survived: every kept entry is
                        corroborated by construction, so a rate over survivors
                        would read 100% however much fiction was written.

          false         lines written about turns where nothing happened. The
                        live stuck-detector: near zero while a run is getting
                        somewhere, and the bulk of the traffic once it is not.

          redundant / movement
                        true, and dropped anyway. High numbers here are not a
                        fault in the model, they are the record refusing to
                        fill up with what it already holds.
        """
        attempted = self.attempted
        reasons: dict[str, int] = {}
        for entry in self.rejected:
            reasons[entry.reason] = reasons.get(entry.reason, 0) + 1
        refused_outcomes: dict[str, int] = {}
        for entry in self.rejected:
            if entry.reason == UNCORROBORATED:
                refused_outcomes[entry.outcome or "unclassified"] = (
                    refused_outcomes.get(entry.outcome or "unclassified", 0) + 1
                )
        outcomes: dict[str, int] = {}
        for entry in self.entries:
            if entry.outcome:
                outcomes[entry.outcome] = outcomes.get(entry.outcome, 0) + 1

        truthful = (
            sum(1 for e in self.entries if e.corroborated)
            + sum(1 for e in self.rejected if e.truthful)
        )
        return {
            "count": len(self.entries),
            "attempted": attempted,
            "rejected": len(self.rejected),
            "truthful": truthful,
            "truthful_pct": round(100 * truthful / attempted, 1) if attempted else 0.0,
            "false": reasons.get(UNCORROBORATED, 0),
            "redundant": reasons.get(REDUNDANT, 0),
            "movement": reasons.get(MOVEMENT, 0),
            "outcomes": outcomes,
            "false_outcomes": refused_outcomes,
            "filtered": self.corroborated_only,
        }
