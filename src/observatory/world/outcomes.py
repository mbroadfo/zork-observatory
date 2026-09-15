"""What actually happened on a turn, and whether it was worth taking.

A transcript hides waste. Reading a hundred turns of "You can't go that way"
leaves an impression; it does not leave a number, and an impression cannot be
compared between two runs. This module turns every turn into one of a small set
of outcomes and keeps the tally.

The outcome that matters most is FUTILE: a command that this agent has already
tried, in this exact room, and already watched fail. Everything else on the list
is a mistake. A futile turn is a mistake the agent had already been told about
and made again, which is the cleanest behavioural signal there is for whether
something is learning from negative feedback — and it is measurable without
knowing anything about the game.

Nothing here is fed back to the agent by default. The agent already has the
transcript; whether it notices is the thing being measured, and a harness that
warns it would be answering its own question.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any

from ..engine.base import Observation, WorldState
from .discovery import NO_SUCH_THING, UNKNOWN_WORD, ERROR_REPLY_MAX_CHARS
from .graph import parse_movement

META_WORDS = {
    "score", "inventory", "i", "look", "l", "save", "restore", "version",
    "verbose", "brief", "diagnose", "wait", "z", "full",
}


class Outcome(str, Enum):
    PROGRESS = "progress"        # the world is not what it was
    BLOCKED = "blocked"          # a real exit refusal: there is no way that way
    UNKNOWN_WORD = "unknown"     # the parser has never heard of it
    ABSENT_NOUN = "absent"       # understood, but that thing is not here
    INERT = "inert"              # understood, happened, changed nothing
    META = "meta"                # addressed the machine, not the world
    FUTILE = "futile"            # already tried here, already seen fail

    @property
    def wasted(self) -> bool:
        return self in (
            Outcome.BLOCKED, Outcome.UNKNOWN_WORD,
            Outcome.ABSENT_NOUN, Outcome.INERT, Outcome.FUTILE,
        )


@dataclass
class TurnOutcome:
    outcome: Outcome
    command: str
    room: str
    repeat_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["outcome"] = self.outcome.value
        d["wasted"] = self.outcome.wasted
        return d


def _is_refusal(text: str, patterns: tuple[str, ...]) -> bool:
    stripped = text.strip()
    if len(stripped) > ERROR_REPLY_MAX_CHARS:
        return False
    return any(p in stripped.lower() for p in patterns)


@dataclass
class OutcomeTally:
    """Running counts, plus the memory of what has already failed where."""

    counts: dict[str, int] = field(default_factory=dict)
    steps: int = 0
    # (room, command) -> how many times it has failed there.
    failed: dict[tuple[str, str], int] = field(default_factory=dict)
    distinct_commands: set[str] = field(default_factory=set)

    def classify(
        self,
        command: str,
        obs: Observation,
        state: WorldState,
        prev_state: WorldState | None,
    ) -> TurnOutcome:
        cmd = " ".join(command.strip().lower().split())
        room = state.location_name or str(state.location_id)
        prev_room = (
            prev_state.location_name or str(prev_state.location_id)
        ) if prev_state else room

        # "Did the world change" is the only test that needs no game knowledge,
        # so it is asked first and everything else is a refinement of *why not*.
        changed = prev_state is None or (
            state.state_hash != prev_state.state_hash
            or state.location_id != prev_state.location_id
            or state.score != prev_state.score
            or state.inventory != prev_state.inventory
        )

        key = (prev_room, cmd)
        if changed:
            outcome = Outcome.PROGRESS
            self.failed.pop(key, None)     # it works here after all
        elif key in self.failed:
            outcome = Outcome.FUTILE
        elif _is_refusal(obs.text, UNKNOWN_WORD):
            outcome = Outcome.UNKNOWN_WORD
        elif _is_refusal(obs.text, NO_SUCH_THING):
            outcome = Outcome.ABSENT_NOUN
        elif parse_movement(cmd) is not None:
            outcome = Outcome.BLOCKED
        elif cmd in META_WORDS or cmd.split()[:1] and cmd.split()[0] in META_WORDS:
            outcome = Outcome.META
        else:
            outcome = Outcome.INERT

        if outcome.wasted and outcome is not Outcome.FUTILE:
            self.failed[key] = self.failed.get(key, 0) + 1
        elif outcome is Outcome.FUTILE:
            self.failed[key] += 1

        self.steps += 1
        self.counts[outcome.value] = self.counts.get(outcome.value, 0) + 1
        if cmd:
            self.distinct_commands.add(cmd)

        return TurnOutcome(
            outcome=outcome,
            command=cmd,
            room=prev_room,
            repeat_count=self.failed.get(key, 0),
        )

    def summary(self) -> dict[str, Any]:
        wasted = sum(
            count for name, count in self.counts.items()
            if Outcome(name).wasted
        )
        futile = self.counts.get(Outcome.FUTILE.value, 0)
        return {
            "steps": self.steps,
            "counts": dict(self.counts),
            "wasted": wasted,
            "futile": futile,
            # The headline pair. Waste is the cost of exploring; futility is the
            # cost of not listening — an agent can have high waste and low
            # futility and be doing perfectly sensible work.
            "wasted_pct": round(100 * wasted / self.steps, 1) if self.steps else 0.0,
            "futile_pct": round(100 * futile / self.steps, 1) if self.steps else 0.0,
            "distinct_commands": len(self.distinct_commands),
            "known_dead_ends": len(self.failed),
        }
