"""What the player keeps when the world takes everything else back.

Restoring a save rewinds the world. It does not rewind the person at the
keyboard. That asymmetry is the entire reason save-and-retry is a way of
learning a game rather than a way of cheating at it: you lose the lamp, the
sword and the twelve points, and you keep the one sentence that matters —
*don't go down there without a light.*

This module is that sentence. A memory survives every restore in the run, and
it is written by the agent in its own words, never by the harness. Nothing here
proposes what a lesson should say; asking "what happened and what will you do
differently" is a question, and the answer is the datum.

Two consequences worth stating plainly, because they are what makes the memory
worth keeping rather than just useful:

  The memory is a belief, not a fact. An agent can draw the wrong lesson from a
  death and carry it for the rest of the run. Reading a confidently wrong
  memory against the object tree is more interesting than reading a right one.

  Memories are comparable across players in a way scores are not. Two models
  that both die four times in the cellar and both end on 45 points may have
  written completely different things down, and the difference is legible to a
  human in a way a scoreline never is.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field, asdict
from typing import Any

# A lesson is one or two sentences. The cap is a guard against a model writing
# an essay on every death and crowding its own transcript out of the context
# window, not an opinion about how much it should have learned.
MAX_LESSON_CHARS = 400
DEFAULT_RECALL = 12


@dataclass
class Lesson:
    """One thing the agent decided was worth keeping."""

    text: str
    turn: int
    kind: str = "death"        # "death" | "note" | "milestone"
    location: str = ""
    life: int = 1
    ts: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class AgentMemory:
    """Lessons that outlive the world state they were learned in."""

    def __init__(self, recall: int = DEFAULT_RECALL) -> None:
        self.lessons: list[Lesson] = []
        self.recall = recall

    def __len__(self) -> int:
        return len(self.lessons)

    def add(self, text: str, turn: int, kind: str = "death", location: str = "", life: int = 1) -> Lesson | None:
        """Record a lesson. Returns None if there was nothing to record."""
        text = " ".join((text or "").split())[:MAX_LESSON_CHARS].strip()
        if not text:
            return None
        lesson = Lesson(text=text, turn=turn, kind=kind, location=location, life=life)
        self.lessons.append(lesson)
        return lesson

    def render(self) -> str:
        """The memory as the agent will see it, or "" when it is empty.

        An empty memory renders as nothing at all rather than as "you have not
        learned anything yet" — at the bottom of the information ladder the
        agent has not established that dying is possible, and a heading that
        mentions deaths would tell it.
        """
        if not self.lessons:
            return ""
        recent = self.lessons[-self.recall:]
        lines = [
            "Things you wrote down earlier and chose to keep:",
        ]
        for i, lesson in enumerate(recent, start=1):
            where = f" ({lesson.location})" if lesson.location else ""
            lines.append(f"{i}. [turn {lesson.turn}{where}] {lesson.text}")
        if len(self.lessons) > len(recent):
            lines.append(f"(+{len(self.lessons) - len(recent)} older notes not shown)")
        return "\n".join(lines)

    def to_dict(self) -> list[dict[str, Any]]:
        return [lesson.to_dict() for lesson in self.lessons]

    def summary(self) -> dict[str, Any]:
        kinds: dict[str, int] = {}
        for lesson in self.lessons:
            kinds[lesson.kind] = kinds.get(lesson.kind, 0) + 1
        return {"count": len(self.lessons), "kinds": kinds}


# The reflection prompt.
#
# It asks and does not tell. It must not name a cause, suggest a remedy, or
# imply that the death was avoidable — an agent that died to something genuinely
# unavoidable should be free to write that down, and an agent that misreads its
# own death should be free to be wrong. Both are data.
REFLECT = """That run has ended.

Write at most two sentences to your future self, who will start again from an \
earlier point with everything you are about to say and nothing else you have \
learned. Say what you believe happened and what, if anything, you intend to do \
differently.

Write only those sentences. No preamble."""
