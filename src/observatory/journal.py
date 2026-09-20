"""What stays true about a world, kept across runs.

Two different things were being called memory, and they behave differently:

  The active state is what is true *now* — where you stand, what you carry,
  which things are open. It belongs to this run and dies with it, because the
  next run starts with the window shut again. That is the episodic record
  (agents/episodic.py), held in memory.

  The journal is what stays true of the world however many times you start
  over: that Behind House lies east of North of House, that a window there
  opens, that the mailbox held a leaflet. Learned once, true in every run.
  That is this module, written to disk after every turn.

The line between them is the reason to keep both. A run that inherits the
journal does not inherit the *state* — it still has to open the window — but it
need not rediscover that there is a window to open.

What is allowed in here is strictly what the player saw the game print:

  topology      a command typed under one heading, and the heading that came
                back. Observed transitions, nothing inferred.
  first sight   the description a place printed the first time it was seen,
                and the words in it — what was there before anything touched it.
  what worked   commands the parser accepted and answered with something other
                than re-describing the place. Not "the right move": only that
                the game did something when it was typed.

Nothing the model says goes in. The notebook (notebook.py) is the other half
of this pair and holds the opposite thing: the agent's own sentences, which
may be wrong. Two memories, kept apart on purpose — one is what the world did,
the other is what the player believed about it.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .world.discovery import ERROR_REPLY_MAX_CHARS, GRAMMAR, NO_SUCH_THING, UNKNOWN_WORD
from .world.graph import parse_movement

if TYPE_CHECKING:
    from .agents.episodic import EpisodicMemory, Heading

# Replies that mean the game understood and did nothing. Kept apart from the
# parser's refusals because they say something true about the world — "there is
# nothing special about the tree" is an answer — but they are not a change.
# The parser asking for a sentence it can read. Not the world answering.
PARSER_ASKING = (
    "there was no verb", "too many nouns", "what do you want to",
    "you must supply a verb", "beg your pardon",
)

DID_NOTHING = (
    "nothing special about", "nothing happens", "that's not something you can",
    "you can't do that", "it is already", "you already have", "already open",
    "already closed", "nothing but dust", "won't fit", "too dark to see",
)

MAX_PLACES = 30
MAX_ACTIONS_PER_PLACE = 6
MAX_REPLY_CHARS = 90
MAX_WORDS_PER_PLACE = 10


def _flat(text: str, limit: int = MAX_REPLY_CHARS) -> str:
    one = " ".join((text or "").split())
    return one if len(one) <= limit else one[: limit - 1].rstrip() + "…"


def _is_refusal(reply: str) -> bool:
    """The parser declining, in its own words. Short replies only, so a long
    description that happens to contain such a phrase is not mistaken for one."""
    low = reply.strip().lower()
    if len(low) > ERROR_REPLY_MAX_CHARS:
        return False
    return any(p in low for p in UNKNOWN_WORD + GRAMMAR + NO_SUCH_THING + PARSER_ASKING)


def _did_nothing(reply: str) -> bool:
    low = reply.strip().lower()
    return len(low) <= ERROR_REPLY_MAX_CHARS and any(p in low for p in DID_NOTHING)


class Journal:
    """Places, the ways between them, and what has been seen to work."""

    def __init__(self, path: Path, story: str, agent: str) -> None:
        self.path = path
        self.story = story
        self.agent = agent
        self.places: dict[str, dict[str, Any]] = {}
        self.runs = 0

    # --- files -----------------------------------------------------------

    @classmethod
    def open(cls, root: Path, story: str, agent: str, fresh: bool = False) -> "Journal":
        from .notebook import slug

        path = Path(root) / slug(story) / f"{slug(agent)}.journal.json"
        if fresh and path.exists():
            stamp = time.strftime("%Y%m%d-%H%M%S")
            path.rename(path.with_name(f"{path.stem}.{stamp}.json"))
        book = cls(path, story, agent)
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            book.places = dict(data.get("places", {}))
            book.runs = int(data.get("runs", 0))
        book.runs += 1
        return book

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.to_dict(), indent=1), encoding="utf-8")
        os.replace(tmp, self.path)

    def to_dict(self) -> dict[str, Any]:
        return {"story": self.story, "agent": self.agent, "runs": self.runs, "places": self.places}

    # --- learning --------------------------------------------------------

    def _place(self, heading: str) -> dict[str, Any]:
        if heading not in self.places:
            self.places[heading] = {
                "first_seen_run": self.runs, "description": "",
                "words": [], "exits": {}, "walls": {}, "worked": {},
            }
        return self.places[heading]

    def absorb(self, record: "EpisodicMemory") -> None:
        """Take everything durable out of this run's record.

        Called after every turn. Cheap: the record is small, and the point of
        writing it out each time is that a run stopped halfway leaves the
        world's shape behind rather than throwing it away.
        """
        for name, heading in record.headings.items():
            if not name:
                continue
            place = self._place(name)
            if not place["description"] and heading.first_text:
                place["description"] = _flat(heading.first_text, 400)
                place["words"] = list(heading.nouns[:MAX_WORDS_PER_PLACE])
            self._absorb_entries(place, heading)

    def _absorb_entries(self, place: dict[str, Any], heading: "Heading") -> None:
        for command, entry in heading.entries.items():
            if entry.stale:
                # Typed where no heading had been printed for a while — a dark
                # room, most often. Which place it belongs to is a guess, and a
                # guess is the one thing this file does not keep.
                continue
            if entry.led_to and entry.led_to != heading.name:
                place["exits"][command] = entry.led_to
                place["walls"].pop(command, None)
                continue
            if command in place["exits"]:
                continue
            reply = entry.reply or ""
            if not reply:
                continue
            if _is_refusal(reply):
                continue        # the parser, not the world
            if parse_movement(command) is not None:
                place["walls"].setdefault(command, _flat(reply))
            elif not _did_nothing(reply):
                place["worked"].setdefault(command, _flat(reply))

    # --- reading ---------------------------------------------------------

    def render(self) -> str:
        """The journal as the player sees it, or "" while it is empty."""
        if not self.places:
            return ""
        lines = [
            f"Your journal, kept across {self.runs - 1} earlier run(s) of this same world. "
            "Everything here is something the game printed for you before; the world starts "
            "over each run, but its shape does not.",
            "",
        ]
        for name, place in list(self.places.items())[:MAX_PLACES]:
            bits = []
            exits = "; ".join(f"{c} → {d}" for c, d in place["exits"].items())
            if exits:
                bits.append(f"    ways out: {exits}")
            if place["walls"]:
                bits.append("    refused: " + ", ".join(place["walls"]))
            worked = list(place["worked"].items())[:MAX_ACTIONS_PER_PLACE]
            if worked:
                bits.append(
                    "    did something: "
                    + "; ".join(f'{c} → "{r}"' for c, r in worked)
                )
            if place["words"]:
                bits.append("    named here at first sight: " + ", ".join(place["words"]))
            lines.append(f"  {name}")
            lines.extend(bits)
        if len(self.places) > MAX_PLACES:
            lines.append(f"  (+{len(self.places) - MAX_PLACES} more places)")
        return "\n".join(lines)

    def summary(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "runs": self.runs,
            "places": len(self.places),
            "exits": sum(len(p["exits"]) for p in self.places.values()),
            "worked": sum(len(p["worked"]) for p in self.places.values()),
        }
