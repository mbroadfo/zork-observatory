"""A record of what was typed and what came back, kept the way a player keeps it.

A transcript is memory in the least useful shape there is. After eighty turns
in a forest a person does not replay eighty exchanges before each move; they
remember that north of the clearing the trees close in, and that they have
found that out four times. A language model handed the eighty exchanges has to
do the replay itself on every turn, and a small one does it badly: the
observatory has watched an 8B model walk into the same wall eleven times with
the evidence sitting in its own context.

So this is the other shape. Every exchange is filed under the heading the text
was showing when the command was typed, and repeats are counted rather than
repeated. Two failures can then be told apart:

  forgetting    - the record says "tried 4 times", and the player never had
                  that line in front of it before (transcript mode);
  perseveration - the line was in front of it, and it typed the command anyway.

What this is careful not to be:

  It is not the map. Places are keyed by the heading line the game printed,
  never by the engine's room id. Zork calls five different rooms "Forest";
  a player reading the screen cannot tell them apart without working at it,
  and neither can this record. Handing over the engine's disambiguation would
  be handing over the map.

  It is not advice. It stores the reply the game gave, verbatim and shortened,
  and a count. It never says "blocked", "dead end" or "don't". Whether four
  refusals are enough is the player's call, and making it is the thing being
  measured. The harness's own classification of turns (world/outcomes.py)
  stays on the observatory's side of the line.

  It is not normalised. "n" and "north" are filed separately, because that
  they are the same command is itself something to discover.

The one assumption it makes about the text is that a short, capitalised line
with no sentence punctuation is a heading. That is a fact about how the screen
is laid out, the same thing a person sees as the bold line at the top of a
room, and it is the only reading of the text the harness does.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from ..world.graph import DIRECTIONS, parse_movement
from .base import TurnContext
from .simple import harvest_nouns

NO_HEADING = ""               # before the text has shown any heading
MAX_REPLY_CHARS = 90          # a reply as filed; enough to recognise it
MAX_REPLY_CHARS_ELSEWHERE = 48
MAX_HERE = 20                 # entries shown for the current heading
MAX_ELSEWHERE_HEADINGS = 15
MAX_ELSEWHERE_ENTRIES = 8
MAX_HEADING_CHARS = 40
MAX_OUTCOMES_HERE = 4         # distinct replies listed per entry
MAX_OUTCOMES_ELSEWHERE = 2
MAX_INVENTORY_CHARS = 300
MAX_AGENDA_NOUNS = 10

# The compass as a player reads it, rather than whatever order the set gives.
COMPASS_ORDER = [
    "north", "south", "east", "west", "northeast", "northwest",
    "southeast", "southwest", "up", "down", "in", "out",
]
assert set(COMPASS_ORDER) == set(DIRECTIONS), "the graph knows a direction this list does not"

# A direction word in a room description ("a path leads south") is not a thing
# to act on, and the directions have their own line in the agenda.
_NOT_A_THING = set(COMPASS_ORDER) | {"n", "s", "e", "w", "ne", "nw", "se", "sw", "u", "d"}

# Pinned only once the player has typed one of these itself, so the pin never
# teaches the word. The parser rung's prompt names "inventory"; "i" is the
# abbreviation a player may try.
INVENTORY_WORDS = {"inventory", "i", "inv"}

_NOT_IN_A_HEADING = re.compile(r"[.!?:;,\"/()\[\]*0-9<>=]")


def heading_of(text: str) -> str:
    """The first line of `text` laid out like a heading, or "".

    Short, starts with a capital, no sentence punctuation or digits, and every
    word longer than three letters capitalised ("West of House", "Up a Tree").
    That last rule is what keeps an inventory line like "A pile of leaves"
    from reading as a place.
    """
    for raw in (text or "").splitlines():
        line = raw.strip()
        if not line or line.startswith("("):
            continue
        if len(line) > MAX_HEADING_CHARS or not line[0].isupper():
            continue
        if _NOT_IN_A_HEADING.search(line):
            continue
        words = line.split()
        if len(words) > 5:
            continue
        if all(w[0].isupper() for w in words if len(w) > 3):
            return line
    return ""


def _shorten(text: str, limit: int) -> str:
    flat = " ".join((text or "").split())
    return flat if len(flat) <= limit else flat[: limit - 1].rstrip() + "…"


@dataclass
class Entry:
    """One command, as typed under one heading."""

    command: str
    count: int = 0
    first: int = 0            # the step it was first typed, for ordering
    last: int = 0
    reply: str = ""           # the latest reply, flattened, when no heading came back
    led_to: str = ""          # the heading the latest reply showed, if any
    # Every distinct outcome and how often it came back. Showing only the
    # latest one hid the evidence: Zork refuses an impossible action with a
    # rotating set of quips, and nineteen refusals rendered as one quip plus
    # "replies varied" reads as though the result might change.
    outcomes: dict[str, int] = field(default_factory=dict)
    latest: str = ""

    @property
    def varied(self) -> bool:
        return len(self.outcomes) > 1

    @staticmethod
    def outcome_key(led_to: str, reply: str, was_under: str = "") -> str:
        """How one reply reads in the record.

        A heading that is the one already showing is said to be the same one.
        Rendering `look` as "→ heading West of House" made re-describing where
        you stand look exactly like arriving somewhere, and a small model read
        that as progress and typed `look` nine times.
        """
        if not led_to:
            return reply
        if led_to == was_under:
            return f"heading {led_to} again (the one you were already under)"
        return f"heading {led_to}"


@dataclass
class Heading:
    name: str
    shown: int = 0            # times the text has displayed this heading
    last: int = 0
    entries: dict[str, Entry] = field(default_factory=dict)
    # Words the text printed under this heading that look like things. Kept in
    # the order they were read, so the agenda reads like the description does.
    nouns: list[str] = field(default_factory=list)

    def untried_directions(self) -> list[str]:
        tried = {parse_movement(cmd) for cmd in self.entries}
        return [d for d in COMPASS_ORDER if d not in tried]

    def untouched_nouns(self) -> list[str]:
        """Things named here that no command typed here has mentioned."""
        spoken = {word for cmd in self.entries for word in cmd.split()}
        return [n for n in self.nouns if n not in spoken]


class EpisodicMemory:
    """Everything typed so far, filed by heading. Survives a rollback — the
    world forgets, the player does not."""

    def __init__(self) -> None:
        self.headings: dict[str, Heading] = {}
        self.here = NO_HEADING
        self.step = 0
        # The latest reply to an inventory command the player typed, whole,
        # and the step it came back at. Only ever the game's words.
        self.inventory: tuple[str, int] | None = None
        # First words the parser has and has not taken, anywhere. A model that
        # spends thirty turns on `use` and `try` while its own record shows
        # `open` and `pour` working is missing something it already knows.
        self.verbs_taken: list[str] = []
        self.verbs_refused: list[str] = []
        # What was last typed and when, so the next observation can be
        # matched to it — or recognised as not a reply to it at all.
        self._pending: str | None = None
        self._turn = 0
        self._life = 1

    # --- filing ----------------------------------------------------------

    def _at(self, name: str) -> Heading:
        if name not in self.headings:
            self.headings[name] = Heading(name)
        return self.headings[name]

    def _see(self, text: str) -> None:
        name = heading_of(text)
        if name:
            place = self._at(name)
            place.shown += 1
            place.last = self.step
            # Only text that carries the heading describes the place, so only
            # that text is read for things. A refusal names nothing.
            for noun in harvest_nouns(text):
                if noun not in place.nouns and noun not in _NOT_A_THING and noun not in name.lower():
                    place.nouns.append(noun)
            self.here = name

    def record(self, command: str, reply: str) -> Entry:
        """File one exchange under the heading that was showing when it was typed."""
        self.step += 1
        key = " ".join(command.strip().lower().split())
        place = self._at(self.here)
        place.last = self.step
        entry = place.entries.get(key)
        if entry is None:
            entry = place.entries[key] = Entry(command=key, first=self.step)
        led_to = heading_of(reply)
        flat = "" if led_to else _shorten(reply, MAX_REPLY_CHARS)
        outcome = Entry.outcome_key(led_to, flat, was_under=self.here)
        entry.outcomes[outcome] = entry.outcomes.get(outcome, 0) + 1
        entry.latest = outcome
        entry.count += 1
        entry.last = self.step
        entry.led_to, entry.reply = led_to, flat
        if key in INVENTORY_WORDS:
            self.inventory = (" ".join(reply.split())[:MAX_INVENTORY_CHARS], self.step)
        self._note_verb(key, reply)
        self._see(reply)
        return entry

    def repeat_of(self, command: str) -> Entry | None:
        """This command, if it has been typed here before and never once done
        anything but reprint the same reply.

        Deliberately strict. An entry whose replies differ, or that ever showed
        a new heading, is not a repeat: the world may have changed, and saying
        otherwise would be the harness deciding what is worth trying.
        """
        key = " ".join(command.strip().lower().split())
        place = self.headings.get(self.here)
        entry = place.entries.get(key) if place else None
        if entry is None or not entry.count or len(entry.outcomes) != 1:
            return None
        outcome = next(iter(entry.outcomes))
        if outcome.startswith("heading ") and "again (" not in outcome:
            return None    # it took you somewhere
        return entry

    def _note_verb(self, key: str, reply: str) -> None:
        """Whether the parser knew the first word, by its own words.

        Only the one reply that names a word it does not know counts as a
        refusal. Everything else — refused for any other reason, or not — is a
        word it took, which is what the player needs to know.
        """
        verb = key.split()[0] if key.split() else ""
        if not verb or verb in self.verbs_taken or verb in self.verbs_refused:
            return
        unknown = re.search(r'know the word ["“]?([a-z\'-]+)', reply.lower())
        if unknown and unknown.group(1) == verb:
            self.verbs_refused.append(verb)
        elif not unknown:
            self.verbs_taken.append(verb)

    def relocate(self, texts: list[str]) -> None:
        """The world moved under the player (a rollback, a restore): take the
        heading from the most recent text that shows one, and file nothing."""
        for text in texts:
            name = heading_of(text)
            if name:
                self.here = name
                self._at(name).last = self.step
                return

    # --- the agent's side ------------------------------------------------

    def before_move(self, ctx: TurnContext) -> None:
        """Bring the record up to date with what the agent is now looking at.

        The observation is filed as the reply to the last command only when
        it plainly is one: same life, next turn. Anything else means the world
        was put back, and the observation is somewhere to stand, not an answer.
        """
        recent = [ctx.observation] + [r for _, r in reversed(ctx.transcript)]
        if ctx.life != self._life or (self._pending is not None and ctx.turn != self._turn + 1):
            self._life = ctx.life
            self.relocate(recent)
        elif self._pending is not None:
            self.record(self._pending, ctx.observation)
        elif not self.headings:
            self._see(ctx.observation)
        self._pending = None
        self._turn = ctx.turn

    def after_move(self, command: str) -> None:
        self._pending = command

    def before_reflection(self, ctx: TurnContext) -> None:
        """A reflection comes straight after the move that ended things, before
        any rollback, so this is the one chance to file that move's reply."""
        if self._pending is not None and ctx.turn == self._turn and ctx.life == self._life:
            self.record(self._pending, ctx.observation)
        self._pending = None

    # --- reading ---------------------------------------------------------

    def _ago(self, step: int) -> str:
        n = self.step - step
        return "just now" if n == 0 else f"{n} command{'s' if n != 1 else ''} ago"

    def _line(self, e: Entry, limit: int, max_outcomes: int, age: bool = False) -> str:
        times = f" ×{e.count}" if e.count > 1 else ""
        when = f" (last {self._ago(e.last)})" if age else ""
        # Most frequent first; the count is the evidence, so it is always shown
        # when there is more than one kind of reply.
        ranked = sorted(e.outcomes.items(), key=lambda kv: (-kv[1], kv[0] != e.latest))
        parts = []
        for outcome, n in ranked[:max_outcomes]:
            text = outcome if outcome.startswith("heading ") else f'"{_shorten(outcome, limit)}"'
            mark = " (latest)" if e.varied and outcome == e.latest else ""
            parts.append(f"{text}{f' ×{n}' if e.varied else ''}{mark}")
        if len(ranked) > max_outcomes:
            parts.append(f"+{len(ranked) - max_outcomes} other replies")
        return f"> {e.command}{times}{when} → " + " · ".join(parts)

    def render(self) -> str:
        """The record as the agent sees it, or "" before anything is in it."""
        if not any(h.entries for h in self.headings.values()):
            return ""
        lines = [
            "Your record so far: every command you have typed and the reply it got, "
            "filed under the heading line the text was showing at the time. "
            "A count means you typed it that many times there; a reply's count is "
            "how many of those times it came back.",
            "",
        ]
        if self.verbs_taken:
            lines.append("First words the parser has taken from you: " + ", ".join(self.verbs_taken))
        if self.verbs_refused:
            lines.append(
                "First words it said it does not know: " + ", ".join(self.verbs_refused)
            )
        if self.verbs_taken or self.verbs_refused:
            lines.append("")
        if self.inventory:
            text, step = self.inventory
            lines.append(f"Latest reply to your inventory command ({self._ago(step)}): \"{text}\"")
            lines.append("")
        here = self.headings.get(self.here)
        label = self.here or "(no heading yet)"
        shown = f", shown {here.shown} times" if here and here.shown > 1 else ""
        lines.append(f"Under the current heading, {label}{shown}:")
        entries = sorted(here.entries.values(), key=lambda e: -e.last) if here else []
        for e in entries[:MAX_HERE]:
            lines.append("  " + self._line(e, MAX_REPLY_CHARS, MAX_OUTCOMES_HERE, age=True))
        if not entries:
            lines.append("  nothing typed here yet")
        elif len(entries) > MAX_HERE:
            lines.append(f"  (+{len(entries) - MAX_HERE} older)")

        # What is left here, which is the thing a record of what you did does
        # not tell you. Both lines are subtraction, not advice: directions you
        # have not typed here, and words this place's own text printed that
        # none of your commands here have mentioned.
        if here:
            untried = here.untried_directions()
            if untried:
                lines.append("  Directions not yet typed here: " + ", ".join(untried))
            untouched = here.untouched_nouns()[:MAX_AGENDA_NOUNS]
            if untouched:
                lines.append("  Words this place's text used that you have not: " + ", ".join(untouched))

        others = sorted(
            (h for h in self.headings.values() if h.name != self.here and h.entries),
            key=lambda h: -h.last,
        )
        if others:
            lines.append("")
            lines.append("Under other headings, most recent first:")
            for h in others[:MAX_ELSEWHERE_HEADINGS]:
                items = sorted(h.entries.values(), key=lambda e: -e.last)
                shown = f", shown {h.shown} times" if h.shown > 1 else ""
                body = "; ".join(
                    self._line(e, MAX_REPLY_CHARS_ELSEWHERE, MAX_OUTCOMES_ELSEWHERE)
                    for e in items[:MAX_ELSEWHERE_ENTRIES]
                )
                more = f"; +{len(items) - MAX_ELSEWHERE_ENTRIES} more" if len(items) > MAX_ELSEWHERE_ENTRIES else ""
                lines.append(f"  {h.name or '(no heading yet)'}{shown}: {body}{more}")
            if len(others) > MAX_ELSEWHERE_HEADINGS:
                lines.append(f"  (+{len(others) - MAX_ELSEWHERE_HEADINGS} headings not shown)")
        return "\n".join(lines)

    def summary(self) -> dict[str, Any]:
        entries = [e for h in self.headings.values() for e in h.entries.values()]
        return {
            "headings": sum(1 for h in self.headings.values() if h.name),
            "entries": len(entries),
            "repeats": sum(e.count - 1 for e in entries),
            "here": self.here,
            "verbs_taken": len(self.verbs_taken),
            "verbs_refused": len(self.verbs_refused),
        }
