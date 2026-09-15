"""The discovery ledger.

The puzzles are not the interesting part. Before any of them there is a quieter
sequence, the one a kid at a TRS-80 in 1980 went through without noticing: *oh,
it understands directions* — *oh, the rooms are still there when I come back* —
*oh, things can be inside other things*. That is an ontology being built from
raw interaction, and it is the same ontology every player ends up with.

This module records when each of those realisations becomes observable. Not
what the agent says it learned — what its behaviour demonstrates. A detector
fires on the first turn the transcript contains evidence that the agent has
crossed a particular line, and the turn number is the datum.

Two things make the ledger worth more than a score:

  It works on failure. An agent that dies on turn 30 having never discovered
  containers has told you something specific, where "score: 0" told you nothing.

  It is comparable across games. Rooms and treasures are not, but "turn at which
  it established that the world persists" is.

Detectors are deliberately conservative: they use world-state changes where
possible and fall back to parser idioms, which are shared across Infocom-era
games but are still heuristics. A missed discovery is much less damaging than a
false one, so when in doubt they do not fire.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Callable

from ..engine.base import Observation, WorldState

# Parser idioms. Infocom-era games share these closely; other authoring systems
# vary, which is why nothing critical depends on them alone.
UNKNOWN_WORD = (
    "don't know the word", "don't know that word", "not a verb i recognise",
    "not a verb i recognize", "that's not a verb", "i don't understand",
    "that sentence isn't one i recognize",
)
NO_SUCH_THING = (
    "can't see any such thing", "you can't see that", "isn't here",
    "aren't holding that", "aren't carrying that", "you can't see it",
)

# Error phrases are matched against *short* replies only.
#
# A parser refusal is a one-liner. A room description is not — and room prose
# contains sentences that look exactly like refusals: "There is no door here."
# reads as a reference failure to any substring match, but it is scenery, and
# the move that produced it succeeded. Length plus "did the world change"
# separates the two reliably, and a missed discovery costs far less than a
# phantom one.
ERROR_REPLY_MAX_CHARS = 120
DIRECTION_WORDS = {
    "north", "south", "east", "west", "northeast", "northwest",
    "southeast", "southwest", "up", "down", "in", "out",
}
ABBREVIATIONS = {"n", "s", "e", "w", "ne", "nw", "se", "sw", "u", "d"}
META_COMMANDS = {
    "save", "restore", "restart", "version", "verbose", "brief", "superbrief",
    "script", "unscript", "diagnose", "quit", "help", "info", "commands",
}


@dataclass
class Turn:
    """Everything a detector is allowed to look at for one turn."""

    turn: int
    command: str
    obs: Observation
    state: WorldState
    prev_state: WorldState | None
    visited_rooms: set[str]
    commands_seen: set[str]

    @property
    def cmd(self) -> str:
        return self.command.strip().lower()

    @property
    def text(self) -> str:
        return self.obs.text.lower()

    @property
    def moved(self) -> bool:
        return (
            self.prev_state is not None
            and self.state.location_id != self.prev_state.location_id
        )

    def _refusal(self, patterns: tuple[str, ...]) -> bool:
        """True only for a short reply that reads as a parser refusal."""
        stripped = self.obs.text.strip()
        if len(stripped) > ERROR_REPLY_MAX_CHARS:
            return False
        return any(p in stripped.lower() for p in patterns)

    @property
    def unknown_word(self) -> bool:
        return self._refusal(UNKNOWN_WORD)

    @property
    def missing_noun(self) -> bool:
        # A command that moved you did not fail, whatever the prose says.
        return not self.moved and self._refusal(NO_SUCH_THING)

    @property
    def errored(self) -> bool:
        return self.unknown_word or self.missing_noun

    def player_object(self) -> int | None:
        """The player's own object number, inferred from what it carries.

        Jericho exposes the player's *room*, not the player object, but every
        carried item's parent is the player — so the inventory identifies it
        without hardcoding anything game-specific.
        """
        held = set(self.state.inventory)
        if not held:
            return None
        for o in self.state.objects:
            if o.name in held:
                return o.parent
        return None

    def visible_objects(self) -> set[int]:
        """Object numbers the player could plausibly perceive this turn.

        Everything inside the current room, at any depth, plus everything the
        player is carrying. The rest of the object table is the rest of the
        world, which is running whether the player is watching or not.
        """
        children: dict[int, list[int]] = {}
        for o in self.state.objects:
            children.setdefault(o.parent, []).append(o.num)

        seen: set[int] = set()
        frontier = [self.state.location_id]
        while frontier:
            num = frontier.pop()
            for child in children.get(num, []):
                if child not in seen:
                    seen.add(child)
                    frontier.append(child)
        seen.add(self.state.location_id)
        return seen


@dataclass
class Discovery:
    key: str
    label: str
    reveals: str

    # `turn` is the in-world turn counter, which a rollback rewinds — after a
    # death at turn 52 restores to turn 40, the next turn is 41 again. `step`
    # is the number of turns actually played and never goes backwards.
    #
    # Time-to-discovery must be measured in steps. Using `turn` silently
    # under-reports every run that died, and dying is the interesting case.
    step: int = 0
    turn: int = 0
    life: int = 1
    command: str = ""
    evidence: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


Detector = Callable[[Turn], str | None]

# (key, label, what crossing this line reveals about the universe, detector)
_DETECTORS: list[tuple[str, str, str, Detector]] = []


def detector(key: str, label: str, reveals: str):
    def wrap(fn: Detector) -> Detector:
        _DETECTORS.append((key, label, reveals, fn))
        return fn

    return wrap


# --- the ontology, roughly in the order a player builds it ----------------


@detector("movement", "Movement exists", "Typed words can change where you are.")
def _movement(t: Turn) -> str | None:
    if t.moved:
        return f"{t.cmd!r} → {t.state.location_name}"
    return None


@detector("abbreviation", "Directions abbreviate", "The parser accepts shorthand.")
def _abbreviation(t: Turn) -> str | None:
    if t.cmd in ABBREVIATIONS and t.moved:
        return f"{t.cmd!r} worked as a direction"
    return None


@detector("persistence", "The world persists", "Places still exist when you leave them.")
def _persistence(t: Turn) -> str | None:
    room = t.state.location_name
    if t.moved and room in t.visited_rooms:
        return f"returned to {room}"
    return None


@detector("inventory", "Inventory exists", "The game tracks what you hold, and will say so.")
def _inventory(t: Turn) -> str | None:
    if t.cmd in ("inventory", "i", "inv") and not t.errored:
        return "asked what it was carrying"
    return None


@detector("possession", "Objects can be carried", "The world can be picked up, not only walked through.")
def _possession(t: Turn) -> str | None:
    if t.prev_state and len(t.state.inventory) > len(t.prev_state.inventory):
        gained = set(t.state.inventory) - set(t.prev_state.inventory)
        return f"acquired {', '.join(sorted(gained)) or 'an object'}"
    return None


@detector("release", "Objects can be released", "Possession is reversible; the world accepts things back.")
def _release(t: Turn) -> str | None:
    if (
        t.prev_state
        and len(t.state.inventory) < len(t.prev_state.inventory)
        and not t.obs.lost
    ):
        lost = set(t.prev_state.inventory) - set(t.state.inventory)
        return f"gave up {', '.join(sorted(lost)) or 'an object'}"
    return None


@detector("containers", "Things can be opened", "Some objects have an interior state.")
def _containers(t: Turn) -> str | None:
    if t.cmd.startswith(("open ", "unlock ")) and not t.errored:
        if "reveal" in t.text or "opening" in t.text or "opened" in t.text:
            return f"{t.cmd!r} changed an object's state"
    return None


@detector("nesting", "Objects contain objects", "The world is a tree, not a flat list.")
def _nesting(t: Turn) -> str | None:
    """An object whose parent is itself an object rather than a room.

    Restricted to what the player could actually have seen. The object tree is
    the whole world, and in a real game things move in it constantly without
    the player present — Zork's thief wanders the dungeon from turn one, and an
    unrestricted version of this detector fired on "thief is inside East-West
    Passage" before the agent had left the front lawn. The ledger records what
    the agent's behaviour established, so a detector that reads ground truth
    the agent has no access to is recording the wrong thing entirely.
    """
    if not t.prev_state:
        return None

    visible = t.visible_objects()
    if not visible:
        return None
    prev = {o.num: o.parent for o in t.prev_state.objects}
    by_num = {o.num: o for o in t.state.objects}

    # Being carried is `possession`, a different lesson. Excluding the player
    # keeps the two categories from both firing on a single `take`.
    excluded = {t.state.location_id, t.player_object()}

    for num in visible:
        o = by_num.get(num)
        if o is None or not o.name:
            continue
        # Nested inside another visible object, not the room and not the player.
        if o.parent in visible and o.parent not in excluded and prev.get(num) != o.parent:
            parent = by_num.get(o.parent)
            if parent is not None and parent.name:
                return f"{o.name} is inside {parent.name}"
    return None


@detector("vocabulary_limit", "The vocabulary is finite", "There is a list of known words, and it can be missed.")
def _vocabulary_limit(t: Turn) -> str | None:
    if t.unknown_word:
        return f"{t.cmd!r} was not understood"
    return None


@detector("reference", "Nouns must be present", "Understanding a word is not the same as the thing being here.")
def _reference(t: Turn) -> str | None:
    if t.missing_noun:
        return f"{t.cmd!r} named something absent"
    return None


@detector("grammar", "Commands compose", "Verbs take objects, and objects take prepositions.")
def _grammar(t: Turn) -> str | None:
    words = t.cmd.split()
    if len(words) >= 3 and not t.errored and words[0] not in DIRECTION_WORDS:
        return f"{t.cmd!r} was accepted"
    return None


@detector("reading", "Text can be examined", "Objects carry information, not just presence.")
def _reading(t: Turn) -> str | None:
    if t.cmd.startswith(("read ", "examine ", "x ", "look at ")) and not t.errored:
        if len(t.obs.text.strip()) > 25:
            return f"{t.cmd!r} returned content"
    return None


@detector("darkness", "Darkness exists", "Some places cannot be perceived at all.")
def _darkness(t: Turn) -> str | None:
    if t.state.dark:
        return f"entered the dark at {t.state.location_name or 'an unlit place'}"
    return None


@detector("light_state", "Light sources have state", "An object you carry can change what you can perceive.")
def _light_state(t: Turn) -> str | None:
    if t.prev_state and t.prev_state.dark and not t.state.dark and not t.moved:
        return f"{t.cmd!r} restored vision without moving"
    return None


@detector("mortality", "Death exists", "Actions can end the run. The world is not safe.")
def _mortality(t: Turn) -> str | None:
    if t.obs.lost:
        return f"died at {t.state.location_name or 'an unknown place'}"
    return None


@detector("scoring", "Score exists", "There is a measure of progress, and it can be queried.")
def _scoring(t: Turn) -> str | None:
    if t.cmd in ("score", "full", "full score") and not t.errored:
        return "asked for a score"
    if t.prev_state and t.state.score > t.prev_state.score:
        return f"score rose to {t.state.score}"
    return None


@detector("meta", "The program has meta-commands", "Some words address the machine, not the world.")
def _meta(t: Turn) -> str | None:
    head = t.cmd.split()[0] if t.cmd.split() else ""
    if head in META_COMMANDS:
        return f"tried {t.cmd!r}"
    return None


class DiscoveryLedger:
    """Records the first turn each realisation becomes observable."""

    def __init__(self) -> None:
        self.found: dict[str, Discovery] = {}
        self._visited: set[str] = set()
        self._commands: set[str] = set()

    @property
    def total(self) -> int:
        return len(_DETECTORS)

    def observe(
        self,
        turn: int,
        command: str,
        obs: Observation,
        state: WorldState,
        prev_state: WorldState | None,
        step: int = 0,
        life: int = 1,
    ) -> list[Discovery]:
        ctx = Turn(
            turn=turn,
            command=command,
            obs=obs,
            state=state,
            prev_state=prev_state,
            visited_rooms=set(self._visited),
            commands_seen=set(self._commands),
        )

        new: list[Discovery] = []
        for key, label, reveals, fn in _DETECTORS:
            if key in self.found:
                continue
            try:
                evidence = fn(ctx)
            except Exception:
                # A detector is an observation, never a failure mode for the run.
                continue
            if evidence:
                discovery = Discovery(
                    key=key, label=label, reveals=reveals,
                    step=step or turn, turn=turn, life=life,
                    command=command, evidence=evidence,
                )
                self.found[key] = discovery
                new.append(discovery)

        if state.location_name:
            self._visited.add(state.location_name)
        if command:
            self._commands.add(command.strip().lower())
        return new

    def manifest(self) -> list[dict[str, Any]]:
        """Every discovery, found or not — the ledger is as informative empty."""
        return [
            (
                self.found[key].to_dict()
                if key in self.found
                else {"key": key, "label": label, "reveals": reveals, "turn": 0}
            )
            for key, label, reveals, _ in _DETECTORS
        ]

    def summary(self) -> dict[str, Any]:
        return {
            "found": len(self.found),
            "total": self.total,
            # Keyed on `step`, the rollback-proof counter — this is the one
            # downstream analysis should use.
            "steps": {k: d.step for k, d in self.found.items()},
            "turns": {k: d.turn for k, d in self.found.items()},
        }
