"""Agents that cost nothing to run.

These exist to make the observatory useful before you spend a cent, and to
give LLM runs something to be measured against. A random agent is the floor;
if a model can't beat it, the harness is telling you something.
"""

from __future__ import annotations

import asyncio
import random

from .base import Agent, AgentAction, TurnContext

DIRECTIONS = [
    "north", "south", "east", "west", "northeast", "northwest",
    "southeast", "southwest", "up", "down", "in", "out",
]

# Generic English actions. These are language, not world knowledge — a player
# in 1980 arrived knowing what "open" means. A specific noun is different:
# "mailbox" is a fact about Zork, and a baseline that knows it is not a floor.
GENERIC_VERBS = [
    "look", "inventory", "wait", "take", "drop", "open", "close", "read",
    "examine", "push", "pull", "move", "enter", "climb", "search", "touch",
    "eat", "drink", "attack", "throw", "wear", "light", "score",
]
TRANSITIVE = {
    "take", "drop", "open", "close", "read", "examine", "push", "pull",
    "move", "climb", "search", "touch", "eat", "drink", "attack", "throw",
    "wear", "light",
}

# Words that appear in prose but are never the noun you want.
STOPWORDS = frozenset("""
a an the this that these those there here is are was were be been being am
you your yours i me my mine it its he she they them we us of in on at to from
by with for and or but not no nor so as if then than too very can cannot could
would should will shall may might must do does did done have has had having
which who whom whose what when where why how all any both each few more most
other some such only own same s t just now up down out off over under again
further once into through during before after above below between about against
standing seeing looking appears seems something anything nothing everything
""".split())


class RandomAgent(Agent):
    """The honest floor: no game knowledge, only what it has been shown.

    Nouns are harvested from the text the game has actually printed, so the
    agent's vocabulary grows exactly as fast as its exposure does. Nothing
    about any particular game is baked in — point it at a different story file
    and it starts equally ignorant, which is what makes it a usable baseline
    rather than a sandbagged one.

    With `--valid-actions` the engine's own action set is used instead. That is
    a much *stronger* baseline, not a weaker one: it amounts to an oracle
    listing every move that would change the world. Useful as a ceiling for
    random play; never confuse it with the floor.
    """

    name = "random"
    kind = "baseline"

    def __init__(self, seed: int = 0) -> None:
        self._rng = random.Random(seed)
        self._nouns: list[str] = []
        self._seen: set[str] = set()

    def _harvest(self, text: str) -> None:
        """Collect candidate nouns from whatever the game just printed."""
        word = ""
        for ch in text.lower():
            if ch.isalpha() or ch == "'":
                word += ch
                continue
            if len(word) > 2 and word not in STOPWORDS and word not in self._seen:
                self._seen.add(word)
                self._nouns.append(word)
            word = ""
        if len(word) > 2 and word not in STOPWORDS and word not in self._seen:
            self._seen.add(word)
            self._nouns.append(word)

    async def act(self, ctx: TurnContext) -> AgentAction:
        if ctx.valid_actions:
            return AgentAction(
                command=self._rng.choice(ctx.valid_actions),
                thought="(uniform over the engine's valid actions — an oracle, not a floor)",
            )

        self._harvest(ctx.observation)

        # Weighted toward movement, which is how the map gets discovered at all.
        if self._rng.random() < 0.5 or not self._nouns:
            return AgentAction(
                command=self._rng.choice(DIRECTIONS),
                thought="(uniform over compass directions)",
            )

        verb = self._rng.choice(GENERIC_VERBS)
        if verb in TRANSITIVE:
            noun = self._rng.choice(self._nouns)
            return AgentAction(
                command=f"{verb} {noun}",
                thought=f"(random verb + a noun harvested from the game's own text; {len(self._nouns)} words seen)",
            )
        return AgentAction(command=verb, thought="(random intransitive verb)")


class ScriptedAgent(Agent):
    """Replays a fixed command list. The canonical-walkthrough control arm."""

    name = "scripted"
    kind = "baseline"

    def __init__(self, commands: list[str], loop_tail: str = "look") -> None:
        self.commands = list(commands)
        self.loop_tail = loop_tail
        self._i = 0

    async def act(self, ctx: TurnContext) -> AgentAction:
        if self._i < len(self.commands):
            cmd = self.commands[self._i]
            self._i += 1
            return AgentAction(command=cmd, thought=f"script step {self._i}/{len(self.commands)}")
        return AgentAction(command=self.loop_tail, thought="script exhausted")


class HumanAgent(Agent):
    """Waits for a command pushed in from the UI."""

    name = "human"
    kind = "human"

    def __init__(self) -> None:
        self._queue: asyncio.Queue[str] = asyncio.Queue()

    def submit(self, command: str) -> None:
        self._queue.put_nowait(command)

    async def act(self, ctx: TurnContext) -> AgentAction:
        command = await self._queue.get()
        return AgentAction(command=command, thought="")


# A short, correct opening for the mock world — handy as a smoke test and as
# the "what does a competent run look like" reference in the UI.
MOCK_WALKTHROUGH = [
    "open mailbox", "read leaflet", "north", "east", "west",
    "west", "take lantern", "turn on lantern", "east", "up",
    "take painting", "down", "west", "put painting in trophy case",
    "down", "take jeweled egg", "up", "put jeweled egg in trophy case",
]
