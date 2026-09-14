"""Agents that cost nothing to run.

These exist to make the observatory useful before you spend a cent, and to
give LLM runs something to be measured against. A random agent is the floor;
if a model can't beat it, the harness is telling you something.
"""

from __future__ import annotations

import asyncio
import random

from .base import Agent, AgentAction, TurnContext

DIRECTIONS = ["north", "south", "east", "west", "up", "down"]
VERBS = ["look", "inventory", "open mailbox", "read leaflet", "take lamp", "take sword"]


class RandomAgent(Agent):
    """Uniform over plausible commands, or over the engine's valid actions."""

    name = "random"
    kind = "baseline"

    def __init__(self, seed: int = 0) -> None:
        self._rng = random.Random(seed)

    async def act(self, ctx: TurnContext) -> AgentAction:
        if ctx.valid_actions:
            return AgentAction(command=self._rng.choice(ctx.valid_actions), thought="(uniform over valid actions)")
        pool = DIRECTIONS * 3 + VERBS
        return AgentAction(command=self._rng.choice(pool), thought="(uniform over a fixed command pool)")


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
