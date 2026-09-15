"""Agent interface.

An agent sees exactly what a human player sees: the text the game printed.
Ground truth — the object tree, the discovered map, the state hash — stays on
the observatory side of this boundary. Keeping that line clean is the whole
point; an agent that can read the object tree isn't playing the game.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from .memory import AgentMemory


@dataclass
class TurnContext:
    """The player's-eye view of the situation."""

    turn: int
    observation: str
    score: int
    moves: int
    transcript: list[tuple[str, str]] = field(default_factory=list)  # (command, response)
    valid_actions: list[str] | None = None
    life: int = 1          # how many times the world has been restarted under it
    lives_left: int = 0


@dataclass
class AgentAction:
    command: str
    thought: str = ""
    meta: dict[str, Any] = field(default_factory=dict)


class Agent(ABC):
    name: str = "agent"
    kind: str = "generic"

    _memory: AgentMemory | None = None

    @property
    def memory(self) -> AgentMemory:
        """Lessons that survive every restore. Lazily created so the simple
        agents don't need an __init__ they have no other use for."""
        if self._memory is None:
            self._memory = AgentMemory()
        return self._memory

    @abstractmethod
    async def act(self, ctx: TurnContext) -> AgentAction:
        ...

    async def reflect(self, ctx: TurnContext, cause: str) -> str | None:
        """Called when the world is about to be rolled back under the agent.

        Return one or two sentences to carry forward, or None to carry nothing.
        Agents that don't learn return None, which is the honest answer for a
        random baseline and makes it the control arm for memory as well.
        """
        return None

    async def on_start(self, intro: str) -> None:
        pass

    async def on_end(self, reason: str) -> None:
        pass

    def usage(self) -> dict[str, Any]:
        """Cumulative cost/latency, surfaced in the UI. Free agents report zeros."""
        return {"input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0, "calls": 0}

    def describe(self) -> dict[str, Any]:
        """Everything that shaped this agent's behaviour, recorded into the trace.

        For an LLM agent this includes the verbatim system prompt. A score is
        only evidence about a model if you can see what the model was told —
        so the prompt travels with the run rather than living only in the
        source tree that produced it.
        """
        return {"name": self.name, "kind": self.kind}
