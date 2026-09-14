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


@dataclass
class TurnContext:
    """The player's-eye view of the situation."""

    turn: int
    observation: str
    score: int
    moves: int
    transcript: list[tuple[str, str]] = field(default_factory=list)  # (command, response)
    valid_actions: list[str] | None = None


@dataclass
class AgentAction:
    command: str
    thought: str = ""
    meta: dict[str, Any] = field(default_factory=dict)


class Agent(ABC):
    name: str = "agent"
    kind: str = "generic"

    @abstractmethod
    async def act(self, ctx: TurnContext) -> AgentAction:
        ...

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
