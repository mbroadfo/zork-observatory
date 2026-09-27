from . import prompts
from .base import Agent, AgentAction, TurnContext
from .journal import Journal
from .llm import DEFAULT_HISTORY

# Kept here so the CLI and server can name the defaults without importing the
# Ollama agent (which they only import when one is actually built).
DEFAULT_NUM_CTX = 16384
# 0.7 is a setting for writing prose. A player following a rule wants less
# invention and more follow-through, but how much less is a measurement, so
# the default is left where it was and the dial is exposed.
DEFAULT_TEMPERATURE = 0.7
from .simple import MOCK_WALKTHROUGH, HumanAgent, RandomAgent, ScriptedAgent

__all__ = [
    "Agent",
    "AgentAction",
    "TurnContext",
    "Journal",
    "prompts",
    "RandomAgent",
    "ScriptedAgent",
    "HumanAgent",
    "MOCK_WALKTHROUGH",
    "build_agent",
    "parse_think",
    "AGENTS",
    "DEFAULT_HISTORY",
    "DEFAULT_NUM_CTX",
    "DEFAULT_TEMPERATURE",
]


AGENTS = ("random", "scripted", "human", "claude", "ollama")


def parse_think(value: str | None) -> bool | str | None:
    """The thinking switch as a form or flag spells it. `default` leaves the
    model alone; low/medium/high is the graded form some models take."""
    v = (value or "default").strip().lower()
    return {"default": None, "off": False, "on": True}.get(v, v)


def build_agent(kind: str, **kwargs) -> Agent:
    """Agent factory. The model players import lazily so neither an API key
    nor a model server is needed to run the baselines."""
    if kind == "random":
        return RandomAgent(seed=kwargs.get("seed", 0))
    if kind == "scripted":
        # A real game's own walkthrough when the engine has one; the fixture's
        # otherwise. The mock has no walkthrough of its own on purpose — it
        # would make the fixture depend on the agents package.
        if kwargs.get("commands"):
            return ScriptedAgent(kwargs["commands"], source="walkthrough")
        return ScriptedAgent(MOCK_WALKTHROUGH, source="mock script")
    if kind == "human":
        return HumanAgent()
    history = kwargs.get("history_turns")
    if history is None:
        history = DEFAULT_HISTORY
    # Off unless asked for. A player that keeps no journal is the control arm
    # for keeping one, and it is the one that has been measured.
    journal = Journal() if kwargs.get("journal") else None
    # Likewise off unless asked for: a player that cannot look past its window
    # is the control arm for one that can.
    search = bool(kwargs.get("search"))
    # The ceiling on what a player may say per turn. None keeps each agent's
    # own default. It is a cost lever as much as a safety one: on the opus run
    # of 2026-09-27, 274 output tokens of reasoning per turn were 55% of the
    # bill, against ~1,100 input tokens that were half cache-read.
    max_tokens = kwargs.get("max_tokens")
    if kind == "claude":
        from .claude_agent import ClaudeAgent

        return ClaudeAgent(
            model=kwargs.get("model") or "claude-opus-5",
            effort=kwargs.get("effort", "medium"),
            history_turns=history,
            info_level=kwargs.get("info_level", "parser"),
            journal=journal,
            search=search,
            **({"max_tokens": max_tokens} if max_tokens else {}),
        )
    if kind == "ollama":
        from .ollama_agent import OllamaAgent

        return OllamaAgent(
            model=kwargs.get("model") or "",
            history_turns=history,
            info_level=kwargs.get("info_level", "parser"),
            think=kwargs.get("think"),
            seed=kwargs.get("seed"),
            num_ctx=kwargs.get("num_ctx") or DEFAULT_NUM_CTX,
            temperature=kwargs.get("temperature", DEFAULT_TEMPERATURE),
            journal=journal,
            search=search,
            **({"max_tokens": max_tokens} if max_tokens else {}),
        )
    raise ValueError(f"Unknown agent: {kind!r} (expected {', '.join(AGENTS)})")
