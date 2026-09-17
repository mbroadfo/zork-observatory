from . import prompts
from .base import Agent, AgentAction, TurnContext
from .llm import RECALL_MODES, default_history
from .simple import MOCK_WALKTHROUGH, HumanAgent, RandomAgent, ScriptedAgent

__all__ = [
    "Agent",
    "AgentAction",
    "TurnContext",
    "prompts",
    "RandomAgent",
    "ScriptedAgent",
    "HumanAgent",
    "MOCK_WALKTHROUGH",
    "build_agent",
    "parse_think",
    "default_history",
    "AGENTS",
    "RECALL_MODES",
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
    recall = kwargs.get("recall") or "transcript"
    history = kwargs.get("history_turns")
    if history is None:
        history = default_history(recall)
    if kind == "claude":
        from .claude_agent import ClaudeAgent

        return ClaudeAgent(
            model=kwargs.get("model") or "claude-opus-5",
            effort=kwargs.get("effort", "medium"),
            history_turns=history,
            info_level=kwargs.get("info_level", "parser"),
            recall=recall,
        )
    if kind == "ollama":
        from .ollama_agent import OllamaAgent

        return OllamaAgent(
            model=kwargs.get("model") or "",
            history_turns=history,
            info_level=kwargs.get("info_level", "parser"),
            think=kwargs.get("think"),
            seed=kwargs.get("seed"),
            recall=recall,
        )
    raise ValueError(f"Unknown agent: {kind!r} (expected {', '.join(AGENTS)})")
