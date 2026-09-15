from . import prompts
from .base import Agent, AgentAction, TurnContext
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
]


def build_agent(kind: str, **kwargs) -> Agent:
    """Agent factory. The Claude agent imports lazily so no API key is needed
    to run the baselines."""
    if kind == "random":
        return RandomAgent(seed=kwargs.get("seed", 0))
    if kind == "scripted":
        return ScriptedAgent(kwargs.get("commands") or MOCK_WALKTHROUGH)
    if kind == "human":
        return HumanAgent()
    if kind == "claude":
        from .claude_agent import ClaudeAgent

        return ClaudeAgent(
            model=kwargs.get("model", "claude-opus-5"),
            effort=kwargs.get("effort", "medium"),
            history_turns=kwargs.get("history_turns", 30),
            info_level=kwargs.get("info_level", "parser"),
        )
    raise ValueError(f"Unknown agent: {kind!r} (expected random, scripted, human or claude)")
