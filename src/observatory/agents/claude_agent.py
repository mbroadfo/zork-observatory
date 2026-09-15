"""A Claude player.

It gets the transcript and nothing else — no object tree, no walkthrough, no
valid-action list. The structured output has two fields on purpose: `command`
is what the game receives, `reasoning` is what the observatory displays. The
second is not decoration. Reading a model's stated plan next to the ground
truth is how you catch it believing things that aren't true.
"""

from __future__ import annotations

import hashlib
import json
import time
from typing import Any

from . import prompts
from .base import Agent, AgentAction, TurnContext

# How much the agent is told is an experimental variable — see prompts.py for
# the ladder and for why "cold" does not produce a naive player. Every level
# below `coached` is held to one rule: describe the interface, never the world.
# tests/test_prompt_hygiene.py enforces it.
DEFAULT_INFO_LEVEL = "parser"

# Kept as module-level names because the hygiene tests and older callers refer
# to them; both track the default rung.
SYSTEM = prompts.get(DEFAULT_INFO_LEVEL)
SYSTEM_FINGERPRINT = hashlib.sha256(SYSTEM.encode()).hexdigest()[:12]

# Output stays deliberately small — this is called once per turn, hundreds of
# times per run, and a chatty schema is the difference between a $2 run and $30.
MOVE_SCHEMA = {
    "type": "json_schema",
    "schema": {
        "type": "object",
        "properties": {
            # Phrased without presupposing rooms, objects or a parser — at the
            # "cold" rung the agent has not established that any of those exist,
            # and a schema that assumes them would leak the answer.
            "reasoning": {
                "type": "string",
                "description": "Two or three sentences: what you currently believe, and what you are trying next.",
            },
            "command": {
                "type": "string",
                "description": "The single line of text to type next.",
            },
        },
        "required": ["reasoning", "command"],
        "additionalProperties": False,
    },
}

# USD per million tokens (input, output).
PRICING: dict[str, tuple[float, float]] = {
    "claude-fable-5-1": (10.00, 50.00),
    "claude-fable-5": (10.00, 50.00),
    "claude-opus-5": (5.00, 25.00),
    "claude-opus-4-8": (5.00, 25.00),
    "claude-opus-4-7": (5.00, 25.00),
    "claude-opus-4-6": (5.00, 25.00),
    "claude-sonnet-5": (2.00, 10.00),
    "claude-sonnet-4-6": (3.00, 15.00),
    "claude-haiku-4-5": (1.00, 5.00),
}


def estimate_cost(model: str, usage: Any) -> float:
    rate_in, rate_out = PRICING.get(model, (5.00, 25.00))
    inp = getattr(usage, "input_tokens", 0) or 0
    out = getattr(usage, "output_tokens", 0) or 0
    written = getattr(usage, "cache_creation_input_tokens", 0) or 0
    read = getattr(usage, "cache_read_input_tokens", 0) or 0
    return (
        inp * rate_in
        + written * rate_in * 1.25
        + read * rate_in * 0.10
        + out * rate_out
    ) / 1_000_000


class ClaudeAgent(Agent):
    kind = "llm"

    def __init__(
        self,
        model: str = "claude-opus-5",
        effort: str = "medium",
        history_turns: int = 30,
        max_tokens: int = 2000,
        info_level: str = DEFAULT_INFO_LEVEL,
        api_key: str | None = None,
    ) -> None:
        import anthropic

        self.model = model
        self.effort = effort
        self.history_turns = history_turns
        self.max_tokens = max_tokens
        self.info_level = info_level
        self.system = prompts.get(info_level)
        self.name = f"claude:{model}" + ("" if info_level == DEFAULT_INFO_LEVEL else f"/{info_level}")
        self._client = anthropic.AsyncAnthropic(api_key=api_key) if api_key else anthropic.AsyncAnthropic()
        self._totals = {
            "input_tokens": 0,
            "output_tokens": 0,
            "cache_read_tokens": 0,
            "cost_usd": 0.0,
            "calls": 0,
            "latency_ms_total": 0.0,
        }

    # --- prompt ----------------------------------------------------------

    def _messages(self, ctx: TurnContext) -> list[dict[str, Any]]:
        """One user turn carrying a windowed transcript.

        A rolling window rather than the whole history: past a few dozen turns
        the early transcript stops informing the next move and starts costing
        real money. Raise `history_turns` if you want the model to have more
        rope; it is the single biggest cost lever here.
        """
        lines = []
        for command, response in ctx.transcript[-self.history_turns:]:
            if command:
                lines.append(f"> {command}")
            lines.append(response.strip())
        lines.append(f"\n[Turn {ctx.turn}. Score {ctx.score}. Moves {ctx.moves}.]")
        lines.append("What is your next command?")
        return [{"role": "user", "content": "\n".join(lines)}]

    # --- play ------------------------------------------------------------

    async def act(self, ctx: TurnContext) -> AgentAction:
        import anthropic

        started = time.perf_counter()
        try:
            response = await self._client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=[
                    {
                        "type": "text",
                        "text": self.system,
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
                output_config={"effort": self.effort, "format": MOVE_SCHEMA},
                messages=self._messages(ctx),
            )
        except anthropic.APIStatusError as exc:
            return AgentAction(
                command="look",
                thought=f"[api error {exc.status_code}: {exc.message}]",
                meta={"error": True},
            )
        except anthropic.APIConnectionError as exc:
            return AgentAction(command="look", thought=f"[connection error: {exc}]", meta={"error": True})

        latency_ms = (time.perf_counter() - started) * 1000

        if response.stop_reason == "refusal":
            detail = getattr(response.stop_details, "explanation", "") if response.stop_details else ""
            return AgentAction(command="look", thought=f"[refusal: {detail}]", meta={"error": True})

        cost = estimate_cost(self.model, response.usage)
        self._totals["input_tokens"] += getattr(response.usage, "input_tokens", 0) or 0
        self._totals["output_tokens"] += getattr(response.usage, "output_tokens", 0) or 0
        self._totals["cache_read_tokens"] += getattr(response.usage, "cache_read_input_tokens", 0) or 0
        self._totals["cost_usd"] += cost
        self._totals["calls"] += 1
        self._totals["latency_ms_total"] += latency_ms

        text = next((b.text for b in response.content if b.type == "text"), "")
        try:
            data = json.loads(text)
            command = str(data.get("command", "look")).strip()
            reasoning = str(data.get("reasoning", "")).strip()
        except (json.JSONDecodeError, AttributeError):
            command, reasoning = "look", f"[unparseable response: {text[:200]}]"

        return AgentAction(
            command=command or "look",
            thought=reasoning,
            meta={
                "model": self.model,
                "effort": self.effort,
                "latency_ms": round(latency_ms),
                "cost_usd": round(cost, 6),
                "input_tokens": getattr(response.usage, "input_tokens", 0),
                "output_tokens": getattr(response.usage, "output_tokens", 0),
                "cache_read_tokens": getattr(response.usage, "cache_read_input_tokens", 0),
            },
        )

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "model": self.model,
            "effort": self.effort,
            "history_turns": self.history_turns,
            "max_tokens": self.max_tokens,
            "info_level": self.info_level,
            "system_prompt": self.system,
            "system_fingerprint": prompts.fingerprint(self.info_level),
        }

    def usage(self) -> dict[str, Any]:
        totals = dict(self._totals)
        calls = totals["calls"] or 1
        totals["latency_ms_avg"] = round(totals["latency_ms_total"] / calls)
        totals["cost_usd"] = round(totals["cost_usd"], 4)
        return totals
