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

from . import llm, prompts
from .base import Agent, AgentAction, TurnContext
from .journal import Journal

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
MOVE_SCHEMA = {"type": "json_schema", "schema": llm.MOVE_FIELDS}

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


# Models that take `output_config.effort`. Haiku 4.5 does not; sending it
# there would fail every call, and a failed call plays `look`.
NO_EFFORT_PREFIXES = ("claude-haiku-",)


def takes_effort(model: str) -> bool:
    return not model.startswith(NO_EFFORT_PREFIXES)


def output_config(model: str, effort: str, fmt: dict[str, Any] | None = None) -> dict[str, Any]:
    config: dict[str, Any] = {}
    if takes_effort(model):
        config["effort"] = effort
    if fmt is not None:
        config["format"] = fmt
    return config


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
        journal: Journal | None = None,
        api_key: str | None = None,
    ) -> None:
        import anthropic

        self.model = model
        self.effort = effort
        self.history_turns = history_turns
        self.max_tokens = max_tokens
        self.info_level = info_level
        self.journal = journal
        self.system = (
            prompts.with_journal(info_level) if journal is not None
            else prompts.get(info_level)
        )
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
        """The windowed transcript; `history_turns` is the single biggest cost
        lever here. The wording lives in llm.py, shared with every model."""
        return [{"role": "user", "content": llm.turn_prompt(
            self.memory, ctx, self.history_turns, self.journal)}]

    def _schema(self) -> dict[str, Any]:
        return {"type": "json_schema", "schema": llm.move_fields(self.journal is not None)}

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
                output_config=output_config(self.model, self.effort, self._schema()),
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
        journalled = ""
        try:
            data = json.loads(text)
            command = llm.clean_command(str(data.get("command", "look")))
            reasoning = str(data.get("reasoning", "")).strip()
            journalled = str(data.get("journal") or "").strip()
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
                **({"journal": journalled} if journalled and self.journal is not None else {}),
            },
        )

    async def reflect(self, ctx: TurnContext, cause: str) -> str | None:
        """One extra call, at the moment the run is about to be rolled back.

        Deliberately given the tail of the transcript and nothing else — no
        object tree, no cause of death from the engine, not even the harness's
        own word for what happened beyond what the game printed. If the agent
        misreads its own death, that misreading is the thing worth recording.
        """
        import anthropic

        try:
            response = await self._client.messages.create(
                model=self.model,
                max_tokens=400,
                system=[{"type": "text", "text": self.system, "cache_control": {"type": "ephemeral"}}],
                output_config=output_config(self.model, self.effort),
                messages=[{"role": "user", "content": llm.reflection_prompt(
                    self.memory, ctx, self.journal)}],
            )
        except (anthropic.APIStatusError, anthropic.APIConnectionError):
            return None

        if response.stop_reason == "refusal":
            return None

        self._totals["input_tokens"] += getattr(response.usage, "input_tokens", 0) or 0
        self._totals["output_tokens"] += getattr(response.usage, "output_tokens", 0) or 0
        self._totals["cost_usd"] += estimate_cost(self.model, response.usage)
        self._totals["calls"] += 1
        self._totals["reflections"] = self._totals.get("reflections", 0) + 1

        text = next((b.text for b in response.content if b.type == "text"), "").strip()
        return text or None

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "model": self.model,
            "effort": self.effort if takes_effort(self.model) else None,
            "history_turns": self.history_turns,
            "max_tokens": self.max_tokens,
            "info_level": self.info_level,
            "journal": getattr(self, "journal", None) is not None,
            "system_prompt": self.system,
            # Of the prompt actually sent, which the journal adds to. A run
            # whose system prompt differs must not report the same fingerprint
            # as one it does not match.
            "system_fingerprint": hashlib.sha256(self.system.encode()).hexdigest()[:12],
        }

    def usage(self) -> dict[str, Any]:
        totals = dict(self._totals)
        calls = totals["calls"] or 1
        totals["latency_ms_avg"] = round(totals["latency_ms_total"] / calls)
        totals["cost_usd"] = round(totals["cost_usd"], 4)
        return totals
