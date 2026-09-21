"""A player served by Ollama, on this machine.

Same words as the Claude player — prompts, transcript window and memory all
come from llm.py — so a difference between the two is a difference between
models, not between harnesses. What changes is the transport, and three things
a hosted API decides for you and a local server does not:

  Context length. Ollama truncates a prompt that overflows `num_ctx` from the
  front, silently. Thirty turns of transcript overflows the historical default,
  and the first thing lost would be the system prompt and the memory. So the
  window is set explicitly and recorded with the run.

  Sampling. Temperature and seed are set and recorded, so a run can be repeated
  and two models are sampled the same way.

  Which model. A tag like `qwen3:8b` is moved when the library is updated; the
  digest is not. The digest travels with the trace.

Local inference costs nothing per token, and every token is still counted:
how much inference a player needed to get somewhere is a measurement whether
or not anyone was billed for it.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import time
import urllib.error
import urllib.request
from typing import Any, Awaitable, Callable

from . import llm, prompts
from .base import Agent, AgentAction, TurnContext
from .episodic import EpisodicMemory

DEFAULT_HOST = "http://localhost:11434"
DEFAULT_INFO_LEVEL = "parser"
DEFAULT_NUM_CTX = 16384
# A ceiling on one reply, thinking included. A move is ~60 tokens and the
# longest healthy deliberation seen from an 8B model was ~2k; a model past
# this is going round in circles, and without a ceiling it goes round until
# the request times out.
DEFAULT_MAX_TOKENS = 4096
REQUEST_TIMEOUT_S = 300.0   # generous: the first call also loads the weights

# Servers that cannot carry a reply made under a JSON schema, in their own
# words. Matched against the error text rather than the model name, so a
# version that fixes it needs no change here.
_FORMAT_UNSUPPORTED = re.compile(r"parsing tool call|harmony|does not support (?:format|structured)")

# Models known to fail that way before the first call is made. gpt-oss speaks
# the harmony format, and a minute and a half is too long to learn it twice.
PLAIN_MODELS = ("gpt-oss",)

# (method, path, body) -> parsed JSON. Swappable so tests need no server.
Transport = Callable[[str, str, dict[str, Any] | None], Awaitable[dict[str, Any]]]


class OllamaError(Exception):
    def __init__(self, message: str, status: int | None = None) -> None:
        super().__init__(message)
        self.status = status


def resolve_host(host: str | None = None) -> str:
    """OLLAMA_HOST is the variable Ollama itself reads, and on a server it is
    often a bind address (`0.0.0.0:11434`) rather than something to dial."""
    raw = (host or os.environ.get("OLLAMA_HOST") or DEFAULT_HOST).strip().rstrip("/")
    if "://" not in raw:
        raw = "http://" + raw
    return raw.replace("//0.0.0.0", "//localhost")


def http_transport(host: str, timeout: float = REQUEST_TIMEOUT_S) -> Transport:
    """Plain urllib in a worker thread: no client library to install or pin."""

    def call(method: str, path: str, body: dict[str, Any] | None) -> dict[str, Any]:
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(
            host + path, data=data, method=method,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read() or b"{}")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode(errors="replace")
            try:
                detail = json.loads(detail).get("error", detail)
            except (ValueError, AttributeError):
                pass
            raise OllamaError(str(detail).strip(), status=exc.code) from None
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            reason = getattr(exc, "reason", exc)
            raise OllamaError(f"cannot reach Ollama at {host}: {reason}") from None

    async def transport(method: str, path: str, body: dict[str, Any] | None) -> dict[str, Any]:
        return await asyncio.to_thread(call, method, path, body)

    return transport


async def list_models(host: str | None = None, transport: Transport | None = None) -> list[dict[str, Any]]:
    """Installed models, smallest first — the order someone picking one wants."""
    host = resolve_host(host)
    send = transport or http_transport(host, timeout=5.0)
    data = await send("GET", "/api/tags", None)
    models = []
    for m in data.get("models", []):
        details = m.get("details") or {}
        models.append({
            "name": m.get("name") or m.get("model", ""),
            "size": m.get("size", 0),
            "digest": m.get("digest", ""),
            "parameter_size": details.get("parameter_size", ""),
            "quantization": details.get("quantization_level", ""),
            "family": details.get("family", ""),
        })
    return sorted(models, key=lambda m: m["size"])


def _same_model(a: str, b: str) -> bool:
    norm = lambda s: s if ":" in s else s + ":latest"  # noqa: E731
    return norm(a) == norm(b)


class OllamaAgent(Agent):
    kind = "llm"

    def __init__(
        self,
        model: str,
        host: str | None = None,
        history_turns: int = 30,
        info_level: str = DEFAULT_INFO_LEVEL,
        think: bool | str | None = None,
        num_ctx: int = DEFAULT_NUM_CTX,
        temperature: float | None = 0.7,
        seed: int | None = None,
        max_tokens: int | None = DEFAULT_MAX_TOKENS,
        transport: Transport | None = None,
        recall: str = "transcript",
        nudge: bool = False,
        candidates: bool = False,
        agenda: bool = False,
        vocabulary: bool = False,
        journal: Any = None,
    ) -> None:
        if not model:
            raise ValueError("an Ollama model name is required, e.g. qwen3:8b")
        llm.default_history(recall)   # validates
        self.model = model
        self.host = resolve_host(host)
        self.history_turns = history_turns
        self.recall = recall
        # One re-ask when the model picks a command its own record shows doing
        # nothing here. The record is kept either way when nudging is on; with
        # `transcript` recall it is used for this and never shown.
        self.nudge = nudge
        # Ask for a ranked list and play the first the record does not already
        # know to be inert here. Cheaper than the nudge (one call, not two) and
        # the skip is an act rather than a request.
        self.candidates = candidates
        # Two further scaffolds over the same record, off unless asked for.
        # They give away more than the record does, so what they are worth has
        # to be measurable rather than assumed. See EpisodicMemory.render.
        self.agenda = agenda
        self.vocabulary = vocabulary
        # What stays true of the world between runs: topology, first sight,
        # and what has been seen to do something. Opened by whoever knows the
        # story (cli.py, server.py) and written after every turn. See journal.py.
        self.journal = journal
        self.episodes = (
            EpisodicMemory()
            if recall == "episodic" or nudge or candidates or journal is not None
            else None
        )
        self.info_level = info_level
        self.system = prompts.get(info_level)
        # None leaves the model's own default; False/True or low/medium/high
        # (the gpt-oss form) set it. Models that cannot think reject the field,
        # and that is remembered rather than retried every turn.
        self.think = think
        self.num_ctx = num_ctx
        self.temperature = temperature
        self.seed = seed
        self.max_tokens = max_tokens
        # Ask in prose rather than under a schema. Set for models known to
        # choke on one, and latched the first time a server says so.
        self.plain = model.startswith(PLAIN_MODELS)
        self.digest = ""
        self.details: dict[str, Any] = {}
        self._send = transport or http_transport(self.host)
        suffix = "" if info_level == DEFAULT_INFO_LEVEL else f"/{info_level}"
        # Part of the name because it is part of the player: a notebook kept by
        # one memory architecture is not the other's to inherit.
        suffix += "" if recall == "transcript" else f"+{recall}"
        suffix += "+nudge" if nudge else ""
        suffix += "+candidates" if candidates else ""
        suffix += "+agenda" if agenda else ""
        suffix += "+vocab" if vocabulary else ""
        suffix += "+journal" if journal is not None else ""
        self.name = f"ollama:{model}{suffix}"
        self._totals: dict[str, Any] = {
            "input_tokens": 0,
            "output_tokens": 0,
            "cost_usd": 0.0,
            "calls": 0,
            "latency_ms_total": 0.0,
        }

    # --- server ----------------------------------------------------------

    def context_warning(self, prompt_chars: int) -> str:
        """Near the window, or past it. Ollama drops the overflow from the
        front without a word, and the front is the system prompt and the
        memory — so a run that quietly stopped being the run you configured
        should say so rather than just scoring badly."""
        # ~4 characters a token is rough, and rough is enough: the point is to
        # notice a prompt approaching the window, not to count it exactly.
        estimate = prompt_chars // 4
        if estimate > self.num_ctx:
            return f"prompt is about {estimate} tokens, past num_ctx={self.num_ctx}: the oldest part was dropped"
        if estimate > self.num_ctx * 0.85:
            return f"prompt is about {estimate} tokens, close to num_ctx={self.num_ctx}"
        return ""

    async def preflight(self) -> str | None:
        """Why this run cannot start, or None. Asked once, before turn one, so
        a missing server is an error message and not four hundred turns of an
        agent that silently types `look`."""
        try:
            models = await list_models(self.host, self._send)
        except OllamaError as exc:
            return f"{exc} — is the ollama service running?"
        match = next((m for m in models if _same_model(m["name"], self.model)), None)
        if match is None:
            have = ", ".join(m["name"] for m in models) or "none"
            return f"model {self.model!r} is not pulled (installed: {have}). Try: ollama pull {self.model}"
        self.digest = match["digest"]
        self.details = {k: match[k] for k in ("parameter_size", "quantization", "family")}
        return None

    def _options(self, max_tokens: int | None) -> dict[str, Any]:
        opts: dict[str, Any] = {"num_ctx": self.num_ctx}
        if self.temperature is not None:
            opts["temperature"] = self.temperature
        if self.seed is not None:
            opts["seed"] = self.seed
        if max_tokens is not None:
            opts["num_predict"] = max_tokens
        return opts

    async def _chat(
        self, user: str, fmt: dict[str, Any] | None, max_tokens: int | None,
        think: bool | str | None = None, override: bool = False,
    ) -> dict[str, Any]:
        """One request. `override` sends `think` for this call only, leaving
        the configured setting alone."""
        body: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self.system},
                {"role": "user", "content": user},
            ],
            "stream": False,
            "options": self._options(max_tokens),
            "keep_alive": "30m",
        }
        if fmt is not None:
            body["format"] = fmt
        if fmt is not None and self.plain:
            body.pop("format", None)
        wanted = think if override else self.think
        if wanted is not None:
            body["think"] = wanted
        try:
            return await self._send("POST", "/api/chat", body)
        except OllamaError as exc:
            reason = str(exc).lower()
            if not override and self.think is not None and "think" in reason:
                self.think = None
                body.pop("think", None)
                return await self._send("POST", "/api/chat", body)
            if fmt is not None and not self.plain and _FORMAT_UNSUPPORTED.search(reason):
                # gpt-oss on Ollama 0.13.5: a schema plus the harmony reply
                # format makes the server read the answer as a tool call and
                # fail it — "error parsing tool call: raw='open mailbox'" — at
                # a minute and a half a turn. The model answered; the server
                # could not carry it. Asked in prose from here on, and the
                # first line of the reply is still what it chose to type.
                self.plain = True
                body.pop("format", None)
                return await self._send("POST", "/api/chat", body)
            raise

    def _count(self, data: dict[str, Any], latency_ms: float) -> tuple[int, int]:
        # prompt_eval_count covers only the tokens Ollama had to evaluate; a
        # prefix it still held from the previous turn is not recounted.
        inp = int(data.get("prompt_eval_count") or 0)
        out = int(data.get("eval_count") or 0)
        self._totals["input_tokens"] += inp
        self._totals["output_tokens"] += out
        self._totals["calls"] += 1
        self._totals["latency_ms_total"] += latency_ms
        return inp, out

    # --- play ------------------------------------------------------------

    @staticmethod
    def _parse(message: dict[str, Any]) -> tuple[str, str, bool]:
        """(command, reasoning, structured) from a reply."""
        command, _, reasoning, structured = OllamaAgent._parse_ranked(message)
        return command, reasoning, structured

    @staticmethod
    def _parse_ranked(message: dict[str, Any]) -> tuple[str, list[str], str, bool]:
        """(command, alternatives, reasoning, structured) from a reply."""
        text = (message.get("content") or "").strip()
        try:
            parsed = json.loads(text)
            raw = parsed.get("alternatives") or []
            alternatives = [
                cleaned for cleaned in (llm.clean_command(str(a)) for a in raw if isinstance(a, str))
                if cleaned
            ]
            return (
                llm.clean_command(str(parsed.get("command", ""))),
                alternatives,
                str(parsed.get("reasoning", "")).strip(),
                True,
            )
        except (json.JSONDecodeError, AttributeError):
            # Small models sometimes answer in prose despite the format. The
            # first line is still what it chose to type, and is kept as such.
            return llm.clean_command(text), [], "[reply was not in the requested format]", False

    async def act(self, ctx: TurnContext) -> AgentAction:
        action = await self._act(ctx)
        # A ranked list that is inert all the way down is the moment the player
        # is most stuck, and the one moment the filter has nothing left to do.
        # Ask again there, whether or not the nudge was asked for.
        wants_nudge = self.nudge or action.meta.get("all_candidates_inert")
        if wants_nudge and self.episodes is not None and not action.meta.get("error"):
            action = await self._nudged(action, ctx)
        if self.episodes is not None:
            self.episodes.after_move(action.command)
        if self.journal is not None and self.episodes is not None:
            # Written every turn: a run stopped halfway leaves the world's
            # shape behind rather than throwing it away.
            self.journal.absorb(self.episodes)
            self.journal.save()
        return action

    async def _nudged(self, action: AgentAction, ctx: TurnContext) -> AgentAction:
        """Ask once more when the chosen command has already done nothing here.

        Only once. A model that repeats itself through the nudge has made its
        choice, and the turn stands as its own — flagged, so the run can be
        read as what it is.
        """
        assert self.episodes is not None
        entry = self.episodes.repeat_of(action.command)
        if entry is None:
            return action
        note = llm.repeat_nudge(entry.command, entry.count, next(iter(entry.outcomes)))
        if action.meta.get("all_candidates_inert"):
            note = llm.exhausted_nudge(action.meta.get("offered") or [action.command], note)
        again = await self._act(ctx, extra=note)
        again.meta["nudged"] = note
        again.meta["nudged_from"] = action.command
        # Why the re-ask happened belongs with the move it produced.
        for key in ("all_candidates_inert", "offered", "skipped"):
            if key in action.meta:
                again.meta.setdefault(key, action.meta[key])
        for key in ("input_tokens", "output_tokens"):
            again.meta[key] = action.meta.get(key, 0) + again.meta.get(key, 0)
        if self.episodes.repeat_of(again.command) is not None:
            again.meta["nudge_ignored"] = True
        return again

    async def _act(self, ctx: TurnContext, extra: str = "") -> AgentAction:
        record = ""
        if self.episodes is not None:
            if not extra:
                self.episodes.before_move(ctx)
            record = (
                self.episodes.render(agenda=self.agenda, vocabulary=self.vocabulary)
                if self.recall == "episodic" else ""
            )
        if self.journal is not None:
            known = self.journal.render()
            record = f"{known}\n\n{record}" if record else known
        prompt = llm.turn_prompt(self.memory, ctx, self.history_turns, record)
        if extra:
            prompt += "\n\n" + extra
        crowded = self.context_warning(len(prompt) + len(self.system))
        started = time.perf_counter()
        schema = llm.MOVE_FIELDS_RANKED if self.candidates else llm.MOVE_FIELDS
        retried = ""
        try:
            data = await self._chat(prompt, schema, self.max_tokens)
        except OllamaError as exc:
            # One more attempt before the turn is spent. A local server fails
            # in ways a hosted one does not — Ollama 0.13.5 mangles gpt-oss
            # replies in its own parser, intermittently — and a turn lost to
            # that is a turn of the run reported as if the player had played
            # `look`. The second failure is the honest one.
            retried = str(exc)
            try:
                data = await self._chat(prompt, schema, self.max_tokens)
            except OllamaError as second:
                return AgentAction(
                    command="look", thought=f"[ollama error: {second}]",
                    meta={"error": True, "first_error": retried},
                )
        inp, out = self._count(data, (time.perf_counter() - started) * 1000)

        message = data.get("message") or {}
        command, alternatives, reasoning, structured = self._parse_ranked(message)
        thinking = message.get("thinking") or ""
        overran = data.get("done_reason") == "length" and not command

        meta: dict[str, Any] = {"model": self.model, "cost_usd": 0.0}
        if retried:
            meta["retried_after"] = retried
        if crowded:
            meta["context_warning"] = crowded
        if self.episodes is not None:
            # The record exactly as the model saw it this turn: when a player
            # repeats itself, whether the repeat was on the page is the question.
            meta["record"] = record
            meta["record_stats"] = self.episodes.summary()
        if thinking:
            # Kept whole in the trace: a model that talked itself in circles
            # is only diagnosable from what it said.
            meta["thinking_chars"] = len(thinking)
            meta["thinking"] = thinking
        if data.get("load_duration", 0) > 1e9:
            meta["load_ms"] = round(data["load_duration"] / 1e6)

        if overran:
            # It hit the ceiling before answering — in practice a thinking
            # loop. Ask again for this turn only, without thinking, and say so:
            # the move that follows was not made under the configured setting.
            meta["overran"] = True
            meta["overrun_tokens"] = out
            retry_started = time.perf_counter()
            try:
                data = await self._chat(prompt, schema, self.max_tokens, think=False, override=True)
            except OllamaError as exc:
                data = {}
                reasoning = f"[ran past {out} tokens without answering; retry failed: {exc}]"
            if data:
                i2, o2 = self._count(data, (time.perf_counter() - retry_started) * 1000)
                inp, out = inp + i2, out + o2
                command, reasoning, structured = self._parse(data.get("message") or {})
                if data.get("done_reason") == "length" and not command:
                    reasoning = "[ran past the output ceiling twice without answering]"

        meta.update({
            "latency_ms": round((time.perf_counter() - started) * 1000),
            "input_tokens": inp,
            "output_tokens": out,
        })
        if not structured:
            meta["unstructured"] = True

        if self.candidates and self.episodes is not None and command:
            command = self._first_untried([command, *alternatives], meta)
        if not command:
            meta["error"] = True

        return AgentAction(command=command or "look", thought=reasoning, meta=meta)

    def _first_untried(self, ranked: list[str], meta: dict[str, Any]) -> str:
        """The model's own ranking, minus what the record has watched do nothing.

        If every candidate is one of those, the first stands: the choice was
        the model's, and the harness has no better idea than it does.
        """
        assert self.episodes is not None
        skipped = []
        for candidate in ranked:
            if self.episodes.repeat_of(candidate) is None:
                if skipped:
                    meta["skipped"] = skipped
                    meta["offered"] = ranked
                return candidate
            skipped.append(candidate)
        meta["skipped"] = skipped[:-1]
        meta["offered"] = ranked
        meta["all_candidates_inert"] = True
        return ranked[0]

    async def reflect(self, ctx: TurnContext, cause: str) -> str | None:
        if self.episodes is not None:
            self.episodes.before_reflection(ctx)
        started = time.perf_counter()
        try:
            # The tight cap only when thinking is known to be off; a model left
            # to its default may think, and would spend 400 tokens doing so.
            data = await self._chat(
                llm.reflection_prompt(self.memory, ctx), None,
                400 if self.think is False else self.max_tokens,
            )
        except OllamaError:
            return None
        self._count(data, (time.perf_counter() - started) * 1000)
        self._totals["reflections"] = self._totals.get("reflections", 0) + 1
        text = ((data.get("message") or {}).get("content") or "").strip()
        return text or None

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "provider": "ollama",
            "model": self.model,
            "digest": self.digest,
            "details": self.details,
            "host": self.host,
            "think": self.think,
            "structured_output": not self.plain,
            "options": self._options(self.max_tokens),
            "history_turns": self.history_turns,
            "recall": self.recall,
            "nudge": self.nudge,
            "candidates": self.candidates,
            "agenda": self.agenda,
            "vocabulary": self.vocabulary,
            "journal": self.journal.summary() if self.journal is not None else None,
            "info_level": self.info_level,
            "system_prompt": self.system,
            "system_fingerprint": prompts.fingerprint(self.info_level),
        }

    def usage(self) -> dict[str, Any]:
        totals = dict(self._totals)
        calls = totals["calls"] or 1
        totals["latency_ms_avg"] = round(totals["latency_ms_total"] / calls)
        return totals
