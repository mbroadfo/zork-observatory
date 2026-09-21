"""The local model player, against a stand-in for the Ollama server.

No server, no weights: the transport is swapped for a fake that records what
it was sent and answers with whatever the test scripted. What is pinned here is
the harness side — what the model is shown, what is kept from its reply, and
that a missing server is a clear error before turn one rather than a run of
silent `look`s.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from observatory.agents import build_agent, llm, parse_think, prompts
from observatory.agents.base import TurnContext
from observatory.agents.ollama_agent import OllamaAgent, OllamaError, list_models, resolve_host
from observatory.engine.mock_engine import MockEngine
from observatory.events import EventBus
from observatory.session import Session, SessionConfig

TAGS = {
    "models": [
        {"name": "qwen3:8b", "size": 5_200_000_000, "digest": "abc123",
         "details": {"family": "qwen3", "parameter_size": "8.2B", "quantization_level": "Q4_K_M"}},
        {"name": "llama3.2:latest", "size": 2_000_000_000, "digest": "def456", "details": {}},
    ]
}


def reply(command: str, reasoning: str = "trying something", **extra) -> dict:
    return {
        "message": {"role": "assistant", "content": json.dumps({"reasoning": reasoning, "command": command}), **extra},
        "prompt_eval_count": 300,
        "eval_count": 25,
    }


class FakeOllama:
    def __init__(self, answers=None, *, down=False, rejects_think=False):
        self.answers = list(answers or [])
        self.down = down
        self.rejects_think = rejects_think
        self.sent: list[dict] = []

    async def __call__(self, method, path, body):
        if self.down:
            raise OllamaError("cannot reach Ollama at http://localhost:11434: connection refused")
        if path == "/api/tags":
            return TAGS
        self.sent.append(dict(body))
        if self.rejects_think and "think" in body:
            raise OllamaError('"llama3.2" does not support thinking', status=400)
        if not self.answers:
            return reply("look")
        answer = self.answers.pop(0)
        if isinstance(answer, Exception):
            raise answer      # a scripted failure, for the turn that must survive one
        return answer if isinstance(answer, dict) else reply(answer)


def ctx(**kw) -> TurnContext:
    base: dict[str, Any] = dict(turn=3, observation="A room.", score=0, moves=2,
                transcript=[("", "Welcome."), ("north", "A room.")])
    base.update(kw)
    return TurnContext(**base)


def agent(fake, **kw) -> OllamaAgent:
    return OllamaAgent(kw.pop("model", "qwen3:8b"), transport=fake, **kw)


class TestWhatTheModelIsShown:
    async def test_the_same_words_as_every_other_model(self):
        fake = FakeOllama(["south"])
        a = agent(fake)
        c = ctx()
        await a.act(c)

        messages = fake.sent[0]["messages"]
        assert messages[0] == {"role": "system", "content": prompts.get("parser")}
        assert messages[1]["content"] == llm.turn_prompt(a.memory, c, 30)
        assert fake.sent[0]["format"] == llm.MOVE_FIELDS

    async def test_the_context_window_is_set_rather_than_left_to_truncate(self):
        fake = FakeOllama(["south"])
        await agent(fake, seed=7).act(ctx())
        opts = fake.sent[0]["options"]
        assert opts["num_ctx"] >= 8192
        assert opts["seed"] == 7
        assert fake.sent[0]["stream"] is False

    async def test_the_rung_changes_the_system_prompt(self):
        fake = FakeOllama(["hello"])
        a = agent(fake, info_level="cold")
        await a.act(ctx())
        assert fake.sent[0]["messages"][0]["content"] == prompts.get("cold")
        assert a.name == "ollama:qwen3:8b/cold"

    async def test_memory_rides_above_the_transcript(self):
        fake = FakeOllama(["south"])
        a = agent(fake)
        a.memory.add("the dark place killed me", turn=9)
        await a.act(ctx())
        content = fake.sent[0]["messages"][1]["content"]
        assert content.index("the dark place killed me") < content.index("Welcome.")

    async def test_a_zero_window_is_stateless(self):
        """`[-0:]` is the whole list in Python; zero must mean none of it."""
        fake = FakeOllama(["south"])
        c = ctx(observation="A room.", transcript=[("", "Welcome."), ("north", "A room.")])
        await agent(fake, history_turns=0).act(c)
        content = fake.sent[0]["messages"][1]["content"]
        assert "A room." in content
        assert "Welcome." not in content
        assert "> north" not in content

    async def test_the_session_hands_a_stateless_agent_no_history(self):
        fake = FakeOllama(["north", "east", "west"])
        session = Session(MockEngine(), agent(fake, history_turns=0), EventBus(),
                          config=SessionConfig(delay=0.0, max_turns=3, history_turns=0))
        await session.run()
        assert session._context().transcript == []
        assert "> north" not in fake.sent[-1]["messages"][1]["content"]

    async def test_thinking_is_left_alone_unless_asked(self):
        fake = FakeOllama(["south", "south"])
        await agent(fake).act(ctx())
        await agent(fake, think=False).act(ctx())
        assert "think" not in fake.sent[0]
        assert fake.sent[1]["think"] is False


class TestWhatIsKeptFromTheReply:
    async def test_command_and_reasoning(self):
        action = await agent(FakeOllama([reply("open the door", "A door is here.")])).act(ctx())
        assert action.command == "open the door"
        assert action.thought == "A door is here."
        assert action.meta["input_tokens"] == 300
        assert action.meta["output_tokens"] == 25
        assert action.meta["cost_usd"] == 0.0

    async def test_decoration_around_a_command_is_stripped(self):
        action = await agent(FakeOllama(["> `take box`\nand then look"])).act(ctx())
        assert action.command == "take box"

    async def test_a_prose_reply_still_yields_what_it_chose_to_type(self):
        prose = {"message": {"content": "north\n\nI'll head north to see."}, "eval_count": 9}
        action = await agent(FakeOllama([prose])).act(ctx())
        assert action.command == "north"
        assert action.meta["unstructured"] is True

    async def test_an_empty_reply_is_flagged(self):
        action = await agent(FakeOllama([reply("   ")])).act(ctx())
        assert action.command == "look"
        assert action.meta["error"] is True

    async def test_thinking_is_measured_not_shown(self):
        answer = reply("west", thinking="hmm " * 50)
        action = await agent(FakeOllama([answer])).act(ctx())
        assert action.meta["thinking_chars"] == 200
        assert "hmm" not in action.thought

    async def test_usage_counts_tokens_and_costs_nothing(self):
        a = agent(FakeOllama(["north", "south"]))
        await a.act(ctx())
        await a.act(ctx())
        u = a.usage()
        assert (u["calls"], u["input_tokens"], u["output_tokens"], u["cost_usd"]) == (2, 600, 50, 0.0)


class TestRunawayThinking:
    """qwen3:8b on a live run: thinking grew turn on turn until one reply
    never finished, and the run sat for ten minutes on the request timeout."""

    def overrun(self, n=4096):
        return {"message": {"content": "", "thinking": "wait, " * 900},
                "done_reason": "length", "prompt_eval_count": 2000, "eval_count": n}

    async def test_every_reply_has_a_ceiling(self):
        fake = FakeOllama(["north"])
        await agent(fake).act(ctx())
        assert fake.sent[0]["options"]["num_predict"] >= 1024

    async def test_an_overrun_is_retried_once_without_thinking_and_flagged(self):
        fake = FakeOllama([self.overrun(), "north"])
        a = agent(fake)
        action = await a.act(ctx())
        assert action.command == "north"
        assert action.meta["overran"] is True
        assert action.meta["overrun_tokens"] == 4096
        assert action.meta["output_tokens"] == 4096 + 25
        assert "think" not in fake.sent[0]
        assert fake.sent[1]["think"] is False
        # Only that turn: the configured setting is untouched.
        assert a.think is None
        await a.act(ctx())
        assert "think" not in fake.sent[2]

    async def test_two_overruns_are_an_error_turn(self):
        action = await agent(FakeOllama([self.overrun(), self.overrun()])).act(ctx())
        assert action.meta["error"] is True
        assert "twice" in action.thought

    async def test_the_thinking_is_kept_for_the_trace(self):
        action = await agent(FakeOllama([reply("west", thinking="the tree, perhaps")])).act(ctx())
        assert action.meta["thinking"] == "the tree, perhaps"

    async def test_a_reflection_is_not_starved_by_thinking(self):
        fake = FakeOllama([{"message": {"content": "Stay out of the dark."}}])
        await agent(fake).reflect(ctx(), cause="died")
        assert fake.sent[0]["options"]["num_predict"] > 400
        fake = FakeOllama([{"message": {"content": "Stay out of the dark."}}])
        await agent(fake, think=False).reflect(ctx(), cause="died")
        assert fake.sent[0]["options"]["num_predict"] == 400


class TestTheServer:
    async def test_preflight_passes_and_records_the_digest(self):
        a = agent(FakeOllama())
        assert await a.preflight() is None
        d = a.describe()
        assert d["digest"] == "abc123"
        assert d["details"]["parameter_size"] == "8.2B"
        assert d["system_prompt"] == prompts.get("parser")

    async def test_an_untagged_name_means_latest(self):
        assert await agent(FakeOllama(), model="llama3.2").preflight() is None

    async def test_a_model_not_pulled_says_how_to_get_it(self):
        problem = await agent(FakeOllama(), model="mistral:7b").preflight()
        assert "ollama pull mistral:7b" in problem
        assert "qwen3:8b" in problem

    async def test_no_server_is_an_error_before_turn_one(self):
        problem = await agent(FakeOllama(down=True)).preflight()
        assert "cannot reach" in problem

    async def test_a_server_that_dies_mid_run_is_visible_in_the_turn(self):
        fake = FakeOllama()
        a = agent(fake)
        fake.down = True
        action = await a.act(ctx())
        assert action.meta["error"] is True
        assert "ollama error" in action.thought

    async def test_a_model_that_cannot_think_is_asked_once_then_not_again(self):
        fake = FakeOllama(["north", "south"], rejects_think=True)
        a = agent(fake, think=True)
        assert (await a.act(ctx())).command == "north"
        assert (await a.act(ctx())).command == "south"
        assert [("think" in b) for b in fake.sent] == [True, False, False]
        assert a.think is None

    async def test_models_are_listed_smallest_first(self):
        models = await list_models(transport=FakeOllama())
        assert [m["name"] for m in models] == ["llama3.2:latest", "qwen3:8b"]
        assert models[1]["quantization"] == "Q4_K_M"


class TestAServerThatCannotCarryASchema:
    """gpt-oss on Ollama 0.13.5: a JSON schema plus the harmony reply format
    makes the server read the answer as a tool call and fail the request —
    "error parsing tool call: raw='open mailbox'" — after a minute and a half.
    The model answered; the server could not carry it."""

    def harmony(self):
        return OllamaError("error parsing tool call: raw='open mailbox', err=invalid character 'o'")

    async def test_the_turn_is_asked_again_in_prose_and_still_played(self):
        fake = FakeOllama([self.harmony(), {"message": {"content": "open mailbox\n\nA box."}}])
        a = agent(fake, model="qwen3:8b")
        action = await a.act(ctx())
        assert action.command == "open mailbox"
        assert "format" in fake.sent[0] and "format" not in fake.sent[1]

    async def test_it_is_learned_once_not_every_turn(self):
        fake = FakeOllama([self.harmony(), {"message": {"content": "north"}},
                           {"message": {"content": "south"}}])
        a = agent(fake, model="qwen3:8b")
        await a.act(ctx())
        assert (await a.act(ctx())).command == "south"
        assert [("format" in b) for b in fake.sent] == [True, False, False]
        assert a.describe()["structured_output"] is False

    async def test_a_model_known_to_choke_is_never_asked_under_a_schema(self):
        fake = FakeOllama([{"message": {"content": "open mailbox"}}])
        a = agent(fake, model="gpt-oss:20b")
        assert (await a.act(ctx())).command == "open mailbox"
        assert "format" not in fake.sent[0]

    async def test_every_other_model_is_still_asked_for_json(self):
        fake = FakeOllama(["north"])
        a = agent(fake)
        await a.act(ctx())
        assert fake.sent[0]["format"] == llm.MOVE_FIELDS
        assert a.describe()["structured_output"] is True

    async def test_an_unrelated_error_is_not_swallowed(self):
        fake = FakeOllama([OllamaError("model runner has unexpectedly stopped")] * 2)
        action = await agent(fake).act(ctx())
        assert action.meta["error"] is True
        assert "unexpectedly stopped" in action.thought

    async def test_one_failure_costs_a_retry_rather_than_the_turn(self):
        """A turn lost to a server bug is reported as a player who typed
        `look`. Ollama 0.13.5 does this to gpt-oss replies intermittently."""
        fake = FakeOllama([self.harmony(), "north"])
        a = agent(fake, model="gpt-oss:20b")
        action = await a.act(ctx())
        assert action.command == "north"
        assert "parsing tool call" in action.meta["retried_after"]
        assert len(fake.sent) == 2

    async def test_twice_is_the_honest_answer(self):
        fake = FakeOllama([self.harmony(), self.harmony()])
        action = await agent(fake, model="gpt-oss:20b").act(ctx())
        assert action.command == "look"
        assert action.meta["error"] is True
        assert "parsing tool call" in action.meta["first_error"]


class TestConfiguration:
    def test_host_forms(self, monkeypatch):
        monkeypatch.delenv("OLLAMA_HOST", raising=False)
        assert resolve_host() == "http://localhost:11434"
        monkeypatch.setenv("OLLAMA_HOST", "0.0.0.0:11434")
        assert resolve_host() == "http://localhost:11434"
        monkeypatch.setenv("OLLAMA_HOST", "http://ollama:11434/")
        assert resolve_host() == "http://ollama:11434"

    def test_think_spellings(self):
        assert [parse_think(v) for v in ("default", None, "off", "on", "high")] == [None, None, False, True, "high"]

    def test_a_model_is_required(self):
        with pytest.raises(ValueError, match="model"):
            build_agent("ollama", model=None)

    async def test_temperature_is_a_setting_and_travels_with_the_run(self):
        """0.7 is a setting for prose. Whether a player follows a rule better
        at 0.3 is a measurement, so it is a dial and it is recorded."""
        fake = FakeOllama(["north"])
        a = build_agent("ollama", model="qwen3:14b", temperature=0.3)
        a._send = fake
        await a.act(ctx())
        assert fake.sent[0]["options"]["temperature"] == 0.3
        assert a.describe()["options"]["temperature"] == 0.3
        assert build_agent("ollama", model="q").describe()["options"]["temperature"] == 0.7

    def test_the_context_window_is_a_setting(self):
        a = build_agent("ollama", model="qwen3:14b", num_ctx=8192)
        assert a.describe()["options"]["num_ctx"] == 8192
        assert build_agent("ollama", model="qwen3:8b").describe()["options"]["num_ctx"] == 16384

    async def test_a_prompt_near_the_window_is_flagged(self):
        """Overflow is dropped from the front — the system prompt and the
        memory — without a word from the server."""
        fake = FakeOllama(["north", "south"])
        a = agent(fake, num_ctx=1024)
        assert (await a.act(ctx())).meta.get("context_warning") is None
        big = ctx(transcript=[("look", "x " * 3000)])
        warning = (await a.act(big)).meta["context_warning"]
        assert "past num_ctx=1024" in warning

    def test_the_factory_builds_one(self):
        a = build_agent("ollama", model="qwen3:8b", info_level="game", think=False, seed=3)
        assert isinstance(a, OllamaAgent)
        assert (a.info_level, a.think, a.seed) == ("game", False, 3)


class TestInARun:
    async def test_it_plays_dies_reflects_and_remembers(self):
        # Into the unlit attic and wait there: the mock kills the player.
        answers = ["north", "east", "west", "up", "look", "look", "look", "look"]
        fake = FakeOllama(answers)
        fake.answers.insert(5, {"message": {"content": "Going up without light was fatal."}})
        a = agent(fake)
        bus = EventBus()
        seen: list = []
        bus.subscribe(seen.append)
        session = Session(MockEngine(), a, bus, config=SessionConfig(delay=0.0, max_turns=10, lives=1, checkpoint_every=2))
        await session.run()

        assert [lesson.text for lesson in a.memory.lessons] == ["Going up without light was fatal."]
        started = next(e for e in seen if e.type == "session.started")
        assert started.payload["agent"] == "ollama:qwen3:8b"
        ended = next(e for e in seen if e.type == "session.ended")
        assert ended.payload["usage"]["output_tokens"] > 0
        assert any(e.type == "run.restored" for e in seen)
        # The lesson is in front of the model on the turns after the restore.
        moves = [b for b in fake.sent if "format" in b]
        assert "Going up without light was fatal." in moves[-1]["messages"][1]["content"]
