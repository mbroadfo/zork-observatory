"""A turn in which the player looks something up before answering.

Both model players do this the same way and for the same reason: a reply that
carries a lookup instead of a command is answered from the record, and the
question is put again with both in view. It is deliberately not a tool call —
the journal field settled that argument once, when Ollama's tool-call path
turned out to be the one that fails on gpt-oss.

What matters here is that the lookup costs a request and not a turn, that the
budget actually binds, that a player without a record cannot spend requests
asking for one, and that everything the run is charged — tokens, latency, cost
— accumulates across the requests rather than reporting only the last.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from observatory.agents import llm
from observatory.agents.base import TurnContext
from observatory.agents.ollama_agent import OllamaAgent
from observatory.world.recall import Exchange, Recall

RECORD = Recall([
    Exchange(turn=7, command="examine grating", response="The grating is locked.",
             room="Clearing"),
    Exchange(turn=9, command="take leaves", response="Taken.", room="Clearing"),
])


def context(**kw) -> TurnContext:
    base = dict(turn=40, observation="You are in a small room.", score=25, moves=40,
                transcript=[("look", "You are in a small room.")], recall=RECORD)
    base.update(kw)
    return TurnContext(**base)  # type: ignore[arg-type]


def said(command: str = "", search: str = "", reasoning: str = "thinking") -> dict:
    body: dict[str, Any] = {"reasoning": reasoning, "command": command}
    if search:
        body["search"] = search
    return {
        "message": {"role": "assistant", "content": json.dumps(body)},
        "prompt_eval_count": 300,
        "eval_count": 25,
    }


class Server:
    """Answers with whatever the test scripted, and keeps what it was sent."""

    def __init__(self, *answers: dict) -> None:
        self.answers = list(answers)
        self.prompts: list[str] = []

    async def __call__(self, method, path, body):
        self.prompts.append(body["messages"][-1]["content"])
        return self.answers.pop(0) if self.answers else said("look")


def player(server: Server, **kw) -> OllamaAgent:
    return OllamaAgent(model="qwen3:8b", search=True, transport=server, **kw)


class TestTheLookupRoundTrip:
    async def test_a_lookup_is_answered_and_the_question_asked_again(self):
        server = Server(said(search="grating"), said(command="unlock grating"))
        action = await player(server).act(context())

        assert action.command == "unlock grating"
        assert len(server.prompts) == 2
        assert "The grating is locked." in server.prompts[1]

    async def test_the_results_arrive_in_the_same_turn_s_prompt(self):
        """Not as a fresh question: the player answers with its lookup and its
        window both in view."""
        server = Server(said(search="grating"), said(command="unlock grating"))
        await player(server).act(context())

        assert server.prompts[1].startswith(server.prompts[0])

    async def test_what_was_looked_up_is_recorded_on_the_action(self):
        server = Server(said(search="grating"), said(command="unlock grating"))
        action = await player(server).act(context())

        assert action.meta["searches"] == [
            {"query": "grating", "hits": 1, "turns": [7]}
        ]

    async def test_a_lookup_that_finds_nothing_still_answers(self):
        server = Server(said(search="screwdriver"), said(command="look"))
        action = await player(server).act(context())

        assert action.meta["searches"][0]["hits"] == 0
        assert "Nothing in the 2 turns" in server.prompts[1]

    async def test_an_ordinary_turn_costs_one_request(self):
        server = Server(said(command="north"))
        action = await player(server).act(context())

        assert len(server.prompts) == 1
        assert "searches" not in action.meta


class TestTheBudget:
    async def test_lookups_are_capped_and_the_turn_ends(self):
        """A player that could look things up forever would never move."""
        server = Server(*[said(search=f"thing {i}") for i in range(6)])
        action = await player(server, max_searches=2).act(context())

        assert len(action.meta["searches"]) == 2
        assert len(server.prompts) == 3        # two lookups, then the answer
        assert action.command == "look"        # nothing was typed; the floor

    async def test_the_player_is_told_how_many_are_left(self):
        """Not an invitation to use them — it is what stops a player being cut
        off mid-thought without knowing why."""
        server = Server(said(search="grating"), said(search="leaves"), said(command="wait"))
        await player(server, max_searches=2).act(context())

        assert "look back 1 more time" in server.prompts[1]
        assert "last lookup available" in server.prompts[2]

    async def test_a_single_lookup_budget_works(self):
        server = Server(said(search="grating"), said(command="unlock grating"))
        action = await player(server, max_searches=1).act(context())
        assert len(action.meta["searches"]) == 1
        assert action.command == "unlock grating"


class TestWhenThereIsNothingToSearch:
    async def test_a_player_with_no_record_does_not_spend_a_request(self):
        """`recall` is None when searching is switched off, or when a cold
        takeover has nothing of its own yet."""
        server = Server(said(search="grating", command="north"))
        action = await player(server).act(context(recall=None))

        assert len(server.prompts) == 1
        assert action.command == "north"
        assert "searches" not in action.meta

    async def test_a_player_built_without_search_is_not_told_it_can(self):
        quiet = OllamaAgent(model="qwen3:8b", transport=Server())
        assert "search" not in quiet.system.lower()
        assert "search" not in json.dumps(llm.move_fields(False, False))

    async def test_the_schema_offers_the_field_only_when_it_is_on(self):
        assert "search" in llm.move_fields(False, True)["properties"]
        assert "search" not in llm.move_fields(True, False)["properties"]
        assert "journal" in llm.move_fields(True, True)["properties"]


class TestAccounting:
    async def test_tokens_are_summed_across_every_request_of_the_turn(self):
        """A turn that took three requests cost three requests. Reporting only
        the last would make lookups look free."""
        server = Server(said(search="grating"), said(search="leaves"), said(command="wait"))
        agent = player(server, max_searches=2)
        action = await agent.act(context())

        assert action.meta["input_tokens"] == 900      # 3 × 300
        assert action.meta["output_tokens"] == 75      # 3 × 25
        assert agent.usage()["calls"] == 3

    async def test_the_run_records_whether_looking_back_was_available(self):
        on = OllamaAgent(model="qwen3:8b", search=True, transport=Server()).describe()
        off = OllamaAgent(model="qwen3:8b", transport=Server()).describe()

        assert on["search"] is True and off["search"] is False
        # Being told it may look things up is a different prompt, and has to
        # fingerprint as one or two arms become indistinguishable in the trace.
        assert on["system_fingerprint"] != off["system_fingerprint"]


class TestThroughTheSession:
    async def test_a_lookup_reaches_the_event_stream(self):
        """A run where searching helped is only distinguishable from one where
        it did not if the trace says what was asked for."""
        from observatory.engine.mock_engine import MockEngine
        from observatory.events import EventBus
        from observatory.session import Session, SessionConfig

        bus = EventBus()
        seen: list = []
        bus.subscribe(seen.append)

        server = Server(said(search="mailbox"), said(command="open mailbox"))
        session = Session(
            MockEngine(), player(server), bus, config=SessionConfig(delay=0.0),
        )
        await session.start()
        await session.step_once()

        search = next(e for e in seen if e.type == "agent.search")
        assert search.payload["query"] == "mailbox"
        assert search.payload["turn"] == 1
        # It searched the opening room description, which is on the record.
        assert search.payload["hits"] >= 1

    async def test_the_lookup_is_emitted_before_the_command(self):
        from observatory.engine.mock_engine import MockEngine
        from observatory.events import EventBus
        from observatory.session import Session, SessionConfig

        bus = EventBus()
        seen: list = []
        bus.subscribe(seen.append)
        server = Server(said(search="mailbox"), said(command="open mailbox"))
        session = Session(
            MockEngine(), player(server), bus, config=SessionConfig(delay=0.0),
        )
        await session.start()
        seen.clear()
        await session.step_once()

        order = [e.type for e in seen]
        assert order.index("agent.search") < order.index("command.issued")
