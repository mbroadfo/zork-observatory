"""Changing players without moving the world.

A handoff is the one operation in this repository that puts two players in one
run, so what is pinned here is mostly about honesty rather than mechanics: the
world is exactly where it was, the abandoned turn does not count, the incoming
player is told where it is standing whether or not it is told how anyone got
there, and the trace says who typed what. A score that spans a handoff belongs
to neither player, and the record has to make that unmissable.
"""

from __future__ import annotations

import asyncio

import pytest

from observatory.agents.base import Agent, AgentAction, TurnContext
from observatory.agents.journal import Journal
from observatory.agents.simple import HumanAgent, RandomAgent, ScriptedAgent
from observatory.engine.mock_engine import MockEngine
from observatory.events import EventBus
from observatory.session import HANDOFF_TURNS, Session, SessionConfig


class Recorder(Agent):
    """A player that remembers what it was shown."""

    name = "recorder"
    kind = "test"

    def __init__(self, command: str = "look") -> None:
        self.seen: list[TurnContext] = []
        self.intro: str | None = None
        self.command = command

    async def act(self, ctx: TurnContext) -> AgentAction:
        self.seen.append(ctx)
        return AgentAction(command=self.command)

    async def on_start(self, intro: str) -> None:
        self.intro = intro


def make_session(agent=None, **cfg) -> tuple[Session, list]:
    bus = EventBus()
    seen: list = []
    bus.subscribe(seen.append)
    session = Session(
        MockEngine(),
        agent or ScriptedAgent(["open mailbox", "north", "east", "west"]),
        bus,
        config=SessionConfig(delay=0.0, **cfg),
    )
    return session, seen


async def played(session: Session, turns: int) -> None:
    await session.start()
    for _ in range(turns):
        await session.step_once()


class TestTheWorldDoesNotMove:
    async def test_the_new_player_inherits_the_position_exactly(self):
        session, _ = make_session()
        await played(session, 3)
        before = (session.turn, session.steps, session._last_state.score,
                  session._last_state.location_name, session.map.stats())

        await session.handoff(Recorder())
        after = (session.turn, session.steps, session._last_state.score,
                 session._last_state.location_name, session.map.stats())
        assert before == after

    async def test_the_next_command_comes_from_the_new_player(self):
        session, _ = make_session()
        await played(session, 2)
        taker = Recorder(command="open mailbox")

        await session.handoff(taker)
        await session.step_once()
        assert len(taker.seen) == 1
        assert session.transcript[-1][0] == "open mailbox"

    async def test_the_map_keeps_what_the_previous_player_discovered(self):
        """The world changes hands; the observatory's record of it does not."""
        session, _ = make_session()
        await played(session, 4)
        rooms = session.map.stats()["rooms"]
        assert rooms > 1

        await session.handoff(Recorder())
        assert session.map.stats()["rooms"] == rooms


class TestWhatTheNewPlayerIsTold:
    async def test_by_default_it_reads_the_turns_it_did_not_play(self):
        session, _ = make_session()
        await played(session, 3)
        taker = Recorder()

        await session.handoff(taker, inherit_transcript=True)
        await session.step_once()
        assert len(taker.seen[0].transcript) == len(session.transcript) - 1

    async def test_cold_means_no_history_at_all(self):
        session, _ = make_session()
        await played(session, 3)
        taker = Recorder()

        await session.handoff(taker, inherit_transcript=False)
        await session.step_once()
        assert taker.seen[0].transcript == []

    async def test_cold_still_says_which_room_it_is_standing_in(self):
        """You can take over a game without being told how it was played. You
        cannot take one over without being told where you are."""
        session, _ = make_session()
        await played(session, 3)
        here = session.transcript[-1][1]
        taker = Recorder()

        await session.handoff(taker, inherit_transcript=False)
        await session.step_once()
        assert taker.seen[0].observation == here
        assert taker.intro == here

    async def test_a_cold_player_accumulates_its_own_history(self):
        session, _ = make_session()
        await played(session, 3)
        taker = Recorder()

        await session.handoff(taker, inherit_transcript=False)
        await session.step_once()
        await session.step_once()
        assert len(taker.seen[1].transcript) == 1

    async def test_the_observatory_transcript_is_never_trimmed(self):
        """Withholding history from a player is not editing the record."""
        session, _ = make_session()
        await played(session, 3)
        full = len(session.transcript)

        await session.handoff(Recorder(), inherit_transcript=False)
        await session.step_once()
        assert len(session.transcript) == full + 1

    async def test_the_history_window_can_be_narrowed_for_the_taker(self):
        session, _ = make_session(history_turns=30)
        await played(session, 4)

        taker = Recorder()
        await session.handoff(taker, history_turns=2)
        await session.step_once()
        assert len(taker.seen[0].transcript) == 2


class TestTheAbandonedTurn:
    async def test_a_human_who_never_typed_does_not_use_up_a_turn(self):
        """The loop is parked inside `act`, before anything reached the engine.
        Counting that as a turn would credit the run with a move nobody made."""
        session, _ = make_session(HumanAgent())
        await session.start()
        task = session.start_background()
        for _ in range(20):        # let the loop reach the queue
            await asyncio.sleep(0)
        assert session.turn == 1   # begun, waiting to be typed into

        await session.handoff(Recorder())
        assert session.turn == 0 and session.steps == 0
        assert task.done()

    async def test_the_taker_plays_the_turn_that_was_abandoned(self):
        session, _ = make_session(HumanAgent())
        await session.start()
        session.start_background()
        for _ in range(20):
            await asyncio.sleep(0)

        taker = Recorder(command="north")
        await session.handoff(taker)
        await session.step_once()
        assert session.turn == 1
        assert session.transcript[-1][0] == "north"

    async def test_a_command_the_old_player_left_unfiled_is_dropped(self):
        session, _ = make_session()
        await played(session, 1)
        session._pending_journal = "something the old player wrote"

        await session.handoff(Recorder())
        assert session._pending_journal == ""


class TestTheRecord:
    async def test_the_handoff_is_emitted_with_both_names_and_the_position(self):
        session, seen = make_session()
        await played(session, 3)
        seen.clear()

        await session.handoff(Recorder(), inherit_transcript=False)
        event = next(e for e in seen if e.type == "run.handoff")
        assert event.payload["from_agent"] == "scripted"
        assert event.payload["agent"] == "recorder"
        assert event.payload["turn"] == 3
        assert event.payload["inherits"] is False
        assert event.payload["room"]

    async def test_the_summary_says_the_run_had_more_than_one_player(self):
        session, _ = make_session()
        await played(session, 2)
        await session.handoff(Recorder())

        handoffs = session.summary()["handoffs"]
        assert handoffs == [
            {"turn": 2, "from": "scripted", "to": "recorder", "inherits": True}
        ]
        assert session.summary()["agent"] == "recorder"

    async def test_the_end_of_the_run_carries_them_too(self):
        """A trace read back months later has to be able to tell that its final
        score belongs to two players."""
        session, seen = make_session(max_turns=3)
        await played(session, 2)
        await session.handoff(Recorder(), max_turns=3)
        await session.step_once()
        await session.step_once()

        end = next(e for e in seen if e.type == "session.ended")
        assert [h["to"] for h in end.payload["handoffs"]] == ["recorder"]

    async def test_an_ordinary_run_reports_no_handoffs(self):
        session, seen = make_session(max_turns=2)
        await session.run()
        assert session.summary()["handoffs"] == []
        end = next(e for e in seen if e.type == "session.ended")
        assert end.payload["handoffs"] == []

    async def test_the_trace_records_it(self, tmp_path):
        from observatory.trace import TraceWriter, read_events

        bus = EventBus()
        session = Session(
            MockEngine(), ScriptedAgent(["north"]), bus,
            config=SessionConfig(delay=0.0),
            trace=TraceWriter(tmp_path / "t.jsonl", meta={"game": "mock"}),
        )
        await played(session, 1)
        await session.handoff(Recorder())
        session.trace.close()

        types = [e.type for e in read_events(tmp_path / "t.jsonl")]
        assert "run.handoff" in types


class TestBudget:
    async def test_a_takeover_gets_turns_to_play(self):
        """The budget was set for somebody else. Handing over at the end of it
        and immediately reporting "turn budget exhausted" would be a joke."""
        session, _ = make_session(max_turns=3)
        await played(session, 3)

        await session.handoff(Recorder())
        assert session.config.max_turns == 3 + HANDOFF_TURNS
        assert await session.step_once() is True
        assert not session.finished

    async def test_turns_from_here_are_counted_after_the_turn_is_given_back(self):
        """`add_turns` is measured against the turn the new player actually
        starts on, which is not the number the old player had reached while it
        was still thinking."""
        session, _ = make_session(HumanAgent(), max_turns=500)
        await session.start()
        session.start_background()
        for _ in range(20):
            await asyncio.sleep(0)
        assert session.turn == 1

        await session.handoff(Recorder(), add_turns=6)
        assert session.turn == 0 and session.config.max_turns == 6

    async def test_an_explicit_budget_is_honoured(self):
        session, _ = make_session(max_turns=100)
        await played(session, 2)

        await session.handoff(Recorder(), max_turns=4)
        assert session.config.max_turns == 4

    async def test_a_budget_already_past_is_still_extended(self):
        session, _ = make_session(max_turns=3)
        await played(session, 3)
        await session.handoff(Recorder(), max_turns=2)
        assert session.config.max_turns == 3 + HANDOFF_TURNS


class TestRefusals:
    async def test_nothing_to_hand_over_before_the_run_starts(self):
        session, _ = make_session()
        with pytest.raises(RuntimeError, match="not started"):
            await session.handoff(Recorder())

    async def test_a_finished_run_cannot_change_players(self):
        session, _ = make_session(max_turns=1)
        await session.run()
        assert session.finished
        with pytest.raises(RuntimeError, match="over"):
            await session.handoff(Recorder())


class TestWhatTheTakerKeeps:
    async def test_a_journal_from_the_notebook_is_carried_not_claimed(self, tmp_path):
        """The incoming player is handed what earlier runs wrote. Those lines
        are not its work and must not be counted as its work."""
        from observatory.agents.journal import Entry
        from observatory.notebook import Notebook

        book = Notebook.open(tmp_path, "mock", "recorder", fresh=True)
        book.write(Entry(text="the window opens", turn=4, room="Behind House",
                         command="open window", outcome="progress", run=1))

        session, _ = make_session()
        await played(session, 2)
        taker = Recorder()
        taker.journal = Journal()

        await session.handoff(taker, notebook=book)
        assert len(taker.journal) == 1
        assert taker.journal.carried == 1
        assert taker.journal.summary()["attempted"] == 0

    async def test_the_taker_brings_its_own_memory(self):
        session, _ = make_session(RandomAgent())
        await played(session, 2)
        session.agent.memory.add(text="the old player's lesson", turn=1)

        taker = Recorder()
        await session.handoff(taker)
        assert len(taker.memory) == 0
        assert session.summary()["memory"] == []


class TestTheRoute:
    async def test_no_session_is_refused(self, monkeypatch):
        from observatory import server

        monkeypatch.setattr(server.hub, "session", None)
        res = await server.handoff(server.Handoff(agent="random"))
        assert res.status_code == 400

    async def test_the_walkthrough_cannot_take_over_midgame(self, monkeypatch):
        """It starts at the first move of a fresh game. From turn 83 it is not
        a walkthrough, it is a list of commands for a world that moved on."""
        from observatory import server

        session, _ = make_session()
        await played(session, 2)
        monkeypatch.setattr(server.hub, "session", session)

        res = await server.handoff(server.Handoff(agent="scripted"))
        assert res.status_code == 400
        assert b"first move" in res.body

    async def test_a_finished_run_is_refused_by_the_route(self, monkeypatch):
        from observatory import server

        session, _ = make_session(max_turns=1)
        await session.run()
        monkeypatch.setattr(server.hub, "session", session)

        res = await server.handoff(server.Handoff(agent="random"))
        assert res.status_code == 400

    async def test_a_takeover_swaps_the_player_and_keeps_the_score(self, monkeypatch):
        from observatory import server

        session, _ = make_session()
        await played(session, 3)
        score = session._last_state.score
        monkeypatch.setattr(server.hub, "session", session)

        res = await server.handoff(
            server.Handoff(agent="random", add_turns=50, start=False)
        )
        import json

        body = json.loads(res.body)
        assert body["session"]["agent"] == "random"
        assert body["session"]["turn"] == 3
        assert session._last_state.score == score
        assert session.config.max_turns == 53

    async def test_handing_back_to_a_human_does_not_start_a_loop(self, monkeypatch):
        """Resuming a human player means waiting for them, not playing for
        them. A background loop here would park on an empty queue and make the
        Run button lie about what is happening."""
        from observatory import server

        session, _ = make_session()
        await played(session, 2)
        monkeypatch.setattr(server.hub, "session", session)

        await server.handoff(server.Handoff(agent="human"))
        assert isinstance(session.agent, HumanAgent)
        assert session._task is None

    async def test_an_unknown_notebook_mode_is_refused(self, monkeypatch):
        from observatory import server

        session, _ = make_session()
        await played(session, 1)
        monkeypatch.setattr(server.hub, "session", session)

        res = await server.handoff(
            server.Handoff(agent="random", notebook="everything")
        )
        assert res.status_code == 400
