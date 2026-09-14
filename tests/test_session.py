"""The turn loop, the event stream, branching, and round-tripping a trace."""

from __future__ import annotations

from observatory.agents.simple import MOCK_WALKTHROUGH, RandomAgent, ScriptedAgent
from observatory.engine.mock_engine import MockEngine
from observatory.events import EventBus
from observatory.session import Session, SessionConfig
from observatory.trace import TraceWriter, read_events, trace_header


def make_session(agent=None, **cfg) -> tuple[Session, EventBus, list]:
    bus = EventBus()
    seen: list = []
    bus.subscribe(seen.append)
    session = Session(
        MockEngine(),
        agent or ScriptedAgent(MOCK_WALKTHROUGH),
        bus,
        config=SessionConfig(delay=0.0, **cfg),
    )
    return session, bus, seen


def types_of(events) -> list[str]:
    return [e.type for e in events]


class TestSessionLoop:
    async def test_walkthrough_wins_and_reports_it(self):
        session, _, seen = make_session(max_turns=40)
        await session.run()

        assert session.finished
        assert session.end_reason == "victory"
        end = seen[-1]
        assert end.type == "session.ended"
        assert end.payload["final_score"] == end.payload["max_score"] == 50

    async def test_start_emits_the_opening_room_before_any_turn(self):
        session, _, seen = make_session()
        await session.start()

        assert types_of(seen)[0] == "session.started"
        assert "West of House" in seen[1].payload["text"]
        assert session.turn == 0

    async def test_each_turn_emits_command_then_observation(self):
        session, _, seen = make_session()
        await session.start()
        seen.clear()
        await session.step_once()

        order = types_of(seen)
        assert order.index("command.issued") < order.index("observation")
        assert "turn.begin" in order

    async def test_turn_limit_stops_the_run(self):
        session, _, seen = make_session(agent=RandomAgent(seed=3), max_turns=5)
        await session.run()

        assert session.turn == 5
        assert seen[-1].payload["reason"] == "turn limit reached"

    async def test_sequence_numbers_are_gapless(self):
        _, bus, seen = make_session()
        session, _, _ = make_session()
        session.bus = bus
        await session.run()

        assert [e.seq for e in seen] == list(range(1, len(seen) + 1))

    async def test_map_grows_as_the_agent_explores(self):
        session, _, _ = make_session(max_turns=40)
        await session.run()

        stats = session.map.stats()
        assert stats["rooms"] >= 6
        assert stats["edges"] >= 6

    async def test_an_agent_that_raises_ends_the_session_instead_of_crashing(self):
        class Exploding(RandomAgent):
            async def act(self, ctx):
                raise RuntimeError("model went to lunch")

        session, _, seen = make_session(agent=Exploding())
        await session.run()

        assert session.finished
        assert "error" in types_of(seen)
        assert "model went to lunch" in session.end_reason


class TestBranching:
    async def test_rewind_restores_score_and_position(self):
        session, _, _ = make_session(max_turns=40)
        await session.start()
        for _ in range(11):           # far enough to have banked the painting
            await session.step_once()

        cp = session.checkpoint("before the cellar")
        score_at_checkpoint = session._last_state.score
        turn_at_checkpoint = session.turn

        for _ in range(6):
            await session.step_once()
        assert session._last_state.score > score_at_checkpoint

        assert session.rewind(cp.id)
        assert session.turn == turn_at_checkpoint
        assert session._last_state.score == score_at_checkpoint

    async def test_rewind_keeps_the_discovered_map(self):
        """The world forgets; the observatory does not."""
        session, _, _ = make_session(max_turns=40)
        await session.start()
        for _ in range(3):
            await session.step_once()
        cp = session.checkpoint()
        rooms_at_checkpoint = session.map.stats()["rooms"]

        for _ in range(10):
            await session.step_once()
        rooms_explored = session.map.stats()["rooms"]
        assert rooms_explored > rooms_at_checkpoint

        session.rewind(cp.id)
        assert session.map.stats()["rooms"] == rooms_explored

    async def test_replaying_a_branch_is_deterministic(self):
        session, _, _ = make_session(max_turns=40)
        await session.start()
        for _ in range(5):
            await session.step_once()

        cp = session.checkpoint()
        first = [session.engine.step(c)[0].text for c in ["north", "east", "west"]]
        session.rewind(cp.id)
        second = [session.engine.step(c)[0].text for c in ["north", "east", "west"]]

        assert first == second

    async def test_rewind_to_an_unknown_checkpoint_is_refused(self):
        session, _, _ = make_session()
        await session.start()
        assert session.rewind("nope") is False

    async def test_checkpoints_appear_on_a_timer_without_being_asked_for(self):
        """You rarely know a turn mattered until later, so the fork points have
        to exist before anyone decides they want them."""
        session, _, seen = make_session(
            agent=RandomAgent(seed=1), max_turns=12, checkpoint_every=4
        )
        await session.run()

        autos = [c for c in session.checkpoints.values() if c.auto]
        assert sorted(c.turn for c in autos) == [4, 8, 12]
        assert [e.type for e in seen].count("checkpoint.created") == 3

    async def test_eviction_drops_old_auto_marks_but_never_manual_ones(self):
        session, _, _ = make_session(
            agent=RandomAgent(seed=1), max_turns=30,
            checkpoint_every=1, max_auto_checkpoints=5,
        )
        await session.start()
        for _ in range(3):
            await session.step_once()
        manual = session.checkpoint("mine")
        for _ in range(20):
            await session.step_once()

        assert manual.id in session.checkpoints
        assert sum(1 for c in session.checkpoints.values() if c.auto) == 5


class TestTraces:
    async def test_trace_round_trips_the_event_stream(self, tmp_path):
        path = tmp_path / "run.jsonl"
        bus = EventBus()
        live: list = []
        bus.subscribe(live.append)
        session = Session(
            MockEngine(),
            ScriptedAgent(MOCK_WALKTHROUGH),
            bus,
            config=SessionConfig(delay=0.0, max_turns=40),
            trace=TraceWriter(path, meta={"game": "mock", "agent": "scripted"}),
        )
        await session.run()

        recorded = list(read_events(path))
        assert types_of(recorded) == types_of(live)
        assert [e.seq for e in recorded] == [e.seq for e in live]
        assert trace_header(path)["game"] == "mock"

    async def test_a_branch_taken_after_the_run_ended_is_still_recorded(self, tmp_path):
        """Finishing a run closes the trace; rewinding to explore a
        counterfactual must reopen it rather than crash."""
        path = tmp_path / "run.jsonl"
        bus = EventBus()
        session = Session(
            MockEngine(), ScriptedAgent(MOCK_WALKTHROUGH), bus,
            config=SessionConfig(delay=0.0, max_turns=40),
            trace=TraceWriter(path),
        )
        await session.start()
        for _ in range(4):
            await session.step_once()
        cp = session.checkpoint("fork")
        await session.run()
        assert session.finished

        events_at_end = len(list(read_events(path)))
        assert session.rewind(cp.id)
        await session.step_once()

        assert len(list(read_events(path))) > events_at_end

    async def test_replayed_commands_match_what_was_played(self, tmp_path):
        path = tmp_path / "run.jsonl"
        bus = EventBus()
        session = Session(
            MockEngine(), ScriptedAgent(MOCK_WALKTHROUGH), bus,
            config=SessionConfig(delay=0.0, max_turns=40),
            trace=TraceWriter(path),
        )
        await session.run()

        replayed = [e.payload["command"] for e in read_events(path) if e.type == "command.issued"]
        assert replayed == MOCK_WALKTHROUGH
