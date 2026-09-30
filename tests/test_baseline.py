"""Reading many runs back as a distribution.

`tools/baseline.py` is the gate on everything the roadmap proposes: no arm is
reported until its effect clears the spread of a frozen baseline. That only
means anything if the numbers it reports are the numbers the runs actually had,
and if runs that differ only by their seed land in the same row.

Two failure modes are worth more than the rest, and both are here. A run that
never scored must report *nothing* rather than zero, because zero sorts as the
fastest run in the sweep. And steps must be counted rather than in-world turns,
because a rollback rewinds the turn counter and every run that died would
otherwise report reaching things sooner than it did.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

import baseline  # noqa: E402

from observatory.agents.simple import MOCK_WALKTHROUGH, ScriptedAgent  # noqa: E402
from observatory.engine.mock_engine import MockEngine  # noqa: E402
from observatory.events import EventBus  # noqa: E402
from observatory.session import Session, SessionConfig  # noqa: E402
from observatory.trace import TraceWriter  # noqa: E402


async def recorded(path: Path, agent=None, **cfg) -> Session:
    """Play a run to a trace and hand back the session that played it."""
    with TraceWriter(path) as writer:
        session = Session(
            MockEngine(),
            agent or ScriptedAgent(MOCK_WALKTHROUGH),
            EventBus(),
            config=SessionConfig(delay=0.0, max_turns=40, **cfg),
            trace=writer,
        )
        await session.run()
    return session


def synthetic(path: Path, config: dict, snapshots: list[dict]) -> Path:
    """A trace written by hand, for the cases a real run cannot be made to hit.

    One command per snapshot, in order. Nothing here exercises the session; the
    point is to control exactly what the reader is given.
    """
    lines = [{"kind": "trace.header", "version": 1}]
    lines.append({"kind": "event", "seq": 1, "ts": 0.0, "type": "session.started",
                  "payload": {"agent": "test", "agent_config": config}})
    seq = 2
    for snapshot in snapshots:
        lines.append({"kind": "event", "seq": seq, "ts": 0.0, "type": "command.issued",
                      "payload": {"command": snapshot.pop("command", "look")}})
        lines.append({"kind": "event", "seq": seq + 1, "ts": 0.0, "type": "state.snapshot",
                      "payload": snapshot})
        seq += 2
    path.write_text("\n".join(json.dumps(line) for line in lines) + "\n", encoding="utf-8")
    return path


class TestItAgreesWithTheRunItRead:
    async def test_the_score_and_rooms_are_the_ones_the_session_had(self, tmp_path):
        path = tmp_path / "run.jsonl"
        session = await recorded(path)

        measured = baseline.measure(path)
        assert measured is not None
        assert measured["score"] == session._last_state.score
        assert measured["rooms"] == session.map.stats()["rooms"]

    async def test_steps_count_commands_not_in_world_turns(self, tmp_path):
        """The two agree on a run that never rolls back, which is what makes
        the difference invisible until the run that matters."""
        path = tmp_path / "run.jsonl"
        session = await recorded(path)

        commands = sum(
            1 for line in path.read_text(encoding="utf-8").splitlines()
            if json.loads(line).get("type") == "command.issued"
        )
        assert baseline.measure(path)["steps"] == commands == session.steps

    async def test_a_run_that_scored_reports_when(self, tmp_path):
        path = tmp_path / "run.jsonl"
        measured = baseline.measure(await recorded(path) and path)

        assert measured["scoring@"] is not None
        assert 0 < measured["scoring@"] <= measured["steps"]

    async def test_acquiring_something_is_counted(self, tmp_path):
        path = tmp_path / "run.jsonl"
        assert baseline.measure(await recorded(path) and path)["acquired"] > 0

    def test_a_file_that_is_not_a_trace_is_skipped_rather_than_fatal(self, tmp_path):
        junk = tmp_path / "notes.jsonl"
        junk.write_text("not json\n{}\n", encoding="utf-8")
        assert baseline.measure(junk) is None


class TestWhatDidNotHappen:
    def test_a_run_that_never_scored_reports_nothing_not_zero(self, tmp_path):
        """Zero would sort as the fastest run in the table."""
        path = synthetic(tmp_path / "flat.jsonl", {"model": "m"}, [
            {"location_id": 1, "inventory": [], "score": 0},
            {"location_id": 1, "inventory": [], "score": 0},
        ])
        assert baseline.measure(path)["scoring@"] is None

    def test_a_missing_value_is_reported_as_missing_and_counted(self):
        assert baseline.spread([None, None]) == "-"
        assert "(1/3)" in baseline.spread([4, None, None])

    def test_the_median_is_not_taken_over_the_survivors_alone(self):
        """Two runs of three never scoring is a different fact from one that
        did, and averaging what is left says the opposite of what happened."""
        assert baseline.spread([10, None, None]).startswith("10")
        assert "(1/3)" in baseline.spread([10, None, None])

    def test_a_range_is_shown_only_when_the_runs_disagree(self):
        assert baseline.spread([5, 5, 5]) == "5"
        assert baseline.spread([1, 5]) == "3 [1-5]"


class TestCountingRooms:
    def test_rooms_are_keyed_on_identity_not_on_the_printed_name(self, tmp_path):
        """Zork prints `Darkness` for every unlit room. Counting names merges
        them into one and reports a run as more stuck than it was."""
        path = synthetic(tmp_path / "dark.jsonl", {"model": "m"}, [
            {"location_id": 7, "location_name": "Darkness", "inventory": [], "score": 0},
            {"location_id": 8, "location_name": "Darkness", "inventory": [], "score": 0},
        ])
        assert baseline.measure(path)["rooms"] == 2

    def test_the_gap_a_run_ended_on_is_reported_separately(self, tmp_path):
        """It is never closed by a new room, so it never lands in dead_max —
        and it is the most diagnostic number in the row."""
        path = synthetic(tmp_path / "stall.jsonl", {"model": "m"}, [
            {"location_id": 1, "inventory": [], "score": 0},
            {"location_id": 2, "inventory": [], "score": 0},
            *[{"location_id": 2, "inventory": [], "score": 0} for _ in range(5)],
        ])
        measured = baseline.measure(path)
        assert measured["dead_end"] == 5
        # Each room arrived one step after the last, so no gap during the run
        # comes close to the one it finished on.
        assert measured["dead_max"] == 1


class TestCountingWhatWasAcquired:
    def test_an_object_dropped_and_taken_again_is_not_a_second_acquisition(self, tmp_path):
        """Counting it would reward a run for fumbling."""
        path = synthetic(tmp_path / "fumble.jsonl", {"model": "m"}, [
            {"location_id": 1, "inventory": ["lamp"], "score": 0},
            {"location_id": 1, "inventory": [], "score": 0},
            {"location_id": 1, "inventory": ["lamp"], "score": 0},
        ])
        assert baseline.measure(path)["acquired"] == 1

    def test_examining_is_not_acquiring(self, tmp_path):
        path = synthetic(tmp_path / "look.jsonl", {"model": "m"}, [
            {"location_id": 1, "inventory": [], "score": 0},
            {"location_id": 1, "inventory": [], "score": 0},
        ])
        assert baseline.measure(path)["acquired"] == 0


class TestArmsNotFiles:
    """An arm is a configuration, not a filename. Filenames in a sweep are
    chosen by whoever wrote the shell script, and two of them may well describe
    the same experiment."""

    def test_runs_that_differ_only_by_seed_are_one_arm(self):
        first = {"model": "qwen3:14b", "options": {"seed": 1, "temperature": 0.3}}
        second = {"model": "qwen3:14b", "options": {"seed": 2, "temperature": 0.3}}
        assert baseline.fingerprint(first) == baseline.fingerprint(second)

    def test_a_different_temperature_is_a_different_arm(self):
        first = {"model": "qwen3:14b", "options": {"seed": 1, "temperature": 0.3}}
        second = {"model": "qwen3:14b", "options": {"seed": 1, "temperature": 0.7}}
        assert baseline.fingerprint(first) != baseline.fingerprint(second)

    def test_a_scaffold_is_a_different_arm(self):
        bare = {"model": "qwen3:14b", "nudge": False}
        nudged = {"model": "qwen3:14b", "nudge": True}
        assert baseline.fingerprint(bare) != baseline.fingerprint(nudged)

    def test_where_the_model_ran_is_not_part_of_the_experiment(self):
        """A digest and a host describe where inference happened, not what was
        asked of it. Two machines running the same arm are the same arm."""
        here = {"model": "qwen3:14b", "host": "http://ollama:11434", "digest": "aaa"}
        there = {"model": "qwen3:14b", "host": "http://gpu-2:11434", "digest": "bbb"}
        assert baseline.fingerprint(here) == baseline.fingerprint(there)

    def test_the_label_names_the_scaffolds_that_are_on(self):
        shown = baseline.label({
            "model": "qwen3:14b", "info_level": "coached",
            "options": {"temperature": 0.3}, "nudge": True, "agenda": True,
        })
        assert "qwen3:14b" in shown and "coached" in shown
        assert "nudge" in shown and "agenda" in shown

    def test_grouping_reports_how_many_runs_stand_behind_a_row(self, tmp_path):
        config = {"model": "m", "options": {"temperature": 0.3}}
        runs = []
        for seed in (1, 2, 3):
            path = synthetic(tmp_path / f"s{seed}.jsonl",
                             dict(config, options={"temperature": 0.3, "seed": seed}),
                             [{"location_id": 1, "inventory": [], "score": seed}])
            runs.append(baseline.measure(path))

        rows = baseline.group(runs)
        assert len(rows) == 1
        assert rows[0]["n"] == 3
        assert rows[0]["score"] == "2 [1-3]"
