"""The shape of the map, as distinct from its size.

Every measure here exists to separate two runs that a room count calls equal,
so each test builds the pair it is meant to tell apart: the same number of
rooms found by pushing outward and by pacing, the same map walked forward and
walked into a pocket. A measure that gives both the same answer is not doing
its job, and the assertion is on the difference rather than the value.
"""

from __future__ import annotations

from observatory.world.frontier import (
    components,
    radius,
    reachable,
    since_new_room,
    stalest,
    structure,
    tried_directions,
)
from observatory.world.graph import MapGraph


def corridor(length: int) -> MapGraph:
    """`length` rooms in a line, walked once from one end to the other."""
    g = MapGraph()
    g.observe_room("r0", "Room 0", turn=0)
    for i in range(1, length):
        g.observe_transition(f"r{i - 1}", "north", f"r{i}", f"Room {i}", turn=i)
    return g


class TestTriedDirections:
    def test_counts_walls_as_tried(self):
        g = corridor(2)
        g.observe_transition("r1", "east", "r1", "Room 1", turn=2, response="The wall is solid.")

        tried = tried_directions(g)
        assert tried["r0"] == {"north"}
        assert tried["r1"] == {"east"}

    def test_a_room_only_arrived_in_has_tried_nothing(self):
        assert tried_directions(corridor(2))["r1"] == set()


class TestRadius:
    def test_grows_with_depth_not_with_room_count(self):
        """Four rooms off a hub and four rooms in a line are both four rooms."""
        line = corridor(5)

        hub = MapGraph()
        hub.observe_room("r0", "Hub", turn=0)
        for i, d in enumerate(["north", "south", "east", "west"], start=1):
            hub.observe_transition("r0", d, f"r{i}", f"Spoke {i}", turn=i)
            hub.observe_room("r0", "Hub", turn=i + 10)     # walked back each time

        assert len(hub.rooms) == len(line.rooms) == 5
        assert radius(line) == 4
        assert radius(hub) == 1

    def test_an_empty_map_has_no_radius(self):
        assert radius(MapGraph()) == 0

    def test_counts_the_way_back_even_when_only_one_way_was_walked(self):
        """The agent stood there, so it got that far. Whether it can return is
        `reachable`'s question, deliberately kept separate."""
        g = corridor(3)
        assert radius(g) == 2


class TestComponentsAndReach:
    def test_a_respawn_somewhere_unconnected_fragments_the_map(self):
        g = corridor(3)
        g.observe_room("far", "Somewhere Else", turn=9)      # a death put it here
        assert len(components(g)) == 2

    def test_reach_is_directional_so_a_one_way_drop_is_a_pocket(self):
        g = corridor(3)      # walked north only; nothing goes back south
        assert reachable(g, "r0") == 3
        assert reachable(g, "r2") == 1

    def test_reach_of_nowhere_is_nothing(self):
        assert reachable(corridor(2), None) == 0
        assert reachable(corridor(2), "r9") == 0


class TestStuckness:
    def test_since_new_room_is_the_gap_since_the_map_last_grew(self):
        g = corridor(3)      # last new room on turn 2
        assert since_new_room(g, turn=2) == 0
        assert since_new_room(g, turn=300) == 298

    def test_stalest_ignores_the_room_being_stood_in(self):
        g = corridor(3)
        name, turns = stalest(g, turn=100, current="r2")
        assert (name, turns) == ("Room 0", 100)

    def test_revisiting_refreshes_a_room(self):
        g = corridor(2)
        g.observe_room("r0", "Room 0", turn=80)
        assert stalest(g, turn=100, current="r1") == ("Room 0", 20)


class TestStructure:
    def test_pacing_and_exploring_differ_where_the_room_count_does_not(self):
        explore = corridor(6)

        pace = corridor(6)
        for turn in range(6, 30):                            # back and forth
            pace.observe_room("r4" if turn % 2 else "r5", "Room", turn=turn)

        assert len(explore.rooms) == len(pace.rooms)
        assert structure(explore, 5)["revisit_ratio"] == 0.0
        assert structure(pace, 29)["revisit_ratio"] > 0.7

    def test_an_unopened_run_reports_zeroes_rather_than_dividing_by_them(self):
        s = structure(MapGraph(), turn=0)
        assert s["revisit_ratio"] == 0.0 and s["tried_per_room"] == 0.0
        assert s["stale_room"] is None

    def test_walls_retested_after_something_changed_are_not_counted_unretried(self):
        g = corridor(2)
        g.observe_transition("r1", "east", "r1", "Room 1", turn=2, response="It is locked.")
        assert structure(g, 2)["unretried_walls"] == 1

        g.observe_transition("r1", "east", "r1", "Room 1", turn=40, response="It is locked.")
        assert structure(g, 40)["unretried_walls"] == 0

    def test_probed_once_finds_rooms_walked_straight_through(self):
        g = corridor(4)
        assert structure(g, 3)["probed_once"] == 4      # three exits used, one arrival

    def test_the_payload_is_json_safe(self):
        import json

        json.dumps(structure(corridor(3), turn=3, current="r2"))


class TestReplayAgreesWithTheLiveRun:
    """tools/structure.py rebuilds the map from a trace so that runs recorded
    before these measures existed can be read with them. That is only worth
    anything if the rebuilt map is the map the run had — including across a
    death, where the world rolls back under a map that does not."""

    async def test_a_recorded_run_replays_to_the_same_shape(self, tmp_path):
        import sys
        from pathlib import Path

        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
        import structure as replay

        from observatory.agents.simple import RandomAgent
        from observatory.engine.mock_engine import MockEngine
        from observatory.events import EventBus
        from observatory.session import Session, SessionConfig
        from observatory.trace import TraceWriter

        path = tmp_path / "run.jsonl"
        with TraceWriter(path) as writer:
            session = Session(
                MockEngine(), RandomAgent(seed=7), EventBus(),
                config=SessionConfig(delay=0.0, max_turns=40, lives=3),
                trace=writer,
            )
            await session.run()

        live = session.summary()["structure"]
        graph, _, ended = replay.rebuild(path)
        assert replay.structure(graph, ended["turns"], graph.path[-1]) == live
        assert len(graph.rooms) == session.map.stats()["rooms"]
