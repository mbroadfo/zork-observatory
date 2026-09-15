"""Coverage — ground covered against ground available.

The denominators are the point. Eleven rooms is an achievement in one game and
a rounding error in another, and the object table knows which.
"""

from __future__ import annotations

from observatory.agents.simple import MOCK_WALKTHROUGH, ScriptedAgent
from observatory.engine.mock_engine import MockEngine
from observatory.events import EventBus
from observatory.session import Session, SessionConfig
from observatory.world.coverage import Coverage
from observatory.world.discovery import Turn as DiscoveryTurn


def cover(commands: list[str]) -> Coverage:
    engine = MockEngine()
    obs, state = engine.reset()
    coverage = Coverage()
    prev = None
    for command in [""] + commands:
        if command:
            obs, state = engine.step(command)
        probe = DiscoveryTurn(
            turn=0, command=command, obs=obs, state=state, prev_state=prev,
            visited_rooms=set(), commands_seen=set(),
        )
        coverage.observe(state, probe.visible_objects(), probe.player_object())
        prev = state
    return coverage


class TestDenominators:
    def test_room_total_is_derived_from_the_object_table(self):
        """Rooms are the siblings of the room you are standing in — true of
        every Z-machine game, and needs no per-game special case."""
        assert cover([]).summary()["rooms_total"] == 9

    def test_the_denominator_is_known_from_the_first_turn(self):
        """You don't have to find the rooms to know how many there are."""
        early, late = cover([]), cover(MOCK_WALKTHROUGH)
        assert early.summary()["rooms_total"] == late.summary()["rooms_total"]

    def test_objects_exclude_rooms_and_the_player(self):
        """The player is identified by being the thing that moves rooms with
        you, so this is exact from the first move — before anything is carried."""
        assert cover(["north"]).summary()["objects_total"] == 8

    def test_the_player_is_identified_without_carrying_anything(self):
        coverage = cover(["north"])
        assert coverage.player_object is not None
        assert coverage.player_object not in coverage.objects_seen


class TestCounting:
    def test_only_the_starting_room_is_seen_at_first(self):
        s = cover([]).summary()
        assert s["rooms_seen"] == 1
        assert s["rooms_pct"] == round(100 / 9, 1)

    def test_walking_around_raises_the_count(self):
        assert cover(["north", "east"]).summary()["rooms_seen"] == 3

    def test_revisiting_does_not(self):
        assert cover(["north", "south", "north", "south"]).summary()["rooms_seen"] == 2

    def test_coverage_never_exceeds_one_hundred_percent(self):
        """The player object is inferred from its inventory, so it is unknown
        while the player is empty-handed — and both Zork and the mock start you
        that way. Counting it as a discovered object pushed this to 112.5%."""
        for commands in ([], ["north"], MOCK_WALKTHROUGH):
            s = cover(commands).summary()
            assert s["objects_pct"] <= 100.0, commands
            assert s["objects_seen"] <= s["objects_total"], commands

    def test_a_full_walkthrough_sees_every_object(self):
        s = cover(MOCK_WALKTHROUGH).summary()
        assert s["objects_seen"] == s["objects_total"] == 8
        assert s["objects_pct"] == 100.0

    def test_objects_held_counts_anything_ever_carried(self):
        """Ever held, not currently held — putting the painting in the case
        must not decrement it."""
        s = cover(MOCK_WALKTHROUGH).summary()
        assert s["objects_held"] == 3        # lantern, painting, egg

    def test_stowed_counts_things_put_inside_containers(self):
        """The generic form of "treasures in the case"."""
        assert cover(MOCK_WALKTHROUGH).summary()["stowed"] >= 2

    def test_score_is_reported_as_a_fraction_of_the_maximum(self):
        s = cover(MOCK_WALKTHROUGH).summary()
        assert (s["score"], s["max_score"], s["score_pct"]) == (50, 50, 100.0)

    def test_an_unstarted_run_does_not_divide_by_zero(self):
        assert Coverage().summary()["rooms_pct"] == 0.0


class TestSessionIntegration:
    async def test_coverage_rides_along_with_every_snapshot(self):
        bus = EventBus()
        seen: list = []
        bus.subscribe(seen.append)
        session = Session(
            MockEngine(), ScriptedAgent(MOCK_WALKTHROUGH), bus,
            config=SessionConfig(delay=0.0, max_turns=40),
        )
        await session.run()

        snapshots = [e for e in seen if e.type == "state.snapshot"]
        assert all("coverage" in e.payload for e in snapshots)

        # Monotonic: you cannot un-see a room.
        counts = [e.payload["coverage"]["rooms_seen"] for e in snapshots]
        assert counts == sorted(counts)
        assert seen[-1].payload["coverage"]["rooms_seen"] == 7
