"""The mock engine's contract — every backend must honour the same one."""

from __future__ import annotations

from observatory.engine.mock_engine import MockEngine


class TestSnapshotRestore:
    """Snapshot/restore is load-bearing: branching, tournaments, and
    counterfactual replay all rest on it."""

    def test_restore_undoes_a_move(self):
        engine = MockEngine()
        engine.reset()
        before = engine.snapshot()

        engine.step("north")
        assert engine.world_state().location_name == "North of House"

        engine.restore(before)
        assert engine.world_state().location_name == "West of House"

    def test_restore_undoes_score_and_inventory(self):
        engine = MockEngine()
        engine.reset()
        for cmd in ["north", "east", "west", "west", "take lantern", "turn on lantern",
                    "east", "up", "take painting"]:
            engine.step(cmd)
        mid = engine.snapshot()
        mid_state = engine.world_state()

        for cmd in ["down", "west", "put painting in trophy case"]:
            engine.step(cmd)
        assert engine.world_state().score == 25

        engine.restore(mid)
        now = engine.world_state()
        assert now.score == mid_state.score == 0
        assert sorted(now.inventory) == sorted(mid_state.inventory)
        assert now.state_hash == mid_state.state_hash

    def test_snapshot_is_not_a_live_view(self):
        """A snapshot taken before a mutation must not observe that mutation."""
        engine = MockEngine()
        engine.reset()
        blob = engine.snapshot()

        engine.step("open mailbox")
        engine.step("north")
        engine.restore(blob)

        assert engine.world_state().location_name == "West of House"
        assert "leaflet" not in engine.step("look")[0].text

    def test_state_hash_matches_for_identical_worlds(self):
        a, b = MockEngine(), MockEngine()
        a.reset()
        b.reset()
        for cmd in ["north", "east"]:
            a.step(cmd)
            b.step(cmd)

        assert a.world_state().state_hash == b.world_state().state_hash

    def test_state_hash_differs_after_a_real_change(self):
        engine = MockEngine()
        engine.reset()
        before = engine.world_state().state_hash
        engine.step("north")

        assert engine.world_state().state_hash != before


class TestParser:
    def test_blocked_exit_reports_why(self):
        engine = MockEngine()
        engine.reset()
        obs, state = engine.step("east")

        assert "boarded" in obs.text
        assert state.location_name == "West of House"

    def test_unknown_word_is_rejected(self):
        engine = MockEngine()
        engine.reset()
        assert "don't know that word" in engine.step("xyzzy")[0].text

    def test_taking_moves_the_object_to_the_player(self):
        engine = MockEngine()
        engine.reset()
        engine.step("open mailbox")
        engine.step("take leaflet")

        assert "leaflet" in engine.world_state().inventory

    def test_containers_hide_contents_until_opened(self):
        engine = MockEngine()
        obs, _ = engine.reset()
        assert "leaflet" not in obs.text

        assert "leaflet" in engine.step("open mailbox")[0].text


class TestDarkness:
    def test_a_dark_room_is_reported_as_dark(self):
        engine = MockEngine()
        engine.reset()
        for cmd in ["north", "east", "west", "up"]:
            engine.step(cmd)
        assert engine.world_state().dark is True

    def test_a_grue_eventually_eats_you(self):
        engine = MockEngine()
        engine.reset()
        for cmd in ["north", "east", "west", "up"]:
            engine.step(cmd)
        obs, _ = engine.step("look")

        assert obs.done and obs.lost
        assert "grue" in obs.text

    def test_a_lit_lamp_keeps_the_dark_room_safe(self):
        engine = MockEngine()
        engine.reset()
        for cmd in ["north", "east", "west", "west", "take lantern", "turn on lantern",
                    "east", "up"]:
            engine.step(cmd)
        obs, state = engine.step("look")

        assert state.dark is False
        assert not obs.done
        assert "painting" in obs.text


class TestObjectTree:
    """The reported tree should have the same shape Jericho reports, so the
    state pane and the object diff behave identically on both backends."""

    def test_the_player_is_an_object_inside_the_current_room(self):
        from observatory.world.objects import build_tree

        engine = MockEngine()
        engine.reset()
        engine.step("north")
        tree = build_tree(engine.world_state().objects)

        north = next(n for n in tree if n["name"] == "North of House")
        assert any(child["name"] == "you" for child in north["children"])

    def test_carried_items_hang_off_the_player(self):
        from observatory.world.objects import build_tree

        engine = MockEngine()
        engine.reset()
        engine.step("open mailbox")
        engine.step("take leaflet")
        tree = build_tree(engine.world_state().objects)

        room = next(n for n in tree if n["name"] == "West of House")
        player = next(c for c in room["children"] if c["name"] == "you")
        assert [c["name"] for c in player["children"]] == ["leaflet"]

    def test_moving_rooms_shows_up_as_the_player_moving(self):
        from observatory.world.objects import diff_objects

        engine = MockEngine()
        _, before = engine.reset()
        _, after = engine.step("north")

        (change,) = [c for c in diff_objects(before.objects, after.objects) if c.name == "you"]
        assert change.kind == "moved"


class TestValidActions:
    def test_valid_actions_reflect_the_current_room(self):
        engine = MockEngine()
        engine.reset()
        actions = engine.valid_actions()

        assert "north" in actions
        assert "open small mailbox" in actions
        assert "take painting" not in actions  # two rooms and a dark climb away
