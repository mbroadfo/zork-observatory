"""Map building and object diffing — the two places a subtle bug would quietly
corrupt every run."""

from __future__ import annotations

from observatory.engine.base import WorldObject
from observatory.world.graph import MapGraph, parse_movement
from observatory.world.objects import build_tree, diff_objects


class TestParseMovement:
    def test_canonicalizes_abbreviations(self):
        assert parse_movement("n") == "north"
        assert parse_movement("NE") == "northeast"
        assert parse_movement("go west") == "west"
        assert parse_movement("  D  ") == "down"

    def test_rejects_non_movement(self):
        assert parse_movement("take lamp") is None
        assert parse_movement("northwind") is None
        assert parse_movement("") is None


class TestMapGraph:
    def test_builds_edge_from_a_transition(self):
        g = MapGraph()
        g.observe_room("r1", "West of House", turn=0)
        delta = g.observe_transition("r1", "north", "r2", "North of House", turn=1)

        assert delta["new_room"]["name"] == "North of House"
        assert delta["new_edge"]["direction"] == "north"
        assert g.stats() == {"rooms": 2, "edges": 1, "blocked": 0, "one_way_edges": 1, "deaths": 0}

    def test_marks_reciprocal_when_the_return_trip_is_observed(self):
        g = MapGraph()
        g.observe_room("r1", "A", turn=0)
        g.observe_transition("r1", "north", "r2", "B", turn=1)
        g.observe_transition("r2", "south", "r1", "A", turn=2)

        assert g.edges["r1|north"].reciprocal is True
        assert g.edges["r2|south"].reciprocal is True
        assert g.stats()["one_way_edges"] == 0

    def test_one_way_passage_stays_one_way(self):
        """Down a chute, then back up a different route — a real Zork shape."""
        g = MapGraph()
        g.observe_room("r1", "Top", turn=0)
        g.observe_transition("r1", "down", "r2", "Bottom", turn=1)
        g.observe_transition("r2", "west", "r3", "Side", turn=2)

        assert g.edges["r1|down"].reciprocal is False
        assert g.stats()["one_way_edges"] == 2

    def test_failed_movement_records_a_blocked_exit(self):
        g = MapGraph()
        g.observe_room("r1", "West of House", turn=0)
        delta = g.observe_transition(
            "r1", "east", "r1", "West of House", turn=1,
            response="The door is boarded and you can't remove the boards.",
        )

        assert delta["new_edge"] is None
        assert delta["new_blocked"]["direction"] == "east"
        assert "boarded" in g.blocked["r1|east"].message
        assert g.stats()["edges"] == 0

    def test_repeat_attempts_count_rather_than_duplicate(self):
        g = MapGraph()
        g.observe_room("r1", "A", turn=0)
        for turn in range(3):
            g.observe_transition("r1", "east", "r1", "A", turn=turn, response="Blocked.")

        assert len(g.blocked) == 1
        assert g.blocked["r1|east"].attempts == 3

    def test_non_movement_commands_never_create_edges(self):
        g = MapGraph()
        g.observe_room("r1", "A", turn=0)
        g.observe_transition("r1", "take lamp", "r1", "A", turn=1, response="Taken.")

        assert g.stats()["edges"] == 0
        assert g.stats()["blocked"] == 0

    def test_room_id_prefers_engine_object_number(self):
        assert MapGraph.room_id(42, "Maze") == "r42"

    def test_room_id_falls_back_to_a_name_slug(self):
        """Some games report no location object; identical names must still merge."""
        assert MapGraph.room_id(0, "West of House") == MapGraph.room_id(0, "West of House")
        assert MapGraph.room_id(0, "West of House") != MapGraph.room_id(0, "North of House")

    def test_revisiting_counts_visits_without_duplicating(self):
        g = MapGraph()
        g.observe_room("r1", "A", turn=0)
        g.observe_room("r1", "A", turn=5)

        assert g.rooms["r1"].visits == 2
        assert g.stats()["rooms"] == 1


class TestObjectDiff:
    @staticmethod
    def obj(num, name, parent):
        return WorldObject(num=num, name=name, parent=parent, child=0, sibling=0)

    def test_detects_a_parent_change_as_a_move(self):
        before = [self.obj(1, "lamp", 10)]
        after = [self.obj(1, "lamp", 20)]

        (change,) = diff_objects(before, after)
        assert (change.kind, change.before, change.after) == ("moved", 10, 20)

    def test_quiet_turn_produces_no_changes(self):
        objs = [self.obj(1, "lamp", 10), self.obj(2, "sword", 10)]
        assert diff_objects(objs, list(objs)) == []

    def test_detects_appearance_and_disappearance(self):
        before = [self.obj(1, "lamp", 10)]
        after = [self.obj(2, "thief", 10)]

        kinds = {c.kind for c in diff_objects(before, after)}
        assert kinds == {"appeared", "vanished"}

    def test_build_tree_nests_by_parent(self):
        objs = [
            self.obj(1, "mailbox", 0),
            self.obj(2, "leaflet", 1),
            self.obj(3, "sword", 0),
        ]
        tree = build_tree(objs)

        assert [n["name"] for n in tree] == ["mailbox", "sword"]
        assert [n["name"] for n in tree[0]["children"]] == ["leaflet"]

    def test_build_tree_survives_a_parent_cycle(self):
        """A bad VM read can produce cycles; the state pane must not hang."""
        objs = [self.obj(1, "a", 2), self.obj(2, "b", 1)]
        assert build_tree(objs) == []  # nothing is reachable from the root
