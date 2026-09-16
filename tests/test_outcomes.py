"""Turn classification and the futility count.

The distinction the whole panel rests on: *wasted* is the price of exploring and
a good player pays it; *futile* is repeating a failure you have already watched
happen in this exact room, which is the price of not listening. Conflating them
would make a careful explorer look as bad as a broken loop.
"""

from __future__ import annotations

from observatory.agents.simple import RandomAgent
from observatory.engine.mock_engine import MockEngine
from observatory.world.outcomes import Outcome, OutcomeTally


def classify(commands: list[str]) -> list[Outcome]:
    engine = MockEngine()
    _, state = engine.reset()
    tally = OutcomeTally()
    out = []
    for command in commands:
        prev = state
        obs, state = engine.step(command)
        out.append(tally.classify(command, obs, state, prev).outcome)
    return out


class TestClassification:
    def test_a_move_that_works_is_progress(self):
        assert classify(["north"]) == [Outcome.PROGRESS]

    def test_a_wall_is_blocked_not_unknown(self):
        assert classify(["east"]) == [Outcome.BLOCKED]

    def test_an_unknown_verb_is_its_own_category(self):
        assert classify(["quibbleflarn"]) == [Outcome.UNKNOWN_WORD]

    def test_naming_something_absent_is_its_own_category(self):
        assert classify(["take lantern"]) == [Outcome.ABSENT_NOUN]

    def test_taking_something_present_is_progress(self):
        assert classify(["open mailbox", "take leaflet"])[1] == Outcome.PROGRESS

    def test_looking_is_meta_not_waste(self):
        assert classify(["look"]) == [Outcome.META]

    def test_progress_is_judged_on_world_state_not_on_prose(self):
        """The only test that needs no knowledge of the game."""
        engine = MockEngine()
        _, before = engine.reset()
        obs, after = engine.step("north")
        tally = OutcomeTally()

        assert before.state_hash != after.state_hash
        assert tally.classify("north", obs, after, before).outcome is Outcome.PROGRESS


class TestFutility:
    def test_the_first_failure_is_not_futile(self):
        assert classify(["east"]) == [Outcome.BLOCKED]

    def test_repeating_a_known_failure_in_the_same_room_is(self):
        assert classify(["east", "east", "east"]) == [
            Outcome.BLOCKED, Outcome.FUTILE, Outcome.FUTILE,
        ]

    def test_the_same_command_in_a_different_room_is_not_futile(self):
        """`west` failing at one place says nothing about `west` at another."""
        outcomes = classify(["west", "north", "west"])
        assert outcomes[0] is Outcome.BLOCKED
        assert outcomes[2] is not Outcome.FUTILE

    def test_repeat_counts_accumulate(self):
        engine = MockEngine()
        _, state = engine.reset()
        tally = OutcomeTally()
        results = []
        for _ in range(4):
            prev = state
            obs, state = engine.step("east")
            results.append(tally.classify("east", obs, state, prev))

        assert [r.repeat_count for r in results] == [1, 2, 3, 4]

    def test_a_command_that_starts_working_is_forgiven(self):
        """Open the mailbox and `take leaflet` stops being a dead end."""
        engine = MockEngine()
        _, state = engine.reset()
        tally = OutcomeTally()

        prev, (obs, state) = state, engine.step("take leaflet")   # still closed
        assert tally.classify("take leaflet", obs, state, prev).outcome is Outcome.ABSENT_NOUN

        prev, (obs, state) = state, engine.step("open mailbox")
        tally.classify("open mailbox", obs, state, prev)

        prev, (obs, state) = state, engine.step("take leaflet")
        assert tally.classify("take leaflet", obs, state, prev).outcome is Outcome.PROGRESS
        assert ("West of House", "take leaflet") not in tally.failed

    def test_whitespace_and_case_do_not_hide_a_repeat(self):
        assert classify(["east", "  EAST  "]) == [Outcome.BLOCKED, Outcome.FUTILE]


class TestSummary:
    def test_waste_and_futility_are_reported_separately(self):
        engine = MockEngine()
        _, state = engine.reset()
        tally = OutcomeTally()
        # East of West of House is boarded: one real wall, then two repeats.
        for command in ["east", "east", "east", "north"]:
            prev = state
            obs, state = engine.step(command)
            tally.classify(command, obs, state, prev)

        summary = tally.summary()
        assert summary["steps"] == 4
        assert summary["wasted"] == 3          # one real wall, two repeats
        assert summary["futile"] == 2
        assert summary["wasted_pct"] == 75.0
        assert summary["futile_pct"] == 50.0

    def test_a_perfect_run_reports_no_waste(self):
        from observatory.agents.simple import MOCK_WALKTHROUGH

        engine = MockEngine()
        _, state = engine.reset()
        tally = OutcomeTally()
        for command in MOCK_WALKTHROUGH:
            prev = state
            obs, state = engine.step(command)
            tally.classify(command, obs, state, prev)

        assert tally.summary()["futile"] == 0

    def test_an_empty_tally_does_not_divide_by_zero(self):
        assert OutcomeTally().summary()["wasted_pct"] == 0.0


class TestHarvesterHygiene:
    """The bugs visible in a real run: `search holds` and `open don't`."""

    async def test_verbs_in_room_prose_do_not_become_nouns(self):
        agent = RandomAgent(seed=1)
        agent._harvest("A table holds an elongated brown sack.")

        assert "holds" not in agent._nouns
        assert "sack" in agent._nouns

    async def test_nothing_is_learned_from_a_parser_refusal(self):
        """Harvesting error text made the agent type `open don't` — learning
        its vocabulary from the errors its bad vocabulary caused."""
        agent = RandomAgent(seed=1)
        agent._harvest("I don't know that word.")
        agent._harvest("You can't see any such thing.")
        agent._harvest("You can't go that way.")

        assert agent._nouns == []

    async def test_contractions_and_adverbs_are_rejected(self):
        agent = RandomAgent(seed=1)
        agent._harvest("One small window is slightly ajar and won't budge.")

        assert "slightly" not in agent._nouns
        assert "won't" not in agent._nouns
        assert "window" in agent._nouns

    async def test_real_nouns_still_get_through(self):
        agent = RandomAgent(seed=1)
        agent._harvest("There is a brass lantern here. A trophy case sits open.")

        for noun in ("brass", "lantern", "trophy", "case"):
            assert noun in agent._nouns

    async def test_the_baseline_knows_nothing_before_it_is_shown_anything(self):
        assert RandomAgent(seed=1)._nouns == []


class TestTheWinningMove:
    def test_winning_is_progress_even_if_nothing_else_changed(self):
        """Zork's last move ends the game without moving the player."""
        from observatory.engine.base import Observation, WorldState
        from observatory.world.outcomes import Outcome, OutcomeTally

        s = WorldState(location_id=178, location_name="Stone Barrow", score=350, state_hash="x")
        won = Observation(text="Inside the Barrow ... Your score is 350.", done=True, won=True)
        assert OutcomeTally().classify("w", won, s, s).outcome is Outcome.PROGRESS


class TestEndingMovesAreNotWalls:
    def test_a_move_that_ends_the_game_draws_no_wall(self):
        from observatory.world.graph import MapGraph

        g = MapGraph()
        g.observe_transition(None, "", "r178", "Stone Barrow", 0)
        delta = g.observe_transition("r178", "w", "r178", "Stone Barrow", 1, ended=True)
        assert delta["new_blocked"] is None
        assert g.blocked == {}

    def test_an_ordinary_refusal_still_does(self):
        from observatory.world.graph import MapGraph

        g = MapGraph()
        g.observe_transition(None, "", "r178", "Stone Barrow", 0)
        g.observe_transition("r178", "e", "r178", "Stone Barrow", 1, response="You can't go that way.")
        assert "r178|east" in g.blocked
