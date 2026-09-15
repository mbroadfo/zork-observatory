"""The discovery ledger.

These tests pin two things: that a detector fires on the right evidence, and —
more important — that it does not fire on the wrong evidence. A false discovery
is worse than a missed one, because it silently shifts the number the whole
experiment reports.
"""

from __future__ import annotations

from observatory.agents.simple import MOCK_WALKTHROUGH, ScriptedAgent
from observatory.engine.mock_engine import MockEngine
from observatory.events import EventBus
from observatory.session import Session, SessionConfig
from observatory.world.discovery import DiscoveryLedger


def run(commands: list[str]) -> DiscoveryLedger:
    """Play a command list through the mock world and return the ledger."""
    engine = MockEngine()
    obs, state = engine.reset()
    ledger = DiscoveryLedger()
    prev = None
    ledger.observe(0, "", obs, state, prev)
    prev = state
    for turn, command in enumerate(commands, start=1):
        obs, state = engine.step(command)
        ledger.observe(turn, command, obs, state, prev)
        prev = state
    return ledger


class TestDetectorsFire:
    def test_movement_is_noticed_on_the_first_successful_move(self):
        ledger = run(["north"])
        assert ledger.found["movement"].turn == 1

    def test_abbreviations_are_a_separate_discovery_from_movement(self):
        ledger = run(["north", "s"])
        assert ledger.found["movement"].turn == 1
        assert ledger.found["abbreviation"].turn == 2

    def test_persistence_needs_a_return_trip(self):
        ledger = run(["north"])
        assert "persistence" not in ledger.found

        ledger = run(["north", "south"])
        assert ledger.found["persistence"].turn == 2

    def test_asking_for_inventory_counts(self):
        assert "inventory" in run(["inventory"]).found

    def test_possession_fires_when_the_inventory_grows(self):
        ledger = run(["open mailbox", "take leaflet"])
        assert ledger.found["possession"].turn == 2
        assert "leaflet" in ledger.found["possession"].evidence

    def test_release_is_distinct_from_possession(self):
        ledger = run(["open mailbox", "take leaflet", "drop leaflet"])
        assert ledger.found["possession"].turn == 2
        assert ledger.found["release"].turn == 3

    def test_opening_something_registers_containers(self):
        assert run(["open mailbox"]).found["containers"].turn == 1

    def test_vocabulary_limit_fires_on_an_unknown_word(self):
        ledger = run(["xyzzy"])
        assert "xyzzy" in ledger.found["vocabulary_limit"].evidence

    def test_reference_failure_is_a_different_lesson_from_vocabulary(self):
        ledger = run(["take lantern"])   # verb known, object absent
        assert "reference" in ledger.found
        assert "vocabulary_limit" not in ledger.found

    def test_climbing_into_an_unlit_room_establishes_darkness(self):
        # West of House → North of House → Behind House → Kitchen → Attic (dark)
        ledger = run(["north", "east", "west", "up"])
        assert ledger.found["darkness"].turn == 4
        assert "light_state" not in ledger.found   # nothing was lit, only entered

    def test_turning_on_a_light_registers_object_state(self):
        ledger = run([
            "north", "east", "west", "west", "take lantern", "east", "up",
            "turn on lantern",
        ])
        assert ledger.found["light_state"].turn == 8

    def test_death_is_recorded(self):
        ledger = run(["north", "east", "west", "up", "look"])
        assert "mortality" in ledger.found

    def test_scoring_fires_when_points_arrive(self):
        ledger = run(MOCK_WALKTHROUGH)
        assert "scoring" in ledger.found

    def test_meta_commands_are_noticed(self):
        assert "meta" in run(["version"]).found

    def test_composed_commands_register_grammar(self):
        """Three words accepted — the first evidence that commands have structure
        beyond verb-noun."""
        ledger = run(["north", "east", "west", "west", "take lantern", "turn on lantern"])
        assert ledger.found["grammar"].turn == 6
        assert ledger.found["grammar"].command == "turn on lantern"


class TestDetectorsStaySilent:
    def test_a_blocked_move_is_not_movement(self):
        ledger = run(["east"])   # boarded door; goes nowhere
        assert "movement" not in ledger.found

    def test_a_rejected_take_is_not_possession(self):
        ledger = run(["take unicorn"])
        assert "possession" not in ledger.found

    def test_dying_does_not_count_as_dropping_things(self):
        """Death empties your hands; that is not a lesson about releasing objects."""
        ledger = run([
            "north", "east", "west", "west", "take lantern", "east", "up", "look",
        ])
        assert "release" not in ledger.found

    def test_opening_something_that_is_not_a_container_does_not_fire(self):
        ledger = run(["open leaflet"])
        assert "containers" not in ledger.found

    def test_a_three_word_direction_is_not_grammar(self):
        ledger = run(["go north"])
        assert "grammar" not in ledger.found

    def test_room_prose_that_reads_like_a_refusal_is_not_a_refusal(self):
        """North of House is described as "There is no door here." — scenery
        that a substring match reads as a missing noun, on a move that
        *succeeded*. A phantom discovery silently corrupts every run."""
        ledger = run(["north"])

        assert ledger.found["movement"].turn == 1
        assert "reference" not in ledger.found
        assert "vocabulary_limit" not in ledger.found

    def test_nothing_fires_on_a_turn_that_did_nothing(self):
        ledger = run(["look"])
        assert set(ledger.found) <= {"reading"}


class TestLedgerShape:
    def test_manifest_lists_undiscovered_rows_too(self):
        """An empty row is data — 'never established X' is a finding."""
        ledger = run(["north"])
        manifest = ledger.manifest()

        assert len(manifest) == ledger.total
        assert any(row["turn"] == 0 for row in manifest)
        assert all("label" in row and "reveals" in row for row in manifest)

    def test_a_discovery_is_recorded_once_at_its_first_turn(self):
        ledger = run(["north", "south", "north", "south"])
        assert ledger.found["movement"].turn == 1

    def test_summary_counts_match_the_found_set(self):
        ledger = run(MOCK_WALKTHROUGH)
        summary = ledger.summary()

        assert summary["found"] == len(ledger.found)
        assert summary["total"] == ledger.total
        assert set(summary["turns"]) == set(ledger.found)


class TestSessionIntegration:
    async def test_the_session_emits_discoveries_as_they_happen(self):
        bus = EventBus()
        seen: list = []
        bus.subscribe(seen.append)
        session = Session(
            MockEngine(), ScriptedAgent(MOCK_WALKTHROUGH), bus,
            config=SessionConfig(delay=0.0, max_turns=40),
        )
        await session.run()

        events = [e for e in seen if e.type == "discovery.made"]
        assert events, "a full walkthrough should establish several discoveries"
        assert [e.payload["key"] for e in events] == list(session.ledger.found)

        end = seen[-1]
        assert end.payload["discoveries"]["found"] == len(session.ledger.found)

    async def test_a_run_that_learns_nothing_reports_an_empty_ledger(self):
        """The interesting negative case: the instrument must not invent progress."""
        class Mute(ScriptedAgent):
            def __init__(self):
                super().__init__(["xyzzy"] * 5)

        bus = EventBus()
        session = Session(
            MockEngine(), Mute(), bus,
            config=SessionConfig(delay=0.0, max_turns=5),
        )
        await session.run()

        assert set(session.ledger.found) == {"vocabulary_limit"}
