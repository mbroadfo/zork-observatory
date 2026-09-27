"""Looking back through what has already been printed.

The window holds thirty exchanges; a puzzle worth solving arrives two hundred
turns after the sentence that explains it. This is the player going to look.

What is pinned here is mostly what the search must *not* do. It must not reveal
anything that was not printed on the screen — that is what keeps it on the
agent's side of the line that holds the object tree on ours. It must not
summarise, because a summary is a second player whispering. And it must respect
the handoff floor, because a cold takeover that can search its predecessor's
play is not cold at all.
"""

from __future__ import annotations

import pytest

from observatory.world.recall import Exchange, Recall, terms


def record(*rows: tuple[int, str, str, str]) -> Recall:
    return Recall([
        Exchange(turn=t, command=c, response=r, room=room) for t, c, r, room in rows
    ])


SAMPLE = (
    (3, "open mailbox", "Opening the small mailbox reveals a leaflet.", "West of House"),
    (12, "take knife", "Taken.", "Attic"),
    (41, "cut nails with knife", "The nasty knife is not the tool for that.", "Attic"),
    (88, "read paper", "Congratulations! You are the privileged owner.", "Studio"),
)


class TestFindingThings:
    def test_a_word_finds_the_turn_that_printed_it(self):
        hits = record(*SAMPLE).search("leaflet")
        assert [h.exchange.turn for h in hits] == [3]

    def test_a_word_in_a_command_counts_as_much_as_one_in_a_reply(self):
        """Half of what a player needs to recall is what it already tried."""
        hits = record(*SAMPLE).search("nails")
        assert [h.exchange.turn for h in hits] == [41]

    def test_more_matching_words_ranks_higher(self):
        hits = record(*SAMPLE).search("knife nails")
        assert hits[0].exchange.turn == 41 and hits[0].matched == 2
        assert hits[1].exchange.turn == 12

    def test_ties_go_to_the_most_recent(self):
        """The world may have been changed since. The older sighting is the
        stale one."""
        rec = record(
            (5, "look", "There is a rope here.", "Attic"),
            (60, "look", "There is a rope here.", "Attic"),
        )
        assert [h.exchange.turn for h in rec.search("rope")] == [60, 5]

    def test_matching_is_case_insensitive(self):
        assert record(*SAMPLE).search("MAILBOX")

    def test_a_query_of_only_noise_words_finds_nothing(self):
        """"what is the" would otherwise match most of a transcript and rank
        none of it."""
        assert record(*SAMPLE).search("what is the") == []
        assert terms("what is the") == []

    def test_the_limit_is_respected(self):
        rows = tuple((i, "look", "a rope", "Attic") for i in range(1, 30))
        assert len(record(*rows).search("rope", limit=4)) == 4

    def test_nothing_recorded_finds_nothing(self):
        assert Recall().search("anything") == []


class TestWhatComesBack:
    def test_a_hit_carries_the_turn_the_command_and_the_place(self):
        rendered = record(*SAMPLE).render("nails", record(*SAMPLE).search("nails"))
        assert "turn 41" in rendered
        assert "Attic" in rendered
        assert "> cut nails with knife" in rendered
        assert "not the tool for that" in rendered

    def test_a_long_reply_is_shown_by_its_most_relevant_line(self):
        """Six room descriptions is a prompt, not an answer."""
        long = "\n".join([
            "You are in a large cave.",
            "A rusty iron grating is set into the ceiling here.",
            "Passages lead off in several directions, and the air is damp.",
            "There is a pile of rubble in one corner of this unpleasant place.",
        ])
        rec = record((70, "look", long, "Grating Room"))
        rendered = rec.render("grating", rec.search("grating"))
        assert "rusty iron grating" in rendered
        assert "pile of rubble" not in rendered

    def test_a_short_reply_is_shown_whole(self):
        rec = record(*SAMPLE)
        assert "Taken." in rec.render("knife", rec.search("knife"))

    def test_finding_nothing_says_what_that_does_and_does_not_mean(self):
        """"Not in the record" is not "does not exist", and a player told the
        first will act on the second."""
        rec = record(*SAMPLE)
        rendered = rec.render("screwdriver", rec.search("screwdriver"))
        assert "Nothing in the 4 turns" in rendered
        assert "not evidence it does not exist" in rendered

    def test_nothing_is_summarised(self):
        """Every word shown came off the screen. A search that paraphrased
        would be a second player whispering."""
        rec = record(*SAMPLE)
        rendered = rec.render("mailbox", rec.search("mailbox"))
        for line in rendered.splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith(("You looked back", "[turn", ">")):
                continue
            assert stripped in SAMPLE[0][2]


class TestTheHandoffFloor:
    def test_by_default_the_whole_record_is_searchable(self):
        rec = record(*SAMPLE)
        assert len(rec.since(0)) == 4

    def test_a_cold_takeover_cannot_search_what_it_was_not_shown(self):
        """The window withholds the predecessor's play; the search has to
        withhold it too, or the cold arm is not cold."""
        rec = record(*SAMPLE)
        own = rec.since(3)
        assert len(own) == 1
        assert own.search("mailbox") == []
        assert own.search("Congratulations")

    def test_the_original_record_is_not_altered_by_taking_a_view(self):
        rec = record(*SAMPLE)
        rec.since(3)
        assert len(rec) == 4 and rec.search("mailbox")


class TestTheSessionKeepsIt:
    async def test_every_exchange_is_recorded_with_its_turn_and_room(self):
        from observatory.agents.simple import ScriptedAgent
        from observatory.engine.mock_engine import MockEngine
        from observatory.events import EventBus
        from observatory.session import Session, SessionConfig

        session = Session(
            MockEngine(), ScriptedAgent(["open mailbox", "north"]), EventBus(),
            config=SessionConfig(delay=0.0),
        )
        await session.start()
        await session.step_once()
        await session.step_once()

        # The opening room is recorded too: it was printed before any command.
        assert len(session.recall) == 3
        last = session.recall.exchanges[-1]
        assert last.turn == 2 and last.command == "north" and last.room

    async def test_a_player_that_cannot_search_is_given_no_record(self):
        """The control arm for having one. Handing it over anyway and trusting
        the agent not to look would make the arm meaningless."""
        from observatory.agents.simple import RandomAgent
        from observatory.engine.mock_engine import MockEngine
        from observatory.events import EventBus
        from observatory.session import Session, SessionConfig

        session = Session(MockEngine(), RandomAgent(), EventBus(),
                          config=SessionConfig(delay=0.0))
        await session.start()
        assert session._context().recall is None

    async def test_a_cold_handoff_hands_over_a_bounded_record(self):
        from observatory.agents.simple import ScriptedAgent
        from observatory.engine.mock_engine import MockEngine
        from observatory.events import EventBus
        from observatory.session import Session, SessionConfig

        taker = ScriptedAgent(["look"])
        taker.search = True   # type: ignore[attr-defined]

        session = Session(
            MockEngine(), ScriptedAgent(["open mailbox", "north", "east"]),
            EventBus(), config=SessionConfig(delay=0.0),
        )
        await session.start()
        for _ in range(3):
            await session.step_once()

        await session.handoff(taker, inherit_transcript=False)
        bounded = session._context().recall
        assert bounded is not None and len(bounded) == 0

        await session.handoff(taker, inherit_transcript=True)
        # The floor is not lowered again by a second handoff: what was withheld
        # stays withheld.
        assert len(session._context().recall or []) == 0
