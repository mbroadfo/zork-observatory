"""The journal: what stays true of a world when the world starts over.

Two memories, kept apart on purpose. The notebook holds the agent's own
sentences, which may be wrong. The journal holds what the game printed: which
places join which, what each looked like before anything was touched, and what
was seen to do something. The world resets every run; its shape does not.

What must not get in: the model's words, the engine's state, and the harness's
opinion about whether a move was a good idea.
"""

from __future__ import annotations

import json

from observatory.agents.episodic import EpisodicMemory
from observatory.agents.base import TurnContext
from observatory.journal import Journal

from test_episodic import CLEARING, INTRO, PATH, filed
from test_ollama import FakeOllama, agent, reply


def journal(tmp_path, fresh=False) -> Journal:
    return Journal.open(tmp_path, "88-840726", "ollama:qwen3:14b", fresh=fresh)


def absorbed(tmp_path, *exchanges, start=INTRO) -> Journal:
    book = journal(tmp_path)
    book.absorb(filed(*exchanges, start=start))
    return book


BEHIND = ("Behind House\nYou are behind the white house. In one corner of the house there is a "
          "small window which is slightly ajar.")


class TestTopology:
    def test_a_move_that_arrived_is_a_way_out(self, tmp_path):
        book = absorbed(tmp_path, ("north", PATH), ("north", CLEARING))
        assert book.places["West of House"]["exits"] == {"north": "Forest Path"}
        assert book.places["Forest Path"]["exits"] == {"north": "Clearing"}

    def test_a_refused_direction_is_a_wall_in_the_games_words(self, tmp_path):
        book = absorbed(tmp_path, ("east", "The door is boarded and you can't remove the boards."))
        walls = book.places["West of House"]["walls"]
        assert "boarded" in walls["east"]
        assert not book.places["West of House"]["exits"]

    def test_a_way_that_opens_later_stops_being_a_wall(self, tmp_path):
        """The world changes inside a run; the journal keeps the later truth."""
        book = journal(tmp_path)
        mem = filed(("west", "The door is nailed shut."))
        book.absorb(mem)
        assert "west" in book.places["West of House"]["walls"]
        mem.record("west", "Living Room\nYou are in the living room.")
        book.absorb(mem)
        assert book.places["West of House"]["exits"]["west"] == "Living Room"
        assert "west" not in book.places["West of House"]["walls"]


class TestWhatWorked:
    def test_a_command_the_game_answered_is_kept(self, tmp_path):
        book = absorbed(tmp_path, ("open mailbox", "Opening the small mailbox reveals a leaflet."))
        worked = book.places["West of House"]["worked"]
        assert "leaflet" in worked["open mailbox"]

    def test_the_parser_declining_is_not_a_fact_about_the_world(self, tmp_path):
        book = absorbed(tmp_path,
                        ("use mailbox", 'I don\'t know the word "use".'),
                        ("open frobozz", "You can't see any frobozz here!"),
                        ("mailbox open", "That sentence isn't one I recognize."))
        assert book.places["West of House"]["worked"] == {}

    def test_understood_and_did_nothing_is_not_kept(self, tmp_path):
        book = absorbed(tmp_path,
                        ("look at tree", "There's nothing special about the tree."),
                        ("take mailbox", "It is securely anchored."))
        worked = book.places["West of House"]["worked"]
        assert "look at tree" not in worked
        assert "take mailbox" in worked     # a real answer about the world

    def test_first_sight_is_kept_as_first_seen(self, tmp_path):
        """What was there before anything touched it."""
        book = journal(tmp_path)
        mem = filed(("north", CLEARING), ("take leaves", "Taken."))
        book.absorb(mem)
        assert "pile of leaves" in book.places["Clearing"]["description"]
        assert "leaves" in book.places["Clearing"]["words"]
        mem.record("look", "Clearing\nYou are in a clearing. There is a grating here.")
        book.absorb(mem)
        assert "pile of leaves" in book.places["Clearing"]["description"]


class TestWhereItWillNotGuess:
    """qwen3:14b walked up into the unlit attic, which prints no heading, and
    the record went on filing under Kitchen. Those turns must not become facts
    about the kitchen."""

    DARK = "You have moved into a dark place.\nIt is pitch black."

    def test_turns_in_an_unlit_place_are_not_facts_about_the_last_one(self, tmp_path):
        book = absorbed(
            tmp_path,
            ("north", CLEARING),
            ("up", self.DARK),
            ("look", "It is pitch black."),
            ("take rope", "Taken."),
            ("light lamp", "Attic\nThis is the attic."),
        )
        assert "take rope" not in book.places["Clearing"]["worked"]
        assert "up" not in book.places["Clearing"]["exits"]
        assert "light lamp" not in book.places["Clearing"]["worked"]

    def test_a_refused_move_leaves_you_where_you_were(self, tmp_path):
        """The common case, and the opposite reading: the next heading comes
        back one turn later and is the same place."""
        book = absorbed(tmp_path,
                        ("north", CLEARING),
                        ("north", "The forest becomes impenetrable to the north."),
                        ("take leaves", "Taken."),
                        ("look", CLEARING))
        assert "take leaves" in book.places["Clearing"]["worked"]
        assert "north" in book.places["Clearing"]["walls"]

    def test_a_move_answered_by_a_heading_is_never_in_doubt(self, tmp_path):
        book = absorbed(tmp_path,
                        ("east", "The door is boarded and you can't remove the boards."),
                        ("north", PATH))
        assert book.places["West of House"]["exits"] == {"north": "Forest Path"}


class TestAcrossRuns:
    def test_written_every_turn_and_read_back(self, tmp_path):
        book = absorbed(tmp_path, ("north", PATH))
        book.save()
        again = journal(tmp_path)
        assert again.places["West of House"]["exits"] == {"north": "Forest Path"}
        assert again.runs == 2
        assert json.loads(book.path.read_text())["agent"] == "ollama:qwen3:14b"

    def test_a_stopped_run_leaves_its_shape_behind(self, tmp_path):
        """No reflection, no ending — the journal is on disk either way."""
        book = journal(tmp_path)
        mem = EpisodicMemory()
        mem.before_move(TurnContext(turn=1, observation=INTRO, score=0, moves=0))
        for i, (cmd, reply_text) in enumerate([("north", PATH), ("north", CLEARING)], start=2):
            mem.after_move(cmd)
            mem.before_move(TurnContext(turn=i, observation=reply_text, score=0, moves=i))
            book.absorb(mem)
            book.save()
        assert journal(tmp_path).places["Forest Path"]["exits"] == {"north": "Clearing"}

    def test_a_new_journal_sets_the_old_one_aside(self, tmp_path):
        absorbed(tmp_path, ("north", PATH)).save()
        fresh = journal(tmp_path, fresh=True)
        assert fresh.places == {}
        assert len(list(fresh.path.parent.iterdir())) == 1      # the archived one

    def test_later_runs_add_to_what_earlier_ones_found(self, tmp_path):
        absorbed(tmp_path, ("north", PATH)).save()
        second = journal(tmp_path)
        second.absorb(filed(("south", "South of House\nYou are facing the south side.")))
        second.save()
        exits = journal(tmp_path).places["West of House"]["exits"]
        assert exits == {"north": "Forest Path", "south": "South of House"}


class TestWhatItReadsLike:
    def test_it_says_how_many_runs_are_behind_it(self, tmp_path):
        absorbed(tmp_path, ("north", PATH)).save()
        second = journal(tmp_path)
        second.absorb(filed(("north", PATH)))
        assert "kept across 1 earlier run(s)" in second.render()

    def test_places_ways_and_what_worked(self, tmp_path):
        book = absorbed(tmp_path,
                        ("open mailbox", "Opening the small mailbox reveals a leaflet."),
                        ("east", "The door is boarded and you can't remove the boards."),
                        ("north", PATH))
        text = book.render()
        assert "West of House" in text
        assert "ways out: north → Forest Path" in text
        assert "refused: east" in text
        assert 'did something: open mailbox → "Opening the small mailbox reveals a leaflet."' in text

    def test_an_empty_journal_renders_as_nothing(self, tmp_path):
        assert journal(tmp_path).render() == ""

    def test_it_stays_bounded(self, tmp_path):
        book = journal(tmp_path)
        pairs = []
        for i in range(40):
            here = f"Place {chr(65 + i % 26)}{chr(65 + i // 26)}\nA place with a thing in it."
            pairs.append((f"go{i}", here))
            pairs += [(f"poke {j}", f"The thing wobbles {j}.") for j in range(10)]
        book.absorb(filed(*pairs))
        assert "more places" in book.render()
        assert len(book.render()) < 20_000


class TestThePlayer:
    async def test_the_journal_is_in_the_prompt_and_named_in_the_agent(self, tmp_path):
        book = absorbed(tmp_path, ("open mailbox", "Opening the small mailbox reveals a leaflet."))
        book.save()
        fake = FakeOllama(["north"])
        a = agent(fake, recall="episodic", journal=journal(tmp_path))
        await a.act(TurnContext(turn=1, observation=INTRO, score=0, moves=0))
        prompt = fake.sent[0]["messages"][1]["content"]
        assert "Your journal, kept across" in prompt
        assert "open mailbox" in prompt
        assert a.name == "ollama:qwen3:8b+episodic+journal"   # the fixture's model
        assert a.describe()["journal"]["places"] == 1

    async def test_it_is_written_after_every_turn(self, tmp_path):
        book = journal(tmp_path)
        fake = FakeOllama(["north", "north"])
        a = agent(fake, recall="episodic", journal=book)
        await a.act(TurnContext(turn=1, observation=INTRO, score=0, moves=0))
        await a.act(TurnContext(turn=2, observation=PATH, score=0, moves=1))
        on_disk = json.loads(book.path.read_text())
        assert on_disk["places"]["West of House"]["exits"] == {"north": "Forest Path"}

    async def test_without_one_nothing_is_written_or_shown(self, tmp_path):
        fake = FakeOllama(["north"])
        a = agent(fake, recall="episodic")
        await a.act(TurnContext(turn=1, observation=INTRO, score=0, moves=0))
        assert "journal" not in fake.sent[0]["messages"][1]["content"].lower()
        assert a.describe()["journal"] is None
        assert not list(tmp_path.iterdir())


_ = reply
