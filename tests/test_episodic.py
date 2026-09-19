"""The episodic player's record: what it files, what it shows, what it refuses to say.

The record exists to separate forgetting from perseveration, and it only does
that if three things hold: every exchange lands under the heading the player
was looking at, a rollback moves the player without inventing an exchange, and
nothing in the record is the harness's opinion.
"""

from __future__ import annotations

import pytest

from observatory.agents import build_agent, llm
from observatory.agents.base import TurnContext
from observatory.agents.episodic import EpisodicMemory, heading_of
from observatory.engine.mock_engine import MockEngine
from observatory.events import EventBus
from observatory.session import Session, SessionConfig

from test_ollama import FakeOllama, agent, reply

INTRO = (
    "ZORK I: The Great Underground Empire\n"
    "Copyright (c) 1981, 1982, 1983 Infocom, Inc. All rights reserved.\n"
    "ZORK is a registered trademark of Infocom, Inc.\n"
    "Revision 88 / Serial number 840726\n\n"
    "West of House\n"
    "You are standing in an open field west of a white house, with a boarded front door.\n"
    "There is a small mailbox here."
)


class TestHeadings:
    @pytest.mark.parametrize("text, expected", [
        (INTRO, "West of House"),
        ("Clearing\nYou are in a clearing.", "Clearing"),
        ("(up the cliff)\nRocky Ledge\nYou are on a ledge.", "Rocky Ledge"),
        ("Up a Tree\nYou are about 10 feet above the ground.", "Up a Tree"),
        ("Forest", "Forest"),
    ])
    def test_a_heading_is_found(self, text, expected):
        assert heading_of(text) == expected

    @pytest.mark.parametrize("text", [
        "Taken.",
        "The forest becomes impenetrable to the north.",
        "You are carrying:\n  A pile of leaves\n  A leaflet",
        "It is pitch black. You are likely to be eaten by a grue.",
        "What do you want to climb down?",
        "****  You have died  ****",
        "",
    ])
    def test_prose_is_not_a_heading(self, text):
        assert heading_of(text) == ""


def filed(*exchanges, start=INTRO) -> EpisodicMemory:
    mem = EpisodicMemory()
    mem.before_move(TurnContext(turn=1, observation=start, score=0, moves=0))
    for i, (command, response) in enumerate(exchanges, start=1):
        mem.after_move(command)
        mem.before_move(TurnContext(turn=i + 1, observation=response, score=0, moves=i))
    return mem


CLEARING = (
    "Clearing\nYou are in a clearing, with a forest surrounding you on all sides. "
    "A path leads south.\nOn the ground is a pile of leaves."
)
PATH = (
    "Forest Path\nThis is a path winding through a dimly lit forest. The path heads "
    "north-south here. One particularly large tree with some low branches stands at "
    "the edge of the path."
)
WALL = "The forest becomes impenetrable to the north."


class TestFiling:
    def test_each_exchange_lands_under_the_heading_it_was_typed_at(self):
        mem = filed(("north", PATH), ("north", CLEARING), ("north", WALL), ("south", PATH))
        assert set(mem.headings) == {"West of House", "Forest Path", "Clearing"}
        assert mem.headings["West of House"].entries["north"].led_to == "Forest Path"
        assert mem.headings["Clearing"].entries["north"].reply == WALL
        assert mem.here == "Forest Path"

    def test_repeats_are_counted_not_repeated(self):
        mem = filed(("north", PATH), ("north", CLEARING), ("n", WALL), ("n", WALL), ("n", WALL))
        entry = mem.headings["Clearing"].entries["n"]
        assert entry.count == 3
        assert not entry.varied
        assert mem.render().count(WALL) == 1
        assert "> n ×3" in mem.render()

    def test_abbreviations_are_not_merged(self):
        """That `n` and `north` are one command is a discovery, not a given."""
        mem = filed(("north", PATH), ("north", CLEARING), ("n", WALL), ("north", WALL))
        assert set(mem.headings["Clearing"].entries) == {"n", "north"}

    def test_a_reply_that_changes_is_marked(self):
        mem = filed(("open box", "It is locked."), ("open box", "Opened."))
        entry = mem.headings["West of House"].entries["open box"]
        assert entry.varied and entry.reply == "Opened."
        assert '"Opened." ×1 (latest)' in mem.render()
        assert '"It is locked." ×1' in mem.render()

    def test_every_refusal_is_on_the_page_with_its_count(self):
        """qwen3:8b, live: nineteen `take grating`s against a record that
        showed one rotating quip and "replies varied". The whole tally is the
        evidence, so the whole tally is shown."""
        quips = ["What a concept!", "You can't be serious.", "An interesting idea...", "What a concept!",
                 "A valiant attempt.", "What a concept!", "Not likely.", "You can't be serious."]
        mem = filed(("north", CLEARING), *[("take grating", q) for q in quips])
        line = next(ln for ln in mem.render().splitlines() if "> take grating" in ln)
        assert "×8" in line
        assert '"What a concept!" ×3' in line
        assert '"You can\'t be serious." ×2 (latest)' in line
        assert "+1 other replies" in line   # four shown here, five kinds seen
        assert line.index("What a concept") < line.index("serious")

    def test_a_reprinted_heading_does_not_read_as_arriving(self):
        """qwen3:8b, live: nine `look`s in a row at West of House. The record
        rendered each as "→ heading West of House", which reads exactly like a
        move that got somewhere."""
        mem = filed(("look", INTRO), ("look", INTRO))
        line = next(ln for ln in mem.render().splitlines() if "> look" in ln)
        assert "heading West of House again (the one you were already under)" in line
        # A move that does arrive somewhere still reads plainly.
        moved = filed(("north", PATH))
        assert "> north → heading Forest Path" in moved.render()
        assert "again" not in moved.render()

    def test_arriving_and_re_describing_are_counted_apart(self):
        mem = filed(("north", CLEARING), ("s", PATH), ("n", CLEARING), ("look", CLEARING))
        entries = mem.headings["Clearing"].entries
        assert list(entries["look"].outcomes)[0].endswith("(the one you were already under)")
        assert list(entries["s"].outcomes) == ["heading Forest Path"]

    def test_the_current_heading_says_how_long_ago(self):
        mem = filed(("north", CLEARING), ("take grating", "No."), ("look", CLEARING), ("wait", "Time passes."))
        text = mem.render()
        assert "> wait (last just now)" in text
        assert "> take grating (last 2 commands ago)" in text

    def test_the_latest_inventory_is_pinned_once_the_player_has_asked(self):
        carrying = "You are carrying:\n  A pile of leaves"
        assert "inventory command" not in filed(("north", CLEARING), ("take leaves", "Taken.")).render()
        mem = filed(("north", CLEARING), ("inventory", carrying), ("take grating", "No."), ("s", PATH))
        text = mem.render()
        assert 'inventory command (2 commands ago): "You are carrying: A pile of leaves"' in text
        assert text.index("inventory command") < text.index("current heading")
        mem = filed(("i", "You are empty-handed."), ("take x", "Taken."), ("i", carrying))
        assert "(just now)" in mem.render() and "empty-handed" not in mem.render().split("\n\n")[1]

    def test_case_and_spacing_do_not_split_an_entry(self):
        mem = filed(("Open  Mailbox", "Opened."), ("open mailbox", "It is already open."))
        assert mem.headings["West of House"].entries["open mailbox"].count == 2

    def test_long_replies_are_shortened(self):
        mem = filed(("read leaflet", "word " * 100))
        assert len(mem.headings["West of House"].entries["read leaflet"].reply) <= 90

    def test_nothing_is_rendered_before_anything_is_typed(self):
        assert filed().render() == ""


class TestTheAgenda:
    """What is left here, which a record of what you did does not say.

    qwen3:14b spent 15 of 37 turns on `look` and `inventory`, saying almost
    every turn that it should try a new direction and then not naming one.
    Both lines here are subtraction from what the agent itself has seen.
    """

    def listed(self, mem, heading: str) -> list[str]:
        line = next(ln for ln in mem.render().splitlines() if heading in ln)
        return [w.strip() for w in line.split(":", 1)[1].split(",")]

    def test_directions_not_yet_typed_here(self):
        mem = filed(("north", CLEARING), ("s", PATH), ("n", CLEARING), ("e", "It is dark."))
        left = self.listed(mem, "Directions not yet")
        assert "south" not in left and "east" not in left    # both typed here
        assert "north" in left and "up" in left
        # Compass order, not whatever order the set iterates in.
        assert left.index("north") < left.index("up") < left.index("out")

    def test_an_abbreviation_counts_as_the_direction(self):
        mem = filed(("north", CLEARING), ("n", "The forest becomes impenetrable to the north."))
        assert "north" not in self.listed(mem, "Directions not yet")

    def test_words_the_place_named_that_no_command_used(self):
        mem = filed(("north", CLEARING), ("take leaves", "Taken."))
        left = self.listed(mem, "Words this place")
        assert "leaves" not in left          # already spoken here
        assert "path" in left and "forest" in left and "pile" in left

    def test_a_refusal_names_nothing(self):
        """Only text carrying the heading describes the place."""
        mem = filed(("north", CLEARING), ("xyzzy", "A hollow voice says fool."))
        line = next(ln for ln in mem.render().splitlines() if "Words this place" in ln)
        assert "hollow" not in line and "voice" not in line

    def test_directions_are_not_listed_as_things(self):
        """"A path leads south" names a direction, not something to act on."""
        assert "south" not in self.listed(filed(("north", CLEARING)), "Words this place")

    def test_the_agenda_is_for_where_you_stand(self):
        mem = filed(("north", CLEARING), ("s", PATH))
        assert mem.render().count("Directions not yet typed here") == 1
        assert "tree" in self.listed(mem, "Words this place")


class TestWhatTheRecordShows:
    def render(self):
        return filed(
            ("north", PATH), ("north", CLEARING), ("north", WALL), ("north", WALL), ("south", PATH),
        ).render()

    def test_the_current_heading_comes_first(self):
        text = self.render()
        assert text.index("current heading, Forest Path") < text.index("Under other headings")

    def test_other_headings_are_listed_with_what_happened_there(self):
        text = self.render()
        assert "Clearing" in text and "> north ×2" in text and "impenetrable" in text

    @pytest.mark.parametrize("word", [
        "blocked", "dead end", "don't", "do not", "avoid", "stop", "useless", "fail", "exit", "room",
    ])
    def test_it_records_and_never_advises(self, word):
        """The reply and a count. Whether four refusals are enough is the
        player's call; a record that says so is coaching."""
        header = self.render().split("\n")[0].lower()
        body = "\n".join(self.render().split("\n")[1:]).lower()
        assert word not in header
        assert word not in body.replace(WALL.lower(), "")

    def test_it_is_bounded(self):
        exchanges = []
        for i in range(60):
            exchanges.append((f"north{i}", f"Place {chr(65 + i % 26)}{chr(65 + i // 26)}\nSomewhere."))
            exchanges += [(f"poke {j}", "Nothing happens.") for j in range(15)]
        text = filed(*exchanges).render()
        assert "headings not shown" in text
        assert len(text) < 12_000


class TestTheWorldMovingUnderThePlayer:
    def test_a_rollback_moves_the_player_and_files_nothing(self):
        mem = filed(("north", PATH), ("north", CLEARING))
        mem.after_move("up")
        # The world was put back while "up" was pending; this is not its reply.
        mem.before_move(TurnContext(turn=2, observation="Taken.", score=0, moves=1, life=2,
                                    transcript=[("", INTRO), ("north", PATH), ("take", "Taken.")]))
        assert "up" not in mem.headings["Clearing"].entries
        assert mem.here == "Forest Path"

    def test_an_operator_rewind_is_noticed_by_the_turn_number(self):
        mem = filed(("north", PATH), ("north", CLEARING))
        mem.after_move("east")
        mem.before_move(TurnContext(turn=1, observation=INTRO, score=0, moves=0))
        assert "east" not in mem.headings["Clearing"].entries
        assert mem.here == "West of House"

    def test_the_move_that_ended_things_is_filed_before_the_reflection(self):
        mem = filed(("north", PATH))
        mem.after_move("jump")
        mem.before_reflection(TurnContext(turn=2, observation="You died.", score=0, moves=1))
        assert mem.headings["Forest Path"].entries["jump"].reply == "You died."
        # ...and not filed a second time when the next turn begins.
        mem.before_move(TurnContext(turn=3, observation="You died.", score=0, moves=1))
        assert mem.headings["Forest Path"].entries["jump"].count == 1


class TestTheEpisodicPlayer:
    async def test_the_record_is_in_the_prompt_and_the_trace(self):
        fake = FakeOllama(["north", "xyzzy", "xyzzy", "south", "look"])
        a = agent(fake, recall="episodic", history_turns=1)
        bus = EventBus()
        seen: list = []
        bus.subscribe(seen.append)
        session = Session(MockEngine(), a, bus, config=SessionConfig(delay=0.0, max_turns=5, history_turns=1))
        await session.run()

        last = fake.sent[-1]["messages"][1]["content"]
        assert "current heading, West of House" in last
        assert "> xyzzy ×2" in last
        assert "Fool" in last
        # One raw exchange, not the whole run.
        assert "> north\n" not in last
        thoughts = [e for e in seen if e.type == "agent.thought"]
        assert thoughts[-1].payload["meta"]["record"] in last
        assert thoughts[-1].payload["meta"]["record_stats"]["repeats"] == 1

    async def test_it_is_a_different_player(self):
        a = agent(FakeOllama(), recall="episodic", info_level="cold")
        assert a.name == "ollama:qwen3:8b/cold+episodic"
        assert a.describe()["recall"] == "episodic"
        assert agent(FakeOllama()).name == "ollama:qwen3:8b"

    async def test_the_transcript_player_is_unchanged(self):
        fake = FakeOllama(["south"])
        a = agent(fake)
        c = TurnContext(turn=3, observation="A room.", score=0, moves=2,
                        transcript=[("", "Welcome."), ("north", "A room.")])
        await a.act(c)
        assert fake.sent[0]["messages"][1]["content"] == llm.turn_prompt(a.memory, c, 30)
        assert a.episodes is None

    async def test_a_death_is_on_the_record_after_the_rollback(self):
        answers = ["north", "east", "west", "up", "look", "look", "look", "look"]
        fake = FakeOllama(answers)
        fake.answers.insert(5, {"message": {"content": "The dark killed me."}})
        a = agent(fake, recall="episodic")
        session = Session(MockEngine(), a, EventBus(),
                          config=SessionConfig(delay=0.0, max_turns=10, lives=1, checkpoint_every=2, history_turns=1))
        await session.run()
        assert session.deaths >= 1
        after = [b for b in fake.sent if "format" in b][-1]["messages"][1]["content"]
        assert "slavering fangs" in after

    def test_the_factory_picks_the_window_for_the_recall(self):
        assert build_agent("ollama", model="qwen3:8b").history_turns == 30
        episodic = build_agent("ollama", model="qwen3:8b", recall="episodic")
        assert (episodic.recall, episodic.history_turns) == ("episodic", 1)
        assert build_agent("ollama", model="qwen3:8b", recall="episodic", history_turns=5).history_turns == 5

    def test_an_unknown_recall_is_refused(self):
        with pytest.raises(ValueError, match="recall"):
            build_agent("ollama", model="qwen3:8b", recall="photographic")


class TestTheNudge:
    """One re-ask when the model picks a command its own record shows doing
    nothing here. qwen3:14b typed `use leaflet on grating` nine times against a
    record that said `I don't know the word "use"` every time; coaching it not
    to had no effect, so the scaffold does it mechanically."""

    def played(self, *exchanges):
        mem = filed(*exchanges)
        return mem

    def test_it_fires_on_a_command_that_has_only_ever_done_nothing(self):
        mem = self.played(("north", CLEARING), ("use grating", 'I don\'t know the word "use".'))
        entry = mem.repeat_of("use  GRATING")   # case and spacing do not matter
        assert entry is not None and entry.count == 1
        assert mem.repeat_of("open grating") is None       # never tried here

    def test_it_does_not_fire_on_a_command_that_got_somewhere(self):
        mem = self.played(("north", CLEARING), ("s", PATH), ("n", CLEARING))
        assert mem.repeat_of("s") is None

    def test_it_does_not_fire_when_the_replies_have_differed(self):
        """A changed reply means the world may have changed; that is the
        player's to judge, not the harness's."""
        mem = self.played(("open box", "It is locked."), ("open box", "Opened."))
        assert mem.repeat_of("open box") is None

    def test_a_re_described_heading_counts_as_doing_nothing(self):
        mem = self.played(("look", INTRO))
        assert mem.repeat_of("look") is not None

    async def test_the_model_is_asked_again_with_its_own_record_quoted(self):
        fake = FakeOllama(["use grating", "use grating", "open grating"])
        a = agent(fake, recall="episodic", nudge=True)
        c1 = TurnContext(turn=1, observation=CLEARING, score=0, moves=0)
        await a.act(c1)   # first `use grating` — nothing to repeat yet
        c2 = TurnContext(turn=2, observation='I don\'t know the word "use".', score=0, moves=1)
        action = await a.act(c2)

        assert action.command == "open grating"
        assert action.meta["nudged_from"] == "use grating"
        assert "use grating" in action.meta["nudged"] and "not tried" in action.meta["nudged"]
        assert 'I don\'t know the word "use".' in action.meta["nudged"]
        assert "nudge_ignored" not in action.meta
        # Both calls are paid for.
        assert action.meta["output_tokens"] == 50
        assert len(fake.sent) == 3

    async def test_a_model_that_repeats_anyway_has_its_turn_stand_and_flagged(self):
        fake = FakeOllama(["use grating", "use grating", "use grating"])
        a = agent(fake, recall="episodic", nudge=True)
        await a.act(TurnContext(turn=1, observation=CLEARING, score=0, moves=0))
        action = await a.act(TurnContext(turn=2, observation='I don\'t know the word "use".',
                                         score=0, moves=1))
        assert action.command == "use grating"
        assert action.meta["nudge_ignored"] is True

    async def test_without_the_scaffold_nothing_is_re_asked(self):
        fake = FakeOllama(["use grating", "use grating"])
        a = agent(fake, recall="episodic")
        await a.act(TurnContext(turn=1, observation=CLEARING, score=0, moves=0))
        action = await a.act(TurnContext(turn=2, observation='I don\'t know the word "use".',
                                         score=0, moves=1))
        assert action.command == "use grating"
        assert "nudged" not in action.meta
        assert len(fake.sent) == 2

    async def test_it_works_without_showing_the_record(self):
        """A nudged transcript player keeps the record for this and is not
        shown it, so the two scaffolds can be varied independently."""
        fake = FakeOllama(["look", "look", "north"])
        a = agent(fake, nudge=True, history_turns=0)
        await a.act(TurnContext(turn=1, observation=INTRO, score=0, moves=0))
        action = await a.act(TurnContext(turn=2, observation=INTRO, score=0, moves=1))
        assert "Your record so far" not in fake.sent[0]["messages"][1]["content"]
        assert action.command == "north"
        assert action.meta["nudged_from"] == "look"

    def test_a_nudged_player_is_named_as_one(self):
        a = build_agent("ollama", model="qwen3:14b", recall="episodic", nudge=True)
        assert a.name == "ollama:qwen3:14b+episodic+nudge"
        assert a.describe()["nudge"] is True
        assert build_agent("ollama", model="qwen3:14b").describe()["nudge"] is False


def ranked(command, *alternatives, reasoning="trying something"):
    import json as _json
    return {"message": {"content": _json.dumps(
        {"reasoning": reasoning, "command": command, "alternatives": list(alternatives)})},
        "prompt_eval_count": 300, "eval_count": 25}


class TestRankedCandidates:
    """The mechanical form of "punish a repeat": the model ranks three, and the
    harness plays the first its record does not already know to be inert. One
    call, and the skip is an act rather than a request."""

    async def play(self, *answers, **kw):
        fake = FakeOllama(list(answers))
        a = agent(fake, recall="episodic", candidates=True, **kw)
        await a.act(TurnContext(turn=1, observation=CLEARING, score=0, moves=0))
        action = await a.act(TurnContext(turn=2, observation="It is closed.", score=0, moves=1))
        return action, fake

    async def test_the_first_choice_stands_when_it_is_new(self):
        action, fake = await self.play(ranked("open grating", "take grating"),
                                       ranked("unlock grating", "take grating"))
        assert action.command == "unlock grating"
        assert "skipped" not in action.meta
        assert fake.sent[0]["format"]["properties"]["alternatives"]["type"] == "array"

    async def test_a_known_inert_first_choice_is_skipped_for_the_next(self):
        action, _ = await self.play(ranked("open grating", "take grating"),
                                    ranked("open grating", "south", "take grating"))
        assert action.command == "south"
        assert action.meta["skipped"] == ["open grating"]
        assert action.meta["offered"] == ["open grating", "south", "take grating"]

    async def test_when_every_candidate_is_known_inert_the_model_still_chooses(self):
        """The harness has no better idea than the model does."""
        fake = FakeOllama([ranked("open grating", "x"), ranked("look", "x"),
                           ranked("open grating", "look")])
        a = agent(fake, recall="episodic", candidates=True)
        await a.act(TurnContext(turn=1, observation=CLEARING, score=0, moves=0))
        await a.act(TurnContext(turn=2, observation="It is closed.", score=0, moves=1))
        action = await a.act(TurnContext(turn=3, observation=CLEARING, score=0, moves=2))
        assert action.command == "open grating"
        assert action.meta["all_candidates_inert"] is True

    async def test_one_call_per_turn(self):
        _, fake = await self.play(ranked("open grating", "take grating"),
                                  ranked("open grating", "south"))
        assert len(fake.sent) == 2

    async def test_a_prose_reply_still_yields_a_command(self):
        fake = FakeOllama([{"message": {"content": "north\n\nheading off"}, "eval_count": 9}])
        a = agent(fake, recall="episodic", candidates=True)
        action = await a.act(TurnContext(turn=1, observation=CLEARING, score=0, moves=0))
        assert action.command == "north"
        assert action.meta["unstructured"] is True

    def test_a_ranked_player_is_named_as_one(self):
        a = build_agent("ollama", model="qwen3:14b", recall="episodic", candidates=True)
        assert a.name == "ollama:qwen3:14b+episodic+candidates"
        assert a.describe()["candidates"] is True


_ = reply  # re-exported helper, kept importable for other test modules
