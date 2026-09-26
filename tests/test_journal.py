"""The record the agent keeps on purpose.

What is pinned here is the difference between this and the lessons in
memory.py: it is written on the turn the thing happened rather than
reconstructed at the end from a keyhole, it is given back verbatim, it never
displaces the transcript, and every entry carries the engine's verdict on the
turn it was written on — so a model that announces an achievement the world did
not corroborate is caught by the record rather than by a reader.
"""

from __future__ import annotations

import json

import pytest

from observatory.agents import build_agent, llm, prompts
from observatory.agents.base import TurnContext
from observatory.agents.journal import MAX_ENTRY_CHARS, Entry, Journal
from observatory.agents.memory import AgentMemory
from observatory.agents.ollama_agent import OllamaAgent
from observatory.engine.mock_engine import MockEngine
from observatory.events import EventBus
from observatory.notebook import Notebook
from observatory.session import Session, SessionConfig

from test_ollama import FakeOllama, reply


class Journalling(FakeOllama):
    """Answers with a move and, when given one, a journal line.

    The default command moves, because in the mock world moving is one of the
    few things that changes anything — and only a turn that changed something
    keeps what was written about it.
    """

    def __init__(self, entries: list[str] | None = None, command: str = "north", **kw):
        super().__init__(**kw)
        self.entries = list(entries or [])
        self.command = command

    async def __call__(self, method, path, body):
        if path == "/api/chat" and "format" in body:
            self.sent.append(dict(body))
            payload = {"reasoning": "because", "command": self.command}
            if self.entries:
                payload["journal"] = self.entries.pop(0)
            return {"message": {"content": json.dumps(payload)}, "eval_count": 9}
        return await super().__call__(method, path, body)


def kept(journal: Journal, text: str, turn: int = 1, **kw) -> Entry | None:
    """Offer an entry on a turn that changed something."""
    return journal.add(text, turn=turn, outcome=kw.pop("outcome", "progress"), **kw)


class TestTheEntry:
    def test_an_entry_knows_whether_the_world_agreed(self):
        assert Entry("x", turn=1, outcome="progress").corroborated is True
        for verdict in ("inert", "futile", "blocked", "meta", ""):
            assert Entry("x", turn=1, outcome=verdict).corroborated is False

    def test_unknown_fields_in_an_old_file_are_ignored(self):
        assert Entry.from_dict({"text": "x", "turn": 2, "mood": "grim"}).text == "x"


class TestWriting:
    def test_blank_entries_are_not_kept(self):
        j = Journal()
        assert kept(j, "   ") is None and len(j) == 0
        assert j.attempted == 0      # nothing was offered, so nothing refused

    def test_long_entries_are_cut_rather_than_refused(self):
        j = Journal()
        entry = kept(j, "x" * 500)
        assert entry is not None and len(entry.text) == MAX_ENTRY_CHARS

    def test_the_same_line_twice_is_kept_once(self):
        """A model that writes one fact every turn would otherwise fill its own
        window with it — which is exactly what the notebook did across four
        runs of light-source notes. Long enough to name what it refers to, so
        the line itself is the claim; see SELF_DESCRIBING_CHARS."""
        j = Journal()
        kept(j, "The window at the back of the house opens.")
        assert kept(j, "the WINDOW at the BACK of the house opens.", turn=9) is None
        assert len(j) == 1

    def test_the_same_attempt_described_differently_is_kept_once(self):
        """`take grating` came back as four entries in ninety turns because
        each was worded differently. The act is what repeats, not the prose."""
        j = Journal()
        kept(j, "took the grating", room="Clearing", command="take grating")
        assert kept(j, "Attempted to take the grating in Clearing.", turn=9,
                    room="Clearing", command="take grating") is None
        assert len(j) == 1 and len(j.rejected) == 1


class TestTheFilter:
    def test_only_what_the_turn_bore_out_is_kept(self):
        j = Journal()
        assert j.add("Opened it.", turn=1, outcome="progress") is not None
        for verdict in ("inert", "futile", "blocked", "meta", "absent", "unknown"):
            assert j.add(f"Did the {verdict} thing.", turn=2, outcome=verdict) is None
        assert [e.text for e in j.entries] == ["Opened it."]
        assert len(j.rejected) == 6

    def test_a_refused_entry_is_never_shown_back(self):
        """It is counted and traced. It is not read back as though it were
        true, which is the whole point of refusing it."""
        j = Journal()
        j.add("took the grating in Clearing", turn=27, outcome="futile")
        assert j.render() == ""
        assert j.to_dict() == []

    def test_the_unfiltered_journal_is_still_available_for_the_other_arm(self):
        j = Journal(corroborated_only=False)
        assert j.add("took the grating", turn=27, outcome="futile") is not None
        assert len(j) == 1 and j.rejected == []
        assert j.summary()["filtered"] is False

    def test_the_rate_is_of_what_was_claimed_not_of_what_survived(self):
        """Filtering must not be allowed to flatter itself: every kept entry
        is corroborated by construction, so a rate over kept entries would
        read 100% however much fiction was written."""
        j = Journal()
        j.add("true", turn=1, outcome="progress")
        for i in range(9):
            j.add(f"false {i}", turn=2 + i, outcome="futile")
        s = j.summary()
        assert (s["count"], s["attempted"], s["rejected"]) == (1, 10, 9)
        assert s["truthful_pct"] == 10.0
        assert s["false_outcomes"] == {"futile": 9}


class TestWhatIsTrueButNotWorthKeeping:
    """Three reasons to refuse, and only one of them is about the model being
    wrong. Counting them together made a run that repeated itself look like a
    run that was inventing."""

    def test_going_somewhere_is_real_change_the_map_already_holds(self):
        j = Journal()
        assert j.add("Moved north from the forest path.", turn=5,
                     outcome="progress", movement=True) is None
        assert j.rejected[0].reason == "movement"
        assert j.summary()["movement"] == 1

    def test_movement_can_be_kept_for_the_arm_that_wants_it(self):
        j = Journal(keep_movement=True)
        assert j.add("Moved north.", turn=5, outcome="progress", movement=True) is not None

    def test_a_short_acknowledgement_is_not_a_repeat(self):
        """Zork answers every successful take with "Taken.", so matching on
        those six characters kept the first thing a run picked up and refused
        every other. The command beside it is what makes it a claim."""
        j = Journal()
        assert j.add("Taken.", turn=4, room="Up a Tree", command="get egg",
                     outcome="progress") is not None
        assert j.add("Taken.", turn=9, room="Living Room", command="get lamp",
                     outcome="progress") is not None
        assert len(j) == 2

    def test_a_substantive_line_repeated_is_still_a_repeat(self):
        j = Journal()
        j.add("Took the sack from the table.", turn=82, room="Kitchen",
              command="take sack", outcome="progress")
        assert j.add("Took the sack from the table.", turn=83, room="Kitchen",
                     command="w", outcome="progress") is None

    def test_a_turn_stamp_no_longer_makes_a_repeat_look_new(self):
        """qwen3:14b wrote "Turn 82: Took the sack from the table." and then,
        on the turn it walked west, the same sentence stamped "Turn 83:". Both
        turns changed something, so only the repeat check could catch it."""
        j = Journal()
        j.add("Turn 82: Took the sack from the table.", turn=82,
              room="Kitchen", command="take sack", outcome="progress")
        assert j.add("Turn 83: Took the sack from the table.", turn=83,
                     room="Kitchen", command="w", outcome="progress") is None
        assert j.rejected[0].reason == "redundant"

    def test_the_line_is_stored_exactly_as_written(self):
        """The stamp is stripped to compare, never to store. What a model
        chose to say is the datum."""
        j = Journal()
        entry = j.add("Turn 82: Took the sack.", turn=82, outcome="progress")
        assert entry is not None and entry.text == "Turn 82: Took the sack."

    def test_a_line_that_only_records_having_tried_is_refused(self):
        """A turn can change something while the thing the sentence is about
        fails. "Attempted to cut the nails with the elvish sword" was kept on
        exactly such a turn and outlived two runs of sawing at the door."""
        j = Journal()
        for hedge in ["Attempted to cut the nails with the elvish sword.",
                      "Attempting to open the grating.",
                      "Tried to tie the rope to the door.",
                      "I tried the window.",
                      "Turn 25: Attempted to unlock it."]:
            assert j.add(hedge, turn=25, outcome="progress") is None
        assert {e.reason for e in j.rejected} == {"hedged"}
        assert j.summary()["hedged"] == 5

    def test_a_line_that_records_a_result_survives(self):
        j = Journal()
        for real in ["Took the sack from the table.",
                     "Entered the kitchen through the window.",
                     "Opened the mailbox.",
                     "The trophy case is now open."]:
            assert j.add(real, turn=1, room=real, command=real, outcome="progress") is not None

    def test_hedged_lines_were_still_true(self):
        """It did attempt it. The line is useless, not false — counting it as
        fiction would misread the model."""
        j = Journal()
        j.add("Attempted to cut the nails.", turn=25, outcome="progress")
        assert j.summary()["truthful_pct"] == 100.0
        assert j.summary()["false"] == 0

    def test_true_but_dropped_still_counts_as_truthful(self):
        j = Journal()
        j.add("Opened it.", turn=1, room="Hall", command="open door", outcome="progress")
        j.add("Moved north.", turn=2, outcome="progress", movement=True)
        j.add("Opened it.", turn=3, room="Hall", command="open door", outcome="progress")
        j.add("Solved it.", turn=4, outcome="futile")

        s = j.summary()
        assert (s["count"], s["movement"], s["redundant"], s["false"]) == (1, 1, 1, 1)
        assert s["truthful"] == 3 and s["truthful_pct"] == 75.0


class TestReadingBack:
    def test_entries_come_back_verbatim_with_where_and_when(self):
        j = Journal()
        kept(j, "The window at the back of the house opens.", turn=14,
             room="Behind House", run=2)
        rendered = j.render()
        assert "The window at the back of the house opens." in rendered
        assert "run 2, turn 14 (Behind House)" in rendered

    def test_the_command_that_worked_is_shown_beside_the_note(self):
        """Without it the prose gets read as an instruction: a run carrying
        "Took the elvish sword from above the trophy case" typed exactly that,
        learned the parser has no word "above", and burned eight turns before
        trying `take elvish sword` — the command the entry was stored with."""
        j = Journal()
        kept(j, "Took the elvish sword from above the trophy case.", turn=88,
             room="Living Room", command="take elvish sword")
        assert '"take elvish sword" worked: Took the elvish sword' in j.render()

    def test_an_entry_with_no_command_still_renders(self):
        j = Journal()
        kept(j, "Something happened.", turn=3)
        assert "Something happened." in j.render()
        assert '"" worked' not in j.render()

    def test_the_rendering_is_plain_ascii(self):
        """It is worth copying verbatim, so it should not depend on the
        transport carrying anything cleverer than ASCII. Headings included."""
        j = Journal()
        j.adopt([Entry("Taken.", turn=4, room="Up a Tree", command="Get egg",
                       outcome="progress")])
        kept(j, "Took the sack from the table.", turn=82,
             room="Kitchen", command="take sack", run=2)
        j.render().encode("ascii")

    def test_an_empty_journal_renders_as_nothing_at_all(self):
        """Same rule as an empty memory: at the bottom of the ladder a heading
        about a journal it has not written in is a fact it has not earned."""
        assert Journal().render() == ""

    def test_the_oldest_entries_fall_off_and_say_they_did(self):
        j = Journal(shown=3)
        for i in range(6):
            kept(j, f"note {i}", turn=i)
        rendered = j.render()
        assert "note 0" not in rendered and "note 5" in rendered
        assert "3 earlier entries not shown" in rendered


class TestWhatWasGivenIsNotWhatWasEarned:
    """A seeded run rendered `"Get egg" worked: Taken.` with no provenance at
    all, because `render` only printed a run marker when `run` was truthy and a
    seed carries `run=0`. The model read it as a fact about now and spent forty
    turns trying to open an egg it had never picked up, hunting a staircase in
    a forest, and typing `go to attic` at a parser with no such word. Meanwhile
    the summary counted the twenty-two entries it had been handed as twenty-two
    of its own, reading 64% truthful on a run that had kept nothing.
    """

    def seeded(self) -> Journal:
        j = Journal()
        j.adopt([
            Entry("Taken.", turn=4, room="Up a Tree", command="Get egg",
                  outcome="progress"),
            Entry("The brass lantern is now on.", turn=16, room="Attic",
                  command="Light lamp", outcome="progress"),
        ])
        return j

    def test_carried_entries_say_they_are_from_earlier_runs(self):
        rendered = self.seeded().render()
        assert "From earlier runs" in rendered
        assert "none of this is true yet" in rendered
        assert "things you have" in rendered

    def test_carried_and_earned_are_rendered_apart(self):
        j = self.seeded()
        kept(j, "Opening the small mailbox reveals a leaflet.", turn=2,
             room="West of House", command="open mailbox")
        rendered = j.render()
        assert rendered.index("Get egg") < rendered.index("This run")
        assert rendered.index("This run") < rendered.index("open mailbox")

    def test_a_journal_with_nothing_carried_says_nothing_about_earlier_runs(self):
        j = Journal()
        kept(j, "Opening the small mailbox reveals a leaflet.", turn=2)
        assert "From earlier runs" not in j.render()

    def test_the_gift_is_left_out_of_this_run_s_figures(self):
        j = self.seeded()
        s = j.summary()
        assert (s["carried"], s["kept"], s["attempted"]) == (2, 0, 0)
        assert s["truthful_pct"] == 0.0        # it has not earned anything yet

        j.add("Solved it.", turn=9, outcome="futile")
        s = j.summary()
        assert (s["carried"], s["kept"], s["attempted"], s["false"]) == (2, 0, 1, 1)
        assert s["truthful_pct"] == 0.0

        kept(j, "Opening the small mailbox reveals a leaflet.", turn=10)
        s = j.summary()
        assert (s["count"], s["carried"], s["kept"], s["attempted"]) == (3, 2, 1, 2)
        assert s["truthful_pct"] == 50.0

    def test_the_notebook_hands_entries_over_as_carried(self, tmp_path):
        book = Notebook.open(tmp_path, "story", "ollama:qwen3:8b")
        book.write(Entry("Taken.", turn=4, room="Up a Tree", command="Get egg",
                         outcome="progress"))
        j = Journal()
        book.begin_run(AgentMemory(), j)
        assert j.carried == 1 and j.mine == [] and j.attempted == 0


class TestCorroboration:
    def test_the_summary_separates_what_happened_from_what_was_claimed(self):
        j = Journal()
        j.add("opened it", turn=1, outcome="progress")
        j.add("took it", turn=2, outcome="progress")
        j.add("solved it", turn=3, outcome="inert")
        j.add("found the way", turn=4, outcome="futile")

        assert j.summary() == {
            "count": 2,
            "carried": 0,
            "kept": 2,
            "attempted": 4,
            "rejected": 2,
            "truthful": 2,
            "truthful_pct": 50.0,
            "false": 2,
            "redundant": 0,
            "movement": 0,
            "hedged": 0,
            "outcomes": {"progress": 2},
            "false_outcomes": {"inert": 1, "futile": 1},
            "filtered": True,
        }

    def test_an_empty_journal_does_not_divide_by_zero(self):
        assert Journal().summary()["truthful_pct"] == 0.0


class TestThePrompt:
    def test_the_journal_is_beside_the_transcript_not_instead_of_it(self):
        """The one rule the September 2026 sweep settled: a summary that
        replaces the raw text is worse than no summary."""
        j = Journal()
        kept(j, "The window opens.", room="Behind House")
        ctx = TurnContext(
            turn=9, observation="Kitchen", score=0, moves=9,
            transcript=[("open window", "With great effort you open it."),
                        ("west", "Kitchen")],
        )
        prompt = llm.turn_prompt(AgentMemory(), ctx, history_turns=30, journal=j)

        assert "The window opens." in prompt
        assert "With great effort you open it." in prompt      # still there
        assert prompt.index("The window opens.") < prompt.index("> west")

    def test_without_a_journal_the_prompt_is_unchanged(self):
        ctx = TurnContext(turn=1, observation="West of House", score=0, moves=1,
                          transcript=[("", "West of House")])
        assert (llm.turn_prompt(AgentMemory(), ctx, 30)
                == llm.turn_prompt(AgentMemory(), ctx, 30, journal=None))

    def test_the_reflection_gets_to_read_what_the_run_wrote_down(self):
        """The reflection sees twelve exchanges. A run that wrote things down
        at turn 2 should not be asked what it learned with only turns 188-200
        in front of it."""
        j = Journal()
        kept(j, "The mailbox opens and there is something in it.", turn=2)
        ctx = TurnContext(turn=200, observation="Forest", score=0, moves=200,
                          transcript=[("look", "Forest")], next_start="beginning")
        assert "The mailbox opens" in llm.reflection_prompt(AgentMemory(), ctx, j)

    def test_the_schema_offers_the_line_only_when_the_journal_is_on(self):
        assert "journal" not in llm.move_fields(False)["properties"]
        assert "journal" in llm.move_fields(True)["properties"]
        assert "journal" not in llm.move_fields(True)["required"]
        assert llm.move_fields(False) is llm.MOVE_FIELDS      # untouched

    def test_the_mechanic_is_told_at_every_rung_and_the_tactic_only_to_coached(self):
        for level in prompts.LEVEL_ORDER:
            text = prompts.with_journal(level)
            assert "You keep a journal" in text
            assert ("changed the world" in text) == (level == "coached")

    def test_a_journal_run_does_not_fingerprint_as_a_plain_one(self):
        agent = OllamaAgent("qwen3:8b", transport=FakeOllama(), journal=Journal())
        plain = OllamaAgent("qwen3:8b", transport=FakeOllama())
        assert agent.describe()["system_fingerprint"] != plain.describe()["system_fingerprint"]
        assert agent.describe()["journal"] is True
        assert plain.describe()["journal"] is False


class TestInARun:
    async def test_an_entry_is_filed_with_the_engine_s_verdict_on_that_turn(self):
        fake = Journalling(["Walked north out of the field."], command="north")
        # keep_movement, because this is about the verdict riding along with
        # the entry, not about which entries are worth keeping.
        agent = OllamaAgent("qwen3:8b", transport=fake, journal=Journal(keep_movement=True))
        bus = EventBus()
        seen: list = []
        bus.subscribe(seen.append)
        session = Session(MockEngine(), agent, bus,
                          config=SessionConfig(delay=0.0, max_turns=2))
        await session.run()

        entry = agent.journal.entries[0]
        assert entry.text == "Walked north out of the field."
        assert entry.turn == 1
        # Where the command was given, not where it landed.
        assert entry.room == "West of House"
        assert entry.command == "north"
        assert entry.outcome == "progress"
        written = [e for e in seen if e.type == "journal.written"]
        assert len(written) == 1
        assert written[0].payload["summary"]["count"] == 1

    async def test_a_claim_the_world_did_not_bear_out_is_refused_and_said_so(self):
        fake = Journalling(["Opened the mailbox."], command="open mailbox")
        agent = OllamaAgent("qwen3:8b", transport=fake, journal=Journal())
        bus = EventBus()
        seen: list = []
        bus.subscribe(seen.append)
        session = Session(MockEngine(), agent, bus,
                          config=SessionConfig(delay=0.0, max_turns=2))
        await session.run()

        assert len(agent.journal) == 0
        assert len(agent.journal.rejected) == 1
        refused = [e for e in seen if e.type == "journal.rejected"]
        assert len(refused) == 1
        assert refused[0].payload["text"] == "Opened the mailbox."
        assert refused[0].payload["corroborated"] is False
        assert refused[0].payload["summary"]["attempted"] == 1
        assert not [e for e in seen if e.type == "journal.written"]

    async def test_movement_entries_are_refused_in_a_real_run(self, tmp_path):
        fake = Journalling(["Moved north out of the field."], command="north")
        agent = OllamaAgent("qwen3:8b", transport=fake, journal=Journal())
        session = Session(MockEngine(), agent, EventBus(),
                          config=SessionConfig(delay=0.0, max_turns=2))
        await session.run()

        assert len(agent.journal) == 0
        assert agent.journal.rejected[0].reason == "movement"
        # Still truthful: it did move. The map simply already knows.
        assert agent.journal.summary()["truthful_pct"] == 100.0

    async def test_the_notebook_keeps_only_what_survived(self, tmp_path):
        fake = Journalling(["Walked north.", "Opened the mailbox."])
        fake.command = "north"
        agent = OllamaAgent("qwen3:8b", transport=fake,
                            journal=Journal(keep_movement=True))
        book = Notebook.open(tmp_path, "story", agent.name)
        session = Session(MockEngine(), agent, EventBus(),
                          config=SessionConfig(delay=0.0, max_turns=2), notebook=book)
        await session.run()

        # Two offered, and the second repeats the first act, so one survives.
        assert agent.journal.attempted == 2
        assert [e.text for e in book.journal] == [e.text for e in agent.journal.entries]

    async def test_an_empty_journal_still_reports_as_present(self):
        """An empty Journal is falsy, so `if agent.journal` is False before the
        first entry — which reported a journal run as having none, and left the
        front end hiding the panel until something was written."""
        agent = OllamaAgent("qwen3:8b", transport=Journalling([]), journal=Journal())
        bus = EventBus()
        seen: list = []
        bus.subscribe(seen.append)
        session = Session(MockEngine(), agent, bus,
                          config=SessionConfig(delay=0.0, max_turns=1))
        await session.run()

        assert session.summary()["journal"] == []
        assert session.summary()["journal_summary"]["count"] == 0
        started = next(e for e in seen if e.type == "session.started")
        ended = next(e for e in seen if e.type == "session.ended")
        assert started.payload["journal"] == []
        assert ended.payload["journal"]["count"] == 0

    async def test_a_player_without_one_reports_none_rather_than_empty(self):
        agent = OllamaAgent("qwen3:8b", transport=Journalling([]))
        session = Session(MockEngine(), agent, EventBus(),
                          config=SessionConfig(delay=0.0, max_turns=1))
        await session.run()
        assert session.summary()["journal"] is None
        assert session.summary()["journal_summary"] is None

    async def test_nothing_is_written_when_the_agent_writes_nothing(self):
        fake = Journalling([])
        agent = OllamaAgent("qwen3:8b", transport=fake, journal=Journal())
        session = Session(MockEngine(), agent, EventBus(),
                          config=SessionConfig(delay=0.0, max_turns=3))
        await session.run()
        assert len(agent.journal) == 0

    async def test_a_player_without_a_journal_is_never_offered_the_field(self):
        fake = Journalling(["this should be ignored"])
        agent = OllamaAgent("qwen3:8b", transport=fake)
        session = Session(MockEngine(), agent, EventBus(),
                          config=SessionConfig(delay=0.0, max_turns=2))
        await session.run()

        assert agent.journal is None
        assert "journal" not in fake.sent[0]["format"]["properties"]

    async def test_a_reply_that_was_not_structured_writes_nothing(self):
        """A prose fallback has no journal line to recover, and inventing one
        would put words the model did not choose into a permanent record."""
        agent = OllamaAgent("qwen3:8b", transport=FakeOllama(), journal=Journal())
        session = Session(MockEngine(), agent, EventBus(),
                          config=SessionConfig(delay=0.0, max_turns=2))
        await session.run()
        assert len(agent.journal) == 0


class TestAcrossRuns:
    def test_the_notebook_carries_the_journal_and_keeps_it_apart_from_lessons(self, tmp_path):
        book = Notebook.open(tmp_path, "story", "ollama:qwen3:8b")
        book.write(Entry("The window opens.", turn=14, room="Behind House",
                         outcome="progress", run=1))
        assert book.journal[0].corroborated is True

        again = Notebook.open(tmp_path, "story", "ollama:qwen3:8b")
        assert [e.text for e in again.journal] == ["The window opens."]
        assert again.lessons == []
        assert again.summary()["journal"] == 1
        assert again.summary()["journal_corroborated"] == 1

    def test_a_new_run_starts_with_everything_written_so_far(self, tmp_path):
        book = Notebook.open(tmp_path, "story", "ollama:qwen3:8b")
        book.write(Entry("The window opens.", turn=14))
        journal = Journal()
        assert book.begin_run(AgentMemory(), journal) == 1
        assert [e.text for e in journal.entries] == ["The window opens."]

        # A copy: the run writes through the notebook, not behind its back.
        journal.add("stray", turn=1)
        assert len(book.journal) == 1

    def test_a_run_without_a_journal_still_opens(self, tmp_path):
        book = Notebook.open(tmp_path, "story", "ollama:qwen3:8b")
        assert book.begin_run(AgentMemory()) == 1

    def test_a_notebook_written_before_journals_existed_still_loads(self, tmp_path):
        book = Notebook.open(tmp_path, "story", "ollama:qwen3:8b")
        book.path.parent.mkdir(parents=True, exist_ok=True)
        book.path.write_text(json.dumps({"lessons": [], "runs": []}))
        assert Notebook.open(tmp_path, "story", "ollama:qwen3:8b").journal == []


class TestTheSwitch:
    @pytest.mark.parametrize("kind", ["ollama", "claude"])
    def test_off_unless_asked_for(self, kind, monkeypatch):
        if kind == "claude":
            pytest.importorskip("anthropic")
        made = build_agent(kind, model="qwen3:8b")
        assert made.journal is None
        assert build_agent(kind, model="qwen3:8b", journal=True).journal is not None

    def test_the_baselines_never_have_one(self):
        assert build_agent("random").journal is None
        assert build_agent("scripted").journal is None


_ = reply
