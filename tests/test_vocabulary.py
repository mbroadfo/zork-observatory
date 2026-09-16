"""The parser vocabulary the agent has mapped.

Replaces a distinct-command count, which rewarded inventing words: `take
lazuli`, `take grue` and `take zorkmid` are three distinct commands and zero
knowledge.
"""

from __future__ import annotations

from observatory.agents.simple import MOCK_WALKTHROUGH, ScriptedAgent
from observatory.engine.mock_engine import MockEngine
from observatory.events import EventBus
from observatory.session import Session, SessionConfig
from observatory.world.outcomes import Outcome
from observatory.world.vocabulary import Vocabulary

UNKNOWN = Outcome.UNKNOWN_WORD


class TestTheParserNamesTheWord:
    def test_a_real_verb_with_an_invented_noun_is_not_blamed(self):
        """The regression that prompted the rewrite: on real Zork, `open`,
        `examine`, `take` and `drop` were all marked unknown because the agent
        had paired them with invented nouns. Zork says which word it failed on."""
        v = Vocabulary()
        v.observe("open zorkmid", UNKNOWN, 'I don\'t know the word "zorkmid".')

        assert v.unknown == {"zorkmid"}
        assert "open" in v.verbs_ok

    def test_an_invented_verb_is_blamed_when_named(self):
        v = Vocabulary()
        v.observe("quibbleflarn mailbox", UNKNOWN, 'I don\'t know the word "quibbleflarn".')

        assert v.unknown == {"quibbleflarn"}
        assert "quibbleflarn" not in v.verbs_ok
        assert "mailbox" not in v.unknown

    def test_curly_quotes_and_no_quotes_are_read_too(self):
        v = Vocabulary()
        v.observe("take zorkmid", UNKNOWN, "I don’t know the word “zorkmid”.")
        v.observe("take lazuli", UNKNOWN, "I don't know the word lazuli.")
        assert v.unknown == {"zorkmid", "lazuli"}

    def test_when_no_word_is_named_only_the_verb_is_suspected(self):
        v = Vocabulary()
        v.observe("frob mailbox", UNKNOWN, "That's not a verb I recognise.")
        assert v.unknown == {"frob"}

    def test_a_word_later_used_successfully_is_rehabilitated_as_a_verb(self):
        v = Vocabulary()
        v.observe("open zorkmid", UNKNOWN, 'I don\'t know the word "zorkmid".')
        v.observe("open mailbox", Outcome.PROGRESS, "Opening the mailbox reveals a leaflet.")
        assert v.verbs_ok == {"open"}
        assert v.nouns_ok == {"mailbox"}


class TestRealZorkPhrasings:
    """Verbatim from a live Zork I run. Each one was misread before."""

    def _classify(self, command, text):
        from observatory.engine.base import Observation, WorldState
        from observatory.world.outcomes import OutcomeTally

        s = WorldState(location_id=1, location_name="Forest")
        return OutcomeTally().classify(command, Observation(text=text), s, s).outcome

    def test_a_grammar_refusal_is_its_own_outcome(self):
        text = 'You used the word "remove" in a way that I don\'t understand.'
        assert self._classify("attack remove", text) is Outcome.GRAMMAR

    def test_an_unrecognised_sentence_is_grammar_not_vocabulary(self):
        assert self._classify("search behind", "That sentence isn't one I recognize.") is Outcome.GRAMMAR

    def test_zorks_absent_phrasing_is_recognised(self):
        assert self._classify("search mailbox", "You can't see any mailbox here!") is Outcome.ABSENT_NOUN

    def test_grammar_refusals_condemn_no_word(self):
        """This is how `attack`, `move`, `open` and `search` were all wrongly
        listed as words Zork doesn't know."""
        v = Vocabulary()
        v.observe("attack remove", Outcome.GRAMMAR,
                  'You used the word "remove" in a way that I don\'t understand.')
        v.observe("open north", Outcome.GRAMMAR,
                  'You used the word "north" in a way that I don\'t understand.')
        v.observe("search behind", Outcome.GRAMMAR, "That sentence isn't one I recognize.")

        assert v.unknown == set()
        assert {"attack", "open", "search"} <= v.verbs_ok

    def test_a_named_noun_clears_a_previously_suspected_verb(self):
        v = Vocabulary()
        v.observe("frob mailbox", UNKNOWN, "That's not a verb I recognise.")
        assert "frob" in v.unknown
        v.observe("move pearl", UNKNOWN, 'I don\'t know the word "pearl".')
        v.observe("move time", UNKNOWN, 'I don\'t know the word "time".')
        assert "move" in v.verbs_ok
        assert "move" not in v.unknown

    def test_absent_nouns_are_now_collected(self):
        v = Vocabulary()
        v.observe("search mailbox", Outcome.ABSENT_NOUN, "You can't see any mailbox here!")
        assert v.absent == {"mailbox"}


class TestTwoKindsOfRefusal:
    def test_absent_is_not_unknown(self):
        """"You can't see any such thing" means the word is real."""
        v = Vocabulary()
        v.observe("take lantern", Outcome.ABSENT_NOUN, "You can't see any such thing.")

        assert v.absent == {"lantern"}
        assert v.unknown == set()
        assert "take" in v.verbs_ok

    def test_finding_it_later_clears_it_from_absent(self):
        v = Vocabulary()
        v.observe("take lantern", Outcome.ABSENT_NOUN)
        v.observe("take lantern", Outcome.PROGRESS)
        assert v.nouns_ok == {"lantern"}
        assert v.absent == set()

    def test_a_known_unknown_is_never_listed_as_absent(self):
        v = Vocabulary()
        v.observe("take zorkmid", UNKNOWN, 'I don\'t know the word "zorkmid".')
        v.observe("drop zorkmid", Outcome.ABSENT_NOUN)
        assert v.absent == set()


class TestShape:
    def test_filler_words_are_not_nouns(self):
        v = Vocabulary()
        v.observe("put painting in the trophy case", Outcome.PROGRESS)
        assert not {"in", "the"} & v.nouns_ok
        assert {"painting", "trophy", "case"} <= v.nouns_ok

    def test_bare_commands_contribute_no_nouns(self):
        v = Vocabulary()
        v.observe("inventory", Outcome.META)
        v.observe("north", Outcome.BLOCKED)
        assert v.nouns_ok == set()
        assert {"inventory", "north"} <= v.verbs_ok


class TestAgainstARun:
    async def _run(self, commands):
        session = Session(
            MockEngine(), ScriptedAgent(commands), EventBus(),
            config=SessionConfig(delay=0.0, max_turns=len(commands) + 5),
        )
        await session.run()
        return session.vocabulary

    async def test_a_walkthrough_maps_real_vocabulary_and_invents_nothing(self):
        v = await self._run(MOCK_WALKTHROUGH)
        assert {"open", "take", "read", "put"} <= v.verbs_ok
        assert {"mailbox", "leaflet", "painting"} <= v.nouns_ok
        assert v.unknown == set()

    async def test_a_real_verb_survives_being_paired_with_nonsense(self):
        # The mock has no noun dictionary, so it answers "can't see" where real
        # Zork answers "don't know the word". Either way the verb must survive.
        v = await self._run(["open quibbleflarn", "open mailbox"])
        assert "open" in v.verbs_ok
        assert "quibbleflarn" not in v.nouns_ok
