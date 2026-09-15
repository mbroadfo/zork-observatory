"""Guards against the easiest way to ruin this instrument.

An agent prompt is allowed to describe the interface — parser grammar,
direction words, meta-commands. It is not allowed to describe the world. The
moment it names an object or warns about a hazard, the run stops being evidence
about the model and becomes evidence about the prompt, and a published score
built on it is wrong in a way nobody can see from the outside.

Every rung of the information ladder is held to this except `coached`, which
exists precisely to be the contaminated control arm. That exemption is checked
too — a control arm that isn't actually contaminated measures nothing.
"""

from __future__ import annotations

import pytest

from observatory.agents import prompts

# Nouns from real interactive fiction. Naming any of these hands the model a
# fact it should have had to discover.
WORLD_NOUNS = [
    "mailbox", "leaflet", "lantern", "lamp", "trophy", "sword", "elvish",
    "grue", "thief", "troll", "cyclops", "dam", "reservoir", "maze",
    "white house", "cellar", "attic", "kitchen", "painting", "egg",
    "jewel", "zork", "dungeon", "wand", "garlic",
]

# Warnings and tactics. Each one is a discovery the player should make.
STRATEGY_HINTS = [
    "lethal", "deadly", "dangerous", "you will die", "light source",
    "before going anywhere dark", "prefer exploring", "be decisive",
    "open everything", "watch out",
]

CLEAN = prompts.CLEAN_LEVELS


@pytest.mark.parametrize("level", CLEAN)
@pytest.mark.parametrize("noun", WORLD_NOUNS)
def test_no_level_names_an_object_from_a_real_game(level, noun):
    assert noun not in prompts.get(level).lower(), (
        f"The {level!r} prompt mentions {noun!r}. That is world knowledge, not "
        f"interface knowledge — the player has to find it out."
    )


@pytest.mark.parametrize("level", CLEAN)
@pytest.mark.parametrize("hint", STRATEGY_HINTS)
def test_no_level_gives_strategy_or_hazard_warnings(level, hint):
    assert hint not in prompts.get(level).lower(), (
        f"The {level!r} prompt contains {hint!r}. Hazards and tactics are the "
        f"game's to teach, not ours."
    )


class TestTheLadderIsActuallyALadder:
    """Each rung must differ from its neighbours in the intended direction."""

    def test_every_rung_is_distinct(self):
        texts = [prompts.get(level) for level in prompts.LEVEL_ORDER]
        assert len(set(texts)) == len(texts)

    def test_cold_does_not_reveal_that_it_is_a_game(self):
        """The whole point of the bottom rung: it doesn't even know that much."""
        cold = prompts.get("cold").lower()
        for giveaway in ["game", "play", "score", "adventure", "puzzle", "win"]:
            assert giveaway not in cold, f"'cold' leaks {giveaway!r}"

    def test_cold_does_not_reveal_that_a_parser_exists(self):
        cold = prompts.get("cold").lower()
        for giveaway in ["parser", "verb", "noun", "command", "north", "inventory"]:
            assert giveaway not in cold, f"'cold' leaks {giveaway!r}"

    def test_game_says_it_is_a_game_but_not_how_to_play_it(self):
        text = prompts.get("game").lower()
        assert "game" in text
        for giveaway in ["verb", "noun", "north", "inventory", "look"]:
            assert giveaway not in text, f"'game' leaks {giveaway!r}"

    def test_parser_explains_the_interface(self):
        """The complement of the rules above: operating the terminal is fair game.

        Without this a prompt could be stripped to nothing and every test above
        would still pass, which would be a different kind of broken.
        """
        text = prompts.get("parser").lower()
        for word in ["north", "inventory", "look", "verb"]:
            assert word in text

    def test_coached_really_is_contaminated(self):
        """A control arm that isn't contaminated isn't a control arm."""
        text = prompts.get("coached").lower()
        assert any(h in text for h in STRATEGY_HINTS), (
            "'coached' is supposed to carry the hints the other levels forbid; "
            "if it no longer does, the comparison it exists for is meaningless."
        )

    def test_coached_is_a_superset_of_parser(self):
        assert prompts.get("parser") in prompts.get("coached")

    def test_unknown_level_is_refused_clearly(self):
        with pytest.raises(ValueError, match="Unknown info level"):
            prompts.get("hardmode")


class TestPromptsTravelWithTheRun:
    """A score can't be audited by someone who wasn't there unless the prompt
    that produced it is in the trace."""

    def test_describe_reports_the_level_and_its_text(self):
        from observatory.agents.claude_agent import ClaudeAgent

        stub = type("Stub", (), {
            "name": "claude:test", "kind": "llm", "model": "claude-opus-5",
            "effort": "medium", "history_turns": 30, "max_tokens": 2000,
            "info_level": "cold", "system": prompts.get("cold"),
        })()
        described = ClaudeAgent.describe(stub)

        assert described["info_level"] == "cold"
        assert described["system_prompt"] == prompts.get("cold")
        assert described["system_fingerprint"] == prompts.fingerprint("cold")

    def test_fingerprints_differ_between_levels(self):
        seen = {prompts.fingerprint(level) for level in prompts.LEVEL_ORDER}
        assert len(seen) == len(prompts.LEVEL_ORDER)

    def test_fingerprint_tracks_the_text(self):
        import hashlib

        for level in prompts.LEVEL_ORDER:
            expected = hashlib.sha256(prompts.get(level).encode()).hexdigest()[:12]
            assert prompts.fingerprint(level) == expected
