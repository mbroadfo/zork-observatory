"""Guards against the easiest way to ruin this instrument.

The agent's system prompt is allowed to describe the interface — parser
grammar, direction words, meta-commands. It is not allowed to describe the
world. The moment it names an object or warns about a hazard, the run stops
being evidence about the model and becomes evidence about the prompt, and a
published score built on it is wrong in a way nobody can see from the outside.

These tests fail loudly so that a well-meaning "let's help it along" edit can't
land quietly.
"""

from __future__ import annotations

import pytest

from observatory.agents.claude_agent import SYSTEM, SYSTEM_FINGERPRINT

# Nouns from real interactive fiction. Naming any of these hands the model a
# fact it should have had to discover.
WORLD_NOUNS = [
    "mailbox", "leaflet", "lantern", "lamp", "trophy", "sword", "elvish",
    "grue", "thief", "troll", "cyclops", "dam", "reservoir", "maze",
    "white house", "cellar", "attic", "kitchen", "painting", "egg",
    "jewel", "treasure", "zork", "dungeon", "wand", "garlic",
]

# Warnings and tactics. Each one is a discovery the player should make.
STRATEGY_HINTS = [
    "lethal", "deadly", "dangerous", "you will die", "light source",
    "before descending", "prefer exploring", "be decisive", "avoid",
    "make sure you", "remember to", "watch out",
]


@pytest.mark.parametrize("noun", WORLD_NOUNS)
def test_prompt_names_no_object_from_any_real_game(noun):
    assert noun not in SYSTEM.lower(), (
        f"The system prompt mentions {noun!r}. That is world knowledge, not "
        f"interface knowledge — the player has to find it out."
    )


@pytest.mark.parametrize("hint", STRATEGY_HINTS)
def test_prompt_gives_no_strategy_or_hazard_warning(hint):
    assert hint not in SYSTEM.lower(), (
        f"The system prompt contains {hint!r}. Hazards and tactics are the "
        f"game's to teach, not ours."
    )


def test_prompt_does_describe_the_interface():
    """The complement of the rule above: operating the terminal is fair game.

    Without this the prompt could be stripped to nothing and the tests above
    would still pass, which would be a different kind of broken.
    """
    lowered = SYSTEM.lower()
    for interface_word in ["north", "inventory", "look", "verb"]:
        assert interface_word in lowered


def test_prompt_is_recorded_with_every_run():
    """A trace has to carry the prompt that produced it, or its score can't be
    audited by anyone who wasn't there."""
    from observatory.agents.claude_agent import ClaudeAgent

    described = ClaudeAgent.describe(
        type("Stub", (), {
            "name": "claude:test", "kind": "llm", "model": "claude-opus-5",
            "effort": "medium", "history_turns": 30, "max_tokens": 2000,
        })()
    )
    assert described["system_prompt"] == SYSTEM
    assert described["system_fingerprint"] == SYSTEM_FINGERPRINT


def test_fingerprint_tracks_the_prompt():
    import hashlib

    assert SYSTEM_FINGERPRINT == hashlib.sha256(SYSTEM.encode()).hexdigest()[:12]
