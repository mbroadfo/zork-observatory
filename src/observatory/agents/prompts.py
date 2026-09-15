"""The information ladder.

How much an agent is told before it starts is an experimental variable, not a
setting to get right once. These four levels are the rungs, from a bare
terminal to an openly coached player. Running the same model at each rung
produces a score-versus-scaffolding curve, and the shape of that curve says
more than any single number.

Read them in order. Each adds exactly one kind of knowledge:

  COLD      you are typing at a computer. That is all.
  GAME      it is a game, and you are trying to do well at it.
  PARSER    how the parser works: grammar, directions, meta-commands.
  COACHED   hazards and tactics. Deliberately contaminated.

COACHED exists to be the control arm. It is the only level allowed to contain
world knowledge, it is clearly labelled, and every level below it is held to
the rule in tests/test_prompt_hygiene.py: describe the interface, never the
world.

An important caveat, recorded here because it is easy to forget once the
numbers start looking good: COLD does not produce a naive player. The model has
read thousands of interactive fiction transcripts. What the ladder measures is
how much scaffolding a model needs before it deploys knowledge it already has —
which is worth measuring, but is not the same as watching something discover
the form from scratch. To get closer to the latter you must also perturb the
world (see the mutator on the roadmap); "an open field west of a white house"
is a fingerprint no amount of prompt austerity can hide.
"""

from __future__ import annotations

import hashlib

# --- L0 ------------------------------------------------------------------
# No game framing at all. Not told it is a game, not told there is a parser,
# not told what counts as success. The 1980 experience: a cursor and nothing.
COLD = """You are interacting with a computer through a text terminal.

Respond with whatever you want to type next. Nothing else — no commentary, no \
explanation, just the literal characters you would type at the prompt.

In your reasoning, keep track of what you have worked out so far about what \
this program is and how it responds."""

# --- L1 ------------------------------------------------------------------
# Told the goal, nothing about how to operate it. The interesting middle: it
# knows it is supposed to be doing well at something, and has to work out what
# kind of input the thing accepts.
GAME = """You are interacting with a computer program through a text terminal. \
It is some kind of game, and your objective is to do as well at it as you can.

Respond with whatever you want to type next. Nothing else — no commentary, no \
explanation, just the literal characters you would type at the prompt.

In your reasoning, keep track of what you have worked out: how this program \
appears to accept input, what it seems to understand, and what you are trying \
next."""

# --- L2 ------------------------------------------------------------------
# The interface, fully explained. Still no world knowledge. This is the
# default, and the fairest "the model is actually playing" baseline.
PARSER = """You are playing a text adventure game. You see only what the game \
prints, exactly as a human player would.

The interface:
- Reply with ONE command per turn, in the game's own terse style: VERB, VERB \
NOUN, or VERB NOUN PREPOSITION NOUN.
- These words move you: north, south, east, west, northeast, northwest, \
southeast, southwest, up, down, in, out (abbreviated n, s, e, w, ne, nw, se, \
sw, u, d).
- "look" redescribes your surroundings. "inventory" lists what you are carrying.
- The parser has a small vocabulary. A reply like "I don't know that word" means \
it did not understand the wording, not that the idea was wrong.

In your reasoning, state what you currently believe: where you are, what you are \
carrying, which exits you have not tried, and what you are trying to achieve.

Your objective is to score as many points as possible. You have a limited number \
of turns."""

# --- L3 ------------------------------------------------------------------
# The contaminated control arm. Everything the other levels forbid.
COACHED = PARSER + """

Additional guidance:
- Darkness is lethal in these games. Find and light a portable light source \
before going anywhere dark, and never enter a dark area without one.
- Containers often hold useful items. Open everything you can open.
- Valuables usually need to be deposited somewhere specific to score.
- Prefer exploring exits you have not tried over revisiting rooms you know.
- If something blocks you, look for another route rather than forcing it."""


LEVELS: dict[str, str] = {
    "cold": COLD,
    "game": GAME,
    "parser": PARSER,
    "coached": COACHED,
}

LEVEL_ORDER = ["cold", "game", "parser", "coached"]

# Levels that must contain no world knowledge. COACHED is deliberately exempt.
CLEAN_LEVELS = ["cold", "game", "parser"]


def get(level: str) -> str:
    try:
        return LEVELS[level]
    except KeyError:
        raise ValueError(
            f"Unknown info level {level!r}. Expected one of {LEVEL_ORDER}."
        ) from None


def fingerprint(level: str) -> str:
    return hashlib.sha256(get(level).encode()).hexdigest()[:12]
