"""What every language-model player is shown, whoever serves the model.

Kept apart from the transports on purpose. Comparing a local model with Claude
is only a comparison of models if both received the same words, and the
easiest way to guarantee that is for there to be exactly one place the words
are assembled.
"""

from __future__ import annotations

from typing import Any

from . import memory
from .base import TurnContext
from .journal import Journal
from .memory import AgentMemory

# The move format. Two fields on purpose: `command` is what the game receives,
# `reasoning` is what the observatory displays.
MOVE_FIELDS: dict[str, Any] = {
    "type": "object",
    "properties": {
        # Phrased without presupposing rooms, objects or a parser — at the
        # "cold" rung the agent has not established that any of those exist,
        # and a schema that assumes them would leak the answer.
        "reasoning": {
            "type": "string",
            "description": "Two or three sentences: what you currently believe, and what you are trying next.",
        },
        "command": {
            "type": "string",
            "description": "The single line of text to type next.",
        },
    },
    "required": ["reasoning", "command"],
    "additionalProperties": False,
}

# The journal line, when the journal is switched on. Optional on purpose: a
# turn with nothing worth keeping should cost nothing to leave blank, and a
# required field would be filled every turn whether or not anything happened.
#
# Carried on the move rather than fetched by a separate tool call. A second
# call would double the latency of a local model and go through Ollama's
# tool-call path, which is the one that fails on gpt-oss; as a field it is free,
# and it is written in the same breath as the command that caused it.
JOURNAL_FIELD: dict[str, Any] = {
    "type": "string",
    "description": "Optional. One line to keep permanently, if this turn "
                   "established something worth keeping. Omit it otherwise.",
}


# The lookup, when searching is switched on. Same reasoning as the journal
# field: carried on the move rather than fetched through a tool-call path that
# not every local server implements the same way. Unlike the journal, filling
# it costs a second request — the results come back and the player is asked
# again — so the description says what it is for rather than inviting it.
SEARCH_FIELD: dict[str, Any] = {
    "type": "string",
    "description": "Optional. A word or phrase to look back for in everything "
                   "typed and printed so far. Fill this in and leave `command` "
                   "empty; you will be shown what matched and asked again.",
}


def move_fields(journal: bool = False, search: bool = False) -> dict[str, Any]:
    """The reply schema, with whichever optional fields are switched on."""
    extra: dict[str, Any] = {}
    if journal:
        extra["journal"] = JOURNAL_FIELD
    if search:
        extra["search"] = SEARCH_FIELD
    if not extra:
        return MOVE_FIELDS
    return {
        **MOVE_FIELDS,
        "properties": {**MOVE_FIELDS["properties"], **extra},
    }

# A command is one line typed at a prompt. Anything longer is a model talking,
# and the interpreter would only choke on it.
MAX_COMMAND_CHARS = 120

# The transcript window, in exchanges. Thirty is what the September 2026 sweep
# measured as working — see docs/experiments/2026-09-scaffold-sweep.md, where
# the arm that replaced this window with a structured record never once entered
# the house.
DEFAULT_HISTORY = 30


def turn_prompt(
    mem: AgentMemory,
    ctx: TurnContext,
    history_turns: int,
    journal: Journal | None = None,
) -> str:
    """One user turn carrying a windowed transcript.

    A rolling window rather than the whole history: past a few dozen turns the
    early transcript stops informing the next move and starts costing tokens.
    A window of zero is the stateless player: the latest output and nothing
    else, not even the command that produced it.

    The window does more work than it appears to. A room description scrolls
    past once, and everything the player could act on is named in it; thirty
    turns keeps it in view long enough to be tried. A one-turn window let the
    same model walk past the only way into the house nine times.
    """
    lines = []

    # The memory goes first and outside the transcript window. It is the one
    # thing that must not scroll off — it is what the agent kept when the world
    # was rolled back, and dropping it silently would make every death teach
    # nothing.
    remembered = mem.render()
    if remembered:
        lines.append(remembered)
        lines.append("")

    # Beside the transcript, never instead of it, and nearest to it: what the
    # agent wrote down is the last thing it reads before the last thing it saw.
    if journal is not None:
        written = journal.render()
        if written:
            lines.append(written)
            lines.append("")

    if history_turns > 0:
        for command, response in ctx.transcript[-history_turns:]:
            if command:
                lines.append(f"> {command}")
            lines.append(response.strip())
    else:
        lines.append(ctx.observation.strip())
    lines.append(f"\n[Turn {ctx.turn}. Score {ctx.score}. Moves {ctx.moves}.]")
    lines.append("What is your next command?")
    return "\n".join(lines)


def search_reply(results: str, ctx: TurnContext, searches_left: int) -> str:
    """What comes back when a player looks something up, and the question again.

    Appended to the same turn's prompt rather than sent as a fresh one, so the
    player answers with its lookup and the window both in view. Saying how many
    lookups remain is not a nudge to use them — it is what stops a player
    spending its last one and being cut off mid-thought without knowing why.
    """
    lines = ["", results, ""]
    if searches_left > 0:
        lines.append(
            f"You may look back {searches_left} more time"
            f"{'' if searches_left == 1 else 's'} this turn, or answer now."
        )
    else:
        lines.append("That was the last lookup available this turn.")
    lines.append(f"[Turn {ctx.turn}. Score {ctx.score}. Moves {ctx.moves}.]")
    lines.append("What is your next command?")
    return "\n".join(lines)


def reflection_prompt(
    mem: AgentMemory, ctx: TurnContext, journal: Journal | None = None
) -> str:
    """The tail of the transcript and the reflection question, nothing else —
    no object tree, no cause of death from the engine. If the agent misreads
    its own death, that misreading is the thing worth recording.

    The journal is shown when there is one. Twelve exchanges is a keyhole to
    summarize two hundred turns through, and it is why the notes this produces
    are about whatever happened last; a player that wrote things down as it
    went should get to read them before being asked what it learned.
    """
    lines = []
    remembered = mem.render()
    if remembered:
        lines.append(remembered)
        lines.append("")
    if journal is not None:
        written = journal.render()
        if written:
            lines.append(written)
            lines.append("")
    for command, response in ctx.transcript[-12:]:
        if command:
            lines.append(f"> {command}")
        lines.append(response.strip())
    lines.append("")
    lines.append(memory.REFLECT_RESTART if ctx.next_start == "beginning" else memory.REFLECT)
    return "\n".join(lines)


def clean_command(text: str) -> str:
    """The first line, without the prompt character or quotes a model may wrap
    it in. Case and wording are left alone — those are the model's choice."""
    for line in (text or "").splitlines():
        line = line.strip().lstrip(">").strip().strip("`\"'").strip()
        if line:
            return line[:MAX_COMMAND_CHARS]
    return ""
