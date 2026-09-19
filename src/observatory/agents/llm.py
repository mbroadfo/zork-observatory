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

# A command is one line typed at a prompt. Anything longer is a model talking,
# and the interpreter would only choke on it.
MAX_COMMAND_CHARS = 120

# The same move, with fallbacks. Asking for a ranked list costs one call, not
# two, and turns "don't repeat yourself" from an instruction into a choice the
# harness can act on: it plays the first candidate its record does not already
# know to be inert here. The model still decides what is worth trying; the
# harness only skips what it has watched do nothing.
MOVE_FIELDS_RANKED: dict[str, Any] = {
    "type": "object",
    "properties": {
        "reasoning": MOVE_FIELDS["properties"]["reasoning"],
        "command": MOVE_FIELDS["properties"]["command"],
        "alternatives": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Two more lines you would type instead, best first, "
                           "different from the first and from each other.",
        },
    },
    "required": ["reasoning", "command", "alternatives"],
    "additionalProperties": False,
}


RECALL_MODES = ("transcript", "episodic")

# How much raw transcript each kind of recall sees by default. The episodic
# player gets the last exchange only: its record already holds everything
# older, filed rather than replayed.
DEFAULT_HISTORY = {"transcript": 30, "episodic": 1}


def default_history(recall: str) -> int:
    if recall not in RECALL_MODES:
        raise ValueError(f"Unknown recall mode: {recall!r} (expected {', '.join(RECALL_MODES)})")
    return DEFAULT_HISTORY[recall]


def turn_prompt(mem: AgentMemory, ctx: TurnContext, history_turns: int, record: str = "") -> str:
    """One user turn carrying a windowed transcript.

    A rolling window rather than the whole history: past a few dozen turns the
    early transcript stops informing the next move and starts costing tokens.
    A window of zero is the stateless player: the latest output and nothing
    else, not even the command that produced it. `record` is the episodic
    player's filed history (see episodic.py), shown ahead of the window.
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
    if record:
        lines.append(record)
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


def reflection_prompt(mem: AgentMemory, ctx: TurnContext) -> str:
    """The tail of the transcript and the reflection question, nothing else —
    no object tree, no cause of death from the engine. If the agent misreads
    its own death, that misreading is the thing worth recording."""
    lines = []
    remembered = mem.render()
    if remembered:
        lines.append(remembered)
        lines.append("")
    for command, response in ctx.transcript[-12:]:
        if command:
            lines.append(f"> {command}")
        lines.append(response.strip())
    lines.append("")
    lines.append(memory.REFLECT_RESTART if ctx.next_start == "beginning" else memory.REFLECT)
    return "\n".join(lines)


def repeat_nudge(command: str, count: int, outcome: str) -> str:
    """Handed back to a model that just chose a command its own record shows
    doing nothing here, so it may choose again.

    A scaffold, and labelled as one. It says nothing about the game: it quotes
    the agent's own record back at it and asks for something else. Whether the
    model then does something else is the measurement — see the `nudge` option
    in ollama_agent.py, and the name suffix that keeps a nudged run from being
    mistaken for a bare one.
    """
    seen = outcome if outcome.startswith("heading ") else outcome.strip('"')
    if count == 1:
        history = f"You have already typed {command!r} here, and the reply was: {seen}"
    else:
        history = (
            f"You have already typed {command!r} here {count} times, and every time "
            f"the reply was the same: {seen}"
        )
    return f"Wait. {history}\n\nType something you have not tried here."


def exhausted_nudge(offered: list[str], note: str) -> str:
    """When every ranked candidate is one the record has watched do nothing.

    The turn a filter cannot help with: the model is out of ideas and all three
    of them are the same idea. It is told so, and told that the record lists
    what is untouched here — which is where the agenda earns its place.
    """
    tried = ", ".join(f"{c!r}" for c in offered)
    return (
        f"{note}\n\nAll of your choices this turn ({tried}) are ones you have already "
        f"typed here to no effect. Leave that idea. Your record lists the directions "
        f"you have not typed here and the words this place's text used that you have "
        f"not; take one of those."
    )


def clean_command(text: str) -> str:
    """The first line, without the prompt character or quotes a model may wrap
    it in. Case and wording are left alone — those are the model's choice."""
    for line in (text or "").splitlines():
        line = line.strip().lstrip(">").strip().strip("`\"'").strip()
        if line:
            return line[:MAX_COMMAND_CHARS]
    return ""
