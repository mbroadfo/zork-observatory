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


def clean_command(text: str) -> str:
    """The first line, without the prompt character or quotes a model may wrap
    it in. Case and wording are left alone — those are the model's choice."""
    for line in (text or "").splitlines():
        line = line.strip().lstrip(">").strip().strip("`\"'").strip()
        if line:
            return line[:MAX_COMMAND_CHARS]
    return ""
