"""The observatory, as tools an assistant can read while you play.

The division of labour this serves: a local model plays, a person gets it
unstuck, and a strong hosted model advises the person. The third role is the
only one that needs a subscription, and it is interactive — somebody reads the
answer and decides what to type. That keeps the run honest, because the trace
records a human playing those turns, which is what happened.

What these tools deliberately do not do is answer the game.

Asking a frontier model "what do I type in this room" is both a waste and a
contamination: it has read Zork, so the answer is recall, and a run that
absorbs it is no longer measuring anything. Every tool here reports the
*run's behaviour* — where it has been, what it has tried, what keeps refusing
it, where it has stopped making progress — and none reports the game's
solutions. There is no hint tool and no walkthrough tool, and that is a design
constraint rather than an oversight. The question this is built to answer is
"what class of thing is my agent failing at", which is a question about the
agent; "how do I open the grating" is a question about Zork, and the internet
already has it.

Transport is stdio JSON-RPC 2.0, spoken directly. The protocol is small enough
that a dependency would cost more than it saves, and this file has none beyond
the standard library — the observatory's own state is fetched over its HTTP
API, so this runs as a separate process and cannot disturb a live run.
"""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

PROTOCOL_VERSION = "2025-06-18"
SERVER_NAME = "zork-observatory"
DEFAULT_BASE = os.environ.get("OBSERVATORY_URL", "http://127.0.0.1:8000")
TIMEOUT_S = 10

# Enough to see a pattern; small enough that an assistant reads it all.
DEFAULT_TAIL = 24
DEFAULT_HITS = 8


class Observatory:
    """Read-only client for a running observatory."""

    def __init__(self, base: str = DEFAULT_BASE) -> None:
        self.base = base.rstrip("/")

    def get(self, path: str) -> dict[str, Any]:
        url = f"{self.base}{path}"
        request = urllib.request.Request(url, headers={"Accept": "application/json"})
        with urllib.request.urlopen(request, timeout=TIMEOUT_S) as response:
            return json.loads(response.read())

    def session(self) -> dict[str, Any] | None:
        return self.get("/api/state").get("session")


def _no_session() -> str:
    return ("No run is live. Start one in the observatory at "
            f"{DEFAULT_BASE}, or open a recorded trace.")


# --- the tools -----------------------------------------------------------
#
# Each returns plain text rather than JSON. The consumer is a language model
# being asked for advice, and prose it can quote back to the person is more
# use than a structure it has to describe.


def position(obs: Observatory, **_: Any) -> str:
    """Where the run stands right now."""
    s = obs.session()
    if not s:
        return _no_session()
    lines = [
        f"Turn {s['turn']} of {s['max_turns']}, score {s['score']} of {s['max_score']}.",
        f"Player: {s['agent']} ({s['agent_kind']}).",
        f"Room: {s['location']}.",
        f"Carrying: {', '.join(s.get('inventory') or []) or 'nothing'}.",
        f"Deaths: {s['deaths']}; lives left: {s.get('lives_left', 0)}.",
    ]
    if s.get("handoffs"):
        hands = "; ".join(
            f"turn {h['turn']}: {h['from']} to {h['to']}" for h in s["handoffs"])
        lines.append(f"This run has changed hands: {hands}.")
        lines.append("Its score therefore belongs to more than one player.")
    if s.get("finished"):
        lines.append(f"The run is over: {s['end_reason']}"
                     + (" (censored — stopped by a budget, not by the game)"
                        if s.get("censored") else ""))
    usage = s.get("usage") or {}
    if usage.get("cost_usd"):
        lines.append(f"Cost so far: ${usage['cost_usd']:.2f} across the whole run.")
    return "\n".join(lines)


def recent(obs: Observatory, turns: int = DEFAULT_TAIL, **_: Any) -> str:
    """The last few exchanges, as the player saw them."""
    s = obs.session()
    if not s:
        return _no_session()
    events = obs.get("/api/transcript")
    rows = events.get("exchanges") or []
    rows = rows[-max(1, min(int(turns), 60)):]
    out = []
    for row in rows:
        if row.get("command"):
            out.append(f"[{row['turn']}] > {row['command']}")
        out.append(f"    {(row.get('response') or '').strip()}")
    return "\n".join(out) or "Nothing has been played yet."


def search_transcript(obs: Observatory, query: str = "", limit: int = DEFAULT_HITS,
                      **_: Any) -> str:
    """Every turn mentioning a word, however far back."""
    if not (query or "").strip():
        return "Give a word or phrase to look for."
    found = obs.get(
        f"/api/transcript/search?q={urllib.parse.quote(query)}&limit={int(limit)}")
    return found.get("rendered") or "Nothing matched."


def what_was_tried(obs: Observatory, room: str = "", **_: Any) -> str:
    """Commands issued in a room, and what the game said back.

    The question behind most "why is it stuck" is really "what has it already
    been told here", and a player whose window has scrolled cannot answer it.
    """
    found = obs.get(f"/api/transcript/room?name={urllib.parse.quote(room)}"
                    if room else "/api/transcript/room")
    rows = found.get("exchanges") or []
    if not rows:
        return f"Nothing recorded in {room!r}." if room else "Nothing recorded yet."
    head = f"In {found.get('room') or room}, {len(rows)} commands so far:"
    out = [head]
    for row in rows:
        reply = (row.get("response") or "").strip().splitlines()
        out.append(f'  [{row["turn"]}] > {row["command"]}'
                   f'  ->  {reply[0][:90] if reply else ""}')
    return "\n".join(out)


def stuck_report(obs: Observatory, **_: Any) -> str:
    """Why this run looks stuck, in the observatory's own measurements.

    This is the tool the whole server exists for. It reports shape, not
    solutions: how long since anything new, how much of the play is revisiting,
    which refusals keep recurring. Everything here is derived from the run's
    own behaviour and none of it is knowledge about the game.
    """
    s = obs.session()
    if not s:
        return _no_session()
    st = s.get("structure") or {}
    q = s.get("quality") or {}
    lines = [
        f"Turn {s['turn']}, score {s['score']}, in {s['location']}.",
        f"Nothing new for {st.get('since_new_room', '?')} turns"
        f" (last new room: {st.get('stale_room') or 'unknown'}).",
        f"{st.get('revisit_ratio', 0):.0%} of moves are revisits;"
        f" {st.get('moves_between_rooms', 0)} moves over"
        f" {(s.get('map') or {}).get('rooms', 0)} rooms.",
        f"Wasted turns: {q.get('wasted_pct', 0)}%; futile (already tried"
        f" here and refused): {q.get('futile_pct', 0)}%.",
        f"Known dead ends: {q.get('known_dead_ends', 0)}."
        f" Distinct commands used: {q.get('distinct_commands', 0)}.",
    ]
    if st.get("unretried_walls"):
        lines.append(f"{st['unretried_walls']} directions were refused once and"
                     " never tried again since the world changed.")
    if st.get("probed_once"):
        lines.append(f"{st['probed_once']} rooms have had exactly one exit tried.")
    lines.append("")
    lines.append("What this does NOT tell you is the answer to the puzzle, on"
                 " purpose. Advise on the shape of the failure.")
    return "\n".join(lines)


def map_shape(obs: Observatory, **_: Any) -> str:
    """The map's shape rather than its size."""
    s = obs.session()
    if not s:
        return _no_session()
    st = s.get("structure") or {}
    m = s.get("map") or {}
    cov = s.get("coverage") or {}
    return "\n".join([
        f"{m.get('rooms', 0)} rooms, {m.get('edges', 0)} connections,"
        f" {m.get('blocked', 0)} refused directions.",
        f"Depth from the start: {st.get('radius', '?')}."
        f" Separate components: {st.get('components', '?')}.",
        f"Rooms reachable from here: {st.get('reachable', '?')}.",
        f"Coverage: {cov.get('rooms_seen', '?')} of {cov.get('rooms_total', '?')}"
        " rooms the engine knows about.",
    ])


def what_it_remembers(obs: Observatory, **_: Any) -> str:
    """The lessons and journal lines the player is carrying."""
    s = obs.session()
    if not s:
        return _no_session()
    out = []
    memory = s.get("memory") or []
    out.append(f"Lessons kept across deaths ({len(memory)}):")
    for m in memory:
        out.append(f'  [turn {m.get("turn")}, {m.get("kind")}] {m.get("text")}')
    journal = s.get("journal") or []
    out.append(f"Journal lines the world bore out ({len(journal)}):")
    for j in journal[-12:]:
        out.append(f'  [turn {j.get("turn")}] {j.get("text")}')
    summary = s.get("journal_summary") or {}
    if summary:
        out.append(f"Of {summary.get('attempted', 0)} lines it offered,"
                   f" {summary.get('kept', 0)} were kept.")
    return "\n".join(out)


TOOLS: dict[str, dict[str, Any]] = {
    "position": {
        "fn": position,
        "description": "Where the live run stands: turn, score, room, inventory, "
                       "who is playing, and whether it has changed hands.",
        "schema": {"type": "object", "properties": {}},
    },
    "stuck_report": {
        "fn": stuck_report,
        "description": "Why the run looks stuck, measured from its own behaviour: "
                       "turns since anything new, revisit ratio, wasted and futile "
                       "turns, walls never retried. Reports the shape of the "
                       "failure, never the game's solutions.",
        "schema": {"type": "object", "properties": {}},
    },
    "recent": {
        "fn": recent,
        "description": "The last N exchanges exactly as the player saw them.",
        "schema": {
            "type": "object",
            "properties": {"turns": {"type": "integer",
                                     "description": f"How many (default {DEFAULT_TAIL})."}},
        },
    },
    "search_transcript": {
        "fn": search_transcript,
        "description": "Every turn mentioning a word or phrase, however far back, "
                       "verbatim with turn numbers. Use it to find what the game "
                       "already said about something.",
        "schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Word or phrase."},
                "limit": {"type": "integer", "description": f"Max hits (default {DEFAULT_HITS})."},
            },
            "required": ["query"],
        },
    },
    "what_was_tried": {
        "fn": what_was_tried,
        "description": "Commands issued in a room and what the game replied. "
                       "Defaults to the room the player is standing in.",
        "schema": {
            "type": "object",
            "properties": {"room": {"type": "string",
                                    "description": "Room name; omit for the current one."}},
        },
    },
    "map_shape": {
        "fn": map_shape,
        "description": "The discovered map's shape: depth, components, reachable "
                       "rooms, coverage against what the engine knows exists.",
        "schema": {"type": "object", "properties": {}},
    },
    "what_it_remembers": {
        "fn": what_it_remembers,
        "description": "The lessons the player kept across deaths and the journal "
                       "lines the world bore out.",
        "schema": {"type": "object", "properties": {}},
    },
}


# --- the protocol ---------------------------------------------------------


def handle(message: dict[str, Any], obs: Observatory) -> dict[str, Any] | None:
    """One JSON-RPC request in, one response out. None for notifications."""
    method = message.get("method")
    mid = message.get("id")
    params = message.get("params") or {}

    if method == "initialize":
        return _ok(mid, {
            "protocolVersion": PROTOCOL_VERSION,
            "capabilities": {"tools": {}},
            "serverInfo": {"name": SERVER_NAME, "version": "1"},
            "instructions": (
                "Tools for advising a human who is playing a text adventure while "
                "a local model plays alongside them. Report on the run's "
                "behaviour. Do not supply walkthrough answers: the point of the "
                "experiment is what the player can work out, so telling it the "
                "solution destroys the measurement."
            ),
        })
    if method == "notifications/initialized" or mid is None:
        return None
    if method == "tools/list":
        return _ok(mid, {"tools": [
            {"name": name, "description": spec["description"],
             "inputSchema": spec["schema"]}
            for name, spec in TOOLS.items()
        ]})
    if method == "tools/call":
        name = params.get("name")
        spec = TOOLS.get(name or "")
        if spec is None:
            return _err(mid, -32602, f"unknown tool {name!r}")
        try:
            text = spec["fn"](obs, **(params.get("arguments") or {}))
        except urllib.error.URLError as exc:
            text = (f"Cannot reach the observatory at {obs.base}: "
                    f"{getattr(exc, 'reason', exc)}. Is it running?")
        except Exception as exc:   # a broken tool must not kill the session
            return _ok(mid, {
                "content": [{"type": "text", "text": f"{type(exc).__name__}: {exc}"}],
                "isError": True,
            })
        return _ok(mid, {"content": [{"type": "text", "text": text}]})
    return _err(mid, -32601, f"unknown method {method!r}")


def _ok(mid: Any, result: dict[str, Any]) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": mid, "result": result}


def _err(mid: Any, code: int, message: str) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": mid, "error": {"code": code, "message": message}}


def serve(base: str = DEFAULT_BASE,
          stdin: Any = None, stdout: Any = None) -> None:
    """Read JSON-RPC from stdin, write responses to stdout, until EOF."""
    obs = Observatory(base)
    stdin = stdin or sys.stdin
    stdout = stdout or sys.stdout
    for line in stdin:
        line = line.strip()
        if not line:
            continue
        try:
            message = json.loads(line)
        except json.JSONDecodeError:
            continue
        response = handle(message, obs)
        if response is None:
            continue
        stdout.write(json.dumps(response) + "\n")
        stdout.flush()
