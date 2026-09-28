"""The observatory as tools an assistant can read.

Two things are pinned. The protocol, because a stray line on stdout corrupts
the stream and a client that cannot initialize gives no error anyone will see.

And the boundary, which matters more: these tools report the *run's behaviour*
and never the game's answers. A frontier model has read Zork, so a tool that
returned hints would turn advice into recall and quietly destroy the thing the
observatory measures. There is no hint tool, and a test says so — because the
tempting addition, six months from now, is exactly that.
"""

from __future__ import annotations

import io
import json

import pytest

from observatory import mcp_server
from observatory.mcp_server import TOOLS, Observatory, handle

SESSION = {
    "turn": 148, "max_turns": 400, "score": 44, "max_score": 350,
    "agent": "ollama:qwen3:14b/coached", "agent_kind": "llm",
    "location": "Studio", "inventory": ["brass lantern", "nasty knife"],
    "deaths": 7, "lives_left": 3, "finished": False,
    "handoffs": [{"turn": 133, "from": "claude:claude-haiku-4-5",
                  "to": "claude:claude-haiku-4-5/coached", "inherits": True}],
    "usage": {"cost_usd": 0.702},
    "map": {"rooms": 24, "edges": 31, "blocked": 36},
    "structure": {"since_new_room": 61, "stale_room": "Gallery", "radius": 8,
                  "components": 1, "reachable": 9, "revisit_ratio": 0.63,
                  "moves_between_rooms": 91, "unretried_walls": 12, "probed_once": 4},
    "quality": {"wasted_pct": 28.5, "futile_pct": 9.2, "known_dead_ends": 36,
                "distinct_commands": 50},
    "coverage": {"rooms_seen": 24, "rooms_total": 109},
    "memory": [{"turn": 113, "kind": "death", "text": "The troll killed me."}],
    "journal": [{"turn": 33, "text": "The window opens with great effort."}],
    "journal_summary": {"attempted": 9, "kept": 1},
}


class Fake(Observatory):
    """An observatory that answers from a dict instead of over HTTP."""

    def __init__(self, session=SESSION, **routes) -> None:
        super().__init__("http://test.invalid")
        self.session_data = session
        self.routes = routes
        self.asked: list[str] = []

    def get(self, path):
        self.asked.append(path)
        if path.startswith("/api/state"):
            return {"session": self.session_data}
        for key, value in self.routes.items():
            if path.startswith(key):
                return value
        return {}


def call(obs, name, **arguments):
    reply = handle({"jsonrpc": "2.0", "id": 9, "method": "tools/call",
                    "params": {"name": name, "arguments": arguments}}, obs)
    assert reply is not None
    return reply["result"]["content"][0]["text"]


class TestTheProtocol:
    def test_initialize_answers_with_a_version_and_a_name(self):
        reply = handle({"jsonrpc": "2.0", "id": 1, "method": "initialize"}, Fake())
        assert reply["result"]["serverInfo"]["name"] == "zork-observatory"
        assert reply["result"]["protocolVersion"]
        assert "tools" in reply["result"]["capabilities"]

    def test_the_instructions_tell_the_assistant_not_to_answer_the_game(self):
        """The one piece of guidance a client always reads."""
        reply = handle({"jsonrpc": "2.0", "id": 1, "method": "initialize"}, Fake())
        text = reply["result"]["instructions"].lower()
        assert "do not supply walkthrough answers" in text

    def test_notifications_get_no_reply(self):
        """A response to a notification is a protocol error, and clients differ
        in how loudly they complain about it."""
        assert handle({"jsonrpc": "2.0", "method": "notifications/initialized"},
                      Fake()) is None

    def test_every_tool_is_listed_with_a_schema(self):
        reply = handle({"jsonrpc": "2.0", "id": 2, "method": "tools/list"}, Fake())
        listed = reply["result"]["tools"]
        assert {t["name"] for t in listed} == set(TOOLS)
        for tool in listed:
            assert tool["description"] and tool["inputSchema"]["type"] == "object"

    def test_an_unknown_tool_is_a_protocol_error(self):
        reply = handle({"jsonrpc": "2.0", "id": 3, "method": "tools/call",
                        "params": {"name": "hint"}}, Fake())
        assert reply["error"]["code"] == -32602

    def test_an_unknown_method_is_refused(self):
        reply = handle({"jsonrpc": "2.0", "id": 4, "method": "resources/list"}, Fake())
        assert reply["error"]["code"] == -32601

    def test_a_tool_that_raises_returns_an_error_result_not_a_crash(self):
        """One broken tool must not end the session."""
        class Broken(Fake):
            def get(self, path):
                raise ValueError("boom")

        reply = handle({"jsonrpc": "2.0", "id": 5, "method": "tools/call",
                        "params": {"name": "position"}}, Broken())
        assert reply["result"]["isError"] is True
        assert "ValueError" in reply["result"]["content"][0]["text"]

    def test_an_observatory_that_is_not_running_says_so(self):
        import urllib.error

        class Down(Fake):
            def get(self, path):
                raise urllib.error.URLError("connection refused")

        assert "Is it running?" in call(Down(), "position")

    def test_the_stream_is_one_json_object_per_line(self):
        """Anything else on stdout corrupts the transport."""
        messages = [
            {"jsonrpc": "2.0", "id": 1, "method": "initialize"},
            {"jsonrpc": "2.0", "method": "notifications/initialized"},
            {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
        ]
        out = io.StringIO()
        mcp_server.serve(
            stdin=io.StringIO("\n".join(json.dumps(m) for m in messages) + "\n"),
            stdout=out,
        )
        lines = [l for l in out.getvalue().splitlines() if l]
        assert len(lines) == 2          # the notification is not answered
        for line in lines:
            json.loads(line)

    def test_a_malformed_line_is_skipped_rather_than_fatal(self):
        out = io.StringIO()
        mcp_server.serve(
            stdin=io.StringIO('not json\n\n{"jsonrpc":"2.0","id":1,"method":"initialize"}\n'),
            stdout=out,
        )
        assert len(out.getvalue().strip().splitlines()) == 1


class TestWhatTheToolsReport:
    def test_position_gives_the_facts_a_person_needs(self):
        text = call(Fake(), "position")
        assert "Turn 148" in text and "score 44" in text
        assert "Studio" in text
        assert "brass lantern" in text

    def test_position_says_when_a_run_has_changed_hands(self):
        """A score spanning a handoff belongs to neither player, and an adviser
        reading only the number would draw the wrong conclusion."""
        text = call(Fake(), "position")
        assert "changed hands" in text
        assert "belongs to more than one player" in text

    def test_the_cost_is_the_run_s_not_the_current_player_s(self):
        assert "$0.70" in call(Fake(), "position")

    def test_stuck_report_measures_the_shape_of_the_failure(self):
        text = call(Fake(), "stuck_report")
        assert "Nothing new for 61 turns" in text
        assert "63% of moves are revisits" in text
        assert "28.5%" in text and "9.2%" in text
        assert "12 directions were refused once" in text

    def test_stuck_report_says_out_loud_that_it_is_not_a_hint(self):
        assert "does NOT tell you is the answer" in call(Fake(), "stuck_report")

    def test_search_passes_the_query_through_and_returns_what_was_printed(self):
        obs = Fake(**{"/api/transcript/search": {
            "rendered": 'You looked back for "grating".\n  [turn 9] > examine grating'}})
        text = call(obs, "search_transcript", query="grating")
        assert "[turn 9]" in text
        assert "q=grating" in obs.asked[-1]

    def test_search_without_a_query_asks_for_one(self):
        assert "Give a word" in call(Fake(), "search_transcript")

    def test_what_was_tried_defaults_to_the_current_room(self):
        obs = Fake(**{"/api/transcript/room": {
            "room": "Studio",
            "exchanges": [{"turn": 136, "command": "north",
                           "response": "You can't go that way."}]}})
        text = call(obs, "what_was_tried")
        assert "In Studio, 1 commands so far" in text
        assert "north" in text and "can't go that way" in text

    def test_map_shape_reports_shape_not_just_size(self):
        text = call(Fake(), "map_shape")
        assert "24 rooms" in text
        assert "Depth from the start: 8" in text
        assert "109" in text

    def test_what_it_remembers_shows_lessons_and_journal(self):
        text = call(Fake(), "what_it_remembers")
        assert "The troll killed me." in text
        assert "The window opens" in text
        assert "1 were kept" in text

    def test_every_tool_copes_with_no_live_run(self):
        empty = Fake(session=None)
        for name in TOOLS:
            text = call(empty, name, query="x")
            assert text and isinstance(text, str)


class TestTheBoundary:
    """The tools are for advising on the run, never for answering the game.

    A hosted model has read Zork. A tool that returned solutions would make
    advice indistinguishable from recall, which is the exact contamination the
    prompt ladder exists to prevent. The temptation to add one later is real,
    so the absence is a test rather than a comment."""

    def test_there_is_no_hint_or_walkthrough_tool(self):
        for forbidden in ("hint", "solve", "walkthrough", "answer", "next_move",
                          "suggest_command", "solution"):
            assert forbidden not in TOOLS

    def test_no_tool_offers_to_choose_the_next_command(self):
        for name, spec in TOOLS.items():
            described = spec["description"].lower()
            assert "what to type" not in described
            assert "next move" not in described

    def test_no_tool_reaches_the_engine_s_ground_truth(self):
        """Valid actions, the object tree and the state hash stay on the
        observatory's side of the line, exactly as they do for the agent."""
        obs = Fake()
        for name in TOOLS:
            call(obs, name, query="x")
        for path in obs.asked:
            assert "valid_action" not in path and "objects" not in path

    def test_the_tools_are_read_only(self):
        """Nothing here may change a run. An adviser that could type for you
        would make the trace a lie about who played."""
        obs = Fake()
        for name in TOOLS:
            call(obs, name, query="x")
        assert all(p.startswith("/api/") for p in obs.asked)
        # Every write route in the server is a POST; this client cannot POST.
        assert not hasattr(Observatory, "post")
