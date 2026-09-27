"""The Claude player's wiring.

The behaviour pinned here is entirely about failing at the right moment. An
unauthenticated client does not complain when it is built; it complains on the
first request — which, before this existed, arrived as a session that crashed
on turn 1 with an SDK TypeError in place of "no key". Worse, on any path that
catches an API error and plays `look`, it arrives as four hundred turns of what
looks like an extraordinarily bad player.

So: ask once, before turn one, and say which of the four possible things is
wrong — no credentials, wrong credentials, a model this account cannot reach,
or nothing answering.
"""

from __future__ import annotations

import anthropic
import pytest

# Whatever transport the installed SDK builds its errors from. Constructing a
# real APIStatusError needs a real response object, and the SDK moved from httpx
# to httpx2 — which is exactly the kind of detail a test should not pin.
try:
    import httpx2 as httpx
except ModuleNotFoundError:  # pragma: no cover - older anthropic
    import httpx  # type: ignore[no-redef]

from observatory import credentials
from observatory.agents.claude_agent import KEY_VARS, ClaudeAgent, takes_effort


def unset_keys(monkeypatch) -> None:
    """No credential from any of the paths credentials.py knows about.

    Including the two that live outside the process: a key saved in this
    checkout, and a profile on the developer's machine. Without that, these
    tests would pass or fail depending on whether whoever ran them had
    connected Claude in the browser.
    """
    for var in (*KEY_VARS, "ANTHROPIC_BASE_URL"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(credentials, "saved", lambda root=None: None)
    monkeypatch.setattr(credentials, "profile", lambda: None)


def status_error(code: int, message: str = "nope") -> anthropic.APIStatusError:
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    response = httpx.Response(code, request=request, json={
        "error": {"type": "error", "message": message}})
    return anthropic.APIStatusError(message, response=response, body=None)


def agent(monkeypatch, replies=None, key: str = "sk-test", **kw) -> ClaudeAgent:
    """A Claude player whose one preflight request is answered locally."""
    unset_keys(monkeypatch)
    player = ClaudeAgent(api_key=key, **kw)
    calls: list[dict] = []

    async def create(**body):
        calls.append(body)
        outcome = replies() if callable(replies) else replies
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    monkeypatch.setattr(player._client.messages, "create", create)
    player.calls = calls   # type: ignore[attr-defined]
    return player


class TestCredentials:
    async def test_no_credentials_is_reported_as_itself(self, monkeypatch):
        unset_keys(monkeypatch)
        player = ClaudeAgent()

        async def explode(**_):
            raise AssertionError("must not reach the network without a key")

        monkeypatch.setattr(player._client.messages, "create", explode)
        problem = await player.preflight()
        assert problem
        # Both routes out, in the order they cost the reader anything: the
        # panel, which needs no restart, then the environment variable.
        assert "setup panel" in problem
        assert "ANTHROPIC_API_KEY" in problem

    async def test_the_message_says_the_rest_of_the_observatory_still_works(
        self, monkeypatch
    ):
        """A missing key is a missing player, not a broken instrument."""
        unset_keys(monkeypatch)
        problem = await ClaudeAgent().preflight()
        assert problem and "without one" in problem

    @pytest.mark.parametrize("var", KEY_VARS)
    async def test_either_environment_variable_counts_as_credentials(
        self, monkeypatch, var
    ):
        unset_keys(monkeypatch)
        monkeypatch.setenv(var, "sk-test")
        player = ClaudeAgent()
        monkeypatch.setattr(
            player._client.messages, "create", lambda **_: _ok()
        )
        assert await player.preflight() is None

    async def test_a_gateway_may_stand_in_for_a_key(self, monkeypatch):
        """Someone fronting the API with their own proxy has no key to set, and
        refusing to start would be wrong."""
        unset_keys(monkeypatch)
        monkeypatch.setenv("ANTHROPIC_BASE_URL", "http://gateway.local")
        player = ClaudeAgent()
        monkeypatch.setattr(player._client.messages, "create", lambda **_: _ok())
        assert await player.preflight() is None

    async def test_a_key_passed_in_beats_an_empty_environment(self, monkeypatch):
        player = agent(monkeypatch, replies=None)
        assert await player.preflight() is None


async def _ok():
    return object()


class TestWhatTheApiSays:
    async def test_a_good_key_and_model_starts_the_run(self, monkeypatch):
        player = agent(monkeypatch)
        assert await player.preflight() is None

    async def test_the_probe_asks_for_the_model_the_run_will_use(self, monkeypatch):
        """A key that works for one model and not another is a real situation,
        so the question has to be about the model actually requested."""
        player = agent(monkeypatch, model="claude-haiku-4-5")
        await player.preflight()
        assert player.calls[0]["model"] == "claude-haiku-4-5"
        assert player.calls[0]["max_tokens"] == 1

    @pytest.mark.parametrize("code", [401, 403])
    async def test_credentials_the_api_rejects_say_rejected(self, monkeypatch, code):
        player = agent(monkeypatch, replies=status_error(code, "invalid x-api-key"))
        problem = await player.preflight()
        assert problem and "refused these credentials" in problem
        assert str(code) in problem

    async def test_a_model_this_account_cannot_reach_says_so(self, monkeypatch):
        player = agent(
            monkeypatch, model="claude-fable-5-1", replies=status_error(404, "not found")
        )
        problem = await player.preflight()
        assert problem and "claude-fable-5-1" in problem
        assert "not available to this account" in problem

    async def test_any_other_status_is_passed_through_with_its_code(self, monkeypatch):
        player = agent(monkeypatch, replies=status_error(529, "overloaded"))
        problem = await player.preflight()
        assert problem and "529" in problem

    async def test_nothing_answering_is_not_a_crash(self, monkeypatch):
        request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
        player = agent(
            monkeypatch,
            replies=anthropic.APIConnectionError(request=request),
        )
        problem = await player.preflight()
        assert problem and "cannot reach" in problem

    async def test_an_sdk_that_cannot_build_the_request_is_reported(self, monkeypatch):
        """The original failure mode: the SDK raising TypeError from inside the
        call because it could not resolve an authentication method."""
        player = agent(
            monkeypatch,
            replies=TypeError("Could not resolve authentication method"),
        )
        problem = await player.preflight()
        assert problem and "TypeError" in problem


class TestTheRestOfTheWiring:
    def test_the_picker_s_models_all_have_a_price(self):
        """An unpriced model still runs, at the default rate — which means the
        cost readout quietly lies. Every model offered in the UI is listed."""
        from pathlib import Path
        import re

        from observatory.agents.claude_agent import PRICING

        html = Path("src/observatory/web/index.html").read_text(encoding="utf-8")
        block = html.split('id="model"')[1].split("</select>")[0]
        offered = re.findall(r'value="(claude-[^"]+)"', block)
        assert offered
        assert [m for m in offered if m not in PRICING] == []

    def test_haiku_is_not_sent_an_effort_it_does_not_take(self):
        assert takes_effort("claude-opus-5")
        assert not takes_effort("claude-haiku-4-5")

    async def test_the_handoff_route_refuses_before_taking_the_keyboard(
        self, monkeypatch
    ):
        """A handoff that fails preflight must leave the game with the player it
        already had — losing a live position to a missing key would be the
        worst version of this."""
        from observatory import server
        from observatory.agents.simple import ScriptedAgent
        from observatory.engine.mock_engine import MockEngine
        from observatory.events import EventBus
        from observatory.session import Session, SessionConfig

        unset_keys(monkeypatch)
        session = Session(
            MockEngine(), ScriptedAgent(["north"]), EventBus(),
            config=SessionConfig(delay=0.0),
        )
        await session.start()
        await session.step_once()
        monkeypatch.setattr(server.hub, "session", session)

        res = await server.handoff(server.Handoff(agent="claude"))
        assert res.status_code == 400
        assert b"ANTHROPIC_API_KEY" in res.body
        assert session.agent.name == "scripted"
        assert session.handoffs == []
