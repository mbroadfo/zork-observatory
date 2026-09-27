"""Where the Claude player's credential comes from, and what is never leaked.

Two things are worth pinning. The resolution order, because a reader who sets
an environment variable and then wonders why a stale saved key is being used
has been badly served. And the leak surface: no route, status or trace field
may carry the credential itself — a saved key is identified by its last four
characters, which tells two keys apart and nothing else.
"""

from __future__ import annotations

import json

import pytest

from observatory import credentials

KEY = "sk-ant-api03-" + "x" * 40
OTHER = "sk-ant-api03-" + "y" * 40


@pytest.fixture
def clean(monkeypatch, tmp_path):
    """No environment credentials, no profile, and a scratch checkout."""
    for var in (credentials.ENV_KEY, credentials.ENV_TOKEN,
                credentials.ENV_BASE_URL, credentials.ENV_KEY_FILE):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(credentials, "profile", lambda: None)
    return tmp_path


class TestResolutionOrder:
    def test_nothing_configured_says_so(self, clean):
        source, key = credentials.resolve(clean)
        assert (source, key) == ("none", None)
        assert credentials.status(clean)["present"] is False

    def test_the_environment_wins(self, clean, monkeypatch):
        """It is the documented mechanism and the one a reader can see. A key
        saved months ago quietly overriding it would be indefensible."""
        credentials.save(KEY, clean)
        monkeypatch.setenv(credentials.ENV_KEY, "sk-ant-from-the-environment")

        source, key = credentials.resolve(clean)
        assert source == credentials.ENV_KEY
        # None: the SDK reads the variable itself, so nothing is passed in.
        assert key is None

    def test_a_bearer_token_counts(self, clean, monkeypatch):
        monkeypatch.setenv(credentials.ENV_TOKEN, "oauth-bearer-token")
        assert credentials.resolve(clean)[0] == credentials.ENV_TOKEN

    def test_a_saved_key_is_used_when_the_environment_is_empty(self, clean):
        credentials.save(KEY, clean)
        source, key = credentials.resolve(clean)
        assert source == "saved" and key == KEY

    def test_a_profile_on_disk_is_left_to_the_sdk(self, clean, monkeypatch):
        """The SDK refreshes an OAuth profile itself. Reading the token out and
        passing it in would break exactly that."""
        monkeypatch.setattr(credentials, "profile", lambda: "CredentialsFile")
        source, key = credentials.resolve(clean)
        assert source == "profile" and key is None

    def test_a_saved_key_beats_a_profile(self, clean, monkeypatch):
        monkeypatch.setattr(credentials, "profile", lambda: "CredentialsFile")
        credentials.save(KEY, clean)
        assert credentials.resolve(clean)[0] == "saved"

    def test_a_gateway_is_treated_as_configured(self, clean, monkeypatch):
        monkeypatch.setenv(credentials.ENV_BASE_URL, "http://gateway.local")
        status = credentials.status(clean)
        assert status["present"] is True
        assert "gateway" in status["detail"]

    def test_the_key_file_can_be_moved(self, clean, monkeypatch, tmp_path):
        elsewhere = tmp_path / "vault" / "key.json"
        monkeypatch.setenv(credentials.ENV_KEY_FILE, str(elsewhere))
        credentials.save(KEY, clean)
        assert elsewhere.exists()
        assert credentials.resolve(clean)[1] == KEY

    def test_a_profile_lookup_that_explodes_is_not_fatal(self, monkeypatch):
        """Whatever the SDK makes of a malformed profile on disk, the page has to
        render and the local models have to stay playable."""
        import anthropic

        def boom(**_):
            raise anthropic.CredentialsError("configs/default.json is malformed")

        monkeypatch.setattr(anthropic, "default_credentials", boom)
        assert credentials.profile() is None


class TestAnEmptyVariableIsNotACredential:
    """`${ANTHROPIC_BASE_URL:-}` in a compose file, an unset k8s secret and a
    blank line in a .env all produce a variable that exists and is empty. None
    of them mean "use the empty string", and the SDK takes an empty base URL as
    the base URL — then builds a request with no protocol and calls the result a
    connection error, which sends the reader to inspect their network."""

    def test_an_empty_base_url_is_dropped(self, clean, monkeypatch):
        monkeypatch.setenv(credentials.ENV_BASE_URL, "")
        assert credentials.sanitize_environment() == [credentials.ENV_BASE_URL]
        assert credentials.ENV_BASE_URL not in __import__("os").environ

    def test_a_whitespace_only_key_is_dropped(self, clean, monkeypatch):
        monkeypatch.setenv(credentials.ENV_KEY, "   ")
        assert credentials.ENV_KEY in credentials.sanitize_environment()
        assert credentials.resolve(clean)[0] == "none"

    def test_a_real_value_is_left_alone(self, clean, monkeypatch):
        monkeypatch.setenv(credentials.ENV_BASE_URL, "https://gateway.local")
        assert credentials.sanitize_environment() == []
        import os

        assert os.environ[credentials.ENV_BASE_URL] == "https://gateway.local"

    def test_nothing_set_is_nothing_to_do(self, clean):
        assert credentials.sanitize_environment() == []


class TestSaving:
    def test_a_saved_key_round_trips(self, clean):
        result = credentials.save(KEY, clean)
        assert result["ok"] is True
        assert credentials.saved(clean) == KEY

    def test_saving_twice_replaces(self, clean):
        credentials.save(KEY, clean)
        credentials.save(OTHER, clean)
        assert credentials.saved(clean) == OTHER

    def test_a_truncated_paste_is_refused(self, clean):
        result = credentials.save("sk-ant-", clean)
        assert result["ok"] is False and "truncated" in result["detail"]
        assert credentials.saved(clean) is None

    def test_a_paste_with_a_newline_in_it_is_refused(self, clean):
        result = credentials.save(KEY + "\nsome other line", clean)
        assert result["ok"] is False and "whitespace" in result["detail"]

    def test_surrounding_whitespace_is_forgiven(self, clean):
        assert credentials.save(f"  {KEY}\n", clean)["ok"] is True
        assert credentials.saved(clean) == KEY

    def test_nothing_is_created_merely_by_asking(self, clean):
        credentials.status(clean)
        credentials.resolve(clean)
        assert list(clean.iterdir()) == []

    def test_forgetting_removes_it(self, clean):
        credentials.save(KEY, clean)
        assert credentials.forget(clean)["ok"] is True
        assert credentials.saved(clean) is None

    def test_forgetting_nothing_is_not_an_error(self, clean):
        assert credentials.forget(clean)["ok"] is True

    def test_a_corrupt_file_reads_as_no_key(self, clean):
        path = credentials.secret_path(clean)
        path.parent.mkdir(parents=True)
        path.write_text("{not json", encoding="utf-8")
        assert credentials.saved(clean) is None
        assert credentials.status(clean)["present"] is False


class TestNothingLeaks:
    def test_the_status_carries_four_characters_and_no_more(self, clean):
        credentials.save(KEY, clean)
        status = credentials.status(clean)
        blob = json.dumps(status)
        assert KEY not in blob
        assert status["tail"] == "…" + KEY[-4:]

    def test_two_keys_can_be_told_apart(self, clean):
        credentials.save(KEY, clean)
        first = credentials.status(clean)["tail"]
        credentials.save(OTHER, clean)
        assert credentials.status(clean)["tail"] != first

    def test_a_run_records_which_path_it_used_not_the_key(self, clean):
        from observatory.agents.claude_agent import ClaudeAgent

        credentials.save(KEY, clean)
        described = ClaudeAgent(api_key=KEY).describe()
        assert described["credential_source"] == "explicit"
        assert KEY not in json.dumps(described)


class TestTheRoutes:
    async def test_the_status_route_says_not_connected(self, clean, monkeypatch):
        from observatory import server

        monkeypatch.setattr(server, "ASSET_ROOT", clean)
        body = json.loads((await server.credentials_status()).body)
        assert body["present"] is False
        assert body["console_url"].startswith("https://console.anthropic.com")

    async def test_a_malformed_key_is_refused_without_a_request(
        self, clean, monkeypatch
    ):
        from observatory import server
        from observatory.agents import claude_agent

        monkeypatch.setattr(server, "ASSET_ROOT", clean)
        monkeypatch.setattr(
            claude_agent.ClaudeAgent, "preflight",
            lambda self: pytest.fail("must not call the API for a bad paste"),
        )
        res = await server.credentials_save(server.SaveKey(key="sk-ant"))
        assert res.status_code == 400
        assert credentials.saved(clean) is None

    async def test_a_key_the_api_rejects_is_not_saved(self, clean, monkeypatch):
        """Saving first and validating never would mean the reader finds out on
        turn one, which is the failure this whole panel exists to remove."""
        from observatory import server
        from observatory.agents import claude_agent

        monkeypatch.setattr(server, "ASSET_ROOT", clean)

        async def refuse(self):
            return "the Anthropic API refused these credentials (401)"

        monkeypatch.setattr(claude_agent.ClaudeAgent, "preflight", refuse)
        res = await server.credentials_save(server.SaveKey(key=KEY))
        assert res.status_code == 400
        assert b"401" in res.body
        assert credentials.saved(clean) is None

    async def test_a_key_that_works_is_saved_and_reported(self, clean, monkeypatch):
        from observatory import server
        from observatory.agents import claude_agent

        monkeypatch.setattr(server, "ASSET_ROOT", clean)

        async def accept(self):
            return None

        monkeypatch.setattr(claude_agent.ClaudeAgent, "preflight", accept)
        res = await server.credentials_save(server.SaveKey(key=KEY))
        body = json.loads(res.body)
        assert body["ok"] is True and body["source"] == "saved"
        assert KEY not in res.body.decode()
        assert credentials.saved(clean) == KEY

    async def test_forgetting_through_the_route(self, clean, monkeypatch):
        from observatory import server

        monkeypatch.setattr(server, "ASSET_ROOT", clean)
        credentials.save(KEY, clean)
        body = json.loads((await server.credentials_forget()).body)
        assert body["ok"] is True and body["present"] is False

    async def test_a_saved_key_makes_a_claude_run_launchable(
        self, clean, monkeypatch
    ):
        """The point of the whole exercise: paste, and the next run works with
        no restart and nothing exported."""
        from observatory.agents.claude_agent import ClaudeAgent

        # The agent resolves against the process's own working directory, so
        # point the override at the scratch copy rather than fake the lookup.
        monkeypatch.setenv(credentials.ENV_KEY_FILE, str(clean / "key.json"))
        credentials.save(KEY, clean)
        player = ClaudeAgent()
        assert player.credential_source == "saved"
        assert player._client.api_key == KEY
