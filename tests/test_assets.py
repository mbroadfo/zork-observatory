"""The two files this project cannot ship.

What is pinned here is the part that would be embarrassing to get wrong: a
download is written only if it is the exact file the project was measured
against, an existing copy is never overwritten, and a manifest the web setup
panel and the terminal tool both read cannot drift between them because there
is only one of it.
"""

from __future__ import annotations

import hashlib

import pytest

from observatory import assets


def blob(n: int = 64) -> bytes:
    return bytes(range(256)) * n


def fake(asset, payload: bytes):
    """A fetcher that returns `payload` whatever was asked for."""
    return lambda _asset: payload


@pytest.fixture
def one(tmp_path, monkeypatch):
    """A single asset pointing into tmp_path, matching `blob()`."""
    payload = blob()
    asset = assets.Asset(
        key="rom",
        relpath="roms/test.z5",
        url="https://example.invalid/test.z5",
        md5=hashlib.md5(payload).hexdigest(),
        size=len(payload),
        what="a test story file",
        source="nowhere",
        source_url="https://example.invalid/",
    )
    monkeypatch.setattr(assets, "ASSETS", (asset,))
    monkeypatch.setattr(assets, "BY_KEY", {asset.key: asset})
    return asset, payload, tmp_path


class TestTheManifest:
    def test_the_real_manifest_names_both_files_and_their_hashes(self):
        keys = {a.key for a in assets.ASSETS}
        assert keys == {"rom", "map"}
        for asset in assets.ASSETS:
            assert len(asset.md5) == 32 and asset.size > 0
            assert asset.url.startswith("https://")
            assert asset.source_url.startswith("https://")

    def test_only_the_story_file_is_required(self):
        """The chart is a nicety; a run without it draws its own graph."""
        assert assets.BY_KEY["rom"].required is True
        assert assets.BY_KEY["map"].required is False

    def test_by_key_and_assets_are_the_same_list(self):
        assert set(assets.BY_KEY) == {a.key for a in assets.ASSETS}
        assert all(assets.BY_KEY[a.key] is a for a in assets.ASSETS)


class TestWhatIsOnDisk:
    def test_missing_says_missing(self, one):
        asset, _, root = one
        assert assets.status(asset, root) == {**assets.status(asset, root),
                                              "present": False, "detail": "missing"}

    def test_the_right_file_is_recognised(self, one):
        asset, payload, root = one
        asset.path(root).parent.mkdir(parents=True)
        asset.path(root).write_bytes(payload)
        state = assets.status(asset, root)
        assert state["present"] is True and state["detail"] == "ok"

    def test_a_different_release_is_named_as_such(self, one):
        asset, payload, root = one
        asset.path(root).parent.mkdir(parents=True)
        asset.path(root).write_bytes(payload + b"x")
        state = assets.status(asset, root)
        assert state["present"] is False and "wrong size" in state["detail"]

    def test_same_size_wrong_bytes_is_caught_by_the_hash(self, one):
        asset, payload, root = one
        asset.path(root).parent.mkdir(parents=True)
        asset.path(root).write_bytes(b"\x00" * len(payload))
        state = assets.status(asset, root)
        assert state["present"] is False and "wrong file" in state["detail"]

    def test_the_survey_says_whether_a_real_game_can_be_played(self, one):
        asset, payload, root = one
        assert assets.survey(root)["ready"] is False
        asset.path(root).parent.mkdir(parents=True)
        asset.path(root).write_bytes(payload)
        assert assets.survey(root)["ready"] is True


class TestFetching:
    def test_a_verified_download_is_written(self, one):
        asset, payload, root = one
        result = assets.fetch(asset, root, fetcher=fake(asset, payload))
        assert result["ok"] is True
        assert asset.path(root).read_bytes() == payload

    def test_a_download_that_does_not_match_is_not_written(self, one):
        """The walkthrough holds for one release and the chart's room boxes are
        pixels of one scan. Approximately the right file is the failure this
        check exists to catch, not something to warn about and keep."""
        asset, _, root = one
        result = assets.fetch(asset, root, fetcher=fake(asset, b"something else"))
        assert result["ok"] is False and "refused" in result["detail"]
        assert not asset.path(root).exists()

    def test_an_existing_correct_copy_is_left_alone(self, one):
        asset, payload, root = one
        asset.path(root).parent.mkdir(parents=True)
        asset.path(root).write_bytes(payload)

        def explode(_asset):
            raise AssertionError("should not have downloaded anything")

        assert assets.fetch(asset, root, fetcher=explode)["ok"] is True

    def test_someone_else_s_copy_is_never_overwritten(self, one):
        """It may be a release they meant to keep, and "yours is a different
        build" is worth telling apart from "the download failed"."""
        asset, _, root = one
        asset.path(root).parent.mkdir(parents=True)
        asset.path(root).write_bytes(b"a copy someone chose")

        result = assets.fetch(asset, root, fetcher=fake(asset, blob()))
        assert result["ok"] is False and "move it aside" in result["detail"]
        assert asset.path(root).read_bytes() == b"a copy someone chose"

    def test_a_network_failure_is_reported_not_raised(self, one):
        asset, _, root = one

        def refuse(_asset):
            raise ConnectionError("no route to host")

        result = assets.fetch(asset, root, fetcher=refuse)
        assert result["ok"] is False and "could not fetch" in result["detail"]
        assert not asset.path(root).exists()


class TestTheServerOffersIt:
    async def test_the_survey_is_served(self, monkeypatch, tmp_path):
        from observatory import server

        monkeypatch.setattr(server, "ASSET_ROOT", tmp_path)
        res = await server.assets_status()
        import json

        body = json.loads(res.body)
        assert {a["key"] for a in body["assets"]} == {"rom", "map"}
        assert body["ready"] is False

    async def test_an_unknown_asset_is_refused(self, monkeypatch, tmp_path):
        from observatory import server

        monkeypatch.setattr(server, "ASSET_ROOT", tmp_path)
        res = await server.assets_fetch(server.FetchAssets(key="everything"))
        assert res.status_code == 400

    async def test_nothing_is_fetched_without_a_request(self, monkeypatch, tmp_path):
        """No import, no startup hook, no session creation touches this. The
        only path to a download is the route below."""
        from observatory import server

        monkeypatch.setattr(server, "ASSET_ROOT", tmp_path)
        monkeypatch.setattr(assets, "http_fetch", lambda a: pytest.fail("downloaded"))
        await server.state()
        assert not any(tmp_path.iterdir())
