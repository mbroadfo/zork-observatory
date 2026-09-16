"""The chart: identifying the story build, and the atlas that depends on it."""

from __future__ import annotations

import json

import pytest

from observatory import server
from observatory.engine.base import story_id
from observatory.engine.mock_engine import MockEngine

ATLAS = server.WEB_DIR / "atlas" / "zork1-r88.json"


def header(version=3, release=88, serial=b"840726", checksum=0xA129) -> bytes:
    data = bytearray(0x40)
    data[0] = version
    data[2:4] = release.to_bytes(2, "big")
    data[0x12:0x18] = serial
    data[0x1C:0x1E] = checksum.to_bytes(2, "big")
    return bytes(data)


class TestStoryId:
    def test_reads_release_serial_and_checksum(self):
        assert story_id(header()) == "88-840726-a129"

    def test_different_releases_are_different_stories(self):
        """Object numbers move between releases; so must the key."""
        assert story_id(header(release=119, serial=b"880429")) != story_id(header())

    def test_not_a_story_file(self):
        assert story_id(b"PK\x03\x04" + bytes(60)) is None
        assert story_id(b"short") is None

    def test_the_mock_has_no_story(self):
        assert MockEngine().story is None


@pytest.fixture(scope="module")
def atlas():
    return json.loads(ATLAS.read_text(encoding="utf-8"))


class TestAtlasFile:
    def test_keyed_to_the_zork_release_it_was_measured_on(self, atlas):
        assert atlas["story"] == "88-840726-a129"

    def test_every_room_of_the_release_is_charted(self, atlas):
        # Zork I r88 has 110 rooms under object #82. The chart has one box each.
        assert len(atlas["rooms"]) == 110

    def test_boxes_are_well_formed_and_on_the_scan(self, atlas):
        for num, room in atlas["rooms"].items():
            assert num.isdigit(), num
            x0, y0, x1, y1 = room["box"]
            assert 0 <= x0 < x1 <= atlas["width"], num
            assert 0 <= y0 < y1 <= atlas["height"], num

    def test_passages_join_charted_rooms_and_stay_on_the_scan(self, atlas):
        assert len(atlas["paths"]) > 100
        for key, pts in atlas["paths"].items():
            a, b = key.split("|")
            # The browser builds the key with a string sort; so must the tracer.
            assert key == "|".join(sorted((a, b)))
            assert a in atlas["rooms"] and b in atlas["rooms"], key
            assert len(pts) >= 2, key
            for x, y in pts:
                assert 0 <= x <= atlas["width"] and 0 <= y <= atlas["height"], key

    def test_stubs_are_not_traced(self, atlas):
        for key in atlas["stubs"]:
            assert key not in atlas["paths"]

    def test_repeated_names_are_told_apart(self, atlas):
        names = [r["name"] for r in atlas["rooms"].values()]
        assert len(names) == len(set(names))


class TestLookup:
    def test_unknown_or_missing_story_finds_nothing(self):
        assert server.find_atlas("") is None
        assert server.find_atlas("1-000000-0000") is None

    def test_a_match_without_the_image_is_no_match(self, tmp_path, monkeypatch):
        """The scan is supplied locally; a checkout without it falls back to the graph."""
        (tmp_path / "atlas").mkdir()
        (tmp_path / "atlas" / "x.json").write_text(
            json.dumps({"story": "1-2-3", "image": "absent.jpg", "rooms": {}})
        )
        monkeypatch.setattr(server, "WEB_DIR", tmp_path)
        assert server.find_atlas("1-2-3") is None

    def test_a_match_with_the_image_is_served(self, tmp_path, monkeypatch):
        (tmp_path / "atlas").mkdir()
        (tmp_path / "atlas" / "x.json").write_text(
            json.dumps({"story": "1-2-3", "image": "map.jpg", "rooms": {}})
        )
        (tmp_path / "atlas" / "map.jpg").write_bytes(b"\xff\xd8")
        monkeypatch.setattr(server, "WEB_DIR", tmp_path)
        found = server.find_atlas("1-2-3")
        assert found and found["image_url"] == "/static/atlas/map.jpg"

    def test_session_started_carries_the_story(self):
        import asyncio

        from observatory.agents.simple import ScriptedAgent
        from observatory.events import EventBus
        from observatory.session import Session, SessionConfig

        bus = EventBus()
        session = Session(MockEngine(), ScriptedAgent([]), bus, config=SessionConfig(delay=0))
        asyncio.run(session.start())
        started = next(e for e in bus.backlog if e.type == "session.started")
        assert "story" in started.payload
