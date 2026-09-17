"""Runs chained through a notebook.

A series is only an experiment if the notebook is the one thing that crosses
between runs. Pinned here: it is written as it grows, handed over whole at the
start of the next run, kept per player, set aside rather than deleted, and the
last word of a run is asked for in terms that say the next reader starts from
the very beginning.
"""

from __future__ import annotations

import json

import pytest

from observatory import cli, server
from observatory.agents import memory
from observatory.agents.memory import AgentMemory, Lesson
from observatory.agents.ollama_agent import OllamaAgent
from observatory.engine.mock_engine import MockEngine
from observatory.events import EventBus
from observatory.notebook import Notebook, slug
from observatory.session import Session, SessionConfig

from test_ollama import FakeOllama, reply


class Reflective(FakeOllama):
    """Moves when asked for a move; a numbered note when asked to reflect."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.notes = 0

    async def __call__(self, method, path, body):
        if path == "/api/chat" and "format" not in body:
            self.sent.append(dict(body))
            self.notes += 1
            return {"message": {"content": f"Note {self.notes}: try the other way."}, "eval_count": 8}
        return await super().__call__(method, path, body)


def book(tmp_path, agent="ollama:qwen3:8b", story="88-840726", fresh=False) -> Notebook:
    return Notebook.open(tmp_path, story, agent, fresh=fresh)


class TestTheFile:
    def test_one_per_player_per_story(self, tmp_path):
        a = book(tmp_path)
        assert a.path == tmp_path / "88-840726" / "ollama-qwen3-8b.json"
        assert book(tmp_path, agent="ollama:qwen3:8b/cold").path != a.path
        assert book(tmp_path, story="other").path != a.path
        assert slug("  ") == "unnamed"

    def test_written_as_it_grows_and_read_back_whole(self, tmp_path):
        a = book(tmp_path)
        a.begin_run(AgentMemory(), session="s1")
        a.add(Lesson(text="The trees close in to the north.", turn=40, kind="run", run=1))
        a.end_run(reason="turn budget exhausted", censored=True, score=10)

        again = book(tmp_path)
        assert [lesson.text for lesson in again.lessons] == ["The trees close in to the north."]
        assert again.runs[0]["score"] == 10 and again.runs[0]["censored"] is True
        assert json.loads(a.path.read_text())["agent"] == "ollama:qwen3:8b"
        assert not a.path.with_suffix(".tmp").exists()

    def test_a_new_notebook_sets_the_old_one_aside(self, tmp_path):
        a = book(tmp_path)
        a.add(Lesson(text="old belief", turn=1))
        fresh = book(tmp_path, fresh=True)
        assert fresh.lessons == []
        archived = [p for p in a.path.parent.iterdir() if p != a.path]
        assert len(archived) == 1
        assert "old belief" in archived[0].read_text()

    def test_unknown_fields_in_an_old_file_are_ignored(self, tmp_path):
        a = book(tmp_path)
        a.path.parent.mkdir(parents=True)
        a.path.write_text(json.dumps({"lessons": [{"text": "x", "turn": 1, "mood": "grim"}], "runs": []}))
        assert book(tmp_path).lessons[0].text == "x"

    def test_a_run_starts_with_everything_written_so_far(self, tmp_path):
        a = book(tmp_path)
        for i in range(3):
            a.add(Lesson(text=f"n{i}", turn=i, run=1))
        mem = AgentMemory()
        assert a.begin_run(mem) == 1
        assert [lesson.text for lesson in mem.lessons] == ["n0", "n1", "n2"]
        # A copy: the run adds through the notebook, not behind its back.
        mem.lessons.append(Lesson(text="stray", turn=9))
        assert len(a.lessons) == 3
        assert "[run 1, turn 0]" in AgentMemory.render(mem)

    def test_summary(self, tmp_path):
        a = book(tmp_path)
        a.begin_run(AgentMemory())
        a.end_run(reason="death", score=5, max_score=350, turns=12, deaths=1, censored=False)
        s = a.summary()
        assert (s["runs"], s["notes"]) == (1, 0)
        assert s["history"][0]["reason"] == "death"


def run_session(tmp_path, fake, notebook=True, **cfg) -> tuple[Session, OllamaAgent, list]:
    a = OllamaAgent("qwen3:8b", transport=fake)
    bus = EventBus()
    seen: list = []
    bus.subscribe(seen.append)
    nb = book(tmp_path, agent=a.name) if notebook else None
    config = dict(delay=0.0, max_turns=3)
    config.update(cfg)
    return Session(MockEngine(), a, bus, config=SessionConfig(**config), notebook=nb), a, seen


class TestTheLastWord:
    async def test_a_run_with_a_notebook_ends_with_a_note_to_the_next_run(self, tmp_path):
        fake = Reflective()
        session, a, seen = run_session(tmp_path, fake)
        await session.run()

        asked = fake.sent[-1]["messages"][1]["content"]
        assert memory.REFLECT_RESTART in asked
        lesson = a.memory.lessons[-1]
        assert (lesson.kind, lesson.run, lesson.text) == ("run", 1, "Note 1: try the other way.")
        on_disk = book(tmp_path, agent=a.name)
        assert on_disk.lessons[-1].text == lesson.text
        assert on_disk.runs[0]["reason"] == "turn budget exhausted"
        assert on_disk.runs[0]["censored"] is True
        ended = next(e for e in seen if e.type == "session.ended")
        assert ended.payload["notebook"]["notes"] == 1

    async def test_without_a_notebook_there_is_no_last_word(self, tmp_path):
        fake = Reflective()
        session, a, _ = run_session(tmp_path, fake, notebook=False)
        await session.run()
        assert fake.notes == 0 and a.memory.lessons == []

    async def test_the_end_says_nothing_about_why(self):
        """Budget, death and victory get the same words."""
        for word in ("budget", "died", "death", "won", "turns"):
            assert word not in memory.REFLECT_RESTART.lower()

    async def test_a_death_mid_run_still_reflects_toward_the_checkpoint(self, tmp_path):
        fake = Reflective(["north", "east", "west", "up", "look", "look", "look"])
        session, a, _ = run_session(tmp_path, fake, max_turns=8, lives=1, checkpoint_every=2)
        await session.run()
        prompts = [b["messages"][1]["content"] for b in fake.sent if "format" not in b]
        assert memory.REFLECT in prompts[0]
        assert memory.REFLECT_RESTART in prompts[-1]
        assert [lesson.kind for lesson in a.memory.lessons] == ["death", "run"]

    async def test_an_ironman_death_is_the_last_word_and_is_not_asked_twice(self, tmp_path):
        fake = Reflective(["north", "east", "west", "up", "look"])
        session, a, _ = run_session(tmp_path, fake, max_turns=8)
        await session.run()
        assert fake.notes == 1
        assert memory.REFLECT_RESTART in fake.sent[-1]["messages"][1]["content"]

    async def test_the_next_run_reads_what_the_last_one_wrote(self, tmp_path):
        first, _, _ = run_session(tmp_path, Reflective())
        await first.run()
        fake = Reflective()
        second, a, seen = run_session(tmp_path, fake)
        await second.run()
        assert second.run_number == 2
        first_move = fake.sent[0]["messages"][1]["content"]
        assert "Note 1: try the other way." in first_move
        assert "run 1," in first_move
        started = next(e for e in seen if e.type == "session.started")
        assert started.payload["notebook"]["runs"] == 2


@pytest.fixture
def hub(monkeypatch, tmp_path):
    fakes: list[Reflective] = []

    def build(kind, **kw):
        fakes.append(Reflective())
        return OllamaAgent("qwen3:8b", transport=fakes[-1], recall=kw.get("recall", "transcript"),
                           history_turns=kw["history_turns"])

    monkeypatch.setattr(server, "build_agent", build)
    monkeypatch.setattr(server, "NOTEBOOK_DIR", tmp_path / "notebooks")
    monkeypatch.setattr(server, "SERIES_PAUSE_S", 0.0)
    return fakes   # each test tears the hub down itself, inside its own loop


class TestASeriesOnTheServer:
    async def test_runs_chain_and_carry_only_the_notebook(self, hub, tmp_path):
        req = server.NewSession(engine="mock", agent="ollama", model="qwen3:8b", max_turns=2,
                                delay=0.0, record=False, runs=3, notebook="new")
        res = await server.new_session(req)
        assert res.status_code == 200
        try:
            first = server.hub.session
            assert first.series == {"index": 1, "total": 3}
            await first.start_background()
            for _ in range(2):
                await server.hub._series_task
                await server.hub.session._task
            last = server.hub.session
            assert last is not first
            assert last.series == {"index": 3, "total": 3}
            assert last.finished and last.run_number == 3
            # The third player was built fresh and read two notes.
            assert len(hub) == 3
            first_move = hub[2].sent[0]["messages"][1]["content"]
            assert "Note 1" in first_move
            nb = last.notebook
            assert [r["run"] for r in nb.runs] == [1, 2, 3]
            assert len(nb.lessons) == 3
        finally:
            await server.hub.teardown()

    async def test_new_is_only_new_for_the_first_run(self, hub, tmp_path):
        req = server.NewSession(engine="mock", agent="ollama", model="qwen3:8b", max_turns=1,
                                delay=0.0, record=False, runs=2, notebook="new")
        await server.new_session(req)
        try:
            assert server.hub.series.request.notebook == "carry"
        finally:
            await server.hub.teardown()

    async def test_starting_something_else_stops_the_series(self, hub):
        req = server.NewSession(engine="mock", agent="ollama", model="qwen3:8b", max_turns=1,
                                delay=0.0, record=False, runs=5, notebook="carry")
        await server.new_session(req)
        await server.hub.teardown()
        assert server.hub.series is None and server.hub.session is None

    async def test_a_bad_mode_is_refused(self, hub):
        req = server.NewSession(engine="mock", agent="ollama", model="qwen3:8b", notebook="sometimes")
        res = await server.new_session(req)
        assert res.status_code == 400

    async def test_the_episodic_window_default_reaches_the_session(self, hub):
        req = server.NewSession(engine="mock", agent="ollama", model="qwen3:8b", record=False,
                                recall="episodic")
        await server.new_session(req)
        try:
            assert server.hub.session.config.history_turns == 1
            assert server.hub.session.agent.name.endswith("+episodic")
        finally:
            await server.hub.teardown()


class TestASeriesFromTheTerminal:
    def test_play_chains_runs_through_one_notebook(self, monkeypatch, tmp_path, capsys):
        fakes: list[Reflective] = []

        def build(kind, **kw):
            fakes.append(Reflective())
            return OllamaAgent("qwen3:8b", transport=fakes[-1], history_turns=kw["history_turns"])

        monkeypatch.setattr(cli, "build_agent", build)
        monkeypatch.chdir(tmp_path)
        code = cli.main(["play", "--engine", "mock", "--agent", "ollama", "--model", "qwen3:8b",
                         "--turns", "2", "--runs", "2", "--notebook", "new"])
        assert code == 0
        out = capsys.readouterr().out
        assert "run 1/2" in out and "run 2/2" in out
        assert "1 note(s) from 1 earlier run(s)" in out
        files = list((tmp_path / "traces" / "notebooks").rglob("*.json"))
        assert len(files) == 1
        data = json.loads(files[0].read_text())
        assert [r["run"] for r in data["runs"]] == [1, 2]
        assert len(fakes) == 2


_ = reply
