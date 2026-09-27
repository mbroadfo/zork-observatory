"""The observatory server.

One process, one active session — this is a local instrument, not a service.
Browsers connect over a WebSocket and receive the backlog before the live
stream, so a tab opened at turn 200 renders the whole run.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import assets, credentials
from .agents import DEFAULT_HISTORY, DEFAULT_NUM_CTX, DEFAULT_TEMPERATURE, build_agent, parse_think
from .notebook import MODES as NOTEBOOK_MODES, Notebook
from .agents.simple import HumanAgent
from .engine import build_engine, engine_seed
from .events import Event, EventBus
from .session import Session, SessionConfig
from .trace import TraceWriter, read_events, trace_header

WEB_DIR = Path(__file__).parent / "web"
TRACE_DIR = Path("traces")
# Where roms/ and assets/ live, relative to wherever the server was started —
# the same convention TRACE_DIR uses, and /app in the container.
ASSET_ROOT = Path(".")


SERIES_PAUSE_S = 2.0
NOTEBOOK_DIR = TRACE_DIR / "notebooks"
MAX_SERIES = 100


@dataclass
class SeriesPlan:
    request: "NewSession"
    total: int
    index: int = 1


class LaunchError(Exception):
    pass


class Hub:
    """Fans the event bus out to every connected browser."""

    def __init__(self) -> None:
        self.bus = EventBus()
        self.session: Session | None = None
        self._queues: list[asyncio.Queue[dict[str, Any]]] = []
        self._replay_task: asyncio.Task | None = None
        # A chain of runs: what to build next, and how many are left.
        self.series: SeriesPlan | None = None
        self._series_task: asyncio.Task | None = None
        self.bus.subscribe(self._on_event)

    def _on_event(self, event: Event) -> None:
        if (
            event.type == "session.ended"
            and self.series is not None
            and self.session is not None
            and event.payload.get("session_id") == self.session.id
            and self.series.index < self.series.total
        ):
            self._series_task = asyncio.get_running_loop().create_task(self._next_in_series(self.session))
        payload = event.to_dict()
        for q in list(self._queues):
            with contextlib.suppress(asyncio.QueueFull):
                q.put_nowait(payload)

    def connect(self) -> asyncio.Queue[dict[str, Any]]:
        q: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=10_000)
        self._queues.append(q)
        return q

    def disconnect(self, q: asyncio.Queue[dict[str, Any]]) -> None:
        if q in self._queues:
            self._queues.remove(q)

    def backlog(self) -> list[dict[str, Any]]:
        return [e.to_dict() for e in self.bus.backlog]

    async def _next_in_series(self, previous: Session) -> None:
        """Start the next run once the last has finished speaking.

        A fresh engine and a fresh agent every time: the notebook is the only
        thing that carries, so it is the only thing that can explain a change.
        """
        plan = self.series
        assert plan is not None
        await asyncio.sleep(SERIES_PAUSE_S)   # let a watcher read the result
        if self.session is not previous or self.series is not plan:
            return   # someone started something else meanwhile
        if previous._task and not previous._task.done() and previous._task is not asyncio.current_task():
            with contextlib.suppress(asyncio.CancelledError):
                await previous._task
        keep_running = not previous.paused
        previous.engine.close()
        self.session = None
        self.bus.clear()

        plan.index += 1
        try:
            session = await launch(plan.request, series={"index": plan.index, "total": plan.total})
        except LaunchError as exc:
            self.series = None
            self.bus.emit("error", where="series", message=str(exc))
            return
        self.session = session
        await session.start()
        if keep_running:
            session.resume()
            session.start_background()

    async def teardown(self) -> None:
        self.series = None
        if self._series_task and not self._series_task.done() and self._series_task is not asyncio.current_task():
            self._series_task.cancel()
        if self._replay_task and not self._replay_task.done():
            self._replay_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._replay_task
        if self.session:
            self.session.pause()
            if self.session._task and not self.session._task.done():
                self.session._task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await self.session._task
            self.session.engine.close()
        self.session = None
        self.bus.clear()


hub = Hub()
app = FastAPI(title="Zork Observatory")

# Before any agent is built: an Anthropic variable set to the empty string means
# "unset" everywhere it comes from, and means something much worse to the SDK.
credentials.sanitize_environment()


# --- request models ------------------------------------------------------


class NewSession(BaseModel):
    engine: str = "mock"
    rom: str | None = None
    agent: str = "random"
    model: str = "claude-opus-5"
    effort: str = "medium"
    think: str = "default"       # ollama: default | off | on | low | medium | high
    info_level: str = "parser"
    max_turns: int = 400
    max_cost_usd: float = 0.0    # 0 = no ceiling
    stall_limit: int = 0         # end a run that stops finding new rooms
    max_tokens: int = 0          # 0 = the agent's own default
    lives: int = 0
    delay: float = 0.35
    seed: int = 12345
    num_ctx: int = DEFAULT_NUM_CTX   # ollama: smaller keeps a big model on the GPU
    temperature: float = DEFAULT_TEMPERATURE   # ollama: lower wanders less
    history_turns: int = DEFAULT_HISTORY   # exchanges of transcript per turn
    journal: bool = False                  # let the agent keep a record of its own
    search: bool = False                   # let it look back past its window
    valid_actions: bool = False
    record: bool = True
    runs: int = 1                # chained runs; each starts from the first move
    notebook: str = "off"        # off | carry | new — what crosses between runs


class Control(BaseModel):
    action: str  # run | pause | step


class Handoff(BaseModel):
    """Who takes the keyboard next, on the run already in progress."""

    agent: str = "ollama"
    model: str = ""
    effort: str = "medium"
    think: str = "default"
    info_level: str = "parser"
    num_ctx: int = DEFAULT_NUM_CTX
    temperature: float = DEFAULT_TEMPERATURE
    history_turns: int = DEFAULT_HISTORY
    journal: bool = False
    # Whether it may look back through everything typed and printed, including
    # the turns it did not play — unless `inherit_transcript` withheld them, in
    # which case the search is bounded the same way the window is.
    search: bool = True
    notebook: str = "off"
    # Whether the new player is shown the turns it did not play. See
    # Session.handoff — this is the difference between a demonstration and a
    # measurement.
    inherit_transcript: bool = True
    add_turns: int = 200         # budget from here, not from turn zero
    # Ceilings for the incoming player. The cost one spans the whole run, not
    # this leg: a budget that reset on every handoff would be no budget.
    max_cost_usd: float = 0.0
    stall_limit: int = 0
    max_tokens: int = 0
    delay: float = 0.35
    start: bool = True           # begin playing at once


class Command(BaseModel):
    command: str


class Label(BaseModel):
    label: str = ""


class RewindTo(BaseModel):
    id: str


class ReplayRequest(BaseModel):
    path: str
    speed: float = 8.0  # events per second


# --- routes --------------------------------------------------------------


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html")


@app.get("/api/state")
async def state() -> JSONResponse:
    if hub.session is None:
        return JSONResponse({"session": None, "traces": _list_traces()})
    return JSONResponse({"session": hub.session.summary(), "traces": _list_traces()})


@app.get("/api/assets")
async def assets_status() -> JSONResponse:
    """What is on disk of the material this project cannot ship."""
    return JSONResponse(assets.survey(ASSET_ROOT))


class FetchAssets(BaseModel):
    key: str | None = None      # one of them, or all when left out


@app.post("/api/assets/fetch")
async def assets_fetch(req: FetchAssets) -> JSONResponse:
    """Download the copyrighted files, on an explicit request and no other way.

    Reached only from the setup panel's button, which says what it is about to
    do and who owns it first. Nothing here runs at startup: a reader who wants
    the mock world, or who has their own copy, should never be asked.

    Each download is verified against the md5 the project was measured with and
    discarded if it does not match — the walkthrough holds for one release and
    the chart's room boxes are pixels of one scan.
    """
    if req.key is not None and req.key not in assets.BY_KEY:
        return JSONResponse({"error": f"unknown asset {req.key!r}"}, status_code=400)
    wanted = [assets.BY_KEY[req.key]] if req.key else list(assets.ASSETS)

    # Blocking network and disk work; off the event loop so the page stays live.
    results = [
        await asyncio.to_thread(assets.fetch, asset, ASSET_ROOT) for asset in wanted
    ]
    return JSONResponse({"results": results, **assets.survey(ASSET_ROOT)})


class SaveKey(BaseModel):
    key: str = ""


@app.get("/api/credentials")
async def credentials_status() -> JSONResponse:
    """Whether the Claude player can run, and on which credential.

    No secret leaves this route: a saved key is reported by its last four
    characters, which is enough to tell two of them apart and nothing else.
    """
    return JSONResponse(credentials.status(ASSET_ROOT))


@app.post("/api/credentials")
async def credentials_save(req: SaveKey) -> JSONResponse:
    """Take a key from the page, prove it works, then keep it.

    Proving it first is the point. A key that is saved and then fails on turn
    one has taught the reader nothing except that this does not work, and the
    obvious mistakes — a truncated paste, the wrong string entirely, a key the
    API does not accept — are all findable in one request that costs a token.

    There is no browser sign-in to offer instead: the SDK can consume an OAuth
    profile and refresh it, but nothing here can mint one, and a flow that
    borrowed another application's OAuth client to reach someone's account is
    not a shortcut this project takes.
    """
    problem = credentials.check(req.key)
    if problem:
        return JSONResponse({"ok": False, "error": problem}, status_code=400)

    from .agents.claude_agent import ClaudeAgent

    try:
        probe = ClaudeAgent(api_key=req.key.strip())
    except Exception as exc:
        return JSONResponse({"ok": False, "error": f"{type(exc).__name__}: {exc}"}, status_code=400)
    refused = await probe.preflight()
    if refused:
        return JSONResponse({"ok": False, "error": refused}, status_code=400)

    result = await asyncio.to_thread(credentials.save, req.key, ASSET_ROOT)
    if not result["ok"]:
        return JSONResponse({"ok": False, "error": result["detail"]}, status_code=400)
    return JSONResponse({"ok": True, **credentials.status(ASSET_ROOT)})


@app.delete("/api/credentials")
async def credentials_forget() -> JSONResponse:
    """Remove the key saved here. Anything in the environment is untouched."""
    result = await asyncio.to_thread(credentials.forget, ASSET_ROOT)
    if not result["ok"]:
        return JSONResponse({"ok": False, "error": result["detail"]}, status_code=400)
    return JSONResponse({"ok": True, "detail": result["detail"], **credentials.status(ASSET_ROOT)})


@app.get("/api/atlas")
async def atlas(story: str = "") -> JSONResponse:
    """The scanned map for this exact story build, if one is installed.

    Room coordinates are keyed by object number, which only holds for the
    release they were measured against — so the match is on the story header,
    never the filename. No match (or no image on disk) means the browser falls
    back to the graph it draws itself.
    """
    found = find_atlas(story)
    if found is None:
        return JSONResponse({"atlas": None})
    return JSONResponse({"atlas": found})


def find_atlas(story: str) -> dict[str, Any] | None:
    if not story:
        return None
    for path in sorted((WEB_DIR / "atlas").glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if data.get("story") != story:
            continue
        # The scan is not ours to redistribute, so it may legitimately be absent.
        if not (path.parent / data["image"]).exists():
            return None
        data["image_url"] = f"/static/atlas/{data['image']}"
        return data
    return None


@app.get("/api/ollama/models")
async def ollama_models() -> JSONResponse:
    """What the local model server has pulled, for the model picker."""
    from .agents.ollama_agent import OllamaError, list_models, resolve_host

    try:
        return JSONResponse({"host": resolve_host(), "models": await list_models()})
    except OllamaError as exc:
        return JSONResponse({"host": resolve_host(), "models": [], "error": str(exc)})


@app.post("/api/session")
async def new_session(req: NewSession) -> JSONResponse:
    await hub.teardown()
    if req.notebook not in NOTEBOOK_MODES:
        return JSONResponse({"error": f"notebook must be one of {NOTEBOOK_MODES}"}, status_code=400)
    total = max(1, min(req.runs, MAX_SERIES))
    try:
        session = await launch(req, series={"index": 1, "total": total} if total > 1 else None)
    except LaunchError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)
    hub.session = session
    hub.series = SeriesPlan(req.model_copy(update={"notebook": "carry" if req.notebook == "new" else req.notebook}), total) if total > 1 else None
    await session.start()
    trace = session.trace
    return JSONResponse({"session": session.summary(), "trace": str(trace.path) if trace else None})


async def launch(req: NewSession, series: dict[str, int] | None = None) -> Session:
    """Build one run from a request: engine, agent, trace, notebook."""
    try:
        engine = build_engine(req.engine, rom=req.rom, seed=engine_seed(req.agent, req.seed))
    except Exception as exc:
        raise LaunchError(f"{type(exc).__name__}: {exc}") from None

    script = engine.walkthrough() if req.agent == "scripted" else None
    if req.agent == "scripted" and req.engine != "mock" and not script:
        engine.close()
        raise LaunchError("no verified walkthrough for this game — scripted needs one")
    # A replay should run to its end; the budget is for agents that explore.
    max_turns = max(req.max_turns, len(script) + 20) if script else req.max_turns

    try:
        agent = build_agent(
            req.agent,
            commands=script,
            seed=req.seed,
            model=req.model,
            effort=req.effort,
            history_turns=req.history_turns,
            info_level=req.info_level,
            think=parse_think(req.think),
            num_ctx=req.num_ctx,
            temperature=req.temperature,
            journal=req.journal,
            search=req.search,
            max_tokens=req.max_tokens or None,
        )
    except Exception as exc:
        engine.close()
        raise LaunchError(f"{type(exc).__name__}: {exc}") from None

    preflight = getattr(agent, "preflight", None)
    problem = await preflight() if preflight else None
    if problem:
        engine.close()
        raise LaunchError(problem)

    notebook = None
    if req.notebook != "off":
        notebook = Notebook.open(
            NOTEBOOK_DIR, engine.story or engine.name, agent.name, fresh=req.notebook == "new",
        )

    trace = None
    if req.record:
        TRACE_DIR.mkdir(exist_ok=True)
        stamp = asyncio.get_event_loop().time()
        run = f"-run{series['index']}" if series else ""
        name = f"{engine.name}-{req.agent}{run}-{int(stamp * 1000) % 10**9}.jsonl"
        trace = TraceWriter(
            TRACE_DIR / name,
            meta={
                "game": engine.name, "agent": agent.name, "seed": req.seed, "engine": req.engine,
                "series": series, "notebook": str(notebook.path) if notebook else None,
            },
        )

    return Session(
        engine=engine,
        agent=agent,
        bus=hub.bus,
        config=SessionConfig(
            max_turns=max_turns,
            max_cost_usd=req.max_cost_usd,
            stall_limit=req.stall_limit,
            lives=req.lives,
            delay=req.delay,
            collect_valid_actions=req.valid_actions,
            history_turns=req.history_turns,
        ),
        trace=trace,
        notebook=notebook,
        series=series,
    )


@app.post("/api/control")
async def control(req: Control) -> JSONResponse:
    session = hub.session
    if session is None:
        return JSONResponse({"error": "no session"}, status_code=400)
    if req.action == "run":
        session.resume()
        session.start_background()
    elif req.action == "pause":
        session.pause()
    elif req.action == "step":
        session.pause()
        await session.step_once()
    else:
        return JSONResponse({"error": f"unknown action {req.action!r}"}, status_code=400)
    return JSONResponse({"session": session.summary()})


@app.post("/api/handoff")
async def handoff(req: Handoff) -> JSONResponse:
    """Hand the live game to a different player, from wherever it stands.

    The world is untouched: same engine, same room, same score. Only the thing
    being asked for a command changes. Useful in both directions — open the
    house yourself and let a model take it underground, or take the keyboard
    back from a run that has spent forty turns typing `look`.

    What this is not is a way to produce a score for the incoming player. The
    handoff is recorded into the trace for exactly that reason.
    """
    session = hub.session
    if session is None:
        return JSONResponse({"error": "no session"}, status_code=400)
    if session.finished:
        return JSONResponse({"error": "the run is over — start a new one"}, status_code=400)
    if not session.started:
        return JSONResponse({"error": "nothing to hand over yet"}, status_code=400)
    if req.agent == "scripted":
        # The walkthrough starts at the first move of a fresh game. Replaying
        # it from turn 83 is not a run, it is a random command list.
        return JSONResponse(
            {"error": "the walkthrough only makes sense from the first move"},
            status_code=400,
        )
    if req.notebook not in NOTEBOOK_MODES:
        return JSONResponse({"error": f"unknown notebook mode {req.notebook!r}"}, status_code=400)

    try:
        agent = build_agent(
            req.agent,
            seed=session.turn,
            model=req.model,
            effort=req.effort,
            history_turns=req.history_turns,
            info_level=req.info_level,
            think=parse_think(req.think),
            num_ctx=req.num_ctx,
            temperature=req.temperature,
            journal=req.journal,
            search=req.search,
            max_tokens=req.max_tokens or None,
        )
    except Exception as exc:
        return JSONResponse({"error": f"{type(exc).__name__}: {exc}"}, status_code=400)

    # Ask the model server whether it can serve this before the running game
    # loses its player to a name that does not resolve.
    preflight = getattr(agent, "preflight", None)
    problem = await preflight() if preflight else None
    if problem:
        return JSONResponse({"error": problem}, status_code=400)

    notebook = None
    if req.notebook != "off" and session.notebook is None:
        notebook = Notebook.open(
            NOTEBOOK_DIR,
            session.engine.story or session.engine.name,
            agent.name,
            fresh=req.notebook == "new",
        )

    if req.max_cost_usd:
        session.config.max_cost_usd = req.max_cost_usd
    if req.stall_limit:
        session.config.stall_limit = req.stall_limit
    await session.handoff(
        agent,
        inherit_transcript=req.inherit_transcript,
        notebook=notebook,
        add_turns=req.add_turns,
        history_turns=req.history_turns,
        delay=req.delay,
    )
    if req.start and agent.kind != "human":
        session.resume()
        session.start_background()
    return JSONResponse({"session": session.summary()})


@app.post("/api/command")
async def command(req: Command) -> JSONResponse:
    session = hub.session
    if session is None:
        return JSONResponse({"error": "no session"}, status_code=400)
    if not isinstance(session.agent, HumanAgent):
        return JSONResponse({"error": "active agent is not human"}, status_code=400)
    session.agent.submit(req.command)
    session.start_background()
    return JSONResponse({"ok": True})


@app.post("/api/checkpoint")
async def checkpoint(req: Label) -> JSONResponse:
    session = hub.session
    if session is None:
        return JSONResponse({"error": "no session"}, status_code=400)
    cp = session.checkpoint(req.label)
    return JSONResponse({"id": cp.id, "label": cp.label, "turn": cp.turn, "session": session.summary()})


@app.post("/api/rewind")
async def rewind(req: RewindTo) -> JSONResponse:
    session = hub.session
    if session is None:
        return JSONResponse({"error": "no session"}, status_code=400)
    session.pause()
    ok = session.rewind(req.id)
    if not ok:
        return JSONResponse({"error": "unknown checkpoint"}, status_code=404)
    return JSONResponse({"session": session.summary()})


@app.post("/api/replay")
async def replay(req: ReplayRequest) -> JSONResponse:
    """Push a recorded trace through the live bus, at a watchable pace."""
    path = Path(req.path)
    if not path.exists():
        return JSONResponse({"error": f"no such trace: {path}"}, status_code=404)
    await hub.teardown()

    header = trace_header(path)
    events = list(read_events(path))

    async def pump() -> None:
        interval = 1.0 / max(req.speed, 0.1)
        for event in events:
            hub.bus.emit(event.type, **event.payload)
            await asyncio.sleep(interval)

    hub._replay_task = asyncio.create_task(pump())
    return JSONResponse({"replaying": str(path), "events": len(events), "header": header})


@app.websocket("/ws")
async def ws(websocket: WebSocket) -> None:
    await websocket.accept()
    q = hub.connect()
    try:
        await websocket.send_json({"type": "hello", "payload": {"backlog": hub.backlog()}})
        while True:
            event = await q.get()
            await websocket.send_json(event)
    except WebSocketDisconnect:
        pass
    finally:
        hub.disconnect(q)


def _list_traces() -> list[dict[str, Any]]:
    if not TRACE_DIR.exists():
        return []
    out = []
    for p in sorted(TRACE_DIR.glob("*.jsonl"), key=lambda x: x.stat().st_mtime, reverse=True)[:25]:
        out.append({"path": str(p), "name": p.name, "size": p.stat().st_size})
    return out


if WEB_DIR.exists():
    app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")
