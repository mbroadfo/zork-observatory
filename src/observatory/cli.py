"""Command line entry points: serve, play, replay."""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from .agents import (
    AGENTS, DEFAULT_HISTORY, DEFAULT_NUM_CTX, DEFAULT_TEMPERATURE,
    build_agent, parse_think,
)
from .engine import build_engine, engine_seed
from .events import Event, EventBus
from .notebook import MODES as NOTEBOOK_MODES, Notebook
from .session import Session, SessionConfig
from .trace import TraceWriter, read_events, trace_header


def _add_run_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--engine", default="mock", choices=["mock", "jericho"])
    p.add_argument("--rom", default=None, help="path to a .z5/.z3 game file (jericho only)")
    p.add_argument("--agent", default="random", choices=list(AGENTS))
    p.add_argument("--model", default=None,
                   help="claude: defaults to claude-opus-5. ollama: required, e.g. qwen3:8b")
    p.add_argument("--effort", default="medium", choices=["low", "medium", "high", "xhigh", "max"])
    p.add_argument("--think", default="default",
                   help="ollama only: default (the model's own), off, on, or low/medium/high")
    p.add_argument(
        "--info-level", default="parser", choices=["cold", "game", "parser", "coached"],
        help="how much the agent is told before it starts: cold (a bare terminal), "
             "game (it is a game, nothing else), parser (interface explained; default), "
             "coached (hazards and tactics — the contaminated control arm)",
    )
    p.add_argument("--turns", type=int, default=400,
                   help="turn BUDGET, not a rule — the game has no turn limit. "
                        "Runs stopped by it are reported as censored")
    p.add_argument("--max-cost", type=float, default=0.0,
                   help="stop once the agent has spent this many USD (0 = no ceiling)")
    p.add_argument("--lives", type=int, default=0,
                   help="times the world may be rolled back after a death (0 = ironman). "
                        "The agent keeps its memory across a rollback, nothing else")
    p.add_argument("--no-agent-save", action="store_true",
                   help="withhold SAVE/RESTORE from the agent")
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE,
                   help="ollama only: sampling temperature. 0.7 (the default) is a setting for "
                        "prose; 0.2-0.3 wanders less and follows an instruction more closely")
    p.add_argument("--num-ctx", type=int, default=DEFAULT_NUM_CTX,
                   help="ollama only: context window in tokens. Smaller keeps a larger model "
                        "entirely on the GPU; overflow is dropped from the front, silently")
    p.add_argument("--history-turns", type=int, default=DEFAULT_HISTORY,
                   help="exchanges of transcript shown each turn")
    p.add_argument("--journal", action="store_true",
                   help="let the agent write a line of its own to keep, on any turn it "
                        "chooses. Shown back beside the transcript, never instead of it, "
                        "and carried between runs by the notebook")
    p.add_argument("--trace", default=None, help="write a JSONL trace here")
    p.add_argument("--runs", type=int, default=1,
                   help="play this many runs back to back, each from the first move")
    p.add_argument("--notebook", default="off", choices=list(NOTEBOOK_MODES),
                   help="what crosses between runs: off (nothing), carry (the agent's own "
                        "notes, kept per player and game under traces/notebooks), new "
                        "(set the existing notebook aside and start another)")


async def _play(args: argparse.Namespace) -> int:
    runs = max(1, args.runs)
    mode = args.notebook
    for index in range(1, runs + 1):
        series = {"index": index, "total": runs} if runs > 1 else None
        code = await _play_once(args, mode, series)
        if code:
            return code
        mode = "carry" if mode == "new" else mode
    return 0


async def _play_once(args: argparse.Namespace, notebook_mode: str, series: dict[str, int] | None) -> int:
    engine = build_engine(args.engine, rom=args.rom, seed=engine_seed(args.agent, args.seed))
    script = engine.walkthrough() if args.agent == "scripted" else None
    if args.agent == "scripted" and args.engine != "mock" and not script:
        print("no verified walkthrough for this game — scripted needs one", file=sys.stderr)
        return 2
    if script:
        # A replay runs to its end; the budget is for agents that explore.
        args.turns = max(args.turns, len(script) + 20)
    try:
        agent = build_agent(
            args.agent, commands=script, seed=args.seed, model=args.model, effort=args.effort,
            history_turns=args.history_turns, info_level=args.info_level,
            think=parse_think(args.think), num_ctx=args.num_ctx,
            temperature=args.temperature, journal=args.journal,
        )
    except ValueError as exc:
        print(exc, file=sys.stderr)
        engine.close()
        return 2
    preflight = getattr(agent, "preflight", None)
    problem = await preflight() if preflight else None
    if problem:
        print(problem, file=sys.stderr)
        engine.close()
        return 2
    bus = EventBus()

    def printer(event: Event) -> None:
        p = event.payload
        if event.type == "session.started":
            run = f" · run {p['series']['index']}/{p['series']['total']}" if p.get("series") else ""
            print(f"=== {p['game']} · {p['agent']} · max score {p['max_score']}{run} ===")
            nb = p.get("notebook")
            if nb:
                print(f"  \033[33m✎ notebook {nb['path']}: {nb['notes']} note(s) from "
                      f"{nb['runs'] - 1} earlier run(s)\033[0m")
            print()
        elif event.type == "agent.thought":
            print(f"  \033[2m{p['text']}\033[0m")
            if (p.get("meta") or {}).get("context_warning"):
                print(f"  \033[31m⚠ {p['meta']['context_warning']}\033[0m")
        elif event.type == "command.issued":
            print(f"\033[1m> {p['command']}\033[0m")
        elif event.type == "observation":
            print(p["text"].strip() + "\n")
        elif event.type == "map.update":
            s = p["stats"]
            print(f"  \033[36m[map: {s['rooms']} rooms, {s['edges']} edges, {s['blocked']} blocked]\033[0m\n")
        elif event.type == "discovery.made":
            print(f"  \033[35m◆ {p['label']} — {p['reveals']}  ({p['evidence']})\033[0m\n")
        elif event.type == "lesson.learned":
            print(f"  \033[33m✎ kept: {p['text']}\033[0m\n")
        elif event.type == "run.restored":
            print(f"  \033[33m↺ life {p['life']} — world back to turn {p['to_turn']}, "
                  f"carrying {p['carried']} note(s). {p['lives_left']} live(s) left\033[0m\n")
        elif event.type == "session.ended":
            censored = "  [CENSORED: stopped by the harness, not the game]" if p.get("censored") else ""
            print(f"=== {p['reason']} — score {p['final_score']}/{p['max_score']} "
                  f"in {p['turns']} turns, {p.get('deaths', 0)} death(s) ==={censored}")
            print(f"    map: {p['map']}")
            c = p["coverage"]
            print(f"    explored: {c['rooms_seen']}/{c['rooms_total']} rooms ({c['rooms_pct']}%) · "
                  f"{c['objects_seen']}/{c['objects_total']} objects seen ({c['objects_pct']}%) · "
                  f"{c['objects_held']} ever held, {c['stowed']} stowed · "
                  f"score {c['score']}/{c['max_score']} ({c['score_pct']}%)")
            q = p["quality"]
            print(f"    turns: {q['wasted_pct']}% wasted, {q['futile_pct']}% futile "
                  f"(already tried and already failed in that same room) · "
                  f"{q['distinct_commands']} distinct commands, {q['known_dead_ends']} known dead ends")
            print(f"           {q['counts']}")
            d = p["discoveries"]
            # Reported against `step`, which rollbacks don't rewind.
            print(f"    discovered {d['found']}/{d['total']} over {p['steps']} steps: "
                  + ", ".join(f"{k}@{v}" for k, v in sorted(d["steps"].items(), key=lambda kv: kv[1])))
            if p["usage"].get("calls"):
                u = p["usage"]
                print(f"    cost: ${u['cost_usd']} over {u['calls']} calls "
                      f"({u['input_tokens']} in / {u['output_tokens']} out)")
        elif event.type == "error":
            print(f"  \033[31m[{p['where']}] {p['message']}\033[0m")

    bus.subscribe(printer)
    trace_path = args.trace
    if trace_path and series:
        base = Path(trace_path)
        trace_path = base.with_name(f"{base.stem}-run{series['index']}{base.suffix}")
    trace = (
        TraceWriter(trace_path, meta={"game": engine.name, "agent": agent.name, "series": series})
        if trace_path else None
    )
    notebook = None
    if notebook_mode != "off":
        notebook = Notebook.open(
            Path("traces") / "notebooks", engine.story or engine.name, agent.name,
            fresh=notebook_mode == "new",
        )

    session = Session(
        engine, agent, bus,
        config=SessionConfig(
            max_turns=args.turns,
            max_cost_usd=args.max_cost,
            lives=args.lives,
            allow_agent_save=not args.no_agent_save,
            delay=0.0,
            history_turns=args.history_turns,
        ),
        trace=trace,
        notebook=notebook,
        series=series,
    )
    await session.run()
    engine.close()

    if agent.memory.lessons:
        print("\n--- what it wrote down and kept ---")
        for lesson in agent.memory.lessons:
            where = f", {lesson.location}" if lesson.location else ""
            run = f"run {lesson.run}, " if lesson.run else ""
            print(f"  {run}life {lesson.life} (turn {lesson.turn}{where}): {lesson.text}")
    if notebook and len(notebook.runs) > 1:
        print("\n--- the notebook's runs so far ---")
        for r in notebook.runs:
            print(f"  run {r['run']}: {r.get('reason', '?')} · score {r.get('score', '?')} "
                  f"in {r.get('turns', '?')} turns · {r.get('deaths', 0)} death(s)")
    print()
    return 0


def _mcp(args: argparse.Namespace) -> int:
    """Speak MCP on stdin/stdout for as long as the client keeps the pipe open.

    Nothing is printed to stdout that is not a protocol message — a stray line
    here corrupts the stream — so diagnostics go to stderr or nowhere.
    """
    from .mcp_server import DEFAULT_BASE, serve

    serve(args.url or DEFAULT_BASE)
    return 0


def _replay(args: argparse.Namespace) -> int:
    path = Path(args.path)
    if not path.exists():
        print(f"No such trace: {path}", file=sys.stderr)
        return 1
    print(f"header: {trace_header(path)}\n")
    for event in read_events(path):
        p = event.payload
        if event.type == "command.issued":
            print(f"> {p['command']}")
        elif event.type == "observation":
            print(p["text"].strip() + "\n")
        elif event.type == "session.ended":
            print(f"=== {p['reason']} — score {p['final_score']}/{p['max_score']} ===")
    return 0


def _serve(args: argparse.Namespace) -> int:
    import uvicorn

    print(f"Observatory at http://{args.host}:{args.port}")
    # An open browser tab holds a WebSocket that never closes on its own, and
    # without a deadline a --reload waits on it forever with the port dead.
    uvicorn.run(
        "observatory.server:app", host=args.host, port=args.port, reload=args.reload,
        timeout_graceful_shutdown=3,
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    # Windows consoles still default to cp1252, which mangles the box-drawing
    # and bullet characters used below into replacement marks.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
        except (AttributeError, ValueError):
            pass

    # An Anthropic variable exported as empty means "unset" wherever it came
    # from, and something much worse to the SDK. See credentials.py.
    from . import credentials

    credentials.sanitize_environment()

    parser = argparse.ArgumentParser(prog="observatory", description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("serve", help="run the web observatory")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    s.add_argument("--reload", action="store_true")
    s.set_defaults(fn=lambda a: _serve(a))

    p = sub.add_parser("play", help="run a headless game in the terminal")
    _add_run_args(p)
    p.set_defaults(fn=lambda a: asyncio.run(_play(a)))

    r = sub.add_parser("replay", help="print a recorded trace")
    r.add_argument("path")
    r.set_defaults(fn=lambda a: _replay(a))

    m = sub.add_parser(
        "mcp",
        help="serve the live run as MCP tools, for an assistant to advise you",
    )
    m.add_argument("--url", default=None,
                   help="the running observatory (default http://127.0.0.1:8000)")
    m.set_defaults(fn=lambda a: _mcp(a))

    args = parser.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
