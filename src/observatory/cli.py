"""Command line entry points: serve, play, replay."""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from .agents import build_agent
from .engine import build_engine
from .events import Event, EventBus
from .session import Session, SessionConfig
from .trace import TraceWriter, read_events, trace_header


def _add_run_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--engine", default="mock", choices=["mock", "jericho"])
    p.add_argument("--rom", default=None, help="path to a .z5/.z3 game file (jericho only)")
    p.add_argument("--agent", default="random", choices=["random", "scripted", "human", "claude"])
    p.add_argument("--model", default="claude-opus-5")
    p.add_argument("--effort", default="medium", choices=["low", "medium", "high", "xhigh", "max"])
    p.add_argument(
        "--info-level", default="parser", choices=["cold", "game", "parser", "coached"],
        help="how much the agent is told before it starts: cold (a bare terminal), "
             "game (it is a game, nothing else), parser (interface explained; default), "
             "coached (hazards and tactics — the contaminated control arm)",
    )
    p.add_argument("--turns", type=int, default=50)
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--history-turns", type=int, default=30)
    p.add_argument("--trace", default=None, help="write a JSONL trace here")


async def _play(args: argparse.Namespace) -> int:
    engine = build_engine(args.engine, rom=args.rom, seed=args.seed)
    agent = build_agent(
        args.agent, seed=args.seed, model=args.model, effort=args.effort,
        history_turns=args.history_turns, info_level=args.info_level,
    )
    bus = EventBus()

    def printer(event: Event) -> None:
        p = event.payload
        if event.type == "session.started":
            print(f"=== {p['game']} · {p['agent']} · max score {p['max_score']} ===\n")
        elif event.type == "agent.thought":
            print(f"  \033[2m{p['text']}\033[0m")
        elif event.type == "command.issued":
            print(f"\033[1m> {p['command']}\033[0m")
        elif event.type == "observation":
            print(p["text"].strip() + "\n")
        elif event.type == "map.update":
            s = p["stats"]
            print(f"  \033[36m[map: {s['rooms']} rooms, {s['edges']} edges, {s['blocked']} blocked]\033[0m\n")
        elif event.type == "discovery.made":
            print(f"  \033[35m◆ {p['label']} — {p['reveals']}  ({p['evidence']})\033[0m\n")
        elif event.type == "session.ended":
            print(f"=== {p['reason']} — score {p['final_score']}/{p['max_score']} in {p['turns']} turns ===")
            print(f"    map: {p['map']}")
            d = p["discoveries"]
            print(f"    discovered {d['found']}/{d['total']}: "
                  + ", ".join(f"{k}@{v}" for k, v in sorted(d["turns"].items(), key=lambda kv: kv[1])))
            if p["usage"].get("calls"):
                u = p["usage"]
                print(f"    cost: ${u['cost_usd']} over {u['calls']} calls "
                      f"({u['input_tokens']} in / {u['output_tokens']} out)")
        elif event.type == "error":
            print(f"  \033[31m[{p['where']}] {p['message']}\033[0m")

    bus.subscribe(printer)
    trace = TraceWriter(args.trace, meta={"game": engine.name, "agent": agent.name}) if args.trace else None

    session = Session(
        engine, agent, bus,
        config=SessionConfig(max_turns=args.turns, delay=0.0, history_turns=args.history_turns),
        trace=trace,
    )
    await session.run()
    engine.close()
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
    uvicorn.run("observatory.server:app", host=args.host, port=args.port, reload=args.reload)
    return 0


def main(argv: list[str] | None = None) -> int:
    # Windows consoles still default to cp1252, which mangles the box-drawing
    # and bullet characters used below into replacement marks.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
        except (AttributeError, ValueError):
            pass

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

    args = parser.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
