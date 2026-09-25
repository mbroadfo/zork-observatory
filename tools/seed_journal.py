"""Write a journal a player would have kept if it had played well.

The journal is a cross-run transfer mechanism, and every run measured so far
started with an empty one. That is its worst case, and two runs of three
hundred turns each have now measured it: one hundred and seventy-five lines
offered, four kept, eight rooms, no points. The one run that looked like the
journal working — inside the house on turn twenty against eighty — had
inherited its entries from an earlier run. Transfer, not learning.

So this seeds one, from the game's own recorded walkthrough, and the amount of
it is a dial. `--steps 20` hands over the opening and nothing else; `--steps
400` hands over the game. Somewhere between those is the answer to "how good
does the record have to be before it helps", which is the same shape of
question as the information ladder in agents/prompts.py, and is measured the
same way: one variable, several rungs, the curve says more than any rung.

Two things make this honest enough to be worth running.

The entries are the game's words, verbatim. Not the harness's summary and not
a model's paraphrase — the reply the interpreter printed, stored against the
command that produced it. Every failure the journal has had so far came from
paraphrase: "Took the elvish sword from above the trophy case" sent a run to
type that at a parser with no word "above". There is nothing here to
mistranslate.

And they go through exactly the same filter a live entry does — the same
Journal object, the same corroboration against what the player could see
change, the same rules about movement and repetition. A seeded journal is not
allowed to contain anything a played one could not.

It is a contaminated arm and is labelled one: `run=0` marks every seeded
entry, so a trace shows at a glance what was given and what was earned.

    python tools/seed_journal.py --steps 30
    python tools/seed_journal.py --steps 30 --agent ollama:qwen3:14b/coached
    python tools/seed_journal.py --steps 30 --show     # print, write nothing
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from observatory.agents.journal import Journal  # noqa: E402
from observatory.engine.jericho_engine import JerichoEngine  # noqa: E402
from observatory.notebook import Notebook  # noqa: E402
from observatory.world.discovery import Turn as DiscoveryTurn  # noqa: E402
from observatory.world.graph import parse_movement  # noqa: E402
from observatory.world.objects import diff_objects  # noqa: E402
from observatory.world.outcomes import Outcome, OutcomeTally  # noqa: E402

NOTEBOOK_DIR = Path("traces") / "notebooks"

# One line of the game's reply. A room description runs to five, and the part
# that says what changed is the first sentence of it.
MAX_REPLY_CHARS = 160


def first_line(text: str) -> str:
    for line in (text or "").splitlines():
        line = " ".join(line.split())
        if line:
            return line[:MAX_REPLY_CHARS]
    return ""


def play(rom: str, steps: int, route: bool = True) -> tuple[Journal, list[str]]:
    """Walk the recorded walkthrough, keeping what a journal would keep.

    `route` keeps the movement entries a live run drops. A played journal
    refuses them because the model wrote sixteen "Moved north from the forest
    path" for every eight that said anything, and the map already holds the
    topology in more detail. Neither applies here: the walkthrough's moves are
    few and correct, and the agent has no map. Stored against the room they
    were given in, each one reads as an edge —

        [turn 1 (West of House)] "N" worked: North of House

    — and without them the seed is a list of things that worked somewhere,
    with no way to reach any of them. That is precisely how the last seeded
    run failed: it knew the sword could be taken and not that the Living Room
    was through a window.
    """
    # seed=None is the seed Jericho verified its walkthrough under. Zork's
    # combat and thief are random, and the same 396 commands die in the forest
    # at step 34 under any other.
    engine = JerichoEngine(rom, seed=None)
    script = engine.walkthrough()
    if not script:
        engine.close()
        raise SystemExit(
            f"{rom} has no verified walkthrough for its seed — nothing to seed from."
        )

    journal = Journal(keep_movement=route)
    skipped: list[str] = []
    tally = OutcomeTally()
    obs, state = engine.reset()
    prev, prev_objects = state, list(state.objects)

    for turn, command in enumerate(script[:steps], start=1):
        obs, state = engine.step(command)
        probe = DiscoveryTurn(
            turn=turn, command=command, obs=obs, state=state, prev_state=prev,
            visited_rooms=set(), commands_seen=set(),
        )
        changes = [
            c for c in diff_objects(prev_objects, state.objects)
            if c.num in probe.visible_objects()
        ]
        player = probe.player_object()
        outcome = tally.classify(
            command, obs, state, prev, witnessed=any(c.num != player for c in changes)
        )
        reply = first_line(obs.text)
        # The room the command was given in, not the one it landed in.
        kept = journal.add(
            reply,
            turn=turn,
            room=outcome.room,
            command=command,
            outcome=outcome.outcome.value,
            run=0,
            movement=parse_movement(command) is not None,
        )
        if kept is None and outcome.outcome is Outcome.PROGRESS:
            skipped.append(f"{command} ({journal.rejected[-1].reason})")
        prev, prev_objects = state, list(state.objects)

    engine.close()
    return journal, skipped


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--rom", default="roms/zork1.z5")
    ap.add_argument("--steps", type=int, default=30,
                    help="how much of the walkthrough to hand over (default 30)")
    ap.add_argument("--agent", default="ollama:qwen3:14b/coached",
                    help="whose notebook to write, exactly as the agent names itself")
    ap.add_argument("--story", default=None,
                    help="story id; defaults to the engine's")
    ap.add_argument("--no-route", dest="route", action="store_false",
                    help="drop the movement entries, leaving results with no way to reach them")
    ap.add_argument("--show", action="store_true", help="print it and write nothing")
    args = ap.parse_args()

    journal, skipped = play(args.rom, args.steps, route=args.route)
    print(journal.render() or "(nothing survived the filter)")
    print()
    s = journal.summary()
    print(f"{s['count']} kept of {s['attempted']} turns — "
          f"{s['movement']} movement, {s['redundant']} repeated, {s['false']} changed nothing")
    if skipped:
        print(f"progress turns not kept: {', '.join(skipped[:8])}")

    if args.show:
        return 0

    engine = JerichoEngine(args.rom, seed=None)
    story = args.story or engine.story or engine.name
    engine.close()

    book = Notebook.open(NOTEBOOK_DIR, story, args.agent)
    if book.journal:
        print(f"\n{book.path} already holds {len(book.journal)} entries — "
              f"set it aside first if you want only the seed in there.")
        return 1
    for entry in journal.entries:
        book.write(entry)
    print(f"\nwrote {len(journal.entries)} entries to {book.path}")
    print("start the run with notebook=carry to hand them over.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
