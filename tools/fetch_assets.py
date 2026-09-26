"""Fetch the two files this project needs and cannot ship, from a terminal.

    python tools/fetch_assets.py            # fetch whatever is missing
    python tools/fetch_assets.py --check    # verify what is there, download nothing
    python tools/fetch_assets.py --only rom

The manifest — which files, from where, and the md5 each must have — lives in
`src/observatory/assets.py`, because the web observatory offers the same
download from its setup panel and the two must not disagree about the hashes.
This is the same thing with a terminal in front of it.

Nothing else in the project calls it. Fetching someone else's copyrighted work
is a decision to make deliberately rather than discover afterwards in a build
log, and the files it writes are in .gitignore and stay there.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from observatory import assets  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--check", action="store_true",
                    help="report what is present and correct, download nothing")
    ap.add_argument("--only", choices=[a.key for a in assets.ASSETS],
                    help="just one of them")
    args = ap.parse_args()

    wanted = [a for a in assets.ASSETS if not args.only or a.key == args.only]

    if args.check:
        bad = 0
        for asset in wanted:
            state = assets.status(asset, ROOT)
            print(f"{'ok  ' if state['present'] else 'BAD '} {asset.key:<4} "
                  f"{asset.relpath:<34} {state['detail']}")
            bad += not state["present"]
        return 1 if bad else 0

    print("These files are copyrighted by Activision and are not distributed")
    print("with this project. Fetching your own copy is your decision and your")
    print("responsibility.\n")

    failed = 0
    for asset in wanted:
        print(f"{asset.key}: {asset.what}")
        print(f"  from {asset.source}")
        print(f"       {asset.source_url}")
        state = assets.status(asset, ROOT)
        if state["present"]:
            print(f"  already have it ({asset.relpath}), md5 matches\n")
            continue
        if state["detail"] != "missing":
            print(f"  {state['detail']} — leaving it alone\n")
            failed += 1
            continue
        print(f"  downloading {asset.size:,} bytes ...")
        result = assets.fetch(asset, ROOT)
        print(f"  {result['detail']}")
        if not result["ok"]:
            for alternate in asset.alternates:
                print(f"  a different copy lives at {alternate}")
            failed += 1
        print()

    if failed:
        print("Not everything arrived. The mock world needs none of it:")
        print("  observatory play --agent scripted --turns 30")
        return 1

    print("In place. `docker compose up` will find them.")
    if any(a.key == "map" for a in wanted):
        print("For the chart, build the web image:")
        print("  pip install pillow")
        print("  python tools/build_atlas.py assets/zork-1-map-ZUG-1982.jpeg")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
