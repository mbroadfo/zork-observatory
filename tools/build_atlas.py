"""Prepare a scanned map for the chart view, and check its calibration.

    python tools/build_atlas.py assets/zork-1-map-ZUG-1982.jpeg
    python tools/build_atlas.py assets/zork-1-map-ZUG-1982.jpeg --check traces/atlas-check.jpg

The first form writes the web copy named in the atlas file. The second also
draws every calibrated room box over the scan, which is how the coordinates in
the atlas were verified — a box that misses its room is obvious at a glance.

Needs Pillow (`pip install pillow`); nothing else in the project does.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw

ATLAS_DIR = Path(__file__).resolve().parent.parent / "src" / "observatory" / "web" / "atlas"
WEB_WIDTH = 4096   # a common GPU texture limit; plenty to read the smallest label


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scan", type=Path)
    ap.add_argument("--atlas", type=Path, default=ATLAS_DIR / "zork1-r88.json")
    ap.add_argument("--check", type=Path, help="also write a calibration overlay here")
    args = ap.parse_args()

    Image.MAX_IMAGE_PIXELS = None
    atlas = json.loads(args.atlas.read_text(encoding="utf-8"))
    scan = Image.open(args.scan).convert("RGB")
    if scan.size != (atlas["width"], atlas["height"]):
        raise SystemExit(
            f"scan is {scan.size[0]}x{scan.size[1]}, atlas expects "
            f"{atlas['width']}x{atlas['height']} — wrong file, or a different scan"
        )

    out = args.atlas.parent / atlas["image"]
    height = round(scan.height * WEB_WIDTH / scan.width)
    scan.resize((WEB_WIDTH, height), Image.Resampling.LANCZOS).save(
        out, quality=84, optimize=True, progressive=True
    )
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")

    if args.check:
        draw = ImageDraw.Draw(scan)
        for num, room in atlas["rooms"].items():
            x0, y0, x1, y1 = room["box"]
            draw.rectangle((x0, y0, x1, y1), outline=(220, 30, 30), width=8)
            draw.text((x0 + 10, y1 - 22), num, fill=(220, 30, 30))
        preview = scan.resize((2400, round(scan.height * 2400 / scan.width)), Image.Resampling.LANCZOS)
        preview.save(args.check, quality=85)
        print(f"wrote {args.check} — {len(atlas['rooms'])} rooms outlined")


if __name__ == "__main__":
    main()
