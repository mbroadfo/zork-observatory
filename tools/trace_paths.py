"""Trace the drawn passages on the scan, so the chart can uncover them.

    python tools/trace_paths.py SCAN STORY [--check OUT.jpg]

Which rooms connect comes from the story file itself — the exit properties in
the object table — not from the scan and not from a walkthrough. For each
connected pair that is charted, the passage is found as the cheapest route
between the two boxes where cost is low on ink and high on bare paper, so the
route follows the line the cartographers drew. Routes that are mostly paper
(the map draws a stub like "(to Cellar)" instead of a line) are dropped, and
the chart falls back to its short straight reveal for those.

The result is written into the atlas as `paths`, keyed "a|b" with the two
object numbers sorted as strings — the same key the browser builds.

Needs Pillow, numpy and scikit-image.
"""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from skimage.graph import MCP_Geometric
from skimage.morphology import footprint_rectangle, opening

ATLAS_DIR = Path(__file__).resolve().parent.parent / "src" / "observatory" / "web" / "atlas"

SCALE = 3            # trace on a 1/3 image: lines are still 3-4 cells thick
STROKE = 3           # cells; the passage lines survive an opening this size
MARGIN = 260        # search window around the two boxes, in scan pixels
MIN_INK = 0.6        # share of a route that must lie on ink to count as drawn
MAX_DETOUR = 3.2     # a route this much longer than the gap is a wander, not a line

# Zork's direction properties (ZIL assigns them downward from 31).
DIRECTION_PROPS = set(range(19, 32))


def exits(story: bytes) -> set[tuple[int, int]]:
    """Undirected room pairs joined by an exit that names a destination."""
    word = lambda a: struct.unpack(">H", story[a:a + 2])[0]
    base = word(0x0A) + 31 * 2
    entry = lambda n: base + (n - 1) * 9

    def props(n):
        p = word(entry(n) + 7)
        p += 1 + 2 * story[p]
        while story[p]:
            size = story[p]
            num, length = size & 31, (size >> 5) + 1
            yield num, story[p + 1:p + 1 + length]
            p += 1 + length

    count = (word(entry(1) + 7) - base) // 9
    pairs = set()
    for n in range(1, count + 1):
        for num, data in props(n):
            # 1 byte: plain exit. 4: conditional. 5: through a door. The first
            # byte is the destination in all three; 2 and 3 are refusals and
            # routines, which name no room.
            if num in DIRECTION_PROPS and len(data) in (1, 4, 5) and data[0] != n:
                pairs.add(tuple(sorted((n, data[0]))))
    return pairs


def rdp(points, eps):
    if len(points) < 3:
        return points
    a, b = np.array(points[0], float), np.array(points[-1], float)
    ab = b - a
    norm = np.hypot(*ab) or 1.0
    d = [abs(ab[0] * (p[1] - a[1]) - ab[1] * (p[0] - a[0])) / norm for p in points[1:-1]]
    i = int(np.argmax(d)) + 1
    if d[i - 1] <= eps:
        return [points[0], points[-1]]
    return rdp(points[:i + 1], eps)[:-1] + rdp(points[i:], eps)


def dump(atlas: dict) -> str:
    """JSON with one room or path per line, so a diff shows what moved."""
    head = {k: v for k, v in atlas.items() if k not in ("rooms", "paths")}
    lines = ["{"]
    lines += [f"  {json.dumps(k)}: {json.dumps(v, ensure_ascii=False)}," for k, v in head.items()]
    for key in ("rooms", "paths"):
        items = list(atlas.get(key, {}).items())
        lines.append(f'  "{key}": {{')
        lines += [
            f"    {json.dumps(k)}: {json.dumps(v, ensure_ascii=False, separators=(', ', ': '))}"
            + ("," if i < len(items) - 1 else "")
            for i, (k, v) in enumerate(items)
        ]
        lines.append("  }," if key == "rooms" else "  }")
    lines.append("}")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scan", type=Path)
    ap.add_argument("story", type=Path)
    ap.add_argument("--atlas", type=Path, default=ATLAS_DIR / "zork1-r88.json")
    ap.add_argument("--check", type=Path)
    args = ap.parse_args()

    Image.MAX_IMAGE_PIXELS = None
    atlas = json.loads(args.atlas.read_text(encoding="utf-8"))
    rooms = {int(k): v["box"] for k, v in atlas["rooms"].items()}
    scan = Image.open(args.scan).convert("L")
    gray = np.asarray(scan.reduce(SCALE), dtype=np.float32)
    # Passages are drawn as thick, even strokes; the illustrations are hatched
    # in thin ones. An opening keeps only ink at least a stroke wide, so a
    # route can't shortcut through a troll.
    solid = opening(gray < 135, footprint_rectangle((STROKE, STROKE)))
    ink = np.where(solid, 1.0, 0.0)
    base_cost = 1 + 80 * (1 - ink) ** 2
    H, W = gray.shape

    def cells(box, grow=0):
        x0, y0, x1, y1 = box
        return (max(0, int(y0 / SCALE) - grow), min(H, int(y1 / SCALE) + grow + 1),
                max(0, int(x0 / SCALE) - grow), min(W, int(x1 / SCALE) + grow + 1))

    # Never route through somebody else's room, or along its outline.
    blocked = np.zeros_like(gray, dtype=bool)
    for box in rooms.values():
        r0, r1, c0, c1 = cells(box, 2)
        blocked[r0:r1, c0:c1] = True

    # Two corrections the tracer can't make for itself, recorded in the atlas:
    # pairs the map draws only as "(to …)" stubs, and passages whose ink runs
    # into an illustration, traced by hand.
    stubs = set(atlas.get("stubs", []))
    drawn = atlas.get("drawn", {})

    paths, rejected = {}, []
    for a, b in sorted(exits(args.story.read_bytes())):
        if a not in rooms or b not in rooms:
            continue
        key = "|".join(sorted((str(a), str(b))))
        if key in drawn:
            paths[key] = drawn[key]
            continue
        if key in stubs:
            rejected.append((a, b, "drawn as a stub"))
            continue
        ba, bb = rooms[a], rooms[b]
        gap = np.hypot((ba[0] + ba[2] - bb[0] - bb[2]) / 2, (ba[1] + ba[3] - bb[1] - bb[3]) / 2)
        if gap > 2600:
            rejected.append((a, b, "across the page"))
            continue

        x0 = min(ba[0], bb[0]) - MARGIN
        y0 = min(ba[1], bb[1]) - MARGIN
        x1 = max(ba[2], bb[2]) + MARGIN
        y1 = max(ba[3], bb[3]) + MARGIN
        R0, R1, C0, C1 = cells((max(0, x0), max(0, y0), x1, y1))
        cost = np.where(blocked, 5000.0, base_cost)[R0:R1, C0:C1].copy()
        inside = []
        for box in (ba, bb):
            r0, r1, c0, c1 = cells(box, 2)
            cost[r0 - R0:r1 - R0, c0 - C0:c1 - C0] = 1.0
            inside.append((r0 - R0, r1 - R0, c0 - C0, c1 - C0))

        def region(r0, r1, c0, c1):
            return [(r, c) for r in range(r0, r1) for c in range(c0, c1)]

        starts, ends = region(*inside[0]), region(*inside[1])
        if set(starts) & set(ends):
            continue    # boxes touch — nothing between them to uncover
        mcp = MCP_Geometric(cost)
        cum, _ = mcp.find_costs(starts, ends, find_all_ends=False)
        end = min(ends, key=lambda p: cum[p])
        route = mcp.traceback(end)

        def within(p, box):
            r0, r1, c0, c1 = box
            return r0 <= p[0] < r1 and c0 <= p[1] < c1

        # Keep only the stretch between the two rooms.
        outside = [p for p in route if not within(p, inside[0]) and not within(p, inside[1])]
        if not outside:
            continue
        share = float(np.mean([ink[p[0] + R0, p[1] + C0] > 0.5 for p in outside]))
        length = len(outside) * SCALE
        edge_gap = max(1.0, gap - (max(ba[2] - ba[0], ba[3] - ba[1]) + max(bb[2] - bb[0], bb[3] - bb[1])) / 2)
        if share < MIN_INK or length > MAX_DETOUR * edge_gap + 400:
            rejected.append((a, b, f"ink {share:.0%}, {length:.0f}px for a {edge_gap:.0f}px gap"))
            continue

        # One cell of each room on either end, so the line meets the box.
        first = route.index(outside[0])
        last = route.index(outside[-1])
        stretch = route[max(0, first - 1):last + 2]
        pts = [((c + C0) * SCALE + SCALE // 2, (r + R0) * SCALE + SCALE // 2) for r, c in stretch]
        paths["|".join(sorted((str(a), str(b))))] = [list(p) for p in rdp(pts, 5)]

    atlas["paths"] = dict(sorted(paths.items()))
    args.atlas.write_text(dump(atlas), encoding="utf-8")
    print(f"{len(paths)} passages traced, {len(rejected)} left as stubs")
    for a, b, why in rejected:
        print(f"  {atlas['rooms'][str(a)]['name']} — {atlas['rooms'][str(b)]['name']}: {why}")

    if args.check:
        img = Image.open(args.scan).convert("RGB")
        draw = ImageDraw.Draw(img)
        for pts in paths.values():
            draw.line([tuple(p) for p in pts], fill=(20, 110, 255), width=14, joint="curve")
        for a, b, _ in rejected:
            pa, pb = rooms[a], rooms[b]
            draw.line([((pa[0] + pa[2]) / 2, (pa[1] + pa[3]) / 2), ((pb[0] + pb[2]) / 2, (pb[1] + pb[3]) / 2)],
                      fill=(230, 30, 30), width=5)
        img.resize((3000, round(img.height * 3000 / img.width)), Image.Resampling.LANCZOS).save(args.check, quality=85)
        print(f"wrote {args.check}")


if __name__ == "__main__":
    main()
