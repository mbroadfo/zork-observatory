"""The two files this project needs and cannot ship.

Zork's story file and the Zork Users Group's map are both copyrighted by
Activision, and neither belongs in this repository. They do live in public,
citable places, and every run here has been made against one exact copy of
each — so what this module does is name those copies, by URL and by hash, and
fetch them on request.

The manifest lives here rather than in `tools/` because two things need it and
they must not disagree: `tools/fetch_assets.py` for someone at a terminal, and
the setup panel the web observatory shows when a story file is missing. One
list, one pair of hashes.

On the sources, since "reputable" is the whole point:

  The story file comes from the Jericho game suite, the corpus essentially
  every interactive-fiction RL paper benchmarks against. Taking it from there
  is what makes a score here comparable to a score in that literature at all.

  The map comes from Andrew Plotkin's collection of Infocom maps — a named
  author in this field, a plain static directory, each map with a provenance
  note. The same scan circulates at the Internet Archive and MOCAGH as PDFs of
  all three Zork maps; those are different files that will not calibrate
  against the atlas, so they are recorded as alternates to check against, not
  as sources to fetch from.

A download whose md5 does not match is discarded rather than written. The
walkthrough is verified against one release and the chart's room boxes are
pixels of one scan, so approximately the right file is the failure this is
here to prevent, not an inconvenience to warn about.

Nothing here runs on its own. Fetching someone else's copyrighted work is a
decision a reader makes deliberately — from the terminal, or by pressing a
button that says what it is about to do.
"""

from __future__ import annotations

import hashlib
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

TIMEOUT_S = 120
CHUNK = 1 << 16

# Where the files go, relative to the project root. The server runs with the
# repository at /app and a terminal usually does not, so the root is passed in
# rather than guessed.
DEFAULT_ROOT = Path(__file__).resolve().parent.parent.parent


@dataclass(frozen=True)
class Asset:
    key: str
    relpath: str
    url: str
    md5: str
    size: int
    what: str
    source: str
    source_url: str
    holder: str = "Activision"
    alternates: tuple[str, ...] = ()
    # False when the project runs perfectly well without it.
    required: bool = True

    def path(self, root: Path | None = None) -> Path:
        return (root or DEFAULT_ROOT) / self.relpath


ASSETS: tuple[Asset, ...] = (
    Asset(
        key="rom",
        relpath="roms/zork1.z5",
        url="https://raw.githubusercontent.com/BYU-PCCL/z-machine-games"
            "/master/jericho-game-suite/zork1.z5",
        md5="b732a93a6244ddd92a9b9a3e3a46c687",
        size=92160,
        what="Zork I — Release 88 / Serial 840726, Z-machine v3, 350 points",
        source="the Jericho game suite, the corpus interactive-fiction RL papers "
               "benchmark against",
        source_url="https://github.com/BYU-PCCL/z-machine-games",
    ),
    Asset(
        key="map",
        relpath="assets/zork-1-map-ZUG-1982.jpeg",
        url="https://eblong.com/infocom/maps/zork-1-map-ZUG-1982.jpeg",
        md5="cc516cee0dc06ba7141bba52c8aa56fb",
        size=11285455,
        what="the 1982 Zork Users Group map, 6517×5030 — designed by Steve "
             "Meretzky, art by David Ardito",
        source="Andrew Plotkin's collection of Infocom maps",
        source_url="https://eblong.com/infocom/maps/",
        alternates=(
            "https://archive.org/details/zork-i-ii-iii-maps",
            "https://mocagh.org/infocom/zork-zugmap.pdf",
        ),
        # The chart is a nicety. Without it a run draws its own graph.
        required=False,
    ),
)

BY_KEY = {asset.key: asset for asset in ASSETS}

Fetcher = Callable[[Asset], bytes]


def digest(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(CHUNK), b""):
            h.update(block)
    return h.hexdigest()


def http_fetch(asset: Asset) -> bytes:
    request = urllib.request.Request(
        asset.url,
        # Some static hosts refuse the default urllib agent. Saying who is
        # asking is politer than pretending to be a browser.
        headers={"User-Agent": "zork-observatory/assets.py"},
    )
    with urllib.request.urlopen(request, timeout=TIMEOUT_S) as response:
        return response.read()


def status(asset: Asset, root: Path | None = None) -> dict[str, Any]:
    """What is on disk, and whether it is the right file."""
    path = asset.path(root)
    out: dict[str, Any] = {
        "key": asset.key,
        "what": asset.what,
        "path": asset.relpath,
        "source": asset.source,
        "source_url": asset.source_url,
        "url": asset.url,
        "size": asset.size,
        "md5": asset.md5,
        "holder": asset.holder,
        "required": asset.required,
        "alternates": list(asset.alternates),
        "present": False,
        "detail": "missing",
    }
    if not path.exists():
        return out
    found_size = path.stat().st_size
    if found_size != asset.size:
        out["detail"] = f"wrong size: {found_size} bytes, expected {asset.size}"
        return out
    found = digest(path)
    if found != asset.md5:
        out["detail"] = f"wrong file: md5 {found}"
        return out
    out.update(present=True, detail="ok")
    return out


def survey(root: Path | None = None) -> dict[str, Any]:
    items = [status(asset, root) for asset in ASSETS]
    return {
        "assets": items,
        # What the UI needs in one word: can this machine play a real game.
        "ready": all(i["present"] for i in items if i["required"]),
        "writable": writable(root),
    }


def writable(root: Path | None = None) -> bool:
    """Whether the process could plausibly write the files it offers to fetch.

    The container mounted `roms` read-only for a long time, which would have
    made a download button fail at the last step with an oblique OSError. It
    is cheaper to say so before offering.

    Asked of the nearest directory that exists, and asked without touching
    anything: a survey is a question, and one that quietly created `roms/` and
    `assets/` as a side effect of being asked would be a poor one. It is a
    hint rather than a guarantee — `fetch` reports the real error either way.
    """
    for asset in ASSETS:
        directory = asset.path(root).parent
        while not directory.exists() and directory != directory.parent:
            directory = directory.parent
        if not os.access(directory, os.W_OK):
            return False
    return True


def fetch(
    asset: Asset, root: Path | None = None, fetcher: Fetcher | None = None
) -> dict[str, Any]:
    """Get one asset, verify it, and write it only if it is the right file."""
    path = asset.path(root)
    if path.exists():
        current = status(asset, root)
        if current["present"]:
            return {"key": asset.key, "ok": True, "detail": "already present"}
        # Never overwrite. It may be a copy someone meant to keep, and the
        # difference between "yours is a different release" and "the download
        # failed" is worth preserving.
        return {
            "key": asset.key, "ok": False,
            "detail": f"{current['detail']} — move it aside and try again",
        }

    try:
        blob = (fetcher or http_fetch)(asset)
    except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as exc:
        return {"key": asset.key, "ok": False,
                "detail": f"could not fetch: {getattr(exc, 'reason', exc)}"}

    found = hashlib.md5(blob).hexdigest()
    if found != asset.md5:
        return {"key": asset.key, "ok": False,
                "detail": f"refused: md5 {found}, expected {asset.md5}. Nothing written."}

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(blob)
    except OSError as exc:
        return {"key": asset.key, "ok": False, "detail": f"could not write: {exc}"}
    return {"key": asset.key, "ok": True, "detail": f"wrote {asset.relpath}"}
