"""Object-tree snapshots and the diff between two turns.

The diff is the interesting half. A turn's real effect on the world is almost
always a handful of parent-pointer changes, and showing exactly those — rather
than the whole tree again — is what makes the state pane readable.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Iterable

from ..engine.base import WorldObject


@dataclass
class ObjectChange:
    num: int
    name: str
    kind: str          # "moved" | "appeared" | "vanished" | "renamed"
    before: Any = None
    after: Any = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def diff_objects(
    before: Iterable[WorldObject], after: Iterable[WorldObject]
) -> list[ObjectChange]:
    prev = {o.num: o for o in before}
    curr = {o.num: o for o in after}
    changes: list[ObjectChange] = []

    for num, obj in curr.items():
        old = prev.get(num)
        if old is None:
            changes.append(ObjectChange(num, obj.name, "appeared", after=obj.parent))
            continue
        if old.parent != obj.parent:
            changes.append(ObjectChange(num, obj.name, "moved", old.parent, obj.parent))
        if old.name != obj.name:
            changes.append(ObjectChange(num, obj.name, "renamed", old.name, obj.name))

    for num, obj in prev.items():
        if num not in curr:
            changes.append(ObjectChange(num, obj.name, "vanished", before=obj.parent))

    return changes


def build_tree(objects: Iterable[WorldObject], root: int = 0) -> list[dict[str, Any]]:
    """Nest a flat object list by parent pointer, for the state explorer."""
    objs = list(objects)
    by_parent: dict[int, list[WorldObject]] = {}
    known = {o.num for o in objs}
    for o in objs:
        parent = o.parent if o.parent in known else root
        by_parent.setdefault(parent, []).append(o)

    def walk(parent: int, depth: int = 0) -> list[dict[str, Any]]:
        if depth > 12:  # object trees can contain cycles after a bad read
            return []
        out = []
        for child in sorted(by_parent.get(parent, []), key=lambda o: o.num):
            out.append(
                {
                    "num": child.num,
                    "name": child.name or f"#{child.num}",
                    "attributes": child.attributes,
                    "children": walk(child.num, depth + 1),
                }
            )
        return out

    return walk(root)


def name_for(objects: Iterable[WorldObject], num: int) -> str:
    for o in objects:
        if o.num == num:
            return o.name or f"#{num}"
    return f"#{num}"
