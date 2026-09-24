"""What shape the discovered map is in, beyond how big it is.

Room and edge counts say how much was found. They say nothing about how, and
the September 2026 sweep turned entirely on that difference: the arm that
scored 40 and the arm that scored 0 both spent 350 turns walking, and the gap
between them was visible in the shape of the map long before it reached the
score. Twelve rooms found by pacing the same corridor and twelve rooms found by
pushing outward are the same number and not the same run.

Everything here is derived from MapGraph, which is derived from movement, which
is derived from what the player typed. Nothing consults the game, the object
tree or a walkthrough — these numbers are meant to be comparable across stories
the observatory has never seen, and to stay on the observatory's side of the
line that agents/base.py draws.

The measures, and what each is for:

  radius          how far from the starting room the map reaches, in moves
                  along edges the agent found itself. The single number that
                  separates the sweep's arms most cleanly: a player looping
                  around the house has a radius of two whatever its room count
                  says.

  components      fragments with no known route between them. More than one
                  means the agent arrived somewhere it cannot explain — after
                  a death and a respawn, usually, or a one-way drop.

  reachable       rooms it could walk to from where it stands, following only
                  edges in the direction it actually traversed them. A
                  frontier you cannot get back to is not a frontier.

  revisit_ratio   of the moves that changed room, the fraction that landed
                  somewhere already known. Nought is pure outward exploration;
                  a player pacing a corridor approaches one.

  tried_per_room  distinct directions attempted per room discovered, counting
                  walls as attempts. Distinguishes working a room from walking
                  through it. `probed_once` is the tail of that: rooms the
                  agent entered, left by one direction, and never examined.

  since_new_room  turns since the map last grew. The stuck-detector, and the
                  one measure that would have ended the sweep early: the
                  episodic arm found its twelfth room on turn 70 and its
                  thirteenth never, so by turn 105 this read 35 and climbing
                  and there was nothing left to learn from the remaining 245.

  stale           the room left alone longest, and for how many turns. Useful
                  live — "you have not been back to X in two hundred turns" —
                  but weak as a summary statistic, because in practice it
                  names the starting room in every run that ever left it.

  unretried_walls refusals hit exactly once. The coached prompt says a refusal
                  sticks until something changes; this counts the ones the
                  agent never went back to test after something did. A high
                  number is incuriosity, a very low one is the eleven-times
                  loop this project started out trying to explain.

Two honest limits, recorded here rather than left for someone to rediscover.
Radius uses the map as an undirected shape, because a room the agent stood in
was reached whether or not the way back is known; `reachable` is the directed
question and they are deliberately different. And `tried_per_room` counts
against twelve compass directions, of which a typical room has three, so its
absolute value means little — only its spread across arms does.
"""

from __future__ import annotations

from collections import deque
from typing import Any

from .graph import MapGraph


def tried_directions(graph: MapGraph) -> dict[str, set[str]]:
    """Directions attempted from each room, whether they worked or not."""
    out: dict[str, set[str]] = {room_id: set() for room_id in graph.rooms}
    for edge in graph.edges.values():
        out.setdefault(edge.src, set()).add(edge.direction)
    for wall in graph.blocked.values():
        out.setdefault(wall.src, set()).add(wall.direction)
    return out


def _adjacency(graph: MapGraph, undirected: bool) -> dict[str, set[str]]:
    adj: dict[str, set[str]] = {room_id: set() for room_id in graph.rooms}
    for edge in graph.edges.values():
        adj.setdefault(edge.src, set()).add(edge.dst)
        if undirected:
            adj.setdefault(edge.dst, set()).add(edge.src)
    return adj


def _depths(adj: dict[str, set[str]], start: str) -> dict[str, int]:
    """Breadth-first distance from `start`, for the rooms it can get to."""
    depth = {start: 0}
    queue = deque([start])
    while queue:
        room = queue.popleft()
        for nxt in adj.get(room, ()):
            if nxt not in depth:
                depth[nxt] = depth[room] + 1
                queue.append(nxt)
    return depth


def components(graph: MapGraph) -> list[set[str]]:
    """Groups of rooms with some known route between them, ignoring direction."""
    adj = _adjacency(graph, undirected=True)
    seen: set[str] = set()
    groups: list[set[str]] = []
    for room_id in graph.rooms:
        if room_id in seen:
            continue
        group = set(_depths(adj, room_id))
        seen |= group
        groups.append(group)
    return groups


def radius(graph: MapGraph) -> int:
    """Greatest distance from the starting room, over the map's known shape."""
    if not graph.path:
        return 0
    depths = _depths(_adjacency(graph, undirected=True), graph.path[0])
    return max(depths.values(), default=0)


def reachable(graph: MapGraph, room_id: str | None) -> int:
    """Rooms walkable from `room_id` by edges already traversed that way."""
    if not room_id or room_id not in graph.rooms:
        return 0
    return len(_depths(_adjacency(graph, undirected=False), room_id))


def stalest(graph: MapGraph, turn: int, current: str | None = None) -> tuple[str | None, int]:
    """The room left alone longest, and for how long.

    The room being stood in is excluded: it is not abandoned, it is occupied.
    """
    candidates = [
        room for room in graph.rooms.values()
        if room.id != current
    ]
    if not candidates:
        return None, 0
    oldest = max(candidates, key=lambda r: turn - r.last_seen_turn)
    return oldest.name, max(0, turn - oldest.last_seen_turn)


def since_new_room(graph: MapGraph, turn: int) -> int:
    """Turns since the map last grew.

    Exact, thresholdless, and the earliest honest signal that a run is over
    before its turn budget is: a player still finding rooms may yet do
    something, and a player that has not found one in two hundred turns will
    not.
    """
    newest = max((room.first_seen_turn for room in graph.rooms.values()), default=turn)
    return max(0, turn - newest)


def structure(graph: MapGraph, turn: int, current: str | None = None) -> dict[str, Any]:
    """Every measure above, as one payload for the event stream.

    Cheap enough to compute on each map change: the graphs are hundreds of
    rooms at the very most, and in practice tens.
    """
    tried = tried_directions(graph)
    rooms = len(graph.rooms)
    moves = max(0, len(graph.path) - 1)
    revisits = max(0, len(graph.path) - rooms)
    stale_room, stale_turns = stalest(graph, turn, current)

    return {
        "radius": radius(graph),
        "components": len(components(graph)),
        "reachable": reachable(graph, current),
        "moves_between_rooms": moves,
        "revisits": revisits,
        "revisit_ratio": round(revisits / moves, 3) if moves else 0.0,
        "tried_per_room": round(sum(len(d) for d in tried.values()) / rooms, 2) if rooms else 0.0,
        "probed_once": sum(1 for d in tried.values() if len(d) <= 1),
        "unretried_walls": sum(1 for w in graph.blocked.values() if w.attempts == 1),
        "since_new_room": since_new_room(graph, turn),
        "stale_room": stale_room,
        "stale_turns": stale_turns,
    }
