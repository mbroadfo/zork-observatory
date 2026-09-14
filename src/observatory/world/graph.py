"""The map, discovered rather than known.

Nothing here is seeded with Zork's real map. Every room and every edge is
inferred from a movement command and the room the player ended up in, which is
why it works on any game the engine can load.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any

DIRECTION_ALIASES = {
    "n": "north", "s": "south", "e": "east", "w": "west",
    "ne": "northeast", "nw": "northwest",
    "se": "southeast", "sw": "southwest",
    "u": "up", "d": "down",
    "enter": "in", "exit": "out",
}

DIRECTIONS = {
    "north", "south", "east", "west",
    "northeast", "northwest", "southeast", "southwest",
    "up", "down", "in", "out",
}

OPPOSITES = {
    "north": "south", "south": "north", "east": "west", "west": "east",
    "northeast": "southwest", "southwest": "northeast",
    "northwest": "southeast", "southeast": "northwest",
    "up": "down", "down": "up", "in": "out", "out": "in",
}


def parse_movement(command: str) -> str | None:
    """Return the canonical direction a command attempts, or None."""
    c = command.strip().lower()
    c = c.removeprefix("go ").removeprefix("walk ").removeprefix("travel ")
    c = DIRECTION_ALIASES.get(c, c)
    return c if c in DIRECTIONS else None


@dataclass
class Room:
    id: str
    name: str
    first_seen_turn: int
    visits: int = 1
    dark: bool = False
    deaths: int = 0
    objects_seen: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Edge:
    src: str
    dst: str
    direction: str
    first_seen_turn: int
    traversals: int = 1
    reciprocal: bool = False

    @property
    def key(self) -> str:
        return f"{self.src}|{self.direction}"

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["key"] = self.key
        return d


@dataclass
class BlockedExit:
    src: str
    direction: str
    message: str
    attempts: int = 1

    @property
    def key(self) -> str:
        return f"{self.src}|{self.direction}"

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["key"] = self.key
        return d


class MapGraph:
    def __init__(self) -> None:
        self.rooms: dict[str, Room] = {}
        self.edges: dict[str, Edge] = {}
        self.blocked: dict[str, BlockedExit] = {}
        self.path: list[str] = []

    # --- construction ----------------------------------------------------

    @staticmethod
    def room_id(location_id: int, location_name: str) -> str:
        """Prefer the engine's object number — names repeat, numbers don't."""
        if location_id:
            return f"r{location_id}"
        slug = "".join(ch if ch.isalnum() else "-" for ch in location_name.lower()).strip("-")
        return f"n-{slug or 'unknown'}"

    def observe_room(self, room_id: str, name: str, turn: int, dark: bool = False) -> bool:
        """Record presence in a room. Returns True if it is newly discovered."""
        new = room_id not in self.rooms
        if new:
            self.rooms[room_id] = Room(id=room_id, name=name, first_seen_turn=turn, dark=dark)
        else:
            room = self.rooms[room_id]
            room.visits += 1
            room.dark = room.dark or dark
            if name and name != "Unknown":
                room.name = name
        if not self.path or self.path[-1] != room_id:
            self.path.append(room_id)
        return new

    def observe_transition(
        self,
        prev_id: str | None,
        command: str,
        new_id: str,
        new_name: str,
        turn: int,
        response: str = "",
        dark: bool = False,
    ) -> dict[str, Any]:
        """Fold one turn into the map. Returns what changed, for the event stream."""
        delta: dict[str, Any] = {"new_room": None, "new_edge": None, "new_blocked": None}

        new_room = self.observe_room(new_id, new_name, turn, dark)
        if new_room:
            delta["new_room"] = self.rooms[new_id].to_dict()

        direction = parse_movement(command)
        if direction is None or prev_id is None:
            return delta

        if prev_id == new_id:
            # A movement command that didn't move us: a wall, a locked door, a
            # closed window. Worth drawing — it's a fact about the map.
            key = f"{prev_id}|{direction}"
            if key in self.blocked:
                self.blocked[key].attempts += 1
            else:
                self.blocked[key] = BlockedExit(
                    src=prev_id,
                    direction=direction,
                    message=response.strip().split("\n")[0][:160],
                )
                delta["new_blocked"] = self.blocked[key].to_dict()
            return delta

        key = f"{prev_id}|{direction}"
        if key in self.edges:
            self.edges[key].traversals += 1
        else:
            edge = Edge(src=prev_id, dst=new_id, direction=direction, first_seen_turn=turn)
            back = f"{new_id}|{OPPOSITES.get(direction, '')}"
            if back in self.edges and self.edges[back].dst == prev_id:
                edge.reciprocal = True
                self.edges[back].reciprocal = True
            self.edges[key] = edge
            delta["new_edge"] = edge.to_dict()
        return delta

    def record_death(self, room_id: str) -> None:
        if room_id in self.rooms:
            self.rooms[room_id].deaths += 1

    # --- output ----------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "rooms": [r.to_dict() for r in self.rooms.values()],
            "edges": [e.to_dict() for e in self.edges.values()],
            "blocked": [b.to_dict() for b in self.blocked.values()],
            "path": self.path,
        }

    def stats(self) -> dict[str, Any]:
        one_way = sum(1 for e in self.edges.values() if not e.reciprocal)
        return {
            "rooms": len(self.rooms),
            "edges": len(self.edges),
            "blocked": len(self.blocked),
            "one_way_edges": one_way,
            "deaths": sum(r.deaths for r in self.rooms.values()),
        }
