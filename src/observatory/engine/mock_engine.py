"""A miniature interactive fiction world.

Not a Zork clone — a test fixture with teeth. It has the features the
observatory needs to render something meaningful: a room graph with blocked
exits, containers, a dark room that kills you, portable objects, treasures, and
a trophy case that scores. Runs on any OS with no ROM, which means the UI and
the trace format can be developed without Docker in the loop.
"""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

from .base import GameEngine, Observation, WorldObject, WorldState

DIRECTION_ALIASES = {
    "n": "north", "s": "south", "e": "east", "w": "west",
    "ne": "northeast", "nw": "northwest",
    "se": "southeast", "sw": "southwest",
    "u": "up", "d": "down",
}


@dataclass
class Room:
    id: int
    name: str
    description: str
    exits: dict[str, int] = field(default_factory=dict)
    blocked: dict[str, str] = field(default_factory=dict)
    dark: bool = False


@dataclass
class Thing:
    id: int
    name: str
    description: str
    location: int          # room id, or -1 for player inventory, or a container id
    portable: bool = False
    container: bool = False
    is_open: bool = True
    treasure: bool = False
    text: str | None = None


PLAYER = -1        # sentinel for "held", used inside Thing.location
PLAYER_OBJ = 99    # the player's object number in the reported object tree
MAX_SCORE = 50

ROOMS = [
    Room(1, "West of House", "You are standing in an open field west of a white house, with a boarded front door.",
         exits={"north": 2, "south": 3}, blocked={"east": "The door is boarded and you can't remove the boards."}),
    Room(2, "North of House", "You are facing the north side of a white house. There is no door here.",
         exits={"south": 1, "east": 4}),
    Room(3, "South of House", "You are behind a white house, in a path winding through a forest.",
         exits={"north": 1, "east": 4}),
    Room(4, "Behind House", "You are behind the white house. One small window is slightly ajar.",
         exits={"north": 2, "south": 3, "east": 7, "west": 5, "in": 5}),
    Room(5, "Kitchen", "You are in the kitchen of the white house. A table holds an elongated brown sack.",
         exits={"west": 6, "east": 4, "out": 4, "up": 8}),
    Room(6, "Living Room", "You are in the living room. There is a doorway to the east and a wooden door with strange gothic lettering to the west.",
         exits={"east": 5, "down": 9}),
    Room(7, "Forest Clearing", "You are in a clearing, with a forest surrounding you on all sides.",
         exits={"west": 4}),
    Room(8, "Attic", "This is the attic. The only exit is a stairway leading down.",
         exits={"down": 5}, dark=True),
    Room(9, "Cellar", "You are in a dark and damp cellar with a narrow passageway leading north.",
         exits={"up": 6}, dark=True),
]

THINGS = [
    Thing(101, "small mailbox", "A small mailbox, closed.", 1, container=True, is_open=False),
    Thing(102, "leaflet", "A leaflet.", 101, portable=True,
          text="WELCOME TO THE OBSERVATORY!\nA deterministic world, kept under glass."),
    Thing(103, "brown sack", "An elongated brown sack, smelling of hot peppers.", 5, portable=True),
    Thing(104, "brass lantern", "A battery-powered brass lantern.", 6, portable=True),
    Thing(105, "elvish sword", "A sword of elvish workmanship.", 6, portable=True),
    Thing(106, "trophy case", "A trophy case, its glass front standing open.", 6, container=True),
    Thing(107, "painting", "A painting of unparalleled beauty.", 8, portable=True, treasure=True),
    Thing(108, "jeweled egg", "A jewel-encrusted egg, of exquisite workmanship.", 9, portable=True, treasure=True),
]


class MockEngine(GameEngine):
    name = "mock"

    def __init__(self, seed: int = 0) -> None:
        self.seed = seed
        self._rooms = {r.id: r for r in ROOMS}
        self._reset_mutable()

    # --- lifecycle -------------------------------------------------------

    def _reset_mutable(self) -> None:
        self._things = {t.id: copy.copy(t) for t in THINGS}
        self._here = 1
        self._score = 0
        self._moves = 0
        self._done = False
        self._won = False
        self._lost = False
        self._lamp_on = False
        self._banked: set[int] = set()

    def reset(self) -> tuple[Observation, WorldState]:
        self._reset_mutable()
        text = f"{self._rooms[self._here].name}\n{self._describe_room()}"
        return self._observe(text, 0), self.world_state()

    # --- state -----------------------------------------------------------

    @property
    def max_score(self) -> int:
        return MAX_SCORE

    def _lit(self) -> bool:
        room = self._rooms[self._here]
        if not room.dark:
            return True
        lamp = self._things[104]
        return self._lamp_on and self._holding(lamp)

    def _holding(self, thing: Thing) -> bool:
        if thing.location == PLAYER:
            return True
        parent = self._things.get(thing.location)
        return parent is not None and parent.location == PLAYER and parent.is_open

    def _visible_here(self) -> list[Thing]:
        out = []
        for t in self._things.values():
            if t.location == self._here:
                out.append(t)
            elif t.location in self._things:
                parent = self._things[t.location]
                if parent.location == self._here and parent.is_open:
                    out.append(t)
        return out

    def _inventory(self) -> list[Thing]:
        return [t for t in self._things.values() if t.location == PLAYER]

    def _describe_room(self) -> str:
        if not self._lit():
            return "It is pitch black. You are likely to be eaten by a grue."
        room = self._rooms[self._here]
        lines = [room.description]
        for t in self._visible_here():
            if t.location == self._here:
                lines.append(f"There is a {t.name} here.")
            else:
                lines.append(f"The {self._things[t.location].name} contains: {t.name}")
        return "\n".join(lines)

    def world_state(self) -> WorldState:
        room = self._rooms[self._here]

        # Shaped like a real Z-machine object tree: rooms are objects, the
        # player is an object inside one of them, and everything carried hangs
        # off the player. That makes the state pane show the same structure
        # here as it does under Jericho — and it makes "you moved from Kitchen
        # to Living Room" fall out of the object diff for free.
        objects = [
            WorldObject(num=r.id, name=r.name, parent=0, child=0, sibling=0)
            for r in self._rooms.values()
        ]
        objects.append(
            WorldObject(num=PLAYER_OBJ, name="you", parent=self._here, child=0, sibling=0)
        )
        objects += [
            WorldObject(
                num=t.id,
                name=t.name,
                parent=PLAYER_OBJ if t.location == PLAYER else t.location,
                child=0,
                sibling=0,
                attributes=[1] if t.location == PLAYER else [],
            )
            for t in self._things.values()
        ]
        raw = json.dumps(
            {str(t.id): t.location for t in sorted(self._things.values(), key=lambda x: x.id)}
            | {"here": self._here, "score": self._score},
            sort_keys=True,
        )
        return WorldState(
            location_id=room.id,
            location_name=room.name,
            inventory=[t.name for t in self._inventory()],
            objects=objects,
            state_hash=hashlib.sha1(raw.encode()).hexdigest()[:16],
            score=self._score,
            moves=self._moves,
            max_score=MAX_SCORE,
            dark=not self._lit(),
        )

    def _observe(self, text: str, reward: int) -> Observation:
        return Observation(
            text=text,
            score=self._score,
            moves=self._moves,
            reward=reward,
            done=self._done,
            won=self._won,
            lost=self._lost,
        )

    # --- snapshot / restore ---------------------------------------------

    def snapshot(self) -> Any:
        return copy.deepcopy(
            {
                "things": {i: copy.copy(t) for i, t in self._things.items()},
                "here": self._here,
                "score": self._score,
                "moves": self._moves,
                "done": self._done,
                "won": self._won,
                "lost": self._lost,
                "lamp_on": self._lamp_on,
                "banked": set(self._banked),
            }
        )

    def restore(self, blob: Any) -> None:
        blob = copy.deepcopy(blob)
        self._things = blob["things"]
        self._here = blob["here"]
        self._score = blob["score"]
        self._moves = blob["moves"]
        self._done = blob["done"]
        self._won = blob["won"]
        self._lost = blob["lost"]
        self._lamp_on = blob["lamp_on"]
        self._banked = blob["banked"]

    def valid_actions(self) -> list[str] | None:
        acts = ["look", "inventory"]
        if self._lit():
            acts += [d for d in self._rooms[self._here].exits]
            for t in self._visible_here():
                if t.portable and t.location != PLAYER:
                    acts.append(f"take {t.name}")
                if t.container and not t.is_open:
                    acts.append(f"open {t.name}")
        for t in self._inventory():
            acts.append(f"drop {t.name}")
            if t.treasure:
                acts.append(f"put {t.name} in trophy case")
        return sorted(set(acts))

    # --- the parser ------------------------------------------------------

    def step(self, command: str) -> tuple[Observation, WorldState]:
        if self._done:
            return self._observe("The game is over.", 0), self.world_state()

        raw = command.strip().lower()
        self._moves += 1
        before = self._score
        text = self._dispatch(raw)

        # Darkness is lethal one turn after you notice it.
        if not self._lit() and raw not in ("turn on lantern", "turn on lamp", "light lamp"):
            if getattr(self, "_grue_warned", False):
                text = "Oh no! You have walked into the slavering fangs of a lurking grue!\n*** You have died ***"
                self._done = True
                self._lost = True
            else:
                self._grue_warned = True
        else:
            self._grue_warned = False

        return self._observe(text, self._score - before), self.world_state()

    def _find(self, phrase: str, pool: list[Thing]) -> Thing | None:
        phrase = phrase.strip()
        if not phrase:
            return None
        for t in pool:
            if phrase == t.name:
                return t
        for t in pool:
            if phrase in t.name or t.name.split()[-1] == phrase:
                return t
        return None

    def _dispatch(self, raw: str) -> str:
        word = DIRECTION_ALIASES.get(raw, raw)
        room = self._rooms[self._here]

        if word in ("north", "south", "east", "west", "up", "down", "in", "out",
                    "northeast", "northwest", "southeast", "southwest"):
            return self._go(word, room)
        if raw.startswith("go "):
            return self._go(DIRECTION_ALIASES.get(raw[3:], raw[3:]), room)

        if raw in ("look", "l"):
            return self._describe_room()
        if raw in ("inventory", "i"):
            inv = self._inventory()
            if not inv:
                return "You are empty-handed."
            return "You are carrying:\n" + "\n".join(f"  A {t.name}" for t in inv)
        if raw in ("turn on lantern", "turn on lamp", "light lamp", "turn on brass lantern"):
            lamp = self._things[104]
            if not self._holding(lamp):
                return "You don't have the lantern."
            self._lamp_on = True
            return "The brass lantern is now on."
        if raw in ("turn off lantern", "turn off lamp"):
            self._lamp_on = False
            return "The brass lantern is now off."

        if raw.startswith(("take ", "get ", "pick up ")):
            return self._take(raw.split(" ", 1)[1].removeprefix("up "))
        if raw.startswith("drop "):
            return self._drop(raw[5:])
        if raw.startswith("open "):
            return self._open(raw[5:])
        if raw.startswith("read "):
            t = self._find(raw[5:], self._visible_here() + self._inventory())
            if t is None:
                return "You can't see that here."
            return t.text or f"There's nothing written on the {t.name}."
        if raw.startswith("put ") and " in " in raw:
            item, _, dest = raw[4:].partition(" in ")
            return self._put(item, dest)
        if raw in ("score",):
            return f"Your score is {self._score} of a possible {MAX_SCORE}, in {self._moves} moves."

        return "I don't know that word."

    def _go(self, direction: str, room: Room) -> str:
        if direction in room.blocked:
            return room.blocked[direction]
        if direction not in room.exits:
            return "You can't go that way."
        self._here = room.exits[direction]
        dest = self._rooms[self._here]
        if not self._lit():
            return "It is pitch black. You are likely to be eaten by a grue."
        return f"{dest.name}\n{self._describe_room()}"

    def _take(self, phrase: str) -> str:
        if not self._lit():
            return "It's too dark to see."
        t = self._find(phrase, self._visible_here())
        if t is None:
            return "You can't see that here."
        if not t.portable:
            return f"The {t.name} is securely fastened in place."
        t.location = PLAYER
        return "Taken."

    def _drop(self, phrase: str) -> str:
        t = self._find(phrase, self._inventory())
        if t is None:
            return "You aren't carrying that."
        t.location = self._here
        return "Dropped."

    def _open(self, phrase: str) -> str:
        t = self._find(phrase, self._visible_here())
        if t is None:
            return "You can't see that here."
        if not t.container:
            return f"You can't open the {t.name}."
        if t.is_open:
            return f"The {t.name} is already open."
        t.is_open = True
        inside = [x.name for x in self._things.values() if x.location == t.id]
        if inside:
            return f"Opening the {t.name} reveals: {', '.join(inside)}."
        return f"The {t.name} is empty."

    def _put(self, item: str, dest: str) -> str:
        t = self._find(item, self._inventory())
        if t is None:
            return "You aren't carrying that."
        c = self._find(dest, self._visible_here())
        if c is None or not c.container:
            return "You can't put anything in that."
        if not c.is_open:
            return f"The {c.name} is closed."
        t.location = c.id
        if c.id == 106 and t.treasure and t.id not in self._banked:
            self._banked.add(t.id)
            self._score += 25
            won = len(self._banked) == sum(1 for x in THINGS if x.treasure)
            if won:
                self._done = True
                self._won = True
                return ("Done.\nAn almost inaudible voice whispers, 'Look to your treasures for "
                        "the final secret.'\n*** You have won ***")
            return "Done. Your score just went up."
        return "Done."
