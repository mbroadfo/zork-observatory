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

# The parser's vocabulary.
#
# This started at eight verbs, which was fine for a test fixture and badly
# wrong for a world agents are measured in. A real Infocom parser knows well
# over a hundred verbs, and — this is the part that matters — it *distinguishes*
# "I don't know that word" from "you can't see that" from "that does nothing".
# A world that answers every unfamiliar verb with "I don't know that word"
# teaches an agent that its vocabulary is hopeless rather than that its idea
# was wrong, and the failure lands on the engine while looking like the agent's.
VERB_SYNONYMS = {
    "x": "examine", "l": "look", "i": "inventory", "inv": "inventory",
    "z": "wait", "get": "take", "grab": "take", "hold": "take",
    "carry": "take", "hit": "attack", "kill": "attack", "fight": "attack",
    "strike": "attack", "break": "attack", "smash": "attack",
    "discard": "drop", "release": "drop", "put down": "drop",
    "pick up": "take", "look at": "examine", "look in": "search",
    "look inside": "search", "turn on": "light", "switch on": "light",
    "turn off": "extinguish", "switch off": "extinguish", "douse": "extinguish",
    "shut": "close", "peruse": "read", "consume": "eat", "taste": "eat",
    "sip": "drink", "shove": "push", "tug": "pull", "yank": "pull",
    "go in": "enter", "go out": "exit", "get out": "exit", "leave": "exit",
    "walk": "go", "run": "go", "travel": "go", "head": "go",
}

# Verbs the parser recognises but that mostly just report back. Their value is
# that they fail *informatively* — "that does nothing" is a fact about the
# world; "I don't know that word" is a fact about the parser.
INERT_VERBS = {
    "wait": "Time passes.",
    "jump": "You jump on the spot, fooling no one.",
    "pray": "Nothing happens.",
    "sing": "Your singing is abominable.",
    "sleep": "You aren't sleepy.",
    "listen": "You hear nothing unexpected.",
    "smell": "You smell nothing unexpected.",
    "yell": "Aaaarrrrgggghhhh!",
    "shout": "Aaaarrrrgggghhhh!",
    "hello": "Nothing happens here.",
    "xyzzy": "A hollow voice says 'Fool.'",
    "dig": "You have nothing to dig with.",
    "swim": "There is no water here.",
    "fly": "You can't fly.",
    "count": "You have counted. Well done.",
}

TRANSITIVE_VERBS = {
    "take", "drop", "open", "close", "read", "examine", "search", "push",
    "pull", "move", "turn", "attack", "eat", "drink", "climb", "throw",
    "wear", "touch", "knock", "light", "extinguish", "lock", "unlock",
    "enter", "put", "insert", "tie", "burn", "fill", "empty",
}

KNOWN_VERBS = (
    TRANSITIVE_VERBS
    | set(INERT_VERBS)
    | {"look", "inventory", "score", "go", "exit", "save", "restore", "quit", "version"}
)


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

    @staticmethod
    def _split(raw: str) -> tuple[str, str]:
        """Pull the verb off the front, honouring two-word verbs."""
        for phrase in sorted(VERB_SYNONYMS, key=len, reverse=True):
            if " " in phrase and (raw == phrase or raw.startswith(phrase + " ")):
                return VERB_SYNONYMS[phrase], raw[len(phrase):].strip()
        verb, _, rest = raw.partition(" ")
        return VERB_SYNONYMS.get(verb, verb), rest.strip()

    def _dispatch(self, raw: str) -> str:
        room = self._rooms[self._here]

        # Bare directions.
        word = DIRECTION_ALIASES.get(raw, raw)
        if word in ("north", "south", "east", "west", "up", "down", "in", "out",
                    "northeast", "northwest", "southeast", "southwest"):
            return self._go(word, room)

        verb, rest = self._split(raw)
        noun = rest.removeprefix("the ").removeprefix("a ").strip()

        if verb == "go":
            target = DIRECTION_ALIASES.get(noun, noun)
            if not target:
                return "Go where?"
            return self._go(target, room)

        if verb not in KNOWN_VERBS:
            return f'I don\'t know the word "{verb}".'

        if verb in INERT_VERBS and not noun:
            return INERT_VERBS[verb]

        if verb == "look":
            if noun:
                return self._examine(noun)
            return self._describe_room()
        if verb == "inventory":
            inv = self._inventory()
            if not inv:
                return "You are empty-handed."
            return "You are carrying:\n" + "\n".join(f"  A {t.name}" for t in inv)
        if verb == "score":
            return f"Your score is {self._score} of a possible {MAX_SCORE}, in {self._moves} moves."
        if verb == "exit":
            return self._go("out", room)
        if verb == "enter" and not noun:
            return self._go("in", room)
        if verb in ("save", "restore", "quit", "version"):
            # The session intercepts save/restore when the agent is allowed
            # them. Reaching here means it isn't, so say so plainly rather than
            # pretending the word is unknown.
            return f"{verb.capitalize()} is not available in this world."

        # Everything below needs something to act on.
        if verb in TRANSITIVE_VERBS and not noun:
            return f"What do you want to {verb}?"

        if verb == "take":
            return self._take(noun)
        if verb == "drop":
            return self._drop(noun)
        if verb == "open":
            return self._open(noun)
        if verb == "close":
            return self._close(noun)
        if verb == "examine":
            return self._examine(noun)
        if verb == "search":
            return self._search(noun)
        if verb == "read":
            t = self._resolve(noun)
            if isinstance(t, str):
                return t
            return t.text or f"There's nothing written on the {t.name}."
        if verb == "light":
            return self._light(noun, on=True)
        if verb == "extinguish":
            return self._light(noun, on=False)
        if verb in ("put", "insert"):
            for sep in (" in ", " into ", " on "):
                if sep in rest:
                    item, _, dest = rest.partition(sep)
                    return self._put(item.strip(), dest.strip())
            return f"What do you want to put the {noun} in?"
        if verb == "enter":
            return self._go("in", room)

        # Recognised verb, recognised object, nothing to do. This is the most
        # useful failure a parser can produce: the idea was understood and
        # rejected, which is information the agent can act on.
        t = self._resolve(noun)
        if isinstance(t, str):
            return t
        return {
            "push": f"Pushing the {t.name} doesn't do anything.",
            "pull": f"The {t.name} won't budge.",
            "move": f"Moving the {t.name} reveals nothing.",
            "turn": f"The {t.name} won't turn.",
            "attack": f"Attacking the {t.name} accomplishes little.",
            "eat": f"The {t.name} is not something you can eat.",
            "drink": f"The {t.name} is not something you can drink.",
            "climb": f"You can't climb the {t.name}.",
            "throw": f"Throwing the {t.name} would be pointless.",
            "wear": f"You can't wear the {t.name}.",
            "touch": f"Touching the {t.name} tells you nothing.",
            "knock": f"Nobody answers at the {t.name}.",
            "lock": f"The {t.name} has no lock.",
            "unlock": f"The {t.name} has no lock.",
            "tie": f"You can't tie the {t.name} to anything.",
            "burn": f"You have nothing to burn the {t.name} with.",
            "fill": f"You can't fill the {t.name}.",
            "empty": f"The {t.name} is not something you can empty.",
        }.get(verb, f"You can't do that to the {t.name}.")

    def _resolve(self, phrase: str) -> "Thing | str":
        """Find a visible object, or return the refusal to print."""
        if not self._lit():
            return "It's too dark to see."
        t = self._find(phrase, self._visible_here() + self._inventory())
        if t is None:
            return "You can't see any such thing."
        return t

    def _examine(self, phrase: str) -> str:
        t = self._resolve(phrase)
        if isinstance(t, str):
            return t
        extra = ""
        if t.container:
            inside = [x.name for x in self._things.values() if x.location == t.id]
            if not t.is_open:
                extra = f" The {t.name} is closed."
            elif inside:
                extra = f" It contains: {', '.join(inside)}."
            else:
                extra = f" The {t.name} is empty."
        if t.id == 104:
            extra += " It is currently " + ("on." if self._lamp_on else "off.")
        return t.description + extra

    def _search(self, phrase: str) -> str:
        t = self._resolve(phrase)
        if isinstance(t, str):
            return t
        if not t.container:
            return f"You find nothing of interest in the {t.name}."
        if not t.is_open:
            return f"The {t.name} is closed."
        inside = [x.name for x in self._things.values() if x.location == t.id]
        if inside:
            return f"The {t.name} contains: {', '.join(inside)}."
        return f"The {t.name} is empty."

    def _close(self, phrase: str) -> str:
        t = self._resolve(phrase)
        if isinstance(t, str):
            return t
        if not t.container:
            return f"You can't close the {t.name}."
        if not t.is_open:
            return f"The {t.name} is already closed."
        t.is_open = False
        return "Closed."

    def _light(self, phrase: str, on: bool) -> str:
        t = self._find(phrase, self._visible_here() + self._inventory())
        if t is None:
            return "You can't see any such thing."
        if t.id != 104:
            return f"The {t.name} is not a light source."
        if on and not self._holding(t):
            return f"You aren't holding the {t.name}."
        self._lamp_on = on
        return f"The {t.name} is now {'on' if on else 'off'}."

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
