"""Jericho backend — real Z-machine games with the object tree exposed.

Jericho (Microsoft Research) wraps a modified Frotz and hands us the things a
plain interpreter hides: the live object tree, a serializable VM state, and a
world-state hash. `get_state`/`set_state` are the important ones — they turn a
playthrough from a line into a tree you can fork.

Jericho ships a compiled Frotz and is effectively Linux-only. On Windows, run
this inside the container in docker/ or under WSL2.
"""

from __future__ import annotations

import os
from typing import Any

from .base import GameEngine, Observation, WorldObject, WorldState, story_id


class JerichoNotAvailable(RuntimeError):
    pass


def _import_frotz():
    try:
        from jericho import FrotzEnv  # type: ignore
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise JerichoNotAvailable(
            "jericho is not installed or failed to load its native library. "
            "It is effectively Linux-only — run the core in Docker "
            "(docker compose up) or under WSL2, or use --engine mock."
        ) from exc
    return FrotzEnv


class JerichoEngine(GameEngine):
    def __init__(self, rom_path: str, seed: int | None = 12345) -> None:
        """`seed=None` uses the seed Jericho recorded its walkthrough under."""
        if not os.path.exists(rom_path):
            raise FileNotFoundError(
                f"Game file not found: {rom_path}\n"
                "Zork game files are copyrighted and are not distributed with this project. "
                "Place your own .z5/.z3 under roms/ and point --rom at it."
            )
        FrotzEnv = _import_frotz()
        self.rom_path = rom_path
        self.name = os.path.splitext(os.path.basename(rom_path))[0]
        with open(rom_path, "rb") as fh:
            self.story = story_id(fh.read(0x40))
        self._env = FrotzEnv(rom_path, seed=seed)
        self.seed = self._env._seed
        self._room_parent = 0
        self._last_obs = ""

    def walkthrough(self) -> list[str] | None:
        # Jericho's walkthroughs are verified against one seed. Zork's combat
        # and thief are random, and under any other seed the same 396 commands
        # die in the forest at step 34 — so a mismatched seed gets no script
        # rather than one that silently fails.
        bindings = getattr(self._env, "bindings", None) or {}
        if not bindings.get("walkthrough") or bindings.get("seed") != self.seed:
            return None
        return list(self._env.get_walkthrough())

    # --- lifecycle -------------------------------------------------------

    def reset(self) -> tuple[Observation, WorldState]:
        text, _info = self._env.reset()
        self._last_obs = text
        # Every game starts the player standing in a room, so whatever holds
        # that room holds all of them. Learned here rather than hardcoded.
        start = self._safe(lambda: self._env.get_player_location(), None)
        self._room_parent = int(getattr(start, "parent", 0)) if start is not None else 0
        return self._observe(text, 0.0, False), self.world_state()

    def step(self, command: str) -> tuple[Observation, WorldState]:
        text, reward, done, _info = self._env.step(command)
        self._last_obs = text
        return self._observe(text, reward, done), self.world_state()

    def close(self) -> None:
        try:
            self._env.close()
        except Exception:
            pass

    # --- state -----------------------------------------------------------

    @property
    def max_score(self) -> int:
        return self._safe(lambda: self._env.get_max_score(), 0)

    def _observe(self, text: str, reward: float, done: bool) -> Observation:
        won = self._safe(lambda: bool(self._env.victory()), False)
        lost = self._safe(lambda: bool(self._env.game_over()), False)
        return Observation(
            text=text,
            score=self._safe(lambda: int(self._env.get_score()), 0),
            moves=self._safe(lambda: int(self._env.get_moves()), 0),
            reward=int(reward),
            done=bool(done) or won or lost,
            won=won,
            lost=lost,
        )

    @staticmethod
    def _attributes(obj: Any) -> list[int]:
        """Jericho reports attributes as a numpy bool array, not a list.

        Never write `getattr(o, "attr", []) or []` here: `or` forces a truth
        test, and numpy raises on an array with more than one element. Returns
        the indices of the set attribute flags, which is the useful form.
        """
        attrs = getattr(obj, "attr", None)
        if attrs is None:
            return []
        try:
            return [i for i, flag in enumerate(attrs) if bool(flag)]
        except TypeError:
            return []

    def world_state(self) -> WorldState:
        loc = self._safe(lambda: self._env.get_player_location(), None)

        # `or []` is unsafe on anything Jericho returns — several accessors hand
        # back numpy arrays. Normalise with an explicit None check instead.
        raw_objects = self._safe(lambda: self._env.get_world_objects(), None)
        raw_inventory = self._safe(lambda: self._env.get_inventory(), None)

        objects = [
            WorldObject(
                num=int(getattr(o, "num", 0)),
                name=str(getattr(o, "name", "")).strip(),
                parent=int(getattr(o, "parent", 0)),
                child=int(getattr(o, "child", 0)),
                sibling=int(getattr(o, "sibling", 0)),
                attributes=self._attributes(o),
            )
            for o in (raw_objects if raw_objects is not None else [])
            # Jericho returns the whole object table, including a long tail of
            # unused slots with empty names. They are noise in every view.
            if str(getattr(o, "name", "")).strip()
        ]
        inventory = [
            str(getattr(o, "name", "")).strip()
            for o in (raw_inventory if raw_inventory is not None else [])
        ]
        loc_id, loc_name = self._room_of(loc, raw_objects)
        text = (self._last_obs or "").lower()
        return WorldState(
            location_id=loc_id,
            location_name=loc_name,
            inventory=inventory,
            objects=objects,
            state_hash=str(self._safe(lambda: self._env.get_world_state_hash(), "")),
            score=self._safe(lambda: int(self._env.get_score()), 0),
            moves=self._safe(lambda: int(self._env.get_moves()), 0),
            max_score=self.max_score,
            dark="pitch black" in text or "too dark" in text,
        )

    def _room_of(self, loc: Any, raw_objects: Any) -> tuple[int, str]:
        """The room the player is in, even when they are inside something.

        Jericho reports the player's immediate parent. In Zork's magic boat
        that is the boat, so a river trip showed up as a room called "magic
        boat" and the real river rooms went missing from the map. Climb the
        object tree until the parent is the room container.
        """
        if loc is None:
            return 0, "Unknown"
        num = int(getattr(loc, "num", 0))
        name = str(getattr(loc, "name", "Unknown")).strip()
        parent = int(getattr(loc, "parent", 0))
        root = getattr(self, "_room_parent", 0)
        if not root or parent == root or raw_objects is None:
            return num, name

        by_num = {int(getattr(o, "num", 0)): o for o in raw_objects}
        seen = {num}
        while parent and parent != root and parent not in seen and parent in by_num:
            seen.add(parent)
            holder = by_num[parent]
            num, name = parent, str(getattr(holder, "name", "")).strip()
            parent = int(getattr(holder, "parent", 0))
        if parent != root:
            # Not inside any room we know of — report what Jericho said.
            return int(getattr(loc, "num", 0)), str(getattr(loc, "name", "Unknown")).strip()
        return num, name

    # --- branching -------------------------------------------------------

    def snapshot(self) -> Any:
        return self._env.get_state()

    def restore(self, blob: Any) -> None:
        self._env.set_state(blob)

    def valid_actions(self) -> list[str] | None:
        """Expensive — Jericho brute-forces candidates against the world state.

        Fine for a single inspection, far too slow to call every turn in a
        long run, so the session only asks for it on demand.
        """
        return self._safe(lambda: list(self._env.get_valid_actions()), None)

    # --- helpers ---------------------------------------------------------

    @staticmethod
    def _safe(fn, default):
        """Jericho's surface varies a little across versions and games.

        A missing accessor should degrade one field, not take down the run.
        """
        try:
            return fn()
        except Exception:
            return default
