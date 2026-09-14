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

from .base import GameEngine, Observation, WorldObject, WorldState


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
    def __init__(self, rom_path: str, seed: int = 12345) -> None:
        if not os.path.exists(rom_path):
            raise FileNotFoundError(
                f"Game file not found: {rom_path}\n"
                "Zork game files are copyrighted and are not distributed with this project. "
                "Place your own .z5/.z3 under roms/ and point --rom at it."
            )
        FrotzEnv = _import_frotz()
        self.rom_path = rom_path
        self.seed = seed
        self.name = os.path.splitext(os.path.basename(rom_path))[0]
        self._env = FrotzEnv(rom_path, seed=seed)
        self._last_obs = ""

    # --- lifecycle -------------------------------------------------------

    def reset(self) -> tuple[Observation, WorldState]:
        text, _info = self._env.reset()
        self._last_obs = text
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

    def world_state(self) -> WorldState:
        loc = self._safe(lambda: self._env.get_player_location(), None)
        objects = [
            WorldObject(
                num=int(getattr(o, "num", 0)),
                name=str(getattr(o, "name", "")).strip(),
                parent=int(getattr(o, "parent", 0)),
                child=int(getattr(o, "child", 0)),
                sibling=int(getattr(o, "sibling", 0)),
                attributes=list(getattr(o, "attr", []) or []),
            )
            for o in self._safe(lambda: self._env.get_world_objects(), []) or []
        ]
        inventory = [
            str(getattr(o, "name", "")).strip()
            for o in self._safe(lambda: self._env.get_inventory(), []) or []
        ]
        text = (self._last_obs or "").lower()
        return WorldState(
            location_id=int(getattr(loc, "num", 0)) if loc is not None else 0,
            location_name=str(getattr(loc, "name", "Unknown")).strip() if loc is not None else "Unknown",
            inventory=inventory,
            objects=objects,
            state_hash=str(self._safe(lambda: self._env.get_world_state_hash(), "")),
            score=self._safe(lambda: int(self._env.get_score()), 0),
            moves=self._safe(lambda: int(self._env.get_moves()), 0),
            max_score=self.max_score,
            dark="pitch black" in text or "too dark" in text,
        )

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
