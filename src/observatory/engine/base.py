"""The engine interface every backend implements.

Two backends exist: `JerichoEngine` (real Z-machine games, needs Linux) and
`MockEngine` (a tiny hand-built world, runs anywhere). They are interchangeable,
which is what lets the whole pipeline be developed and tested without a ROM.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from typing import Any


@dataclass(frozen=True)
class WorldObject:
    """One node of the game's internal object tree."""

    num: int
    name: str
    parent: int
    child: int
    sibling: int
    attributes: list[int] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Observation:
    """What a player — human or agent — is allowed to see."""

    text: str
    score: int = 0
    moves: int = 0
    reward: int = 0
    done: bool = False
    won: bool = False
    lost: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class WorldState:
    """Ground truth. The agent never sees this; the observatory always does."""

    location_id: int
    location_name: str
    inventory: list[str] = field(default_factory=list)
    objects: list[WorldObject] = field(default_factory=list)
    state_hash: str = ""
    score: int = 0
    moves: int = 0
    max_score: int = 0
    dark: bool = False

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return d


def story_id(data: bytes) -> str | None:
    """Identify a Z-machine story file by its header: `release-serial-checksum`.

    The filename says nothing — `zork1.z5` has shipped in a dozen releases whose
    object numbers differ, and anything keyed to object numbers (an atlas, a
    walkthrough) is only valid for the exact build it was made against.
    """
    if len(data) < 0x40 or not 1 <= data[0] <= 8:
        return None
    release = int.from_bytes(data[2:4], "big")
    serial = data[0x12:0x18].decode("ascii", "replace")
    checksum = int.from_bytes(data[0x1C:0x1E], "big")
    return f"{release}-{serial}-{checksum:04x}"


class GameEngine(ABC):
    """A deterministic, snapshot-able interactive fiction world."""

    name: str = "unknown"
    # Exact build of the story, when the backend can tell. See `story_id`.
    story: str | None = None

    @abstractmethod
    def reset(self) -> tuple[Observation, WorldState]:
        ...

    @abstractmethod
    def step(self, command: str) -> tuple[Observation, WorldState]:
        ...

    @abstractmethod
    def world_state(self) -> WorldState:
        """Ground truth as of right now, without advancing the game."""

    @abstractmethod
    def snapshot(self) -> Any:
        """Opaque blob capturing the entire world. Feeds `restore`."""

    @abstractmethod
    def restore(self, blob: Any) -> None:
        """Rewind to a snapshot. This is what makes branching possible."""

    def walkthrough(self) -> list[str] | None:
        """A known winning command list for this world as seeded, if any."""
        return None

    def valid_actions(self) -> list[str] | None:
        """Actions that actually change world state, if the backend knows."""
        return None

    @property
    def max_score(self) -> int:
        return 0

    def close(self) -> None:
        pass
