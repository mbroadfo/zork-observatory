from .base import GameEngine, Observation, WorldObject, WorldState
from .mock_engine import MockEngine

__all__ = [
    "GameEngine",
    "Observation",
    "WorldObject",
    "WorldState",
    "MockEngine",
    "build_engine",
    "engine_seed",
]


def build_engine(kind: str, rom: str | None = None, seed: int | None = 12345) -> GameEngine:
    """Engine factory. Jericho is imported lazily so Windows can still run mock.

    `seed=None` means "the seed the game's walkthrough was recorded under",
    which is what a scripted replay needs.
    """
    if kind == "mock":
        return MockEngine(seed=12345 if seed is None else seed)
    if kind == "jericho":
        if not rom:
            raise ValueError("--rom is required for the jericho engine")
        from .jericho_engine import JerichoEngine

        return JerichoEngine(rom, seed=seed)
    raise ValueError(f"Unknown engine: {kind!r} (expected 'mock' or 'jericho')")


def engine_seed(agent: str, seed: int) -> int | None:
    """A scripted run replays a recorded walkthrough, so it must roll the same
    dice the recording did; every other agent gets the seed it asked for."""
    return None if agent == "scripted" else seed
