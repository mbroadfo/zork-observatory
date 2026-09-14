from .base import GameEngine, Observation, WorldObject, WorldState
from .mock_engine import MockEngine

__all__ = [
    "GameEngine",
    "Observation",
    "WorldObject",
    "WorldState",
    "MockEngine",
    "build_engine",
]


def build_engine(kind: str, rom: str | None = None, seed: int = 12345) -> GameEngine:
    """Engine factory. Jericho is imported lazily so Windows can still run mock."""
    if kind == "mock":
        return MockEngine(seed=seed)
    if kind == "jericho":
        if not rom:
            raise ValueError("--rom is required for the jericho engine")
        from .jericho_engine import JerichoEngine

        return JerichoEngine(rom, seed=seed)
    raise ValueError(f"Unknown engine: {kind!r} (expected 'mock' or 'jericho')")
