"""The turn loop.

One session drives one engine with one agent and narrates everything it does
onto the event bus. It owns three things the engine doesn't: the discovered
map, the turn-over-turn object diff, and the checkpoint stack that makes
branching possible.

Note what rewinding does *not* undo: the map. The world goes back to turn 34,
the observatory keeps everything it learned. That asymmetry — a world that
forgets and an observer that doesn't — is the reason the map survives
counterfactual exploration.
"""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass
from typing import Any

from .agents.base import Agent, TurnContext
from .engine.base import GameEngine, Observation, WorldState
from .events import EventBus
from .trace import TraceWriter
from .world.discovery import DiscoveryLedger
from .world.graph import MapGraph
from .world.objects import build_tree, diff_objects, name_for


@dataclass
class Checkpoint:
    id: str
    label: str
    turn: int
    blob: Any
    transcript: list[tuple[str, str]]
    location_name: str
    score: int
    auto: bool = False


@dataclass
class SessionConfig:
    max_turns: int = 200
    delay: float = 0.35          # seconds between turns, so a human can watch
    collect_valid_actions: bool = False   # expensive on Jericho; off by default
    history_turns: int = 30

    # Checkpoint on a timer as well as on demand. You almost never know a turn
    # mattered until later — by the time the run walks into a dark room and
    # dies, the decision worth re-running was twenty turns back. Automatic
    # checkpoints mean the fork points exist before you know you want them.
    checkpoint_every: int = 10
    max_auto_checkpoints: int = 40   # VM snapshots are not free; keep a window


class Session:
    def __init__(
        self,
        engine: GameEngine,
        agent: Agent,
        bus: EventBus,
        config: SessionConfig | None = None,
        trace: TraceWriter | None = None,
    ) -> None:
        self.id = uuid.uuid4().hex[:12]
        self.engine = engine
        self.agent = agent
        self.bus = bus
        self.config = config or SessionConfig()
        self.trace = trace

        self.map = MapGraph()
        self.ledger = DiscoveryLedger()
        self.transcript: list[tuple[str, str]] = []
        self.checkpoints: dict[str, Checkpoint] = {}

        self.turn = 0
        self.started = False
        self.finished = False
        self.end_reason = ""
        self._paused = True
        self._resume = asyncio.Event()
        self._prev_objects: list = []
        self._prev_room_id: str | None = None
        self._last_state: WorldState | None = None
        self._task: asyncio.Task | None = None

    # --- event plumbing --------------------------------------------------

    def _emit(self, type_: str, **payload: Any) -> None:
        event = self.bus.emit(type_, session_id=self.id, **payload)  # type: ignore[arg-type]
        if self.trace:
            self.trace.write(event)

    # --- lifecycle -------------------------------------------------------

    async def start(self) -> None:
        if self.started:
            return
        self.started = True
        obs, state = self.engine.reset()

        self._emit(
            "session.started",
            game=self.engine.name,
            agent=self.agent.name,
            agent_kind=self.agent.kind,
            agent_config=self.agent.describe(),
            max_score=self.engine.max_score,
            config={"max_turns": self.config.max_turns},
        )
        self._ingest(command="", obs=obs, state=state)
        await self.agent.on_start(obs.text)

    def _ingest(self, command: str, obs: Observation, state: WorldState) -> None:
        """Fold one engine result into map, transcript, and the event stream."""
        room_id = MapGraph.room_id(state.location_id, state.location_name)

        delta = self.map.observe_transition(
            prev_id=self._prev_room_id,
            command=command,
            new_id=room_id,
            new_name=state.location_name,
            turn=self.turn,
            response=obs.text,
            dark=state.dark,
        )
        if obs.lost:
            self.map.record_death(room_id)

        self._emit(
            "observation",
            text=obs.text,
            score=obs.score,
            moves=obs.moves,
            reward=obs.reward,
            done=obs.done,
            won=obs.won,
            lost=obs.lost,
        )

        changes = diff_objects(self._prev_objects, state.objects)
        if changes:
            self._emit(
                "object.delta",
                changes=[
                    dict(
                        c.to_dict(),
                        before_name=name_for(state.objects, c.before) if isinstance(c.before, int) else None,
                        after_name=name_for(state.objects, c.after) if isinstance(c.after, int) else None,
                    )
                    for c in changes
                ],
            )

        self._emit(
            "state.snapshot",
            location_id=state.location_id,
            location_name=state.location_name,
            room_id=room_id,
            inventory=state.inventory,
            state_hash=state.state_hash,
            score=state.score,
            moves=state.moves,
            max_score=state.max_score,
            dark=state.dark,
            object_count=len(state.objects),
            tree=build_tree(state.objects),
        )

        if any(delta.values()):
            self._emit(
                "map.update",
                current_room=room_id,
                stats=self.map.stats(),
                **{k: v for k, v in delta.items() if v},
            )

        # What the agent has worked out about the shape of the world, judged by
        # what it did rather than what it claimed.
        for discovery in self.ledger.observe(
            self.turn, command, obs, state, self._last_state
        ):
            self._emit("discovery.made", **discovery.to_dict(), summary=self.ledger.summary())

        self.transcript.append((command, obs.text))
        self._prev_objects = list(state.objects)
        self._prev_room_id = room_id
        self._last_state = state

    # --- turns -----------------------------------------------------------

    async def step_once(self) -> bool:
        """Play exactly one turn. Returns False when the run is over."""
        if self.finished:
            return False
        if not self.started:
            await self.start()
        if self.turn >= self.config.max_turns:
            await self._finish("turn limit reached")
            return False

        self.turn += 1
        self._emit("turn.begin", turn=self.turn)

        ctx = TurnContext(
            turn=self.turn,
            observation=self.transcript[-1][1] if self.transcript else "",
            score=self._last_state.score if self._last_state else 0,
            moves=self._last_state.moves if self._last_state else 0,
            transcript=self.transcript[-self.config.history_turns:],
            valid_actions=self.engine.valid_actions() if self.config.collect_valid_actions else None,
        )

        try:
            action = await self.agent.act(ctx)
        except Exception as exc:  # an agent crash should not kill the observatory
            self._emit("error", where="agent.act", message=f"{type(exc).__name__}: {exc}")
            await self._finish(f"agent error: {exc}")
            return False

        if action.thought:
            self._emit("agent.thought", turn=self.turn, text=action.thought, meta=action.meta)

        self._emit("command.issued", turn=self.turn, command=action.command, meta=action.meta)

        obs, state = self.engine.step(action.command)
        self._ingest(command=action.command, obs=obs, state=state)

        if self.config.checkpoint_every and self.turn % self.config.checkpoint_every == 0:
            self.checkpoint(f"turn {self.turn}", auto=True)

        if obs.done:
            await self._finish("victory" if obs.won else "death" if obs.lost else "game over")
            return False
        return True

    async def run(self) -> None:
        """Run until the game ends, the turn budget runs out, or someone pauses."""
        self._paused = False
        self._resume.set()
        if not self.started:
            await self.start()
        while not self.finished:
            if self._paused:
                self._resume.clear()
                await self._resume.wait()
                continue
            if not await self.step_once():
                break
            if self.config.delay:
                await asyncio.sleep(self.config.delay)

    def start_background(self) -> asyncio.Task:
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self.run())
        return self._task

    def pause(self) -> None:
        self._paused = True

    def resume(self) -> None:
        self._paused = False
        self._resume.set()

    @property
    def paused(self) -> bool:
        return self._paused

    async def _finish(self, reason: str) -> None:
        if self.finished:
            return
        self.finished = True
        self.end_reason = reason
        state = self._last_state
        self._emit(
            "session.ended",
            reason=reason,
            turns=self.turn,
            final_score=state.score if state else 0,
            max_score=state.max_score if state else 0,
            map=self.map.stats(),
            discoveries=self.ledger.summary(),
            usage=self.agent.usage(),
        )
        await self.agent.on_end(reason)
        if self.trace:
            self.trace.close()

    # --- branching -------------------------------------------------------

    def checkpoint(self, label: str = "", auto: bool = False) -> Checkpoint:
        """Freeze the world. Cheap, and the basis of every counterfactual."""
        cp = Checkpoint(
            id=uuid.uuid4().hex[:8],
            label=label or f"turn {self.turn}",
            turn=self.turn,
            blob=self.engine.snapshot(),
            transcript=list(self.transcript),
            location_name=self._last_state.location_name if self._last_state else "",
            score=self._last_state.score if self._last_state else 0,
            auto=auto,
        )
        self.checkpoints[cp.id] = cp
        self._evict_old_auto_checkpoints()
        self._emit(
            "checkpoint.created",
            id=cp.id,
            label=cp.label,
            turn=cp.turn,
            score=cp.score,
            location=cp.location_name,
            auto=auto,
        )
        return cp

    def _evict_old_auto_checkpoints(self) -> None:
        """Drop the oldest automatic checkpoints past the window.

        Manual marks are never evicted — someone chose those deliberately.
        """
        autos = [c for c in self.checkpoints.values() if c.auto]
        excess = len(autos) - self.config.max_auto_checkpoints
        if excess <= 0:
            return
        for cp in sorted(autos, key=lambda c: c.turn)[:excess]:
            self.checkpoints.pop(cp.id, None)

    def rewind(self, checkpoint_id: str) -> bool:
        """Put the world back. The map keeps everything it has learned."""
        cp = self.checkpoints.get(checkpoint_id)
        if cp is None:
            return False
        self.engine.restore(cp.blob)
        self.turn = cp.turn
        self.transcript = list(cp.transcript)
        self.finished = False
        self.end_reason = ""
        state = self.engine.world_state()
        self._prev_objects = list(state.objects)
        self._prev_room_id = MapGraph.room_id(state.location_id, state.location_name)
        self._last_state = state
        self._emit(
            "turn.begin",
            turn=self.turn,
            rewound_to=checkpoint_id,
            label=cp.label,
        )
        return True

    def summary(self) -> dict[str, Any]:
        state = self._last_state
        return {
            "id": self.id,
            "game": self.engine.name,
            "agent": self.agent.name,
            "agent_kind": self.agent.kind,
            "turn": self.turn,
            "max_turns": self.config.max_turns,
            "started": self.started,
            "finished": self.finished,
            "paused": self._paused,
            "end_reason": self.end_reason,
            "score": state.score if state else 0,
            "max_score": state.max_score if state else self.engine.max_score,
            "location": state.location_name if state else "",
            "inventory": state.inventory if state else [],
            "map": self.map.stats(),
            "discoveries": self.ledger.manifest(),
            "usage": self.agent.usage(),
            "checkpoints": [
                {
                    "id": c.id, "label": c.label, "turn": c.turn,
                    "score": c.score, "location": c.location_name, "auto": c.auto,
                }
                for c in sorted(self.checkpoints.values(), key=lambda c: c.turn)
            ],
        }
