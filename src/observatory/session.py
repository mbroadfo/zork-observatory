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
from .engine.base import GameEngine, Observation, WorldState  # noqa: F401  (Observation is constructed here)
from .events import EventBus
from .trace import TraceWriter
from .world.coverage import Coverage
from .world.discovery import DiscoveryLedger, Turn as DiscoveryTurn
from .world.graph import MapGraph
from .world.outcomes import OutcomeTally
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
    # A budget, NOT a rule of the game.
    #
    # Zork has no turn limit. It has a lamp that runs down and a world that
    # kills you, and those are the real constraints. This number exists only to
    # stop a runaway loop spending money, and a run that hits it must be read
    # as *censored*, not failed: an agent that had not discovered containers by
    # turn 200 has not failed to discover them, it ran out of budget. Every
    # summary carries `censored` for exactly this reason — treating budget
    # exhaustion as an outcome would corrupt any time-to-discovery statistic.
    max_turns: int = 400
    max_cost_usd: float = 0.0    # 0 = no cost ceiling; the other budget that matters

    delay: float = 0.35          # seconds between turns, so a human can watch
    collect_valid_actions: bool = False   # expensive on Jericho; off by default
    history_turns: int = 30

    # Death handling. `lives` is how many times the world may be rolled back
    # under the agent after a death; 0 is ironman. What the agent keeps across
    # a rollback is its memory, and nothing else — see agents/memory.py.
    lives: int = 0
    allow_agent_save: bool = True   # the agent may type SAVE and RESTORE itself

    # How far back a death rolls the world, when the agent has no save of its
    # own. Restoring to the most recent checkpoint is the obvious choice and
    # the wrong one: checkpoints land on a timer, so the latest one is often
    # *inside* the situation that just killed you — back in the dark room, one
    # turn from the same death. A run would burn every life in four turns
    # without ever acting on what it learned. A margin gives the memory
    # somewhere to be useful, which is the only reason the life exists.
    death_rollback_margin: int = 5

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
        self.outcomes = OutcomeTally()
        self.coverage = Coverage()
        self.transcript: list[tuple[str, str]] = []
        self.checkpoints: dict[str, Checkpoint] = {}

        self.turn = 0
        # Turns actually played. Unlike `turn`, a rollback never rewinds this,
        # so it is the honest x-axis for anything measured over a run.
        self.steps = 0
        self.life = 1
        self.lives_left = self.config.lives
        self.deaths = 0
        self.started = False
        self.finished = False
        self.censored = False
        self.end_reason = ""
        self._agent_save: Checkpoint | None = None
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

        # Classify the turn before anything else reads the new state, so the
        # comparison is against the world as it was when the command was given.
        outcome = None
        if command:
            outcome = self.outcomes.classify(command, obs, state, self._last_state)

        self._emit(
            "observation",
            text=obs.text,
            score=obs.score,
            moves=obs.moves,
            reward=obs.reward,
            done=obs.done,
            won=obs.won,
            lost=obs.lost,
            outcome=outcome.to_dict() if outcome else None,
            quality=self.outcomes.summary() if outcome else None,
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
            coverage=self.coverage.summary(),
            # Which objects the agent has actually laid eyes on. The tree is
            # the whole world; without this the front end cannot tell the
            # difference between "the trophy case holds a painting" and "the
            # trophy case exists somewhere and holds a painting the agent has
            # never seen", and it renders a walkthrough instead of a run.
            seen_objects=sorted(self.coverage.objects_seen),
        )

        if any(delta.values()):
            self._emit(
                "map.update",
                current_room=room_id,
                stats=self.map.stats(),
                **{k: v for k, v in delta.items() if v},
            )

        # Ground covered. Reuses the ledger's notion of "what could the player
        # see from here" so the two panels never disagree about visibility.
        probe = DiscoveryTurn(
            turn=self.turn, command=command, obs=obs, state=state,
            prev_state=self._last_state, visited_rooms=set(), commands_seen=set(),
        )
        self.coverage.observe(state, probe.visible_objects(), probe.player_object())

        # What the agent has worked out about the shape of the world, judged by
        # what it did rather than what it claimed.
        for discovery in self.ledger.observe(
            self.turn, command, obs, state, self._last_state,
            step=self.steps, life=self.life,
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
            await self._finish("turn budget exhausted", censored=True)
            return False
        if self.config.max_cost_usd:
            spent = self.agent.usage().get("cost_usd", 0.0)
            if spent >= self.config.max_cost_usd:
                await self._finish(f"cost budget exhausted (${spent:.2f})", censored=True)
                return False

        self.turn += 1
        self.steps += 1
        self._emit("turn.begin", turn=self.turn, step=self.steps)

        ctx = self._context()

        try:
            action = await self.agent.act(ctx)
        except Exception as exc:  # an agent crash should not kill the observatory
            self._emit("error", where="agent.act", message=f"{type(exc).__name__}: {exc}")
            await self._finish(f"agent error: {exc}")
            return False

        if action.thought:
            self._emit("agent.thought", turn=self.turn, text=action.thought, meta=action.meta)

        self._emit("command.issued", turn=self.turn, command=action.command, meta=action.meta)

        # SAVE and RESTORE never reach the parser.
        #
        # A real Z-machine prompts for a filename, which would deadlock a loop
        # that only knows how to send one line per turn. Handling them here
        # gives the agent the same affordance a human had at a TRS-80 — one
        # slot, restore returns you to it — using the snapshot machinery that
        # already exists for branching.
        if self.config.allow_agent_save:
            handled = self._handle_save_command(action.command)
            if handled is not None:
                self._ingest(command=action.command, obs=handled[0], state=handled[1])
                return True

        obs, state = self.engine.step(action.command)
        self._ingest(command=action.command, obs=obs, state=state)

        if self.config.checkpoint_every and self.turn % self.config.checkpoint_every == 0:
            self.checkpoint(f"turn {self.turn}", auto=True)

        if obs.done:
            if obs.lost:
                self.deaths += 1
                self.map.record_death(self._prev_room_id or "")
                if await self._try_another_life(obs):
                    return True
            await self._finish("victory" if obs.won else "death" if obs.lost else "game over")
            return False
        return True

    def _context(self) -> TurnContext:
        return TurnContext(
            turn=self.turn,
            observation=self.transcript[-1][1] if self.transcript else "",
            score=self._last_state.score if self._last_state else 0,
            moves=self._last_state.moves if self._last_state else 0,
            transcript=self.transcript[-self.config.history_turns:],
            valid_actions=self.engine.valid_actions() if self.config.collect_valid_actions else None,
            life=self.life,
            lives_left=self.lives_left,
        )

    def _handle_save_command(self, command: str) -> tuple[Observation, WorldState] | None:
        """Intercept SAVE / RESTORE. Returns None for anything else."""
        word = command.strip().lower().rstrip(".")
        if word in ("save", "save game"):
            self._agent_save = self.checkpoint(f"agent save, turn {self.turn}")
            return self._synthetic("Ok.")
        if word in ("restore", "restore game", "load", "load game"):
            target = self._agent_save
            if target is None:
                return self._synthetic("There is no saved game.")
            self._restore(target, reason="agent restore")
            state = self.engine.world_state()
            self._emit(
                "run.restored",
                life=self.life,
                lives_left=self.lives_left,
                deaths=self.deaths,
                to_turn=target.turn,
                label=target.label,
                carried=len(self.agent.memory),
                by_agent=True,
            )
            return self._synthetic(f"Ok.\n{state.location_name}")
        return None

    def _synthetic(self, text: str) -> tuple[Observation, WorldState]:
        """A reply from the harness rather than the game."""
        state = self.engine.world_state()
        return (
            Observation(
                text=text,
                score=state.score,
                moves=state.moves,
                reward=0,
                done=False,
            ),
            state,
        )

    async def _try_another_life(self, obs: Observation) -> bool:
        """Let the agent write down what it thinks happened, then roll back.

        The order matters: reflection happens while the death is still the most
        recent thing in the transcript, and the rollback happens after. The
        world goes back; the memory does not.
        """
        if self.lives_left <= 0:
            await self._record_lesson(obs, final=True)
            return False

        await self._record_lesson(obs, final=False)

        # The agent's own save wins: it chose that point, and second-guessing
        # a deliberate choice would make SAVE mean something other than save.
        target = self._agent_save or self._safe_rollback_target()
        if target is None:
            return False

        self.lives_left -= 1
        self.life += 1
        self._restore(target, reason="death")
        self._emit(
            "run.restored",
            life=self.life,
            lives_left=self.lives_left,
            deaths=self.deaths,
            to_turn=target.turn,
            label=target.label,
            carried=len(self.agent.memory),
        )
        return True

    async def _record_lesson(self, obs: Observation, final: bool) -> None:
        try:
            text = await self.agent.reflect(self._context(), cause=obs.text)
        except Exception as exc:
            self._emit("error", where="agent.reflect", message=f"{type(exc).__name__}: {exc}")
            return
        if not text:
            return
        lesson = self.agent.memory.add(
            text,
            turn=self.turn,
            kind="death",
            location=self._last_state.location_name if self._last_state else "",
            life=self.life,
        )
        if lesson:
            self._emit("lesson.learned", **lesson.to_dict(), final=final, deaths=self.deaths)

    def _latest_checkpoint_before(self, turn: int) -> Checkpoint | None:
        earlier = [c for c in self.checkpoints.values() if c.turn < turn]
        return max(earlier, key=lambda c: c.turn) if earlier else None

    def _safe_rollback_target(self) -> Checkpoint | None:
        """The latest checkpoint far enough back to be worth returning to.

        Falls back to the earliest checkpoint when nothing clears the margin —
        early deaths are exactly the case where the margin matters most, and
        the alternative is restarting into the jaws of the same grue.
        """
        cutoff = self.turn - self.config.death_rollback_margin
        target = self._latest_checkpoint_before(cutoff)
        if target is not None:
            return target
        if not self.checkpoints:
            return None
        return min(self.checkpoints.values(), key=lambda c: c.turn)

    def _restore(self, cp: Checkpoint, reason: str) -> None:
        """Put the world back.

        The map and the agent's memory are deliberately untouched. The world
        forgets; neither the observatory nor the player does. `reason` is for
        the caller's readability at the call sites — the events that describe
        a restore are emitted by those callers, which know more than this does.
        """
        self.engine.restore(cp.blob)
        self.turn = cp.turn
        self.transcript = list(cp.transcript)
        self.finished = False
        self.end_reason = ""
        state = self.engine.world_state()
        self._prev_objects = list(state.objects)
        self._prev_room_id = MapGraph.room_id(state.location_id, state.location_name)
        self._last_state = state

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

    async def _finish(self, reason: str, censored: bool = False) -> None:
        if self.finished:
            return
        self.finished = True
        self.end_reason = reason
        self.censored = censored
        state = self._last_state
        self._emit(
            "session.ended",
            reason=reason,
            # True when the harness stopped the run rather than the game.
            # Downstream, these are right-censored observations: "had not
            # discovered X by turn N", never "failed to discover X".
            censored=censored,
            deaths=self.deaths,
            lives_used=self.life,
            memory=self.agent.memory.to_dict(),
            turns=self.turn,
            steps=self.steps,
            final_score=state.score if state else 0,
            max_score=state.max_score if state else 0,
            map=self.map.stats(),
            discoveries=self.ledger.summary(),
            quality=self.outcomes.summary(),
            coverage=self.coverage.summary(),
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
        """Put the world back. The map and the agent's memory are untouched."""
        cp = self.checkpoints.get(checkpoint_id)
        if cp is None:
            return False
        self._restore(cp, reason="operator rewind")
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
            "steps": self.steps,
            "max_turns": self.config.max_turns,
            "started": self.started,
            "finished": self.finished,
            "paused": self._paused,
            "end_reason": self.end_reason,
            "censored": self.censored,
            "deaths": self.deaths,
            "life": self.life,
            "lives_left": self.lives_left,
            "memory": self.agent.memory.to_dict(),
            "score": state.score if state else 0,
            "max_score": state.max_score if state else self.engine.max_score,
            "location": state.location_name if state else "",
            "inventory": state.inventory if state else [],
            "map": self.map.stats(),
            "discoveries": self.ledger.manifest(),
            "quality": self.outcomes.summary(),
            "coverage": self.coverage.summary(),
            "usage": self.agent.usage(),
            "checkpoints": [
                {
                    "id": c.id, "label": c.label, "turn": c.turn,
                    "score": c.score, "location": c.location_name, "auto": c.auto,
                }
                for c in sorted(self.checkpoints.values(), key=lambda c: c.turn)
            ],
        }
