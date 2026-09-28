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
import contextlib
import uuid
from dataclasses import dataclass
from typing import Any

from .agents.base import Agent, TurnContext
from .engine.base import GameEngine, Observation, WorldState  # noqa: F401  (Observation is constructed here)
from .events import EventBus
from .notebook import Notebook
from .trace import TraceWriter
from .world.coverage import Coverage
from .world.discovery import DiscoveryLedger, Turn as DiscoveryTurn
from .world.frontier import since_new_room, structure as map_structure
from .world.graph import MapGraph, parse_movement
from .world.outcomes import OutcomeTally
from .world.recall import Exchange, Recall
from .world.vocabulary import Vocabulary
from .world.objects import build_tree, diff_objects, name_for


# How many more turns a takeover gets when nobody says. A handoff usually
# happens at the end of a budget someone set for a different player.
HANDOFF_TURNS = 200


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

    # End a run that has stopped discovering. 0 leaves it off.
    #
    # The third budget, and the one the other two miss. A run can reach a state
    # where it cannot die and cannot progress — above ground with no light
    # source, say — and then walk into the same refusal until the turn budget
    # runs out. Nothing is learned after the stall begins, but every turn of it
    # is still paid for. Censored like the others: "had not found a new room in
    # N turns" is an observation about the run, never a verdict on the player.
    stall_limit: int = 0

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
        notebook: Notebook | None = None,
        series: dict[str, int] | None = None,
    ) -> None:
        self.id = uuid.uuid4().hex[:12]
        self.engine = engine
        self.agent = agent
        self.bus = bus
        self.config = config or SessionConfig()
        self.trace = trace
        # Carried between runs; None means this run starts and ends with
        # nothing. See notebook.py.
        self.notebook = notebook
        self.run_number = 0
        self.series = series    # {"index": i, "total": n} when runs are chained
        self._final_reflection_done = False

        self.map = MapGraph()
        self.ledger = DiscoveryLedger()
        self.outcomes = OutcomeTally()
        self.coverage = Coverage()
        self.vocabulary = Vocabulary()
        self.transcript: list[tuple[str, str]] = []
        self.recall = Recall()
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
        # What the agent asked to write down this turn, held until the command
        # has run so the entry can be filed with the engine's verdict on it.
        self._pending_journal: str = ""
        # Who has played, and from which turn. Empty for the ordinary case of
        # one player for one run.
        self.handoffs: list[dict[str, Any]] = []
        # What players who have already handed over spent. Usage lives on the
        # agent, so without this a handoff resets the run's cost to zero — and
        # the readout, the trace and the cost ceiling all quietly understate
        # what the run actually cost.
        self._retired_usage: dict[str, Any] = {}
        # Where the current player's view of the transcript begins. Nonzero
        # only after a handoff that deliberately withheld its predecessor's
        # play; the observatory's own record is never trimmed.
        self._agent_floor = 0
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
        notebook = None
        if self.notebook:
            self.run_number = self.notebook.begin_run(
                self.agent.memory, self.agent.journal, session=self.id
            )
            notebook = self.notebook.summary()

        self._emit(
            "session.started",
            game=self.engine.name,
            story=self.engine.story,
            agent=self.agent.name,
            agent_kind=self.agent.kind,
            agent_config=self.agent.describe(),
            max_score=self.engine.max_score,
            config={"max_turns": self.config.max_turns},
            notebook=notebook,
            series=self.series,
            memory=self.agent.memory.to_dict(),
            journal=self.agent.journal.to_dict() if self.agent.journal is not None else None,
        )
        self._ingest(command="", obs=obs, state=state)
        await self.agent.on_start(obs.text)

    async def handoff(
        self,
        agent: Agent,
        *,
        inherit_transcript: bool = True,
        notebook: Notebook | None = None,
        max_turns: int | None = None,
        add_turns: int | None = None,
        history_turns: int | None = None,
        delay: float | None = None,
    ) -> None:
        """Put a different player at the keyboard, without moving the world.

        Same engine, same room, same score, same inventory, same map. The only
        thing that changes is who gets asked for the next command. That makes
        one thing possible that separate runs cannot: a position reached by one
        player, continued by another — a human opening the house and a model
        taking it underground, or the reverse when a run gets stuck.

        It also makes one thing impossible, and nothing here pretends
        otherwise: the score afterwards is not the new player's score. The
        trace carries `run.handoff` precisely so that every turn can be
        attributed to whoever actually typed it, and so no summary that spans
        one can be read as a result about either player.

        `inherit_transcript` is the part that matters if you care about the
        answer. Left on, the new player reads the last N exchanges of its
        predecessor's play — which, after a human, is the strongest coaching
        anywhere in this repository: expert moves, in the game's own words,
        against this exact world. Turned off, it starts from the room it is
        standing in and nothing else, which is the only version of a takeover
        comparable with a cold run.
        """
        if not self.started:
            raise RuntimeError("nothing to hand over yet — the run has not started")
        if self.finished:
            raise RuntimeError("the run is over")

        await self._interrupt()

        previous = self.agent
        # Bank what the outgoing player spent before the new one replaces it.
        for key, value in previous.usage().items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                self._retired_usage[key] = self._retired_usage.get(key, 0) + value
        self._retired_usage.pop("latency_ms_avg", None)
        self.agent = agent
        # Whatever the old player asked to write down, it never saw the reply.
        self._pending_journal = ""

        if history_turns is not None:
            self.config.history_turns = history_turns
        if delay is not None:
            self.config.delay = delay
        if max_turns is not None:
            self.config.max_turns = max_turns
        # Turns from here, counted after the interrupt: the turn the old player
        # was still thinking about has been given back by now, and a budget
        # measured against a turn number that then moved would be off by one.
        if add_turns is not None:
            self.config.max_turns = self.turn + max(add_turns, 1)
        # A budget already spent would end the run on the takeover's first
        # turn. Someone who asks for a handoff is asking for turns.
        if self.turn >= self.config.max_turns:
            self.config.max_turns = self.turn + HANDOFF_TURNS

        if not inherit_transcript:
            self._agent_floor = len(self.transcript)

        if notebook is not None:
            self.notebook = notebook
        if self.notebook is not None:
            self.run_number = self.notebook.begin_run(
                agent.memory, agent.journal, session=self.id,
                handoff_from=previous.name, handoff_turn=self.turn,
            )

        state = self._last_state
        record = {
            "turn": self.turn,
            "from": previous.name,
            "to": agent.name,
            "inherits": inherit_transcript,
        }
        self.handoffs.append(record)
        self._emit(
            "run.handoff",
            turn=self.turn,
            steps=self.steps,
            score=state.score if state else 0,
            room=state.location_name if state else "",
            from_agent=previous.name,
            from_kind=previous.kind,
            agent=agent.name,
            agent_kind=agent.kind,
            agent_config=agent.describe(),
            # False means the new player was handed the room it stands in and
            # nothing else — no record of how anyone got there.
            inherits=inherit_transcript,
            config={
                "max_turns": self.config.max_turns,
                "history_turns": self.config.history_turns,
            },
            memory=agent.memory.to_dict(),
            journal=agent.journal.to_dict() if agent.journal is not None else None,
            notebook=self.notebook.summary() if self.notebook else None,
        )
        await agent.on_start(self.transcript[-1][1] if self.transcript else "")

    def usage(self) -> dict[str, Any]:
        """What this run has cost, across every player that has had it.

        A handoff builds a new agent, and usage lives on the agent — so asking
        the current player alone reports the cost of the last leg and calls it
        the run. Anything summing money or tokens has to ask here.
        """
        total = dict(self._retired_usage)
        for key, value in self.agent.usage().items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                total[key] = total.get(key, 0) + value
            else:
                total.setdefault(key, value)
        calls = total.get("calls") or 1
        if "latency_ms_total" in total:
            total["latency_ms_avg"] = round(total["latency_ms_total"] / calls)
        if "cost_usd" in total:
            total["cost_usd"] = round(total["cost_usd"], 6)
        return total

    async def _interrupt(self) -> None:
        """Stop the loop mid-turn and wait for it to actually be stopped.

        Whatever it was awaiting is inside `agent.act` — a human who has not
        typed yet, or a model call whose answer nobody is going to read now.
        The engine has not been told anything at that point, so cancelling
        there costs a wasted inference and no world state at all.
        """
        self.pause()
        task, self._task = self._task, None
        if task is None or task.done() or task is asyncio.current_task():
            return
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

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
            ended=obs.done,
        )
        if obs.lost:
            self.map.record_death(room_id)

        # What the agent could perceive this turn. Established once, up front,
        # because three separate panels depend on it agreeing with itself.
        probe = DiscoveryTurn(
            turn=self.turn, command=command, obs=obs, state=state,
            prev_state=self._last_state, visited_rooms=set(), commands_seen=set(),
        )
        visible = probe.visible_objects()

        # What changed that the agent could have witnessed.
        #
        # The object tree is the whole world and it keeps moving without the
        # player: Zork's thief wanders the dungeon from turn one, so an
        # unfiltered delta announced his position every few turns to an agent
        # who had never met him. That is the game's knowledge, not the run's.
        changes = [
            c for c in diff_objects(self._prev_objects, state.objects)
            if c.num in visible
        ]

        # Everything above except the player, which is the observer rather than
        # part of what is observed. Zork flips a bit on the player object every
        # time a command is understood at all: `examine tree` moves nothing in
        # the world and still changes `cretin`, so counting it would call
        # looking at something progress. Measured rather than assumed —
        # examining in Zork does not touch the thing examined, while opening a
        # box sets its open bit and closing it clears the same bit again.
        player = probe.player_object()
        witnessed = any(c.num != player for c in changes)

        # Classify the turn before anything else reads the new state, so the
        # comparison is against the world as it was when the command was given.
        outcome = None
        if command:
            outcome = self.outcomes.classify(
                command, obs, state, self._last_state, witnessed=witnessed
            )
            self.vocabulary.observe(command, outcome.outcome, obs.text)

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

        # Ground covered. Reuses the same visibility set as everything else so
        # the panels never disagree about what the agent could see — and done
        # BEFORE the snapshot that carries it. Updating afterwards shipped the
        # previous turn's coverage with this turn's score, so the header and
        # the coverage panel disagreed on screen by exactly one turn.
        self.coverage.observe(
            state, visible, probe.player_object(), command=command, text=obs.text
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
            opened_containers=sorted(self.coverage.opened),
            vocabulary=self.vocabulary.summary(),
            # Carried so that a snapshot outside the ordinary flow of a turn —
            # after a rollback — can correct the turn counter too.
            turn=self.turn,
        )

        if any(delta.values()):
            self._emit(
                "map.update",
                current_room=room_id,
                stats=self.map.stats(),
                # How the map is shaped, not just how big it is. Emitted only
                # when the map changed, which is the only time most of it can
                # change; staleness drifts between those points, and the
                # post-hoc timeline in tools/structure.py is where that is read.
                structure=map_structure(self.map, self.turn, room_id),
                **{k: v for k, v in delta.items() if v},
            )

        # What the agent has worked out about the shape of the world, judged by
        # what it did rather than what it claimed.
        for discovery in self.ledger.observe(
            self.turn, command, obs, state, self._last_state,
            step=self.steps, life=self.life,
        ):
            self._emit("discovery.made", **discovery.to_dict(), summary=self.ledger.summary())

        # The journal entry the agent asked for, filed now that the turn has
        # been classified. The engine's verdict rides along with it: PROGRESS
        # means the world really did change, and anything else means the agent
        # recorded an achievement the world did not corroborate. Neither is
        # corrected or withheld — it wrote what it wrote — but the difference
        # is the measurement, so it is stored rather than inferred later.
        if self._pending_journal and self.agent.journal is not None:
            journal = self.agent.journal
            before = len(journal.rejected)
            entry = journal.add(
                self._pending_journal,
                turn=self.turn,
                # Where the command was given, not where it landed: an entry
                # about opening a window belongs to the room with the window.
                room=outcome.room if outcome else state.location_name,
                command=command,
                outcome=outcome.outcome.value if outcome else "",
                run=self.run_number,
                # Going somewhere is real change, and the map already has it —
                # every room, every passage, in more detail and without the
                # mistakes. Told here rather than worked out in agents/,
                # which does not import from world/.
                movement=parse_movement(command) is not None,
            )
            if entry is not None:
                if self.notebook:
                    self.notebook.write(entry)
                self._emit("journal.written", **entry.to_dict(), summary=journal.summary())
            elif len(journal.rejected) > before:
                # Refused, and said so. A claim the world did not bear out is
                # not kept, but it is not swallowed either: it is the half of
                # the record that says when the player stopped being reliable.
                self._emit(
                    "journal.rejected",
                    **journal.rejected[-1].to_dict(),
                    summary=journal.summary(),
                )
        self._pending_journal = ""

        self.transcript.append((command, obs.text))
        # The same exchange, keyed by turn and place, for a player that wants to
        # look further back than its window reaches. Append-only across a
        # rollback, like the map: the world goes back, having read something
        # does not un-happen.
        self.recall.add(Exchange(
            turn=self.turn, command=command, response=obs.text,
            room=state.location_name,
        ))
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
            # The run's cost, not the current player's. Asking the agent alone
            # made a handoff reset the ceiling to zero, so a run could be
            # handed from one expensive player to another and never reach a
            # budget it had already spent twice over.
            spent = self.usage().get("cost_usd", 0.0)
            if spent >= self.config.max_cost_usd:
                await self._finish(f"cost budget exhausted (${spent:.2f})", censored=True)
                return False
        if self.config.stall_limit:
            idle = since_new_room(self.map, self.turn)
            if idle >= self.config.stall_limit:
                await self._finish(
                    f"stalled — nothing new in {idle} turns", censored=True)
                return False

        self.turn += 1
        self.steps += 1
        self._emit("turn.begin", turn=self.turn, step=self.steps)

        ctx = self._context()

        try:
            action = await self.agent.act(ctx)
        except asyncio.CancelledError:
            # Someone took the keyboard, or the session is being torn down,
            # while this player was still deciding. Nothing reached the engine,
            # so this turn did not happen and must not be counted as one.
            self.turn -= 1
            self.steps -= 1
            raise
        except Exception as exc:  # an agent crash should not kill the observatory
            self._emit("error", where="agent.act", message=f"{type(exc).__name__}: {exc}")
            await self._finish(f"agent error: {exc}")
            return False

        # Before the thought: the lookups happened while it was deciding, and
        # the transcript should read in the order things occurred.
        for lookup in action.meta.get("searches") or []:
            self._emit(
                "agent.search",
                turn=self.turn,
                query=lookup.get("query", ""),
                hits=lookup.get("hits", 0),
                turns=lookup.get("turns", []),
                # What it was allowed to look through, which after a cold
                # handoff is less than the run.
                of=len(self.recall) - self._agent_floor,
            )

        if action.thought:
            self._emit("agent.thought", turn=self.turn, text=action.thought, meta=action.meta)

        self._emit("command.issued", turn=self.turn, command=action.command, meta=action.meta)

        self._pending_journal = str(action.meta.get("journal") or "")

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
        # The player's own history, which after a handoff may start later than
        # the run does. The current observation is never withheld: you can
        # take over a game without being told how it was played, but not
        # without being told where you are standing.
        own = self.transcript[self._agent_floor:]
        return TurnContext(
            turn=self.turn,
            observation=self.transcript[-1][1] if self.transcript else "",
            score=self._last_state.score if self._last_state else 0,
            moves=self._last_state.moves if self._last_state else 0,
            # [-0:] is the whole list; a window of zero means no history.
            transcript=own[-self.config.history_turns:] if self.config.history_turns > 0 else [],
            # The same boundary as the window: a player handed the room and
            # nothing else must not be able to search for the rest either.
            #
            # Asked of the player rather than of a setting, so that a handoff
            # to someone who can look things up brings the record with it, and
            # a handoff to someone who cannot takes it away, with nothing to
            # keep in sync.
            recall=(
                self.recall.since(self._agent_floor)
                if getattr(self.agent, "search", False) else None
            ),
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
            await self._record_lesson(obs.text, final=True)
            return False

        await self._record_lesson(obs.text, final=False)

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

    async def _record_lesson(self, cause: str, final: bool, kind: str = "death") -> None:
        ctx = self._context()
        if final:
            self._final_reflection_done = True
            if self.notebook:
                # The reader is the next run, which starts from the first move.
                ctx.next_start = "beginning"
        try:
            text = await self.agent.reflect(ctx, cause=cause)
        except Exception as exc:
            self._emit("error", where="agent.reflect", message=f"{type(exc).__name__}: {exc}")
            return
        if not text:
            return
        lesson = self.agent.memory.add(
            text,
            turn=self.turn,
            kind=kind,
            location=self._last_state.location_name if self._last_state else "",
            life=self.life,
            run=self.run_number,
        )
        if lesson and self.notebook:
            self.notebook.add(lesson)
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
        self._emit_snapshot(state)

    def _emit_snapshot(self, state: WorldState) -> None:
        """Say where the world is now, outside the ordinary flow of a turn.

        A rollback moves the score, the room and the inventory without a
        command having been typed, so nothing else would announce it. Without
        this the readouts kept showing the dead run's numbers — after a death
        at turn 270 the header still read turn 270 and 133 points while the
        engine was back at turn 162 with 104, which is the kind of quiet
        disagreement that makes a watcher distrust every other panel.
        """
        self._emit(
            "state.snapshot",
            location_id=state.location_id,
            location_name=state.location_name,
            room_id=MapGraph.room_id(state.location_id, state.location_name),
            inventory=state.inventory,
            state_hash=state.state_hash,
            score=state.score,
            moves=state.moves,
            max_score=state.max_score,
            dark=state.dark,
            object_count=len(state.objects),
            tree=build_tree(state.objects),
            coverage=self.coverage.summary(),
            seen_objects=sorted(self.coverage.objects_seen),
            opened_containers=sorted(self.coverage.opened),
            vocabulary=self.vocabulary.summary(),
            turn=self.turn,
        )

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
        # A run that will be followed by another gets the last word, whatever
        # ended it — unless the agent itself is what broke.
        if self.notebook and not self._final_reflection_done and not reason.startswith("agent error"):
            await self._record_lesson(reason, final=True, kind="run")
        self.finished = True
        self.end_reason = reason
        self.censored = censored
        state = self._last_state
        if self.notebook:
            self.notebook.end_run(
                reason=reason, censored=censored, turns=self.turn, steps=self.steps,
                deaths=self.deaths, score=state.score if state else 0,
                max_score=state.max_score if state else 0,
                rooms=self.map.stats()["rooms"], usage=self.usage(),
                trace=str(self.trace.path) if self.trace else None,
            )
        self._emit(
            "session.ended",
            reason=reason,
            # True when the harness stopped the run rather than the game.
            # Downstream, these are right-censored observations: "had not
            # discovered X by turn N", never "failed to discover X".
            censored=censored,
            deaths=self.deaths,
            lives_used=self.life,
            # Empty for an ordinary run. Anything in it means these numbers
            # belong to more than one player.
            handoffs=list(self.handoffs),
            memory=self.agent.memory.to_dict(),
            journal=self.agent.journal.summary() if self.agent.journal is not None else None,
            turns=self.turn,
            steps=self.steps,
            final_score=state.score if state else 0,
            max_score=state.max_score if state else 0,
            map=self.map.stats(),
            structure=map_structure(self.map, self.turn, self._prev_room_id),
            discoveries=self.ledger.summary(),
            quality=self.outcomes.summary(),
            coverage=self.coverage.summary(),
            vocabulary=self.vocabulary.summary(),
            usage=self.usage(),
            notebook=self.notebook.summary() if self.notebook else None,
            series=self.series,
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
            # Who played before, and from which turn. A summary spanning one of
            # these is a fact about the position, not about either player.
            "handoffs": list(self.handoffs),
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
            "journal": self.agent.journal.to_dict() if self.agent.journal is not None else None,
            "journal_summary": self.agent.journal.summary() if self.agent.journal is not None else None,
            "notebook": self.notebook.summary() if self.notebook else None,
            "run_number": self.run_number,
            "series": self.series,
            "score": state.score if state else 0,
            "max_score": state.max_score if state else self.engine.max_score,
            "location": state.location_name if state else "",
            "inventory": state.inventory if state else [],
            "map": self.map.stats(),
            "structure": map_structure(self.map, self.turn, self._prev_room_id),
            "discoveries": self.ledger.manifest(),
            "quality": self.outcomes.summary(),
            "coverage": self.coverage.summary(),
            "vocabulary": self.vocabulary.summary(),
            "usage": self.usage(),
            "checkpoints": [
                {
                    "id": c.id, "label": c.label, "turn": c.turn,
                    "score": c.score, "location": c.location_name, "auto": c.auto,
                }
                for c in sorted(self.checkpoints.values(), key=lambda c: c.turn)
            ],
        }
