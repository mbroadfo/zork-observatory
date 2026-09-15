"""Death, restore, and the one thing that crosses the boundary.

Restoring rewinds the world and not the player. These tests pin that asymmetry
from both sides: the world really does go back, and the memory really does not.
"""

from __future__ import annotations

from observatory.agents.base import Agent, AgentAction
from observatory.agents.memory import MAX_LESSON_CHARS, AgentMemory
from observatory.agents.simple import ScriptedAgent
from observatory.engine.mock_engine import MockEngine
from observatory.events import EventBus
from observatory.session import Session, SessionConfig

# Walk into the unlit attic and stay there: turn 4 enters the dark, turn 5 dies.
SUICIDE = ["north", "east", "west", "up", "look"]


class Learner(ScriptedAgent):
    """A scripted agent that writes down a fixed lesson whenever it dies."""

    def __init__(self, commands, lesson: str | None = "It was dark and something killed me."):
        super().__init__(commands)
        self.lesson = lesson
        self.reflections = 0
        self.memory_at_reflect: list[int] = []

    async def reflect(self, ctx, cause: str) -> str | None:
        self.reflections += 1
        self.memory_at_reflect.append(len(self.memory))
        return self.lesson


class Doomed(Learner):
    """Walks back into the dark every time, however many lives it is given.

    The control for memory: it writes lessons down and acts on none of them.
    """

    def __init__(self) -> None:
        super().__init__([])
        self._i = 0

    async def act(self, ctx) -> AgentAction:
        command = SUICIDE[self._i % len(SUICIDE)]
        self._i += 1
        return AgentAction(command=command, thought="(walking the same path again)")


def make(agent: Agent, **cfg) -> tuple[Session, list]:
    bus = EventBus()
    seen: list = []
    bus.subscribe(seen.append)
    session = Session(
        MockEngine(), agent, bus,
        config=SessionConfig(delay=0.0, max_turns=cfg.pop("max_turns", 40), **cfg),
    )
    return session, seen


class TestAgentMemory:
    def test_an_empty_memory_renders_as_nothing_at_all(self):
        """Not "you have not learned anything yet" — at the bottom of the
        information ladder that sentence would reveal that dying is possible."""
        assert AgentMemory().render() == ""

    def test_lessons_render_with_their_turn_and_place(self):
        memory = AgentMemory()
        memory.add("Do not go down without light.", turn=12, location="Cellar")

        rendered = memory.render()
        assert "turn 12" in rendered and "Cellar" in rendered
        assert "Do not go down without light." in rendered

    def test_blank_lessons_are_not_recorded(self):
        memory = AgentMemory()
        assert memory.add("   ", turn=1) is None
        assert memory.add("", turn=1) is None
        assert len(memory) == 0

    def test_a_long_lesson_is_truncated_rather_than_dropped(self):
        memory = AgentMemory()
        lesson = memory.add("x" * (MAX_LESSON_CHARS + 500), turn=1)

        assert lesson is not None
        assert len(lesson.text) == MAX_LESSON_CHARS

    def test_only_recent_lessons_are_recalled_but_none_are_lost(self):
        memory = AgentMemory(recall=3)
        for i in range(10):
            memory.add(f"lesson {i}", turn=i)

        rendered = memory.render()
        assert "lesson 9" in rendered
        assert "lesson 0" not in rendered
        assert "+7 older notes" in rendered
        assert len(memory) == 10

    def test_every_agent_has_a_memory_without_needing_an_init(self):
        agent = ScriptedAgent(["look"])
        agent.memory.add("something", turn=1)
        assert len(agent.memory) == 1

    def test_memories_are_not_shared_between_agents(self):
        a, b = ScriptedAgent(["look"]), ScriptedAgent(["look"])
        a.memory.add("mine", turn=1)
        assert len(b.memory) == 0


class TestLivesAndRestore:
    async def test_ironman_is_the_default_and_death_ends_the_run(self):
        session, _ = make(Learner(SUICIDE))
        await session.run()

        assert session.finished
        assert session.end_reason == "death"
        assert session.deaths == 1
        assert session.life == 1

    async def test_a_life_rolls_the_world_back_and_the_run_continues(self):
        session, seen = make(Learner(SUICIDE + ["look"] * 20), lives=1, checkpoint_every=2)
        await session.run()

        restores = [e for e in seen if e.type == "run.restored"]
        assert len(restores) == 1
        assert restores[0].payload["life"] == 2
        assert session.life == 2

    async def test_lives_run_out_for_an_agent_that_will_not_learn(self):
        """A memory the agent ignores buys it nothing; the lives still run out."""
        session, seen = make(Doomed(), lives=2, checkpoint_every=2)
        await session.run()

        assert session.lives_left == 0
        assert session.deaths == 3           # two rollbacks, then the last one
        assert session.end_reason == "death"
        assert len([e for e in seen if e.type == "run.restored"]) == 2

    async def test_the_world_really_goes_back(self):
        session, _ = make(Learner(SUICIDE + ["look"] * 10), lives=1, checkpoint_every=2)
        await session.start()
        for _ in range(5):        # through the death
            await session.step_once()

        # Restored clear of the lethal attic, not back into it.
        assert session._last_state.location_name != "Attic"
        assert not session.finished

    async def test_the_rollback_clears_the_situation_that_killed_it(self):
        """Restoring to the newest checkpoint would land back in the dark room
        one turn from the same death, burning every life for nothing."""
        session, _ = make(
            Learner(SUICIDE + ["look"] * 10),
            lives=1, checkpoint_every=2, death_rollback_margin=5,
        )
        await session.start()
        for _ in range(5):
            await session.step_once()

        assert session._last_state.dark is False
        assert session.turn < 4      # behind the "up" that entered the attic

    async def test_an_agents_own_save_beats_the_margin(self):
        """A deliberate choice is not second-guessed."""
        session, _ = make(
            Learner(["save"] + SUICIDE + ["look"] * 10),
            lives=1, checkpoint_every=2,
        )
        await session.start()
        for _ in range(6):
            await session.step_once()

        assert session._last_state.location_name == "West of House"

    async def test_the_memory_does_not(self):
        """The whole point: the lamp is gone, the lesson isn't."""
        agent = Learner(SUICIDE + ["look"] * 10)
        session, _ = make(agent, lives=1, checkpoint_every=2)
        await session.start()
        for _ in range(5):
            await session.step_once()

        assert len(agent.memory) == 1
        assert "dark" in agent.memory.lessons[0].text
        assert agent.memory.lessons[0].location == "Attic"

    async def test_reflection_happens_before_the_rollback(self):
        """Ordering matters — the agent must still be able to see the death in
        its transcript when it writes about it."""
        agent = Learner(SUICIDE + ["look"] * 10)
        session, seen = make(agent, lives=1, checkpoint_every=2)
        await session.run()

        types = [e.type for e in seen]
        assert types.index("lesson.learned") < types.index("run.restored")

    async def test_a_lesson_is_recorded_even_on_the_final_death(self):
        """Ironman still learns — the memory is the artifact, not the retry."""
        agent = Learner(SUICIDE)
        session, seen = make(agent)
        await session.run()

        lessons = [e for e in seen if e.type == "lesson.learned"]
        assert len(lessons) == 1
        assert lessons[0].payload["final"] is True
        assert len(agent.memory) == 1

    async def test_an_agent_that_learns_nothing_carries_nothing(self):
        agent = Learner(SUICIDE, lesson=None)
        session, seen = make(agent)
        await session.run()

        assert agent.reflections == 1
        assert len(agent.memory) == 0
        assert not [e for e in seen if e.type == "lesson.learned"]

    async def test_a_crashing_reflect_does_not_take_down_the_run(self):
        class Brittle(ScriptedAgent):
            async def reflect(self, ctx, cause):
                raise RuntimeError("reflection failed")

        session, seen = make(Brittle(SUICIDE))
        await session.run()

        assert session.finished
        assert any(e.type == "error" and e.payload["where"] == "agent.reflect" for e in seen)

    async def test_the_final_summary_carries_the_memory(self):
        agent = Learner(SUICIDE)
        session, seen = make(agent)
        await session.run()

        end = seen[-1].payload
        assert end["deaths"] == 1
        assert len(end["memory"]) == 1
        assert "dark" in end["memory"][0]["text"]


class TestAgentInvokedSaveRestore:
    """The agent may type SAVE and RESTORE itself, as a human at a TRS-80 could."""

    async def test_save_is_answered_by_the_harness_not_the_parser(self):
        session, _ = make(ScriptedAgent(["save"]))
        await session.start()
        await session.step_once()

        assert session.transcript[-1][1] == "Ok."
        assert session._agent_save is not None

    async def test_restore_without_a_save_says_so(self):
        session, _ = make(ScriptedAgent(["restore"]))
        await session.start()
        await session.step_once()

        assert "no saved game" in session.transcript[-1][1].lower()

    async def test_restore_returns_the_world_to_the_saved_point(self):
        session, seen = make(ScriptedAgent(["save", "north", "east", "restore"]))
        await session.run()

        assert session._last_state.location_name == "West of House"
        assert any(e.type == "run.restored" and e.payload["by_agent"] for e in seen)

    async def test_save_commands_never_reach_the_game(self):
        """A real Z-machine prompts for a filename, which would deadlock a loop
        that can only send one line per turn."""
        session, _ = make(ScriptedAgent(["save"]))
        await session.start()
        before = session.engine.world_state().moves
        await session.step_once()

        assert session.engine.world_state().moves == before

    async def test_the_affordance_can_be_withheld(self):
        session, _ = make(ScriptedAgent(["save"]), allow_agent_save=False)
        await session.start()
        await session.step_once()

        assert session._agent_save is None
        assert "not available" in session.transcript[-1][1].lower()
