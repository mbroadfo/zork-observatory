"""What a player carries from one run to the next.

Inside a run, a death rolls the world back and the agent keeps its memory.
A notebook is the same bargain stretched over whole runs: the game starts again
from the first move, the agent starts again from nothing, and the only thing
that crosses is what the agent itself wrote down. No map, no transcript, no
hint from the harness.

That makes a series of runs an experiment in its own right. If run five
scores better than run one, the notebook is the only thing that can have
caused it, and the notebook is short enough to read.

One notebook per player per story. "Player" is the agent's full name — model
and information level — because a note written under a coached prompt is not
something an uncoached player should inherit. "Story" is the release the
notes were written about; notes about one game mean nothing in another.
"""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any

from .agents.memory import AgentMemory, Lesson

MODES = ("off", "carry", "new")


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9.]+", "-", text.lower()).strip("-") or "unnamed"


class Notebook:
    def __init__(self, path: Path, story: str, agent: str) -> None:
        self.path = path
        self.story = story
        self.agent = agent
        self.lessons: list[Lesson] = []
        self.runs: list[dict[str, Any]] = []

    # --- files -----------------------------------------------------------

    @classmethod
    def open(cls, root: Path, story: str, agent: str, fresh: bool = False) -> "Notebook":
        """The notebook for this player and story, created if absent.

        `fresh` starts a new one. The old one is set aside, never deleted: it
        is the record of what an earlier lineage believed.
        """
        path = Path(root) / slug(story) / f"{slug(agent)}.json"
        if fresh and path.exists():
            stamp = time.strftime("%Y%m%d-%H%M%S")
            path.rename(path.with_name(f"{path.stem}.{stamp}.json"))
        book = cls(path, story, agent)
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            book.lessons = [Lesson.from_dict(d) for d in data.get("lessons", [])]
            book.runs = list(data.get("runs", []))
        return book

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.to_dict(), indent=1), encoding="utf-8")
        os.replace(tmp, self.path)

    def to_dict(self) -> dict[str, Any]:
        return {
            "story": self.story,
            "agent": self.agent,
            "runs": self.runs,
            "lessons": [lesson.to_dict() for lesson in self.lessons],
        }

    # --- a run -----------------------------------------------------------

    def begin_run(self, memory: AgentMemory, **meta: Any) -> int:
        """Hand the agent everything written so far and open a new run."""
        memory.lessons = list(self.lessons)
        number = len(self.runs) + 1
        self.runs.append({"run": number, "started": time.time(), "notes_at_start": len(self.lessons), **meta})
        self.save()
        return number

    def add(self, lesson: Lesson) -> None:
        # Written the moment it exists: a run that crashes still leaves its
        # notes behind.
        self.lessons.append(lesson)
        self.save()

    def end_run(self, **result: Any) -> None:
        if self.runs:
            self.runs[-1].update({"ended": time.time(), **result})
            self.save()

    def summary(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "agent": self.agent,
            "runs": len(self.runs),
            "notes": len(self.lessons),
            "history": [
                {k: r.get(k) for k in ("run", "reason", "score", "max_score", "turns", "deaths", "censored")}
                for r in self.runs
            ],
        }
