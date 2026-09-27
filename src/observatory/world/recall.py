"""Everything that has been typed and printed, and a way to look through it.

A transcript window of thirty exchanges is what a player can hold in view. It
is not what a player *has seen* — by the time anyone reaches a puzzle worth
solving, the sentence that matters scrolled past two hundred turns ago. A human
deals with this by remembering that they saw something and going to look. This
is that, for a player made of prompt.

Three things it is deliberately not:

  It is not ground truth. Every word in here was printed on the screen by the
  game in reply to something someone typed. Searching it reveals nothing the
  player was not already shown, which is the property that lets it sit on the
  agent's side of the line that keeps the object tree on ours.

  It is not a summary. No model wrote any of this and nothing is paraphrased,
  so a search cannot return a belief — only a turn number, a command, and the
  reply verbatim.

  It is not retrieval in the machine-learning sense. Scoring is term overlap
  with ties broken by recency: explainable in a sentence, identical on every
  machine, and free. A search that silently reranked by embedding similarity
  would make two runs of the same seed diverge for reasons nobody could read
  out of the trace.

The floor matters as much as the search. After a handoff that withheld its
predecessor's play, the incoming player must not be able to search for it
either — `since()` is what keeps the cold arm cold.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable

# Query words that would match most of a transcript and rank nothing.
STOPWORDS = frozenset("""
a an the this that these those there here is are was were be been am i you your
it its of in on at to from by with for and or but not no so as if then than do
does did done have has had what when where why how all any some my me we us
""".split())

# Shown per search. Enough to see a pattern, few enough that a reply does not
# cost more context than the window it is meant to spare.
DEFAULT_LIMIT = 6

# A reply longer than this is shown by its most relevant line rather than
# whole; a room description is a paragraph and six of them is a prompt.
WHOLE_REPLY_CHARS = 200

MAX_QUERY_CHARS = 80

_WORD = re.compile(r"[a-z0-9']+")


@dataclass(frozen=True)
class Exchange:
    """One command and what the machine printed back."""

    turn: int
    command: str
    response: str
    # Where the player was standing when the reply arrived. Shown with a hit
    # because "I have seen this before" is nearly useless without "and here".
    room: str = ""

    @property
    def text(self) -> str:
        return f"{self.command}\n{self.response}"


@dataclass(frozen=True)
class Hit:
    exchange: Exchange
    matched: int          # how many distinct query terms this exchange carries
    line: str             # the most relevant line of the reply


def terms(query: str) -> list[str]:
    """The words worth matching on, in order, without duplicates."""
    found: list[str] = []
    for word in _WORD.findall((query or "").lower()[:MAX_QUERY_CHARS]):
        if len(word) < 2 or word in STOPWORDS or word in found:
            continue
        found.append(word)
    return found


def _best_line(response: str, wanted: list[str]) -> str:
    """The line of a reply carrying the most query terms; the first otherwise."""
    lines = [line.strip() for line in response.splitlines() if line.strip()]
    if not lines:
        return ""
    best, best_score = lines[0], -1
    for line in lines:
        lowered = line.lower()
        score = sum(1 for term in wanted if term in lowered)
        if score > best_score:
            best, best_score = line, score
    return best


@dataclass
class Recall:
    """The whole record, append-only, searchable.

    Append-only across a rollback on purpose, and for the same reason the map
    is: the world goes back, the player's memory of having read something does
    not. A screen does not un-print.
    """

    exchanges: list[Exchange] = field(default_factory=list)

    def add(self, exchange: Exchange) -> None:
        self.exchanges.append(exchange)

    def __len__(self) -> int:
        return len(self.exchanges)

    def since(self, floor: int) -> "Recall":
        """A view of everything from index `floor` on.

        What a player who took over mid-run is allowed to look through. With a
        floor of zero this is the whole record, which is the ordinary case.
        """
        return Recall(self.exchanges[floor:]) if floor else self

    def search(self, query: str, limit: int = DEFAULT_LIMIT) -> list[Hit]:
        """Turns mentioning the query, best first, ties broken by recency."""
        wanted = terms(query)
        if not wanted:
            return []
        scored: list[tuple[int, int, Exchange, str]] = []
        for exchange in self.exchanges:
            lowered = exchange.text.lower()
            matched = sum(1 for term in wanted if term in lowered)
            if not matched:
                continue
            scored.append((matched, exchange.turn, exchange,
                           _best_line(exchange.response, wanted)))
        # Most terms first; among equals the most recent, because a world that
        # has been changed since makes the older sighting the stale one.
        scored.sort(key=lambda row: (row[0], row[1]), reverse=True)
        return [Hit(exchange=e, matched=m, line=line)
                for m, _turn, e, line in scored[:limit]]

    def render(self, query: str, hits: Iterable[Hit]) -> str:
        """What the player is shown in reply to its own search.

        Verbatim, with turn numbers, most useful first. No summary, no count of
        what it "should" conclude — a search that editorialised would be a
        second player whispering.
        """
        hits = list(hits)
        head = f'You looked back for "{query.strip()}".'
        if not self.exchanges:
            return f"{head} Nothing has been recorded yet."
        if not hits:
            return (
                f"{head} Nothing in the {len(self)} turns on record mentions it. "
                "That is not evidence it does not exist — only that it has not "
                "been typed or printed yet."
            )
        lines = [f"{head} {len(hits)} of {len(self)} turns on record, "
                 "most relevant first:"]
        for hit in hits:
            where = f", {hit.exchange.room}" if hit.exchange.room else ""
            lines.append("")
            lines.append(f"  [turn {hit.exchange.turn}{where}]")
            if hit.exchange.command:
                lines.append(f"  > {hit.exchange.command}")
            body = hit.exchange.response.strip()
            shown = body if len(body) <= WHOLE_REPLY_CHARS else hit.line
            for line in shown.splitlines():
                lines.append(f"    {line.strip()}")
        return "\n".join(lines)
