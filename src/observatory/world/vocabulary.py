"""The parser's vocabulary, as the agent has mapped it.

Counting distinct commands was close to meaningless: an agent that types
`take lazuli`, `take grue` and `take zorkmid` has three distinct commands and
has learned nothing, because none of those words exist. What is worth counting
is which words the parser turned out to know.

The game makes two different refusals and they mean opposite things, so they
are tracked apart:

    I don't know the word "zorkmid".     the word is not in the dictionary
    You can't see any such thing.        the word is fine; the thing is elsewhere

The first closes a door permanently. The second is a hint that the object
exists somewhere — an agent that treats them the same is throwing away the more
useful half of what the parser told it.

Crucially, the parser *names* the word it failed on, so read it rather than
guessing. An earlier version blamed the verb for every unknown-word refusal,
which on a real Zork run marked `open`, `examine`, `take` and `drop` as words
the parser didn't know — all of them real verbs, condemned because the agent
had paired them with invented nouns.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from .outcomes import Outcome

# `I don't know the word "zorkmid".` — Infocom names the offending word, and
# most later authoring systems copy the convention.
QUOTED_WORD = re.compile(
    r"""don['’]?t know the word[:\s]*["'“‘]?([a-z\-]+)""", re.I
)

# `You used the word "remove" in a way that I don't understand.`
USED_WORD = re.compile(r"""used the word[:\s]*["'“‘]?([a-z\-]+)""", re.I)

# Words that are commands in their own right, not verbs taking an object.
BARE = {
    "north", "south", "east", "west", "northeast", "northwest", "southeast",
    "southwest", "up", "down", "in", "out", "n", "s", "e", "w", "ne", "nw",
    "se", "sw", "u", "d", "look", "l", "inventory", "i", "wait", "z", "score",
}

# Standard one-letter forms. Listing `n` beside `north` as two verbs the agent
# found is double counting; the ledger already records, once, that
# abbreviations work at all.
ABBREVIATIONS = {
    "n": "north", "s": "south", "e": "east", "w": "west",
    "ne": "northeast", "nw": "northwest", "se": "southeast", "sw": "southwest",
    "u": "up", "d": "down", "l": "look", "i": "inventory", "z": "wait",
}

FILLER = {"the", "a", "an", "at", "in", "on", "to", "with", "into", "from", "of"}


@dataclass
class Vocabulary:
    verbs_ok: set[str] = field(default_factory=set)
    nouns_ok: set[str] = field(default_factory=set)
    unknown: set[str] = field(default_factory=set)   # not in the dictionary at all
    absent: set[str] = field(default_factory=set)    # real word, not here

    def observe(self, command: str, outcome: Outcome, text: str = "") -> None:
        words = [w for w in command.strip().lower().split() if w not in FILLER]
        if not words:
            return

        verb, nouns = ABBREVIATIONS.get(words[0], words[0]), words[1:]

        if outcome is Outcome.GRAMMAR:
            # "You used the word X in a way that I don't understand." Every word
            # was in the dictionary; the sentence built from them was not. That
            # is evidence *for* the words, never against them.
            used = USED_WORD.search(text or "")
            if used:
                self.unknown.discard(used.group(1).lower())
            if verb not in self.unknown:
                self.verbs_ok.add(verb)
            return

        if outcome is Outcome.UNKNOWN_WORD:
            match = QUOTED_WORD.search(text or "")
            if match:
                # The parser told us exactly which word it choked on. Believe it.
                bad = match.group(1).lower()
                self.unknown.add(bad)
                self.verbs_ok.discard(bad)
                self.nouns_ok.discard(bad)
                if bad != verb:
                    # The parser reads left to right and reports the first word
                    # it fails on, so a named noun means the verb parsed.
                    self.unknown.discard(verb)
                    self.verbs_ok.add(verb)
                return
            # No word named — stay conservative and blame nothing in particular.
            self.unknown.add(verb)
            self.verbs_ok.discard(verb)
            return

        if verb not in self.unknown:
            self.verbs_ok.add(verb)

        if verb in BARE and not nouns:
            return

        for noun in nouns:
            if noun in self.unknown:
                continue
            if outcome is Outcome.ABSENT_NOUN:
                if noun not in self.nouns_ok:
                    self.absent.add(noun)
            else:
                self.nouns_ok.add(noun)
                self.absent.discard(noun)

    def summary(self) -> dict[str, Any]:
        return {
            "verbs_ok": sorted(self.verbs_ok),
            "nouns_ok": sorted(self.nouns_ok),
            "unknown": sorted(self.unknown),
            "absent": sorted(self.absent),
            "verbs_ok_n": len(self.verbs_ok),
            "nouns_ok_n": len(self.nouns_ok),
            "unknown_n": len(self.unknown),
            "absent_n": len(self.absent),
        }
