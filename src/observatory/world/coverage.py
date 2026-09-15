"""How much of the world has been seen, out of how much there is.

Counting rooms visited is easy and nearly useless on its own: eleven rooms is
an achievement in one game and a rounding error in another. What makes the
number mean something is a denominator, and the object table has one — the
engine knows how many rooms exist whether or not the player ever finds them.

The denominators are derived structurally rather than from any knowledge of a
particular game:

  Rooms are the siblings of the room you are standing in. Every Z-machine game
  hangs its rooms off one parent object — Zork I puts all 109 under object #82,
  the mock world puts its 9 under the root — so the room you are in identifies
  the whole set from the first turn, without a map, a walkthrough, or a special
  case per game.

  Objects are everything named that is not a room and not the player.

Coverage is deliberately kept apart from the discovery ledger. The ledger is
about what the agent worked out; this is about how much ground it covered. A
run can score well on one and badly on the other, and those are different
failures worth telling apart.
"""

from __future__ import annotations

from typing import Any

from ..engine.base import WorldState


class Coverage:
    """Ground covered, against ground available."""

    def __init__(self) -> None:
        self.room_parent: int | None = None
        self.rooms_total = 0
        self.objects_total = 0

        # Identifying the player object.
        #
        # Inferring it from carried items is exact but only works once
        # something has been picked up, and both Zork and the mock start you
        # empty-handed — leaving the player counted as an object it had
        # discovered, which pushed coverage past 100%.
        #
        # So there is a second route that converges in one move: the player is
        # the thing that changes rooms when you do. Intersecting the children
        # of each new room across moves leaves exactly one object. Neither
        # route knows anything about a particular game.
        self.player_object: int | None = None
        self._player_candidates: set[int] | None = None
        self._last_location: int | None = None

        self.rooms_seen: set[int] = set()
        self.objects_seen: set[int] = set()
        self.objects_held: set[int] = set()
        self.stowed: set[int] = set()      # put inside a container, not carried

        self.score = 0
        self.max_score = 0

    def observe(self, state: WorldState, visible: set[int], player_object: int | None) -> None:
        if player_object is not None:
            self.player_object = player_object
        elif self.player_object is None and state.location_id:
            if self._last_location is not None and state.location_id != self._last_location:
                movers = {o.num for o in state.objects if o.parent == state.location_id}
                self._player_candidates = (
                    movers if self._player_candidates is None
                    else self._player_candidates & movers
                )
                if len(self._player_candidates) == 1:
                    self.player_object = next(iter(self._player_candidates))
        if state.location_id:
            self._last_location = state.location_id
        player_object = self.player_object

        if state.location_id:
            self.rooms_seen.add(state.location_id)

        here = next((o for o in state.objects if o.num == state.location_id), None)
        if here is not None:
            # Re-derived every turn rather than latched: cheap, and it recovers
            # if the first reading happened somewhere odd.
            self.room_parent = here.parent
            rooms = {o.num for o in state.objects if o.parent == self.room_parent}
            self.rooms_total = len(rooms)
            self.objects_total = sum(
                1 for o in state.objects
                if o.num not in rooms and o.num != player_object and o.name
            )

        self.objects_seen |= {
            n for n in visible if n != state.location_id and n != player_object
        }
        if player_object is not None:
            self.objects_seen.discard(player_object)   # may have been added before it was known

        held = {
            o.num for o in state.objects
            if player_object is not None and o.parent == player_object
        }
        self.objects_held |= held

        # Anything sitting inside something that is neither a room nor the
        # player. In Zork this is the trophy case; generically it is "stowed".
        room_nums = {o.num for o in state.objects if o.parent == self.room_parent}
        for o in state.objects:
            if (
                o.name
                and o.parent not in room_nums
                and o.parent != player_object
                and o.parent != 0
                and o.num in self.objects_seen
            ):
                self.stowed.add(o.num)

        self.score = state.score
        self.max_score = state.max_score

    @staticmethod
    def _pct(seen: int, total: int) -> float:
        return round(100 * seen / total, 1) if total else 0.0

    def summary(self) -> dict[str, Any]:
        return {
            "rooms_seen": len(self.rooms_seen),
            "rooms_total": self.rooms_total,
            "rooms_pct": self._pct(len(self.rooms_seen), self.rooms_total),
            "objects_seen": len(self.objects_seen),
            "objects_total": self.objects_total,
            "objects_pct": self._pct(len(self.objects_seen), self.objects_total),
            "objects_held": len(self.objects_held),
            "stowed": len(self.stowed),
            "score": self.score,
            "max_score": self.max_score,
            "score_pct": self._pct(self.score, self.max_score),
        }
