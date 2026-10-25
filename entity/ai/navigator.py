"""
Where `Navigator` went.

The straight-line navigator that used to live here is `PathNavigation` in
`entity/ai/pathfinding/navigation.py`, and this module is the thin
compatibility layer over it rather than an implementation of anything.

**Why it still exists rather than being deleted.** `tools/check_mob_ai.py`
imports `MAX_TRAVEL_TICKS` and `STUCK_CHECK_INTERVAL` from here, and those
names are not invented by the check tool - they are this module's own
tunables, and a check that imported its own copies of them would be testing a
constant against itself. Anything else importing `Navigator` gets the new
class under the old name, which is a one-line change in one place rather than
a rename across five call sites in `goals.py` and a rewrite of the check.

The five-method contract (`move_to / stop / tick / is_idle / is_stuck`) is
unchanged and is the whole point of the replacement being a replacement rather
than a wrapper: those names are what `goals.py` calls and what `MobAI`
drives. `move_to` gained one *optional* argument - the mob, so that its return
value can be a real reachability answer instead of a standing `True` - and
every existing call site still works without it.
"""

from entity.ai.pathfinding.navigation import (
    MAX_TRAVEL_TICKS,
    REPATH_COOLDOWN,
    STUCK_CHECK_INTERVAL,
    PathNavigation,
)

#: The old name, pointing at the new class. Kept so `from
#: entity.ai.navigator import Navigator` keeps resolving to the thing that
#: actually walks mobs.
Navigator = PathNavigation

__all__ = [
    "MAX_TRAVEL_TICKS",
    "Navigator",
    "PathNavigation",
    "REPATH_COOLDOWN",
    "STUCK_CHECK_INTERVAL",
]