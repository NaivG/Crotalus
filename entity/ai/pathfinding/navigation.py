"""
Where a mob actually walks: the layer between "go there" and "push forward".

A goal says `navigator.move_to(x, y, z, speed)`. This turns that into a
`MoveControl` wanted position every tick, and gives up when the walk is not
going anywhere - so the goal can stop and pick a new target.

**This replaces the straight-line navigator, and the five public methods do
not change.** `move_to / stop / tick / is_idle / is_stuck` are the contract
`goals.py` calls and `MobAI` drives; what changed is that "go there" now means
"follow the node path the search found" rather than "point at the target and
push". On flat ground the two produce the same walk. At a fence, a tree or a
step, this one goes round or over.

**The navigation layer picks waypoints; the controls do the walking.** That
division is the one decision here worth arguing for. Pumpkin's
`tick_ground` writes `movement_input` and `jumping` directly, because Pumpkin
folded `MoveControl` into its navigation. This server did not: `MoveControl`
exists, it turns at 90 degrees per tick, it owns the arrival radius, and it
is the thing `SwimGoal` and `LookAtEntityGoal` already share. Writing
`movement_input` from here would take all of that away from a class that owns
it, to buy nothing. So this file decides *which node to head for* and calls
`set_wanted_position`; turning, acceleration, friction and the step-up jump
all stay in `control.py`. The one thing written outside is the jump request,
because "the next node is a block above me" is a fact about the path that
`MoveControl` cannot see.

**Two stuck detectors, because they catch different failures.** The 100-tick
one compares against a distance threshold scaled by the mob's speed, and
catches a mob wedged against geometry. The per-node one asks how long this
*specific* node should have taken at this mob's speed - `dist / speed * 20`
ticks, times three as slack - and catches a mob that is moving fine but is
stuck on one waypoint, which is what a mob circling a tree looks like from the
outside. Either one stops the walk. A third, coarser backstop (`MAX_TRAVEL_TICKS`)
bounds a walk that is somehow making progress forever, which vanilla has no
equivalent for and which costs one comparison a tick.

**Repathing is throttled, and the throttle is the important part.** A search is
not free - a few hundred node classifications, each a handful of dict lookups
and sometimes a collision test. Recomputing every tick would be visible as
frame time. So a fresh path is not re-planned for `REPATH_COOLDOWN` ticks, and
`needs_new_path` is the other half: the goal moving is a reason to recompute,
but the threshold it uses is scaled by how far the mob has left to walk. A mob
four blocks from its target should abandon its path the moment the target
moves. A mob sixteen blocks out should not throw away a good path every time
the goal re-rolls by a block.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

from entity.ai.control import ARRIVED_DISTANCE_SQ
from entity.ai.pathfinding.finder import PathFinder
from entity.ai.pathfinding.node import PathType
from entity.ai.pathfinding.path import Path

__all__ = [
    "MAX_TRAVEL_TICKS",
    "PathNavigation",
    "REPATH_COOLDOWN",
    "STUCK_CHECK_INTERVAL",
]

#: Ticks between two coarse progress samples. Vanilla's
#: `STUCK_CHECK_INTERVAL`; the fine-grained detectors below are vanilla's, and
#: this is the coarser net underneath them.
STUCK_CHECK_INTERVAL = 20

#: Blocks a mob must cover in `STUCK_CHECK_INTERVAL` ticks to count as making
#: progress. One tenth of a block in a second is well under a cow's eight.
STUCK_EPSILON = 0.1

#: Hard bound on one walk, in ticks.
#:
#: Vanilla bounds path *recomputation*, not the walk, and the two detectors
#: above are what normally end a walk. This is the backstop for the case they
#: both miss: a mob that keeps covering enough ground per sample to look
#: alive while orbiting a wall it will never get round. Sized against the
#: slowest legitimate walk - a ten-block wander leg at a cow's 0.088 blocks
#: per tick is roughly 120 ticks including the acceleration ramp - so 400
#: leaves a real walk three times the headroom before it bounds a fake one.
MAX_TRAVEL_TICKS = 400

#: Ticks a fresh path is trusted before the mob will consider a new one
#: (`repath_cooldown`, `mod.rs:1025`). Not "recompute every N ticks" - it is
#: a quiet period after a recompute, so a mob that recomputed once does not
#: immediately recompute again because `needs_new_path` still says yes.
REPATH_COOLDOWN = 15

#: Vanilla's `MAX_ITERS`-adjacent stuck window: sample every this many ticks.
STUCK_SAMPLE_TICKS = 100

#: How close horizontally counts as having reached a waypoint, as a fraction of
#: the mob's width (`max_distance_to_waypoint`, `mod.rs:1094`). A wide mob has
#: a wider radius because its *centre* passing within a cow's radius still
#: leaves its shoulder in the gap.
#:
#: The formula inverts below 1.5 because the two halves are measuring
#: different things: for a wide mob it is "half my own width" (how far my
#: centre may be from the node), and for a narrow one it is "the gap between
#: my width and the cell", which is why a chicken (0.4) gets 0.55 and a cow
#: (0.9) gets 0.3.
NODE_REACH_Y = 1.0

#: Ticks on one waypoint before the walk is abandoned
#: (`ticks_on_current_node > 100`, `mod.rs:1057`).
MAX_TICKS_ON_NODE = 100

#: Squares a mob must travel in `STUCK_SAMPLE_TICKS` ticks, or it is stuck.
#: Vanilla's `< 4.0`: two blocks in five seconds, for a mob that was trying
#: to move.
STUCK_DISPLACEMENT_SQ = 4.0


class PathNavigation:
    """One per mob, reached as `mob.ai.navigator`.

    Holds the destination, the current path, and the three watches over the
    walk. `mob` is not stored, for the same reason the old `Navigator` did not
    store it: it is handed back to its mob on every call, which keeps the
    object graph one-directional and means a navigator cannot be pointed at a
    mob other than the one that started it.

    The exception is `move_to`'s `mob` argument, which is optional and exists
    only so the return value can be a real answer. See `move_to`.
    """

    __slots__ = ("finder", "path", "target_x", "target_y", "target_z",
                 "speed_modifier", "ticks", "reach_range",
                 "_active", "_stuck",
                 "_repath_cooldown",
                 "_ticks_on_current_node",
                 "_last_node_index", "_path_start_pos",
                 "_last_sample_tick", "_last_sample_pos",
                 "_last_stuck_tick", "_last_stuck_pos",
                 "_timeout_node", "_timeout_ticks", "_timeout_limit",
                 "_total_ticks")

    def __init__(self) -> None:
        self.finder = PathFinder()
        self.path: Optional[Path] = None
        self.target_x = 0.0
        self.target_y = 0.0
        self.target_z = 0.0
        self.speed_modifier = 1.0
        #: Ticks this walk has been running. Read by the check tools and by
        #: `MAX_TRAVEL_TICKS`, and the only piece of this object's state that
        #: predates the pathfinder - the old navigator's public counter.
        self.ticks = 0
        #: Vanilla's `reachRange`: a Manhattan radius in blocks at which the
        #: target counts as reached. 1, so a mob stops *at* the block next to
        #: a player rather than inside them.
        self.reach_range = 1
        self._active = False
        self._stuck = False
        self._repath_cooldown = 0
        self._ticks_on_current_node = 0
        self._last_node_index = 0
        self._path_start_pos: Optional[Tuple[float, float, float]] = None
        self._last_sample_tick = 0
        self._last_sample_pos: Optional[Tuple[float, float, float]] = None
        self._last_stuck_tick = 0
        self._last_stuck_pos: Optional[Tuple[float, float, float]] = None
        self._timeout_node: Optional[Tuple[int, int, int]] = None
        self._timeout_ticks = 0
        self._timeout_limit = 0.0
        self._total_ticks = 0

    # ----- the contract -----

    @property
    def is_idle(self) -> bool:
        return not self._active

    @property
    def is_stuck(self) -> bool:
        """Whether the last walk ended because the mob was not getting anywhere.

        Read by goals that want to give up early rather than wait out the
        timeout. Nothing reads it this round - `WanderAroundGoal` stops on
        `is_idle`, which arrives on the same tick - and it is kept because a
        later goal (breed, follow-parent) needs to distinguish "arrived" from
        "wedged", and the two are indistinguishable from outside.

        **The flag latches.** It is set by a detector and survives `stop`,
        because the question cannot be answered while the walk is still
        running - a mob that is wedged is wedged exactly when the walk has
        already ended. `move_to` is the only thing that clears it, since that
        is the start of a new walk and a new walk is by definition not yet
        stuck. (This is not a change from the straight-line navigator, which
        had the same shape and the same latent emptiness in it.)
        """
        return self._stuck

    def move_to(self, x: float, y: float, z: float,
                speed_modifier: float = 1.0, mob=None) -> bool:
        """Start walking to a point. True when the walk started.

        **The return value is a real answer this round**, which is the one
        behavioural change to this method. It was always True before, because
        there was no reachability test to fail - deciding reachability was
        "the pathfinder's job and that is the gap named in the module
        docstring", and there is no longer a gap. When a `mob` is supplied the
        path is searched for immediately and the answer is whether one was
        found; when it is not, the walk starts unconditionally and the caller
        gets the old permissive answer.

        Searching here rather than deferring to the first tick is what makes
        the answer possible at all, and it costs nothing extra: the path the
        first tick would have computed is the path this computes, and the
        `REPATH_COOLDOWN` set below is what stops the first tick from
        immediately discarding it and computing the same thing again.
        """
        self.target_x = x
        self.target_y = y
        self.target_z = z
        self.speed_modifier = speed_modifier
        self.ticks = 0
        self._total_ticks = 0
        self._active = True
        self._stuck = False
        self._repath_cooldown = REPATH_COOLDOWN
        self._ticks_on_current_node = 0
        self._last_node_index = 0
        self.path = None
        self._path_start_pos = None
        self._last_sample_tick = 0
        self._last_sample_pos = None
        self._last_stuck_tick = 0
        self._last_stuck_pos = None
        self._reset_stuck_timeout()

        if mob is None:
            return True
        path = self.finder.create_path(mob, (x, y, z), self.reach_range)
        self.path = path
        if path is None:
            self._active = False
            return False
        return True

    def stop(self) -> None:
        """End the current walk and drop the path."""
        self._active = False
        self.ticks = 0
        self._total_ticks = 0
        self.path = None
        self._path_start_pos = None
        self._ticks_on_current_node = 0
        self._last_sample_pos = None
        self._last_stuck_pos = None
        self._reset_stuck_timeout()

    def can_reach_within(self, mob, x: float, y: float, z: float,
                         distance: float) -> bool:
        """Whether the mob could get within `distance` of a point.

        Vanilla's public `canReachWithin` (`mod.rs:552`), kept as-is for the
        goals that will need it: `MoveToBlockGoal` ("stand next to this
        block") and `BreachGoal` ("get within reach of this mob") are both
        reachability questions wearing a goal's clothes.
        """
        return self.finder.can_reach_within(mob, (x, y, z), distance)

    def create_path(self, mob, x: float, y: float, z: float,
                    reach_range: int = 1) -> Optional[Path]:
        """Search a path without adopting it.

        Vanilla's `createPath` (`mod.rs:393`). A goal that wants to know
        whether somewhere is walkable - before committing a mob to a detour -
        asks this rather than `move_to`.
        """
        return self.finder.create_path(mob, (x, y, z), reach_range)

    # ----- the tick -----

    def tick(self, mob) -> None:
        """Advance the walk by one tick, or give it up.

        Runs after the goal selector and before the move control, so a goal
        that started a walk this tick and one that has been running all along
        reach `MoveControl` the same way: exactly one `set_wanted_position`
        per tick, from exactly one place.
        """
        if not self._active:
            return
        self.ticks += 1
        self._total_ticks += 1
        if self._repath_cooldown > 0:
            self._repath_cooldown -= 1

        pos = mob.position
        if self._last_sample_pos is None:
            # First tick of a walk: take every baseline now and fall through,
            # so the mob is given a waypoint on the same tick the goal
            # started. Returning here instead would waste a tick per walk and
            # would leave the baseline unset until the third one.
            self._last_sample_pos = (pos.x, pos.y, pos.z)
            self._last_sample_tick = self.ticks
            self._last_stuck_pos = self._last_sample_pos
            self._last_stuck_tick = self.ticks

        if self.needs_new_path():
            self._recompute(mob)

        if self.path is None or not self.path.is_valid():
            # No route at all, not even a partial one. Standing still is the
            # right answer and the goal's `should_continue` sees it on this
            # tick, so it can pick a different target rather than waiting.
            self.stop()
            return

        if self._arrived(pos):
            self.stop()
            return

        self._do_stuck_detection(mob, pos)
        if not self._active:
            return

        self._follow_path(mob, pos)

    # ----- path management -----

    def needs_new_path(self) -> bool:
        """Whether the current path should be discarded and searched again.

        Three ways to say yes: there is no path, the goal has moved far enough
        that this one no longer aims at it, or the walk has been going on long
        enough that a fresh look is worth the cost.

        **The goal-moved threshold is `half the distance left to walk`,
        clamped to 4..16 blocks** (`mod.rs:724`). That is the whole trick. The
        question "has the goal moved too much?" has no fixed answer - a mob
        that has walked to within four blocks of its target should care about
        a one-block drift, and a mob that has sixteen blocks to go should not
        care about a five-block one. So the tolerance is a fraction of the
        remaining walk, bounded at both ends: the floor stops a mob at the very
        end of its path from thrashing over sub-block differences, and the
        ceiling stops a long walk from being thrown away for every small
        re-roll of a wander target.

        `repath_cooldown` is checked first and returns False, which is what
        makes the timeout case cheap: a mob that just finished a 20-tick walk
        whose path is still perfectly good does not search again.
        """
        if self.path is None:
            return True
        if self._repath_cooldown > 0:
            return False
        path_x, path_y, path_z = self.path.get_target()
        goal_x = int(math.floor(self.target_x))
        goal_y = int(math.floor(self.target_y))
        goal_z = int(math.floor(self.target_z))
        dx = path_x - goal_x
        dy = path_y - goal_y
        dz = path_z - goal_z
        distance_sq = dx * dx + dy * dy + dz * dz
        remaining = min(max(self.path.get_remaining_distance(), 4.0), 16.0)
        threshold = remaining * 0.5
        return distance_sq > threshold * threshold

    def _recompute(self, mob) -> None:
        self.path = self.finder.create_path(
            mob, (self.target_x, self.target_y, self.target_z),
            self.reach_range)
        self._ticks_on_current_node = 0
        self._last_node_index = 0
        pos = mob.position
        self._path_start_pos = (pos.x, pos.y, pos.z)
        self._repath_cooldown = REPATH_COOLDOWN
        self._reset_stuck_timeout()

    # ----- giving up -----

    def _arrived(self, pos) -> bool:
        """Whether the mob is close enough to call it there.

        **This is what ends a walk, and it is not optional.** Without it the
        path runs out, `next_node_index` walks off the end, and the mob stands
        still on the spot it was aiming at while the goal's `should_continue`
        ("is the navigator busy?") keeps answering yes. A cow holding a wander
        goal it finished twenty seconds ago is the visible version of that.
        """
        if self.path.is_done():
            return True
        dx = self.target_x - pos.x
        dy = self.target_y - pos.y
        dz = self.target_z - pos.z
        return dx * dx + dy * dy + dz * dz <= ARRIVED_DISTANCE_SQ

    def _reset_stuck_timeout(self) -> None:
        """Clear the per-waypoint timers - but *not* the latched `_stuck`.

        These are two different facts. The timers are "how long has this
        waypoint been current", which is meaningless the moment the walk
        stops. `_stuck` is "the last walk ended because the mob could not
        move", which is the answer a goal asks for afterwards and so has to
        outlive the walk that produced it. `move_to` clears it.
        """
        self._timeout_node = None
        self._timeout_ticks = 0
        self._timeout_limit = 0.0

    def _do_stuck_detection(self, mob, pos) -> None:
        """The three ways a walk is declared finished.

        Vanilla's two (`do_stuck_detection`, `mod.rs:842`) plus this module's
        coarse bound. All three read the mob's position and none of them touch
        the path, so a stuck mob stops walking and the *goal* decides what to
        do next - which is the right division: "I am wedged" is not the goal's
        to fix, but it is the goal's to hear about.
        """
        x, y, z = pos.x, pos.y, pos.z

        # --- vanilla: sample every 100 ticks against a speed-scaled threshold.
        # The threshold is speed^2 * 100 * 0.25 for a mob slower than 1.0/tick
        # and speed * 100 * 0.25 for a fast one - the quadratic is the same
        # shape as the acceleration in `control.py`, so a slow mob gets a
        # correspondingly small "made progress" bar. A 0.2 cow therefore has to
        # cover one block in five seconds, against the eight it would cover
        # walking, and a mob that has covered less is not slow, it is stuck.
        if self.ticks - self._last_stuck_tick > STUCK_SAMPLE_TICKS:
            speed = float(mob.attributes.movement_speed)
            effective_speed = speed if speed >= 1.0 else speed * speed
            threshold = effective_speed * STUCK_SAMPLE_TICKS * 0.25
            last_x, last_y, last_z = self._last_stuck_pos
            dx = x - last_x
            dy = y - last_y
            dz = z - last_z
            moved_sq = dx * dx + dy * dy + dz * dz
            if moved_sq < threshold * threshold:
                self._stuck = True
                self.stop()
                return
            self._last_stuck_tick = self.ticks
            self._last_stuck_pos = (x, y, z)

        # --- vanilla: how long should this one waypoint have taken?
        # `dist / speed * 20` ticks is "twenty ticks to cover this many blocks
        # at this mob's speed", times three as slack for a mob that had to
        # turn first. Without the slack a mob whose first waypoint is
        # diagonally behind it would time out before it finished turning.
        next_pos = self.path.get_next_node_pos()
        if next_pos is not None:
            if next_pos == self._timeout_node:
                self._timeout_ticks += 1
            else:
                self._timeout_node = next_pos
                centre = (next_pos[0] + 0.5, float(next_pos[1]),
                          next_pos[2] + 0.5)
                dx = x - centre[0]
                dy = y - centre[1]
                dz = z - centre[2]
                distance = math.sqrt(dx * dx + dy * dy + dz * dz)
                speed = float(mob.attributes.movement_speed)
                self._timeout_limit = distance / speed * 20.0 if speed > 0 else 0.0
            if (self._timeout_limit > 0.0
                    and self._timeout_ticks > self._timeout_limit * 3.0):
                self._stuck = True
                self.stop()
                return

        # --- this module's coarse bound, and the oldest detector: a windowed
        # sample at STUCK_CHECK_INTERVAL. It is the one that survives a mob
        # moving at exactly the wrong speed to trip either vanilla detector.
        if self._last_sample_pos is not None and (
                self.ticks - self._last_sample_tick >= STUCK_CHECK_INTERVAL):
            last_x, last_y, last_z = self._last_sample_pos
            dx = x - last_x
            dy = y - last_y
            dz = z - last_z
            moved_sq = dx * dx + dy * dy + dz * dz
            # `not (x >= threshold)` rather than `x < threshold`: a position
            # that has gone non-finite makes the comparison false either way,
            # and this spelling is the one that reads "no progress".
            self._stuck = not (moved_sq >= STUCK_EPSILON * STUCK_EPSILON)
            self._last_sample_tick = self.ticks
            self._last_sample_pos = (x, y, z)
            if self._stuck:
                self.stop()
                return

        # --- and the wall clock.
        if self.ticks >= MAX_TRAVEL_TICKS:
            self.stop()

    # ----- following -----

    def _max_distance_to_waypoint(self, width: float) -> float:
        """How close horizontally counts as reaching a waypoint.

        Vanilla's `max_distance_to_waypoint` (`mod.rs:1094`), and the formula
        has two branches that look odd side by side:
        `width > 0.75 ? width * 0.5 : 0.75 - width * 0.5`. Read as "half the
        gap between me and the cell", it stops making sense for a wide mob -
        a 2-block-wide mob would get a 1.0 radius and be considered arrived
        while its centre was a whole block past the node. Vanilla switches to
        half its own width there, which is the right question for a mob that
        fills the gap it is standing in.
        """
        return width * 0.5 if width > 0.75 else 0.75 - width * 0.5

    def _follow_path(self, mob, pos) -> None:
        """Point the move control at the current waypoint, or advance.

        Four reasons to skip a node rather than walk to it, in the order they
        are checked:

        * **The node is already behind and below.** A mob falling past a
          waypoint should not try to walk to it - it is already closer than
          the reach radius in every axis and still descending. Advancing here
          is what stops a mob that jumped off a ledge spending ten ticks
          circling the block it left from.
        * **Close enough**: inside the waypoint radius horizontally and within
          a block vertically.
        * **Corner cut.** If the node *after* this one is nearer than this one
          and roughly in the direction of travel, this node is already
          rounded - the mob walked past it while turning, and insisting on it
          would produce the characteristic zig-zag along a wall. Excluded for
          fire and door nodes, because a mob that has already stepped into
          fire or is threading a door should finish that rather than skip to
          the node past it.
        * **The path ran out.**
        """
        path = self.path
        node_index = path.next_node_index
        if node_index == self._last_node_index:
            self._ticks_on_current_node += 1
        else:
            self._ticks_on_current_node = 0
            self._last_node_index = node_index

        if self._ticks_on_current_node > MAX_TICKS_ON_NODE:
            self._stuck = True
            self.stop()
            return

        # Every 100 ticks, check that the mob has actually gone somewhere since
        # the path was computed. A mob that is walking a stale path - because
        # the world changed under it, or because it is pinned against geometry
        # the path did not see - drifts away from where the path said it would
        # be, and the waypoint cursor stops moving. Two blocks in five seconds
        # is the bar, and it is the same idea as the stuck sample above from
        # the other direction: not "am I moving" but "am I where I said".
        if self._total_ticks % STUCK_SAMPLE_TICKS == 0:
            if self._path_start_pos is not None:
                start_x, start_y, start_z = self._path_start_pos
                dx = pos.x - start_x
                dy = pos.y - start_y
                dz = pos.z - start_z
                if dx * dx + dy * dy + dz * dz < STUCK_DISPLACEMENT_SQ:
                    self._stuck = True
                    self.stop()
                    return
            self._path_start_pos = (pos.x, pos.y, pos.z)

        entity_type = getattr(mob, "entity_type", None)
        width = float(getattr(entity_type, "width", 0.6) or 0.6)
        max_distance = self._max_distance_to_waypoint(width)

        next_block = path.get_next_node_pos()
        if next_block is None:
            self.stop()
            return

        target_x = next_block[0] + 0.5
        target_y = float(next_block[1])
        target_z = next_block[2] + 0.5
        dx = target_x - pos.x
        dy = target_y - pos.y
        dz = target_z - pos.z
        horizontal_dist_sq = dx * dx + dz * dz
        horizontal_dist = math.sqrt(horizontal_dist_sq)

        # Airborne and descending past the waypoint: it is behind us.
        if (not pos.on_ground and horizontal_dist < max_distance and dy < -0.5):
            path.advance()

        close_enough = (horizontal_dist < max_distance and abs(dy) < NODE_REACH_Y)
        node = path.get_next_node()
        corner_cut = (
            node is not None
            and node.path_type not in (PathType.DANGER_FIRE, PathType.DAMAGE_FIRE,
                                       PathType.WALKABLE_DOOR)
            and self._should_target_next_in_direction(pos, path))

        if close_enough or corner_cut:
            path.advance()

        # **Consuming a waypoint does not end the tick.** Pumpkin returns
        # here because its `movement_input` survives the tick on its own; in
        # this server `MobAI.tick` clears `MoveControl` at the top of every
        # tick and the navigator refills it, so returning would leave the mob
        # with no walk input for one tick on *every* waypoint it passed. A
        # one-block waypoint step every ten ticks is a ten-percent slowdown,
        # and it shows up as a gap in the acceleration series rather than as an
        # obvious stutter. So: advance, then fall through and point at the
        # new node, all on the same tick.
        next_block = path.get_next_node_pos()
        if next_block is None:
            # The last node was the last node.
            self.stop()
            return

        target_x = next_block[0] + 0.5
        target_y = float(next_block[1])
        target_z = next_block[2] + 0.5
        dx = target_x - pos.x
        dy = target_y - pos.y
        dz = target_z - pos.z
        horizontal_dist_sq = dx * dx + dz * dz
        horizontal_dist = math.sqrt(horizontal_dist_sq)

        # Everything else: this node is the destination for this tick. The
        # control does the turning and the accelerating.
        #
        # The arrival radius is handed over rather than left at its default,
        # because the two layers have to agree on where "there" is. The
        # control's own radius (0.5) is right for a *destination* and wrong
        # for a waypoint: a cow's waypoint radius is 0.45, so with the
        # default the control would announce arrival half a block early, zero
        # its input, and cost the mob a stationary tick at every node
        # boundary. `MoveControl.set_wanted_position` says more.
        mob.ai.move_control.set_wanted_position(
            target_x, target_y, target_z, self.speed_modifier,
            arrive_distance=max_distance)

        # The jump decision lives here rather than only in `MoveControl`,
        # because it is a statement about the *path*: the next node is a block
        # above where the mob is standing, or the mob is below its level and
        # has to climb to it. `MoveControl` re-derives the first half from the
        # wanted position's height and will catch it too; the second half it
        # cannot see at all.
        #
        # The distance comparison is vanilla's (`mod.rs:1141`):
        # `max(1.0, width)` squared, against the *horizontal* distance. For a
        # 0.6-wide cow that is one block - the step has to be the next thing
        # in front of it, not somewhere off to the side. For a sheep at 1.3
        # wide it is 1.69, because a wider mob needs to start moving earlier to
        # clear the step.
        step_height = float(mob.attributes.step_height)
        jump_distance = max(1.0, width)
        if (dy > step_height or next_block[1] > pos.y) and (
                horizontal_dist_sq < jump_distance * jump_distance):
            mob.ai.jump_control.jump()

    def _should_target_next_in_direction(self, pos, path: Path) -> bool:
        """Whether to skip a waypoint the mob has already rounded.

        Vanilla's `shouldTargetNextNodeInDirection` (`mod.rs:896`), and it is
        a geometry test with four conditions, each of which is a way the naive
        version goes wrong:

        * **There has to be a next node.** Skipping the last node of a path
          leaves nothing to aim at.
        * **The mob has to be near the current node** - within two blocks.
          Beyond that, "the next node is closer than this one" says something
          about the mob's position relative to a node it is nowhere near.
        * **Closer to the next node, or already within half a block of this
          one.** The second clause is the fallback: a mob sitting exactly on a
          node is "past" it even though the next one is not nearer.
        * **And going the other way.** The two offset vectors from the mob to
          the two nodes must point in *opposite* directions - a dot product
          below zero. Without this the test says "the next node is nearer"
          about a mob standing on a path it is walking towards, which is true
          on every tick and would cut every corner in the world.
        """
        if path.next_node_index + 1 >= path.node_count():
            return False
        current_pos = path.get_next_node_pos()
        next_pos = path.get_node(path.next_node_index + 1)
        if current_pos is None or next_pos is None:
            return False

        current_node = (current_pos[0] + 0.5, float(current_pos[1]),
                        current_pos[2] + 0.5)
        next_node = (next_pos.pos[0] + 0.5, float(next_pos.pos[1]),
                     next_pos.pos[2] + 0.5)

        mob_x, mob_y, mob_z = pos.x, pos.y, pos.z
        dx = mob_x - current_node[0]
        dy = mob_y - current_node[1]
        dz = mob_z - current_node[2]
        if dx * dx + dy * dy + dz * dz > 4.0:
            return False

        to_current = (current_node[0] - mob_x, current_node[1] - mob_y,
                      current_node[2] - mob_z)
        to_next = (next_node[0] - mob_x, next_node[1] - mob_y,
                   next_node[2] - mob_z)
        current_sq = to_current[0] ** 2 + to_current[1] ** 2 + to_current[2] ** 2
        next_sq = to_next[0] ** 2 + to_next[1] ** 2 + to_next[2] ** 2

        if not (next_sq < current_sq or current_sq < 0.5):
            return False

        current_len = math.sqrt(current_sq)
        next_len = math.sqrt(next_sq)
        if current_len < 1e-5 or next_len < 1e-5:
            return False
        dot = ((to_next[0] * to_current[0] + to_next[1] * to_current[1]
                + to_next[2] * to_current[2]) / (current_len * next_len))
        return dot < 0.0