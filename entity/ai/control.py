"""
The three controls: what a mob is walking toward, where it is looking, and
whether it is off the ground.

Vanilla splits these because different goals want different ones. A goal that
wants the mob to walk asks for MOVE; a goal that only wants it to turn its head
asks for LOOK. Both can hold their control at once, and the selector
arbitrates. Keeping them apart is what lets a cow walk *and* glance at you.

What is deliberately merged here is the *rotation*. Vanilla has two -
`yRot` (the body, which steering turns) and `yHeadRot` (the head, which the
look control turns, clamped to within `maxHeadRotation` of the body). This
server's `EntityPosition` has exactly one `yaw`, and the wire packets it feeds
(`add_entity`'s `yaw` and `head_yaw`, `entity_position_sync`) both carry that
one value. So `LookControl` and `MoveControl` would be writing the same field,
and the tick order is what decides:

    look first, then move

Move wins, because that is what it is for. A cow that is walking faces where
it is going; a cow that has stopped turns to face you. In vanilla the walking
cow's head still turns and the client renders the two separately - that
half-second of "the cow walks sideways while watching you" is what the second
rotation buys, and it is a declared gap rather than something this file
pretends to reproduce.

The velocity bridge
-------------------
The physics solver owns velocity, gravity and friction. A control does not
move a mob; it writes `mob.movement_input` and `mob.jumping`, and
`MobAI.apply_input` turns those into one `set_velocity` call. Two
consequences worth stating:

* **The write is an acceleration, not a speed.** `apply_input` adds the input
  to the body's *current* velocity, and the solver's ground friction
  (`0.546` per tick on grass) bleeds it back off - Pumpkin's `moveRelative`
  against vanilla's `travel`. The attribute enters the acceleration twice,
  once through the input and once through the factor below, so the steady
  state is `accel / (1 - 0.546)` and a 0.2 cow walks at ~0.088 blocks per
  tick - the speed the 0.2 attribute exists to produce. When the input
  stops, so do the writes, and friction glides the cow the last tenth of a
  block; there is no zeroing write anywhere.
* **Vertical is the solver's, except when jumping.** Gravity goes into the
  *motion* inside `_step_awake`, so a body that is not jumping must have its
  `vy` passed through unchanged. Writing `vy = 0` to "make it stop falling"
  would be a mob that hovers.
"""

from __future__ import annotations

import math
from typing import Optional

#: Vanilla's ground-speed calibration, `0.21600002 / slipperiness^3`, evaluated
#: at grass. Grass has slipperiness 0.6 and 0.6^3 == 0.216 exactly, which is
#: why the constant is written that way. It scales the *acceleration*, not the
#: velocity: on grass the factor is 1.0 and each tick adds
#: `speed_modifier * movement_speed^2` to the body's velocity - the attribute
#: enters once through the input (`MoveControl` writes `modifier * speed`)
#: and once here. Steady state against the solver's ground friction is
#: `accel / (1 - 0.546)`, so a 0.2 cow converges at 0.088 blocks per tick,
#: a 0.23 sheep at 0.116 and a 0.25 chicken at 0.138. That quadratic is
#: vanilla's own: it is why 0.23 zombies amble at a pace a walking player
#: outruns.
#:
#: There is no per-block slipperiness table in this server
#: (`physics/block_shapes.py` has the material columns, the behaviour is
#: declared out of scope), so the factor is fixed at the grass value. Ice,
#: slime and everything else slippery will accelerate at grass speed until
#: that table lands. Written as the ratio rather than as a bare `1.0` so the
#: calibration is legible and so the constant has one obvious place to change.
GROUND_SPEED_FACTOR = 0.21600002 / (0.6 ** 3)

#: Degrees per tick the move control may turn the mob. Vanilla's `MoveControl`
#: turns at 90 regardless of anything else - a mob does not get a rounder
#: wheel at low speed.
MOVE_TURN_RATE = 90.0

#: Below this squared distance a wanted position counts as reached. Vanilla's
#: `MoveControl.ARRIVED_THRESHOLD`, 0.5 blocks. That is right for a
#: *destination*. A pathfinder waypoint is a different claim and has a
#: narrower radius - vanilla's `maxDistanceToWaypoint`, which is a function of
#: the mob's width - so `set_wanted_position` takes it as an argument rather
#: than the two halves disagreeing about what "there" means. Left at this
#: default for a waypoint, a 0.9-wide cow would be declared arrived half a
#: block early and lose a stationary tick at every node boundary.
ARRIVED_DISTANCE = 0.5
ARRIVED_DISTANCE_SQ = ARRIVED_DISTANCE * ARRIVED_DISTANCE

__all__ = [
    "ARRIVED_DISTANCE",
    "ARRIVED_DISTANCE_SQ",
    "GROUND_SPEED_FACTOR",
    "JumpControl",
    "LookControl",
    "MOVE_TURN_RATE",
    "MoveControl",
    "change_angle",
    "wrap_degrees",
    "yaw_to",
]


def wrap_degrees(value: float) -> float:
    """Fold an angle difference into (-180, 180]. Vanilla's `Mth.wrapDegrees`.

    The `%` comes first because Python's `%` already floors toward negative
    infinity, which is what puts -181 at 179 rather than at -181.
    """
    value = value % 360.0
    if value >= 180.0:
        value -= 360.0
    return value


def change_angle(current: float, target: float, max_change: float) -> float:
    """Turn `current` toward `target` by at most `max_change` degrees.

    Takes the short way round: without the wrap, a mob facing 179 degrees told
    to face -179 would spin 358 degrees the wrong way, and the check runs on
    every tick of every turn.
    """
    delta = wrap_degrees(target - current)
    if delta > max_change:
        delta = max_change
    elif delta < -max_change:
        delta = -max_change
    return wrap_degrees(current + delta)


def yaw_to(x: float, y: float, z: float, tx: float, ty: float,
           tz: float) -> Optional[float]:
    """The yaw that faces (tx, ty, tz) from (x, y, z), in degrees.

    `atan2(dz, dx) - 90`, the client's convention: yaw 0 is +Z, and yaw -90 is
    +X. Returns None when the target is directly overhead, where the angle is
    undefined and vanilla's `getTargetYaw` declines rather than returning NaN.
    """
    dx = tx - x
    dz = tz - z
    if abs(dx) < 1e-5 and abs(dz) < 1e-5:
        return None
    return math.degrees(math.atan2(dz, dx)) - 90.0


class LookControl:
    """Turns the mob toward a point, at a bounded rate.

    Vanilla's timer (`look_at_timer = 2` on every `look_at`) exists because
    the look target is set by a goal and acted on by the control one tick
    later. Here goals and control run in the same tick, so a goal that wants
    a look sets the point and `MobAI` calls `tick` straight after - the timer
    would only ever delay a turn by a tick and nothing else.
    """

    __slots__ = ("x", "y", "z", "active")

    def __init__(self) -> None:
        self.x = 0.0
        self.y = 0.0
        self.z = 0.0
        self.active = False

    def look_at(self, x: float, y: float, z: float) -> None:
        """Aim at a world point. Replaced every tick a look goal is running."""
        self.x = x
        self.y = y
        self.z = z
        self.active = True

    def reset(self) -> None:
        """Forget the look target.

        Called at the top of every AI tick, before the goals run, which is what
        makes "the goal that set this stopped" and "there is nothing to look
        at" the same fact. Clearing it in each goal's `stop` instead would be
        one more thing to forget, and a look target left behind is a cow whose
        head keeps drifting toward a player who walked away.
        """
        self.active = False

    def tick(self, mob) -> None:
        """Turn `mob.position.yaw` toward the point, within its rate limit.

        Pitch is written too, but only because the packet carries it: these
        three animals have `shouldStayHorizontal()` true in vanilla (a cow does
        not crane its neck to look at a floating eye), so the pitch is forced
        to 0 rather than computed. Writing it at all is what keeps a stale
        pitch from a previous client-driven entity out of the sync packet.
        """
        if not self.active:
            return
        pos = mob.position
        attributes = mob.attributes
        target_yaw = yaw_to(pos.x, pos.y, pos.z, self.x, self.y, self.z)
        if target_yaw is not None:
            pos.yaw = change_angle(pos.yaw, target_yaw,
                                   attributes.max_look_yaw_change)
        pos.pitch = 0.0


class MoveControl:
    """Turns the mob toward a wanted point and records the walk input.

    The input is `(strafe, 0, forward)` where `forward` is
    `speed_modifier * movement_speed` - Pumpkin's `MoveControl.tick` shape and
    vanilla's `LivingEntity.zza`. It is not a unit vector, and that is the
    point: `apply_input` multiplies by `movement_speed * GROUND_SPEED_FACTOR`
    again, so the acceleration is quadratic in the attribute. That is
    vanilla's own arithmetic, and it is why the attribute values (0.2-0.25
    for these animals) read far larger than the speeds they produce.
    """

    __slots__ = ("wanted_x", "wanted_y", "wanted_z", "speed_modifier",
                 "state", "arrive_sq")

    #: The three states vanilla's `MoveControl.Operation` has. `JUMPING` is a
    #: real one: once a mob has jumped at a wall it keeps pushing forward
    #: until it lands, which is why it clears the step instead of stalling
    #: against it on the tick it left the ground.
    WAIT = 0
    MOVE_TO = 1
    JUMPING = 2

    def __init__(self) -> None:
        self.wanted_x = 0.0
        self.wanted_y = 0.0
        self.wanted_z = 0.0
        self.speed_modifier = 0.0
        self.state = MoveControl.WAIT
        #: Squared radius at which this particular wanted position counts as
        #: reached. It is per-position rather than a constant because a
        #: waypoint and a destination are different claims: a waypoint is
        #: "close enough to have walked it, point me at the next one", and the
        #: radius for that is the mob's own (vanilla's
        #: `maxDistanceToWaypoint`, which is narrower than the default for a
        #: wide mob). Leaving this at the default for a waypoint makes the
        #: control announce arrival half a block early, zero its input, and
        #: cost the mob one stationary tick at *every* node boundary.
        self.arrive_sq = ARRIVED_DISTANCE_SQ

    def set_wanted_position(self, x: float, y: float, z: float,
                            speed_modifier: float,
                            arrive_distance: float = ARRIVED_DISTANCE) -> None:
        """Point the control at a world position.

        Refuses to cancel an in-flight jump, matching vanilla: setting a new
        wanted position while airborne must not stop the mob from finishing
        the step it started.

        `arrive_distance` is the radius at which this position counts as
        reached. A destination uses the default; a pathfinder waypoint passes
        its own narrower radius so the two layers agree on where "there" is.
        """
        self.wanted_x = x
        self.wanted_y = y
        self.wanted_z = z
        self.speed_modifier = speed_modifier
        self.arrive_sq = arrive_distance * arrive_distance
        if self.state != MoveControl.JUMPING:
            self.state = MoveControl.MOVE_TO

    def has_wanted(self) -> bool:
        return self.state != MoveControl.WAIT

    def clear(self) -> None:
        self.state = MoveControl.WAIT
        self.speed_modifier = 0.0
        self.arrive_sq = ARRIVED_DISTANCE_SQ

    def tick(self, mob) -> None:
        """Aim the mob at its wanted point and write `mob.movement_input`.

        Run *after* the look control, so a mob that is walking turns its body
        toward the walk and not toward whatever it was looking at.
        """
        attributes = mob.attributes
        speed = self.speed_modifier * attributes.movement_speed
        pos = mob.position

        if self.state == MoveControl.MOVE_TO:
            dx = self.wanted_x - pos.x
            dz = self.wanted_z - pos.z
            dy = self.wanted_y - pos.y
            if dx * dx + dy * dy + dz * dz < self.arrive_sq:
                mob.movement_input = (0.0, 0.0, 0.0)
                self.state = MoveControl.WAIT
                return

            target_yaw = yaw_to(pos.x, pos.y, pos.z, self.wanted_x,
                                self.wanted_y, self.wanted_z)
            if target_yaw is not None:
                pos.yaw = change_angle(pos.yaw, target_yaw, MOVE_TURN_RATE)
            mob.movement_input = (0.0, 0.0, speed)

            # Vanilla jumps when the wanted point is a full step or more above
            # the mob and close enough to be the next thing it would touch.
            # The width comparison is vanilla's, and it reads oddly: the
            # horizontal *distance* is compared against the mob's own width
            # squared, which is a coarse "the step is right in front of me".
            horizontal_sq = dx * dx + dz * dz
            if dy > attributes.step_height and horizontal_sq < 1.0:
                mob.ai.jump_control.jump()
                self.state = MoveControl.JUMPING
        elif self.state == MoveControl.JUMPING:
            mob.movement_input = (0.0, 0.0, speed)
            if pos.on_ground:
                self.state = MoveControl.WAIT
        else:
            mob.movement_input = (0.0, 0.0, 0.0)


class JumpControl:
    """A one-tick request to leave the ground.

    `jump()` sets a flag and `tick` publishes it onto the mob; the flag is
    cleared immediately after, so a jump is one impulse and not a held button.
    Holding it is what vanilla's `jumpingCooldown` and the swim goal's
    80%-per-tick roll are for, and the swim goal rolls rather than holding.
    """

    __slots__ = ("_requested",)

    def __init__(self) -> None:
        self._requested = False

    def jump(self) -> None:
        self._requested = True

    def tick(self, mob) -> None:
        mob.jumping = self._requested
        self._requested = False
