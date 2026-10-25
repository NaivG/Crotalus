"""
`MobAI` - what a mob thinks, and the order it thinks in.

One per `Mob`, reached as `mob.ai`. It owns four controls, a goal selector and
a navigator, and it is ticked from the mob's own `Mob.tick` - the hook
`entity/entity.py`'s `Entity.tick` docstring has pointed at since before any of
this existed, and the only one `Dimension.tick` calls.

    mob.tick()  ->  ai.tick()  ->  selector, controls, navigator, velocity

**Why the mob's tick and not a round over `Dimension.entities`.** Vanilla runs
the goal selector from `Mob.tick`, so a mob that is not ticking does not think,
and one that is ticks its goals in the same order every time. A server-side
"AI tick" that swept the entity table would have to re-derive that ordering
itself and would eventually disagree with `Dimension.ticks_entity`, which is
the one rule that says whether an entity is processed at all this tick. Running
from `Mob.tick` means there is exactly one place that can be wrong.

The tick order, and why it is that order:

1. `move_control` and `look_control` are *reset*. Vanilla's move control is
   one-shot - it consumes its operation and returns to WAIT, and the navigator
   refills it - and the look control's target is owned by whichever goal set
   it. Resetting here rather than clearing in every `stop` is what makes "the
   look goal stopped" and "the mob is no longer looking" the same statement
   instead of two that can drift.
2. The selector runs. Goals start, stop, and drive the controls.
3. The navigator follows its path: it advances the waypoint cursor, or gives
   up if the mob is stuck or out of time. It writes a `MoveControl` wanted
   position and, when the next node is above the mob, a jump request - and
   nothing else. Turning, accelerating and stopping are the control's.
4. `look_control.tick`, then `move_control.tick`. Look first, move second, so a
   mob that is walking faces where it is going.
5. `jump_control.tick` publishes the one-tick jump request. After the move
   control, because the move control is what requests the step-up jump.
6. `apply_input` writes one velocity, and publishes a rotation if the yaw moved
   without the body moving.

**Step 6 is where the physics solver is written to, and only there.**
`PhysicsSolver.set_velocity` is the only sanctioned way to give a body a speed
(`physics/solver.py` says so at length), and a mob that wrote
`entity.position.velocity_x` directly would be overwritten by the next
`_sync_entities`.

**And it is also where the mob is allowed to be quiet.** `apply_input` writes
nothing at all when the mob wants to stand still. That is not an optimisation
detail, it is the difference between the sleeping bodies the physics solver
works so hard to keep (`physics/solver.py` §"Sleeping is a real skip") and a
world where every mob is re-integrated every tick because it keeps waking
itself to be told it has nowhere to go. The write is an *acceleration* -
Pumpkin's `moveRelative`, added to the body's current velocity - and the
solver's ground friction is what bleeds it off, so there is no stop write
either: a cow that stops walking coasts for a fifth of a block while friction
drains the last of its speed, exactly as vanilla's does, and falls asleep on
its own once the residual drops under the solver's rest threshold.
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

from entity.ai.control import (
    GROUND_SPEED_FACTOR,
    JumpControl,
    LookControl,
    MoveControl,
)
from entity.ai.goal import Goal, GoalSelector
from entity.ai.pathfinding.navigation import PathNavigation

__all__ = ["MobAI"]


class MobAI:
    """One mob's behaviour: its goals, its controls, and where it is going."""

    __slots__ = ("mob", "context", "selector", "look_control",
                 "move_control", "jump_control", "navigator",
                 "vertical_override", "ticks", "no_body_ticks")

    def __init__(self, mob, goals: Optional[List[Tuple[int, Goal]]] = None) -> None:
        #: The mob these goals act on. A back-reference, and the only one: the
        #: controls reach the mob through their argument and the navigator
        #: holds nothing, so the object graph is `Mob -> MobAI -> Mob` with no
        #: path from any control back to the AI.
        self.mob = mob
        #: The world handle. `None` until the spawner binds it, and a mob with
        #: no context does nothing at all rather than guessing - see
        #: `entity/ai/context.py`.
        self.context = None
        self.selector = GoalSelector(goals)
        self.look_control = LookControl()
        self.move_control = MoveControl()
        self.jump_control = JumpControl()
        #: One per mob, as before - `PathNavigation` holds the destination, the
        #: current node path, and the three watches over the walk. No state is
        #: shared between mobs; the search each one runs is its own.
        self.navigator = PathNavigation()
        #: A goal's vertical override for this tick (`SwimGoal` in water).
        #: `None` means "whatever the jump state says", which is the default.
        self.vertical_override: Optional[float] = None
        #: Ticks this AI has run, and how many of them had no physics body.
        #: The second is a real number rather than a log line because "the AI
        #: never wrote a velocity" is indistinguishable, from the outside, from
        #: "the AI had nothing to say".
        self.ticks = 0
        self.no_body_ticks = 0

    def bind(self, context) -> None:
        """Attach the world handle. Called once, by the spawner, at placement.

        Late rather than in the constructor because a mob is built before it is
        in a world: `NaturalSpawner._place` calls this on the same line it
        calls `physics.add_entity`, and a mob bound without a body is a mob the
        AI will correctly refuse to move.
        """
        self.context = context

    # ----- the tick -----

    def tick(self) -> None:
        """One game tick of behaviour. Called from `Mob.tick`.

        An AI with no context does nothing at all, goals included. That is a
        property of the whole brain rather than of each goal: `RandomLookAround`
        needs no world and would happily roll its 2% and turn a cow that has
        nowhere to stand, and a mob whose head moves while its body cannot is
        worse than one that is plainly inert.
        """
        self.ticks += 1
        if self.context is None:
            return
        mob = self.mob
        # Vanilla's one-shot controls: neither survives the tick on its own.
        self.move_control.clear()
        self.look_control.reset()
        self.vertical_override = None
        yaw_before = mob.position.yaw

        self.selector.tick(mob)
        self.navigator.tick(mob)
        self.look_control.tick(mob)
        self.move_control.tick(mob)
        self.jump_control.tick(mob)
        self.apply_input(mob, yaw_before)

    def apply_input(self, mob, yaw_before: float) -> None:
        """Turn this tick's input into one acceleration, and publish any turn.

        The rotation publish is here and not in the tracker because the
        physics solver only pushes bodies that *moved* or that changed ground
        state (`PhysicsSolver._notify_tracker`). A cow turning her head on the
        spot is exactly the case it skips, and without this her head would lag
        her body indefinitely.

        `yaw_before` is captured at the top of `MobAI.tick`, before the
        controls ran - comparing against `pos.yaw` here would compare the
        yaw against itself and find no turn, ever.

        The velocity write is Pumpkin's `moveRelative` against vanilla's
        `travel`: the acceleration is *added* to the body's current velocity,
        and the solver's ground friction (`0.546` per tick on grass) is what
        bleeds it back off. The attribute enters the acceleration twice - the
        input already carries `speed_modifier * movement_speed`, and the
        factor multiplies by `movement_speed` again - so the steady state is
        `accel / (1 - friction)` and a 0.2 cow walks at ~0.088 blocks per
        tick, not 0.2. This is also why nothing is written when the mob wants
        to stand still: a zero acceleration written to a coasting body would
        freeze it mid-glide, and the solver's friction is what brings it to
        rest and lets it sleep.
        """
        context = self.context
        if context is None:
            return

        pos = mob.position
        strafe, _, forward = mob.movement_input
        wants_motion = (strafe != 0.0 or forward != 0.0
                        or mob.jumping or self.vertical_override is not None)

        entity_id = mob.entity_id
        if context.has_body(entity_id):
            self.no_body_ticks = 0
            if wants_motion:
                yaw = math.radians(pos.yaw)
                cos_yaw = math.cos(yaw)
                sin_yaw = math.sin(yaw)
                accel_scale = (mob.attributes.movement_speed
                               * GROUND_SPEED_FACTOR)
                ax = (strafe * cos_yaw - forward * sin_yaw) * accel_scale
                az = (forward * cos_yaw + strafe * sin_yaw) * accel_scale
                # Vertical is the solver's - it owns gravity and wrote the
                # current value back onto the position. Only a jump or an
                # explicit override replaces it; writing 0 here would be a
                # mob that hovers.
                vy = pos.velocity_y
                if self.vertical_override is not None:
                    vy = self.vertical_override
                elif mob.jumping:
                    vy = mob.attributes.jump_power
                context.set_velocity(entity_id,
                                     pos.velocity_x + ax, vy,
                                     pos.velocity_z + az)
        else:
            self.no_body_ticks += 1

        if pos.yaw != yaw_before:
            context.push_rotation(entity_id, pos.x, pos.y, pos.z,
                                  pos.yaw, pos.pitch, pos.on_ground)

    def stop(self) -> None:
        """Release everything: goals, controls, and the navigator's target.

        For the paths that end a mob's life mid-thought - the void pass, the
        spawner's residency sweep. A goal that is not stopped leaves a
        navigator target pointing into a world the mob is no longer in.
        """
        self.selector.stop_all()
        self.navigator.stop()
        self.move_control.clear()
        self.look_control.reset()

    def running_goals(self) -> Tuple[str, ...]:
        """Names of the goals running this tick, for stats and the check tool."""
        return tuple(type(p.goal).__name__ for p in self.selector.running_goals())
