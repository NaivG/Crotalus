"""
The goals themselves.

Five of them, and they are the intersection of what vanilla's cow, sheep and
chicken carry and what this server can answer. The vanilla lists are:

    cow      swim, escapeDanger(2.0), breed, tempt(1.25), followParent(1.25),
             wander(1.0), lookAt(player, 6.0), lookAround
    chicken  swim, escapeDanger(1.4), breed, tempt(1.0), followParent(1.1),
             wander(1.0), lookAt(player, 6.0), lookAround
    sheep    swim, escapeDanger(1.25), breed, tempt(1.1), followParent(1.1),
             eatGrass, wander(1.0), lookAt(player, 6.0), lookAround

which is why the user-facing description of this round is "three mobs that look
alike": the three lists differ only in three floats and one extra goal. The
priorities are preserved exactly, because the gaps between them are what decide
who wins a control - `EscapeDanger` at 1 beating `Wander` at 5 is the whole
reason a hurt cow runs instead of continuing her stroll.

**The four that are not here, and why.** They are listed in
`entity/MOB_AI_PLAN.md` with the same detail; in one line each:

* `Breed` needs the love-mode pair, the baby age counter and the birth timer -
  none of which exist, and a `Breed` that starts and then does nothing is worse
  than no `Breed`.
* `Tempt` needs dropped item entities to be attracted to. There are none: an
  item is not an entity type this server builds.
* `FollowParent` needs `age` / `isBaby` and the metadata that carries it.
* `EatGrass` (sheep only) needs a block change that reaches the client.
  `Dimension.set_block` writes the array and stops - it does not mark the chunk
  dirty and it sends nothing, so a sheep eating grass would leave the client's
  grass standing and the change would evaporate on unload.

`EscapeDanger` is the interesting one: it *is* here and it is wired to
`Mob.panic_source`, which nothing sets yet, because there is no damage path.
It is shipped anyway rather than dropped, because it is the top behaviour a
player sees on a cow and because "nothing sets this yet" is a gap a reader can
see, while a missing goal is a gap a reader has to notice.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

from entity.ai.goal import JUMP, LOOK, MOVE, Goal
from entity.ai.random_pos import random_pos_near

__all__ = [
    "EscapeDangerGoal",
    "LookAtEntityGoal",
    "RabbitHopGoal",
    "RandomLookAroundGoal",
    "SwimGoal",
    "WanderAroundGoal",
]

#: Vanilla's `PathNavigation.randomPos`, halved the way a goal's ticks are:
#: `WanderAroundGoal` rolls `nextInt(toGoalTicks(120))`, i.e. 60, so a cow
#: starts a stroll about once every three seconds.
WANDER_INTERVAL = 120

#: Vanilla's wander radii: ten blocks out, seven up or down. Ten draws of a
#: direction inside a 21x15x21 box, keeping the best, is what makes a cow's
#: stroll a stroll rather than a sprint across the field.
WANDER_HORIZONTAL_DISTANCE = 10
WANDER_VERTICAL_DISTANCE = 7

#: Vanilla's `EscapeDangerGoal` radii.
ESCAPE_HORIZONTAL_DISTANCE = 5
ESCAPE_VERTICAL_DISTANCE = 4

#: `LookAtEntityGoal` as vanilla's three factories build it: six blocks,
#: a 2% chance per tick, looking at players.
LOOK_AT_RANGE = 6.0
LOOK_AT_CHANCE = 0.02

#: `RandomLookAroundGoal`'s per-tick chance. Two percent of twenty ticks per
#: second is a glance at nothing about every two and a half seconds of standing
#: still, which is the cadence a cow actually has.
LOOK_AROUND_CHANCE = 0.02

#: Vanilla's `RabbitJumpGoal` rolls an internal counter of 10 ticks and
#: triggers a hop when it reaches zero, then resets. The visible effect is
#: that a rabbit hops about every 10-30 ticks during wander; the exact
#: distribution is what makes "rabbit on a field" recognisable and what makes
#: a startled rabbit look like a startled rabbit (faster hops, more often).
#: We use a single constant here and let the goal's tick decide - the rabbit
#: alone uses this, and a class constant keeps the loop tight.
HOP_COOLDOWN_TICKS = 12

#: How many ticks a random look lasts, and the extra it rolls.
LOOK_AROUND_TICKS = 20

#: Vanilla's `travelInWater` adds `0.04 * waterViewFactor` to the vertical
#: velocity every tick and applies no gravity at all in a fluid. This server's
#: solver subtracts its `-0.08` unconditionally, into the motion, before
#: anything else sees it - so the AI has to hand the solver a velocity that
#: survives that subtraction. 0.08 cancels it (a neutrally buoyant mob) and
#: 0.12 leaves the vanilla 0.04 of rise.
WATER_GRAVITY = 0.08
WATER_RISE_VELOCITY = WATER_GRAVITY + 0.04


class SwimGoal(Goal):
    """Float, and swim upward, in a fluid.

    Priority 0 on all three animals, which is what makes it win over everything
    including `EscapeDanger`: a cow in a river does not decide whether she is
    more afraid or more wet.

    Vanilla's `is_in_fluid` tests `touching_water && waterHeight > swimHeight`,
    where the swim height is a fraction of the mob's own height - a chicken
    (0.7 tall) is swimming as soon as its feet are wet, a cow (1.4) has to be
    further in. Two block samples reproduce that: the feet block, and the block
    containing two thirds of the way up the body.

    The 0.8-per-tick roll in vanilla's `tick` is over the *input*, not the
    rise, and this server has no swim input to roll. The rise itself is
    unconditional in vanilla, so it is unconditional here.
    """

    controls = JUMP

    def in_fluid(self, mob) -> bool:
        context = mob.ai.context
        if context is None:
            return False
        pos = mob.position
        bx, by, bz = math.floor(pos.x), math.floor(pos.y), math.floor(pos.z)
        if context.is_fluid(bx, by, bz):
            return True
        height = mob.entity_type.height if mob.entity_type else 1.0
        return context.is_fluid(bx, math.floor(pos.y + height * 0.667), bz)

    def can_start(self, mob) -> bool:
        return self.in_fluid(mob)

    def should_continue(self, mob) -> bool:
        return self.in_fluid(mob)

    def tick(self, mob) -> None:
        mob.ai.vertical_override = WATER_RISE_VELOCITY


class EscapeDangerGoal(Goal):
    """Run away from whatever last hurt this mob.

    Vanilla's trigger is `lastDamageType` carrying `#minecraft:panic_causes` -
    suffocation, fall, fire, cactus, drowning, void and a handful more. What
    that reduces to on this server is `Mob.panic_source`, and the reason it is
    a field rather than a health comparison is that vanilla's is a *tag on a
    damage type*, not "am I below half health": being pricked by an arrow makes
    a cow flee and hitting it hard does not.

    Vanilla's on-fire branch - dive for the nearest water within five blocks -
    is absent, because nothing can set a mob on fire. The random-position
    branch below is what runs instead, and it is what vanilla runs when the mob
    is not burning.
    """

    controls = MOVE

    def __init__(self, speed: float) -> None:
        self.speed = speed
        self.target: Optional[Tuple[float, float, float]] = None

    def can_start(self, mob) -> bool:
        if mob.panic_source is None:
            return False
        context = mob.ai.context
        if context is None:
            return False
        self.target = random_pos_near(mob, context,
                                      ESCAPE_HORIZONTAL_DISTANCE,
                                      ESCAPE_VERTICAL_DISTANCE)
        return self.target is not None

    def should_continue(self, mob) -> bool:
        return not mob.ai.navigator.is_idle

    def start(self, mob) -> None:
        if self.target is not None:
            # The mob is passed so `move_to` can answer whether the target is
            # actually walkable to. The answer is not used here - a panic goal
            # runs whether or not the path exists, because "I could not find a
            # route" and "there is no route" both want a different target next
            # tick - but a wander goal with an unreachable target is better off
            # knowing, and the cost is one search either way.
            mob.ai.navigator.move_to(*self.target, self.speed, mob=mob)

    def stop(self, mob) -> None:
        # The navigator is shared state, not this goal's private business -
        # but nothing else can be walking at the same time (only one goal
        # holds MOVE), so "the navigator has a target" and "I have a target"
        # are the same statement. Leaving it set would have the mob walking
        # with no goal running, which `stop_all` at the end of a mob's life
        # would then never clean up.
        self.target = None
        mob.ai.navigator.stop()


class WanderAroundGoal(Goal):
    """Walk somewhere nearby, every few seconds, at walking pace.

    The mob's baseline behaviour. Priority 5 on all three animals: below
    everything that has a reason, above nothing.

    `can_start` rolls first and asks for a position second, which is vanilla's
    order and matters for the cost - the roll is one bounded integer and
    rejects 59 ticks out of 60, while the position search is ten block reads.
    Rolled the other way round it would run ten block reads every tick.
    """

    controls = MOVE

    def __init__(self, speed: float = 1.0,
                 interval: int = WANDER_INTERVAL) -> None:
        self.speed = speed
        self.interval = interval
        self.target: Optional[Tuple[float, float, float]] = None

    def can_start(self, mob) -> bool:
        if mob.random.next_bounded_i32(self.get_tick_count(self.interval)) != 0:
            return False
        context = mob.ai.context
        if context is None:
            return False
        self.target = random_pos_near(mob, context,
                                      WANDER_HORIZONTAL_DISTANCE,
                                      WANDER_VERTICAL_DISTANCE)
        return self.target is not None

    def should_continue(self, mob) -> bool:
        return not mob.ai.navigator.is_idle

    def start(self, mob) -> None:
        if self.target is not None:
            # `random_pos_near` only returns a position with solid floor under
            # it, so this is usually walkable to - but "solid floor" and "on
            # a connected path" are different questions, and the return value
            # is the second one.
            mob.ai.navigator.move_to(*self.target, self.speed, mob=mob)

    def stop(self, mob) -> None:
        self.target = None
        mob.ai.navigator.stop()


class LookAtEntityGoal(Goal):
    """Turn to face a nearby player.

    Holds LOOK only, so it runs *alongside* a walking goal - the cow keeps
    walking toward her destination while facing you. That it can only turn the
    body, not the head, is `entity/ai/control.py`'s single-rotation gap.

    The chance is per tick and tiny (2%), which is what makes it a glance and
    not a stare: `should_continue` then holds the look for 20-39 halved ticks
    (the goal does not run every tick), so a cow looks at you for one or two
    seconds and then goes back to minding her business.
    """

    controls = LOOK

    def __init__(self, look_range: float = LOOK_AT_RANGE,
                 chance: float = LOOK_AT_CHANCE) -> None:
        self.look_range = look_range
        self.chance = chance
        self.target: Optional[Tuple[float, float, float, float]] = None
        self.look_time = 0

    def can_start(self, mob) -> bool:
        if mob.random.next_f32() >= self.chance:
            return False
        context = mob.ai.context
        if context is None:
            return False
        pos = mob.position
        self.target = context.nearest_viewer(pos.x, pos.y, pos.z,
                                             self.look_range)
        return self.target is not None

    def should_continue(self, mob) -> bool:
        if self.target is None:
            return False
        pos = mob.position
        dx = self.target[0] - pos.x
        dz = self.target[2] - pos.z
        if dx * dx + dz * dz > self.look_range * self.look_range:
            return False
        return self.look_time > 0

    def start(self, mob) -> None:
        self.look_time = self.get_tick_count(
            40 + mob.random.next_bounded_i32(40))

    def stop(self, mob) -> None:
        self.target = None

    def tick(self, mob) -> None:
        if self.target is None:
            return
        self.look_time -= 1
        mob.ai.look_control.look_at(self.target[0], self.target[3],
                                     self.target[2])


class RandomLookAroundGoal(Goal):
    """Stop and stare in a random direction for a second.

    The only goal besides `SwimGoal` that asks for more than one control: MOVE
    | LOOK, so it *stops the mob walking* while it runs. That is deliberate and
    is the goal's whole visible effect - a cow that wanders into a spot, stops,
    looks at a tree for twenty ticks, and then carries on is the most
    recognisable thing vanilla passive mobs do.

    The look target is `position + (cos t, ., sin t)` with `t` rolled once at
    `start`, which is a *fixed* compass bearing: the mob turns to a direction
    and holds it, rather than drifting. Getting that wrong produces a cow whose
    head sweeps continuously, which is the classic tell of a re-rolled
    per-tick direction.
    """

    controls = MOVE | LOOK

    def __init__(self, chance: float = LOOK_AROUND_CHANCE) -> None:
        self.chance = chance
        self.delta_x = 0.0
        self.delta_z = 0.0
        self.look_time = 0

    def can_start(self, mob) -> bool:
        return mob.random.next_f32() < self.chance

    def should_continue(self, mob) -> bool:
        return self.look_time >= 0

    def start(self, mob) -> None:
        angle = math.tau * mob.random.next_f64()
        self.delta_x = math.cos(angle)
        self.delta_z = math.sin(angle)
        self.look_time = LOOK_AROUND_TICKS + mob.random.next_between(
            0, LOOK_AROUND_TICKS)

    def tick(self, mob) -> None:
        self.look_time -= 1
        pos = mob.position
        mob.ai.look_control.look_at(pos.x + self.delta_x,
                                     pos.y + mob.eye_height,
                                     pos.z + self.delta_z)


class RabbitHopGoal(Goal):
    """A periodic hop during walking. Priority 5, alongside wander.

    Vanilla's `RabbitJumpGoal` is a `MoveToBlockGoal` subclass that picks a
    target one hop away on grass and walks toward it as a jump. We have no
    pathfinder, so the goal is the much simpler "while walking, hop every so
    often" - the visible effect is the same on a flat field, and the corner
    case (a rabbit that needs to step up to get over a fence) is a gap we
    accept and note, the same way the cow's straight-line wander accepts that
    a wall is hit before a corner is taken.

    JUMP only - the goal asks for no other control, so it runs alongside
    `WanderAroundGoal` without taking MOVE away. The selector arbitrates per
    control, and a goal that wants JUMP never conflicts with one that wants
    MOVE: the rabbit walks toward its wander target and hops over the
    ground in the process. That is the whole design in one sentence.

    `mob.jumping = True` is a one-tick flag (`JumpControl.tick` clears it
    after publishing), so a hop is a one-impulse event - the rabbit takes
    off this tick, lands ~7-10 ticks later, and is back on the ground. The
    goal tracks its own cooldown to keep hops spaced, and decides each tick
    whether the rabbit is in a state to take off (on the ground, not in a
    fluid, cooldown at zero).
    """

    controls = JUMP

    def __init__(self, cooldown: int = HOP_COOLDOWN_TICKS) -> None:
        self.cooldown = cooldown
        self.cooldown_ticks = 0

    def can_start(self, mob) -> bool:
        """A rabbit is a candidate whenever it is on solid ground and in air.

        The cooldown is *not* here - `can_start` runs each tick regardless of
        the cooldown, and `tick` reads the cooldown and refuses to hop when
        it is non-zero. The reason is that `can_start` rolling the dice and
        `tick` acting on it would let a goal win and then do nothing for ten
        ticks, which is the visible sign of an `if` where there should be a
        `while`.
        """
        if not mob.position.on_ground:
            return False
        context = mob.ai.context
        if context is None:
            return False
        bx, by, bz = (math.floor(mob.position.x), math.floor(mob.position.y),
                      math.floor(mob.position.z))
        if context.is_fluid(bx, by, bz):
            return False
        return True

    def should_continue(self, mob) -> bool:
        """Keep running until the rabbit leaves the ground or enters water."""
        if not mob.position.on_ground:
            return False
        context = mob.ai.context
        if context is None:
            return False
        bx, by, bz = (math.floor(mob.position.x), math.floor(mob.position.y),
                      math.floor(mob.position.z))
        if context.is_fluid(bx, by, bz):
            return False
        return True

    def tick(self, mob) -> None:
        """Decide each tick whether the rabbit hops.

        The state machine:
          - `cooldown_ticks > 0`: count it down, do nothing. This is the gap
            between hops; vanilla's `RabbitJumpGoal` resets to 10 the moment
            a hop lands, and 10 ticks at 20 tps is half a second - the time
            the rabbit spends in the air.
          - `cooldown_ticks == 0`: ask `should_hop_now`. If yes, call
            `mob.ai.jump_control.jump()` (the one-tick request the AI turns
            into `mob.jumping = True` at the end of this tick) and reset the
            cooldown. If no, leave the cooldown at zero so the next tick
            asks again.

        The decision lives in a separate method because it is the only part
        of this goal that is genuinely a design choice: vanilla rolls a
        uniform tick counter; an "is the wander goal currently walking"
        predicate would tie hops to wander targets; a "is the rabbit moving
        at all" check would suppress hops on a stopped rabbit. Each of those
        is a recognisably different rabbit.
        """
        if self.cooldown_ticks > 0:
            self.cooldown_ticks -= 1
            return
        if self.should_hop_now(mob):
            mob.ai.jump_control.jump()
            self.cooldown_ticks = self.cooldown

    def should_hop_now(self, mob) -> bool:
        """Whether this tick is the tick the rabbit hops.

        TODO: implement the hop trigger. The inputs you have:
          - `mob.random.next_bounded_i32(n)` returns an int in [0, n).
            `mob.random.next_f32()` returns a float in [0, 1).
          - `mob.position.on_ground` is True (the goal already checked).
          - `mob.ai.navigator.is_idle` is True when no walk target is set -
            the rabbit is not actively wandering. A hopping rabbit that
            is not walking looks stranger than a still rabbit.
          - `mob.ai.vertical_override is not None` means a swim goal is
            overriding vertical motion; the rabbit is in fluid and should
            not hop from there.
        The cooldown is at zero when this is called - the gap between hops
        is enforced by `tick` resetting it after a hop. So this method is
        only asked "is *now* a good moment", not "how long since the last
        hop".

        Trade-offs to weigh:
          - Rolling `next_bounded_i32(N) == 0` for some N gives a flat
            cadence (about one hop every N ticks). Vanilla uses ~10.
          - Reading `not mob.ai.navigator.is_idle` keeps hops tied to
            walking, which is the vanilla look but adds a coupling to the
            wander goal.
          - Using `next_f32() < 0.05` gives a probabilistic hop (5% of
            ticks); less predictable but reads as "rabbits are twitchy".
        Aim for 5-10 lines. Read `mob.ai.navigator` only if you decide the
        wander coupling is worth it - the navigator exists for that.
        """
        raise NotImplementedError("RabbitHopGoal.should_hop_now: see TODO")
