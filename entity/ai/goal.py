"""
Goals, priorities, and the selector that decides which one runs.

A mob's behaviour is a *priority list*, not an `if/elif` chain, and the two
are not the same thing. Vanilla's `GoalSelector` runs several goals at once and
makes them negotiate over four shared resources - move, look, jump, target -
which is why a cow can be walking away (`WanderAround` holds MOVE) and still
turn her head toward you (`LookAtEntity` holds only LOOK). An `if/elif` chain
cannot express that: the moment a lower-priority branch runs, the higher one
stops.

The rules, as vanilla states them and as Pumpkin's `goal_selector.rs` ports
them:

* A goal runs when nothing else holds a control it wants at an equal or
  higher priority. `RandomLookAround` asking for MOVE|LOOK therefore locks a
  walking cow in place for its 20-40 ticks - it *is* how vanilla cows stop and
  stare at nothing.
* `should_continue` failing stops the goal. There is no separate "finish what
  you started"; `EatGrassGoal` restarts from scratch if it is interrupted.
* A goal's timer runs at half speed unless it declares itself
  `should_run_every_tick`. That is `to_goal_ticks`, and it is why a cow
  wanders roughly every three seconds rather than every 1.5: the 120-tick
  interval is halved to 60 chances, one per tick, each a 1-in-60 roll.

**The controls are a mutual-exclusion resource, not a lock.** Only one goal
holds MOVE at a time, and the holder is re-decided every tick from scratch:
a goal that wanted MOVE last tick and lost it does not "resume" when it wins
back, it runs `start` again. That is why `PrioritizedGoal.running` is the only
state the selector keeps per goal.

Why this is a Python class hierarchy rather than a data-driven goal list: the
registries do not carry AI in this server (`assets/entities.json` has no
behaviour section), and Pumpkin's Rust `Box<dyn Goal>` maps onto a plain
Python base class with the same four methods and nothing more.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

__all__ = [
    "JUMP",
    "LOOK",
    "MOVE",
    "TARGET",
    "Goal",
    "GoalSelector",
    "PrioritizedGoal",
    "to_goal_ticks",
]

#: Move control. Held by anything that wants the mob to walk somewhere.
MOVE = 1
#: Look control. Held by anything that wants to aim the mob's head.
LOOK = 2
#: Jump control. Held by anything that wants the mob off the ground.
JUMP = 4
#: Target control. Unused this round - no goal attacks anything.
TARGET = 8

#: The four controls, in the order `GoalSelector` indexes them. Kept as a
#: tuple rather than derived from the constants: the selector's
#: `_NO_GOAL` sentinel array is sized from it, and deriving a count from bit
#: values makes a future fifth control a silent off-by-one instead of a
#: visible NameError.
ALL_CONTROLS: Tuple[int, ...] = (MOVE, LOOK, JUMP, TARGET)

#: "No goal holds this control." A plain index into the selector's goal list
#: would be a valid one, so the empty slot needs a value nothing else can take.
_NO_GOAL = -1


def to_goal_ticks(server_ticks: int) -> int:
    """Halve a goal timer, rounding up. Vanilla's `Goal.toGoalTicks`.

    Ceiling, not truncation: a 1-tick timer must still fire on some tick, and
    `(1 + 1) // 2` truncating to 0 would turn it into a goal that never runs.
    """
    return -(-server_ticks // 2)


def has_control(controls: int, wanted: int) -> bool:
    """Whether a goal's control mask *includes* another mask.

    Subset rather than intersection, and the distinction is load-bearing:
    `RandomLookAround` holds MOVE|LOOK, and `selector` asks whether that mask
    contains LOOK (yes) as well as whether it intersects MOVE (also yes).
    A goal holding nothing asked about returns True, which is what makes
    "this goal uses no control" mean "it can never be displaced".
    """
    return (controls & wanted) == wanted


class Goal:
    """One behaviour a mob may run.

    Every method is a no-op or False by default, so a goal overrides only the
    hooks its behaviour actually needs: `RandomLookAroundGoal` has no
    `can_start` state to release and no target to drop, and spells out neither.
    """

    #: Which controls this goal holds while running. A goal that wants to walk
    #: says `MOVE`; a goal that only wants to aim says `LOOK`.
    controls: int = 0

    def can_start(self, mob) -> bool:
        """Whether this goal wants the tick it is asked about.

        Called for every non-running goal, every tick. Goals that roll dice
        here must be cheap when they say no - `WanderAroundGoal` rolls one
        bounded int and returns.
        """
        return False

    def should_continue(self, mob) -> bool:
        """Whether a *running* goal keeps running. Defaults to `can_start`.

        The default is right for goals whose trigger and its persistence are
        the same question ("am I in water"), and wrong for the ones that need
        to finish what they started - those override it.
        """
        return self.can_start(mob)

    def start(self, mob) -> None:
        """Called once, on the tick the goal wins its controls."""

    def stop(self, mob) -> None:
        """Called once, on the tick the goal loses its controls.

        Releases whatever it took: the navigator's target, a looked-at entity,
        a timer. A `stop` that leaves a wanted position behind is how a cow
        ends up walking into a lake after the goal that sent her there gave up.
        """

    def tick(self, mob) -> None:
        """Called every tick while running (subject to `should_run_every_tick`)."""

    def should_run_every_tick(self) -> bool:
        """Whether `tick` runs every tick rather than every other.

        Goals that hold their own countdown timers and do one discrete thing
        when it expires say True; everything else is halved along with its
        `get_tick_count` intervals.
        """
        return False

    def can_stop(self) -> bool:
        """Whether the selector may take this goal's controls away mid-run."""
        return True

    def get_tick_count(self, ticks: int) -> int:
        """Scale a tick interval for this goal's own timer.

        The one place `should_run_every_tick` leaks into behaviour: a goal that
        runs every tick keeps vanilla's intervals (its timer is decremented by
        one per tick anyway), and one that does not gets them halved with
        `to_goal_ticks` so its two-tick cadence still lands on one tick per
        game tick.
        """
        return ticks if self.should_run_every_tick() else to_goal_ticks(ticks)


class PrioritizedGoal:
    """A goal plus its priority and its running flag.

    The wrapper exists for the same reason vanilla has one: the selector needs
    to remember what it started so it can stop it, and the goal object itself
    must stay a stateless template that any number of mobs share conceptually.
    """

    __slots__ = ("goal", "priority", "running")

    def __init__(self, goal: Goal, priority: int) -> None:
        self.goal = goal
        self.priority = priority
        self.running = False

    @property
    def controls(self) -> int:
        """The wrapped goal's controls. Vanilla's `PrioritizedGoal` implements
        `Goal` and gets this for free; here it is a property, because the
        selector asks every holder "what do you control" and the answer must
        not need unwrapping at each of the four call sites."""
        return self.goal.controls

    def can_be_replaced_by(self, other: "PrioritizedGoal") -> bool:
        """Whether `other` may take the controls this one holds.

        Strictly lower priority, and only while this one *can* stop. Equal
        priorities never displace each other, which is what makes insertion
        order the tie-break rather than an accident of list order.
        """
        return self.goal.can_stop() and other.priority < self.priority

    def __repr__(self) -> str:  # pragma: no cover - diagnostics only
        return (f"<PrioritizedGoal {type(self.goal).__name__} "
                f"p={self.priority} running={self.running}>")


class GoalSelector:
    """Runs a mob's goals and arbitrates over the four controls.

    One per mob, ticked from the mob's own tick - see `entity/ai/__init__.py`.
    Not thread safe, and it does not need to be: it lives on the dimension
    actor thread with everything else the mob owns.
    """

    __slots__ = ("goals", "_by_control", "_disabled")

    def __init__(self, goals: Optional[List[Tuple[int, Goal]]] = None) -> None:
        # Insertion order is preserved because it is the tie-break: two goals
        # of equal priority that both want MOVE are decided by which was added
        # first, and a selector that sorted its list would silently change
        # every mob's behaviour depending on how it was written.
        self.goals: List[PrioritizedGoal] = [
            PrioritizedGoal(goal, priority)
            for priority, goal in (goals or [])
        ]
        self._by_control: List[int] = [_NO_GOAL] * len(ALL_CONTROLS)
        self._disabled: int = 0

    def add_goal(self, priority: int, goal: Goal) -> None:
        """Append a goal. Only ever called while building the selector."""
        self.goals.append(PrioritizedGoal(goal, priority))

    def disable_control(self, control: int) -> None:
        """Stop any goal from taking a control (vanilla's `disableControl`).

        Nothing calls it this round - it exists because a control nobody can
        disable is a control that can only be given away one goal at a time,
        and `EatGrassGoal` disabling JUMP to keep a sheep from hopping while
        it eats is the reason vanilla has the switch.
        """
        self._disabled |= control

    def _control_slot(self, control: int) -> int:
        return ALL_CONTROLS.index(control)

    def _uses_disabled(self, goal: PrioritizedGoal) -> bool:
        return (goal.goal.controls & self._disabled) != 0

    def _can_replace_all(self, candidate: PrioritizedGoal) -> bool:
        """Whether `candidate` outranks every goal holding a control it wants."""
        controls = candidate.goal.controls
        for control in ALL_CONTROLS:
            if not has_control(controls, control):
                continue
            holder_index = self._by_control[self._control_slot(control)]
            if holder_index == _NO_GOAL:
                continue
            if not self.goals[holder_index].can_be_replaced_by(candidate):
                return False
        return True

    def _take_controls(self, index: int, mob) -> None:
        """Give goal `index` every control it asks for, stopping the losers."""
        controls = self.goals[index].goal.controls
        for control in ALL_CONTROLS:
            if not has_control(controls, control):
                continue
            slot = self._control_slot(control)
            held = self._by_control[slot]
            if held != _NO_GOAL and held != index:
                self.goals[held].running = False
                self.goals[held].goal.stop(mob)
            self._by_control[slot] = index

    def tick(self, mob) -> None:
        """One tick of arbitration, then every running goal.

        The three phases are vanilla's and the order is not negotiable:

        1. stop everything whose controls are disabled or whose
           `should_continue` failed;
        2. free the control slots those stops vacated;
        3. offer the slots to the non-running goals, in priority-then-insertion
           order.

        Stopping before offering is what makes "the cow stopped wandering" and
        "the cow started looking at you" the same tick rather than the same
        tick plus one.
        """
        for prioritized in self.goals:
            if prioritized.running and (
                self._uses_disabled(prioritized)
                or not prioritized.goal.should_continue(mob)
            ):
                prioritized.running = False
                prioritized.goal.stop(mob)

        # Phase 2. A slot whose holder stopped must be released before phase 3
        # looks at it, or `_can_replace_all` would compare against a goal that
        # is no longer running and let nothing take its MOVE.
        for slot in range(len(self._by_control)):
            index = self._by_control[slot]
            if index != _NO_GOAL and not self.goals[index].running:
                self._by_control[slot] = _NO_GOAL

        # Phase 3. `can_start` is allowed to have side effects on the goal
        # (WanderAround caches the position it just picked); it is not allowed
        # to have side effects on the *world*, because a goal that fails
        # `can_start` after having moved something would move it again next
        # tick.
        for index in range(len(self.goals)):
            prioritized = self.goals[index]
            if prioritized.running or self._uses_disabled(prioritized):
                continue
            if not self._can_replace_all(prioritized):
                continue
            if not prioritized.goal.can_start(mob):
                continue
            self._take_controls(index, mob)
            prioritized.running = True
            prioritized.goal.start(mob)

        # Every running goal ticks every tick. `should_run_every_tick` is not
        # a gate on that - it is the gate on `get_tick_count`, i.e. on the
        # intervals a goal asks for, and the two are easy to confuse. A goal
        # with its own countdown (EatGrass) runs every tick so the countdown
        # counts down one per tick; a goal with none does not care either way.
        for prioritized in self.goals:
            if prioritized.running:
                prioritized.goal.tick(mob)

    def stop_all(self) -> None:
        """Stop everything and release every control.

        For the paths that end a mob's life mid-idea - a body removed by the
        void pass, an entity taken out of the world. Without it a goal's
        navigator target outlives the mob and the mob's `stop` never runs.
        """
        for prioritized in self.goals:
            if prioritized.running:
                prioritized.running = False
        for slot in range(len(self._by_control)):
            self._by_control[slot] = _NO_GOAL

    def running_goals(self) -> Tuple[PrioritizedGoal, ...]:
        """The goals running this tick. For stats and the check tool."""
        return tuple(p for p in self.goals if p.running)
