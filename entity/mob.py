"""
Mobs: entities that live in `Dimension.entities` and are spawned by the world.

Round one was three animals - chicken, cow, sheep - and they were *static*:
a spawned mob did not walk, did not fall, did not age. Round two is the AI
(`entity/ai/`), and a mob now walks, looks at you, bolts when hurt and floats
up out of a river. The mob class grew four things to carry it and none of them
is behaviour: `attributes` (the vanilla per-class numbers), `random` (the mob's
own RNG stream), `movement_input` (what the controls wrote this tick), and
`ai` (the goal selector itself). Health, hurt cooldown, status effects and
the death path live one level up, on `LivingEntity`, the same way they do in
Pumpkin; the mob class only re-exposes the half the AI cares about.

**`Mob.tick` is the AI's tick, and it is the only entry point.** `Dimension.tick`
already called `entity.tick()` for every non-player entity whose chunk is
entity-ticking, and `PHYSICS_PLAN.md` §4.6 reserved that call for exactly this.
Running the AI from anywhere else - a round over `Dimension.entities`, a
timer - would give mobs a second opinion about whether they are ticking this
tick, and the one authority on that question is the ticket model.

**The registry is still the point of the three classes.** They are empty
subclasses of `Animal`, and what they accomplish is appearing in
`MOB_CLASSES`. That set is what `BiomeSpawns` filters biome spawner rows
against, so a plains `creature` list of six rows becomes three, and a pig never
gets built. Had the spawner been allowed to fall back to a base `Mob` for
unimplemented types, the world would fill with pigs that cannot move, breed, or
be killed - a bug that reads as "spawning works" in every screenshot.
`tools/check_mob_spawning.py` asserts the set is exactly those three, and
`tools/check_mob_ai.py` asserts their AI is wired the same way.
"""

from __future__ import annotations

import uuid as uuid_module
from typing import Dict, Optional, Tuple, Type

from entity.ai import MobAI
from entity.ai.animal import (
    CHICKEN_PROFILE,
    COW_PROFILE,
    PIG_PROFILE,
    RABBIT_PROFILE,
    SHEEP_PROFILE,
    AnimalProfile,
    build_animal_ai,
    build_rabbit_ai,
)
from entity.ai.context import (
    CHICKEN_ATTRIBUTES,
    COW_ATTRIBUTES,
    PIG_ATTRIBUTES,
    RABBIT_ATTRIBUTES,
    SHEEP_ATTRIBUTES,
    AIContext,
    MobAttributes,
)
from entity.entity_type import EntityType
from entity.living_entity import LivingEntity
from entity.metadata import mob_spawn_blob
from entity.mob_category import MobCategory, mob_category
from entity.position import EntityPosition
from world.vanilla.random_source import MASK64, Xoroshiro


class Mob(LivingEntity):
    """An entity that lives in a world rather than in a socket.

    The base class here is deliberately abstract in practice: nothing builds a
    bare `Mob`. Either a type is in `MOB_CLASSES` - and the biome filter has
    already kept the spawner from asking for anything else - or it never
    reaches the world at all.
    """

    #: The vanilla attribute block for this mob class. `None` on the base
    #: means "use `MobAttributes`' defaults", which are a passive animal's:
    #: 0.25 `movement_speed`, 16 blocks of follow range. Concrete classes that
    #: differ - the cow is 0.2, the sheep 0.23 - name their own.
    ai_attributes: Optional[MobAttributes] = None

    #: The AI knobs for this mob class, or None for a mob with no brain.
    #: See `entity/ai/animal.py`.
    ai_profile: Optional[AnimalProfile] = None

    def __init__(self, entity_type: EntityType, x: float, y: float, z: float,
                 dimension: str = "minecraft:overworld",
                 yaw: float = 0.0, health: Optional[float] = None) -> None:
        super().__init__(
            entity_id=0,
            uuid=str(uuid_module.uuid4()),
            entity_type=entity_type,
            position=EntityPosition(x=x, y=y, z=z, yaw=yaw, pitch=0.0,
                                    on_ground=True),
            dimension=dimension,
            removed=False,
            # `health` flows through `LivingEntity`, which clamps it to
            # `[0, max_health]`. The `max_health` resolution is also on
            # `LivingEntity` (`entity_type.max_health` first, then the
            # `DEFAULT_MAX_HEALTH` fallback) so the mob's spawn-time health
            # bar matches the table its class was registered against.
            health=health,
        )

        #: Vanilla's attribute block. Read by the controls (`movement_speed`
        #: is the raw material of the walk *acceleration* - it enters the
        #: per-tick write twice, see `entity/ai/control.py` - `jump_power` the
        #: jump, the two look rates the
        #: head turn).
        self.attributes = type(self).ai_attributes or MobAttributes()

        #: This mob's own RNG stream. Vanilla gives every entity its own
        #: `RandomSource`; here it is seeded from the entity's uuid, which is
        #: already unique and already random, so two cows in the same chunk
        #: never roll the same wander direction on the same tick - the visible
        #: symptom of sharing a stream. Goals take `mob.random` rather than
        #: the module `random`, so a mob's behaviour is a function of its own
        #: history and nothing else.
        self.random = Xoroshiro.from_seed(uuid_module.uuid4().int & MASK64)

        #: What the AI's controls wrote this tick. `MoveControl` fills the
        #: first two (`strafe`, 0, `forward`), `JumpControl` the flag, and
        #: `MobAI.apply_input` turns both into one `set_velocity`. Read-only
        #: from outside the AI: nothing else should be writing a mob's
        #: movement input. `jumping` itself lives on `LivingEntity`, so the
        #: AI's one-tick intent flag has one home across the whole class
        #: hierarchy.
        self.movement_input: Tuple[float, float, float] = (0.0, 0.0, 0.0)

        #: The damage type that last hurt this mob, mirrored from
        #: `LivingEntity.last_damage_type` for the AI's panic check. Vanilla
        #: reads `lastDamageType` and asks whether it carries
        #: `#minecraft:panic_causes`; this is the same question with the tag
        #: folded in, because the only tag membership that matters is "does
        #: this make a passive mob run". `LivingEntity.damage` is what writes
        #: it - the AI never sets `panic_source` directly.
        self.panic_source: Optional[str] = None

        #: The goal selector, or None for a mob with no profile. Built here
        #: and not at spawn because goals are per-mob state (a wander target,
        #: a look timer) and the *context* they need is not available yet -
        #: `NaturalSpawner._place` binds that, on the same line it gives the
        #: mob a physics body.
        profile = type(self).ai_profile
        if profile is None:
            self.ai: Optional[MobAI] = None
        elif profile.name == "rabbit":
            # The rabbit is the first passive animal whose goal list diverges
            # from the shared shape - `RabbitHopGoal` lives here only. Future
            # divergence (sheep's `EatGrass`, goat's `Ram`) will get their own
            # branches alongside this one. Routing on the profile's name
            # rather than the class keeps the dispatch the data's problem.
            self.ai = build_rabbit_ai(self, profile)
        else:
            self.ai = build_animal_ai(self, profile)

    def damage(self, amount: float, source: Optional[str] = None,
               attacker_id: int = 0) -> bool:
        """Apply damage and mirror the result into `panic_source`.

        `LivingEntity.damage` does the actual work (cooldown check, absorption,
        death path); this override is a thin wrapper that, on a successful
        hit, copies the damage type into `panic_source` so `EscapeDangerGoal`
        can ask "did I panic?" without reaching into the LivingEntity layer.
        Runs before any subclass behaviour, and falls back to `False` if the
        base rejected the hit for any reason (cooldown, dead target).
        """
        landed = super().damage(amount, source, attacker_id)
        if landed and source is not None:
            self.panic_source = source
        elif not landed:
            # A rejected hit should not flip panic off either: the previous
            # cause, if any, is still valid.
            pass
        return landed

    def on_death(self) -> None:
        """A mob's death side effects.

        `Mob` itself has no kill-scoring or drops yet (`MOB_AI_PLAN.md`
        §3 lists what is still owed), so the only thing this round does on
        death is clear the panic - a dead mob does not need a reason to run.
        """
        self.panic_source = None

    @property
    def category(self) -> Optional[MobCategory]:
        """This mob's spawn category, from its entity type's `category` name."""
        if self.entity_type is None:
            return None
        return mob_category(self.entity_type.category)

    @property
    def despawn_distance(self) -> int:
        """Blocks from a player past which this mob would be removed.

        The category's, falling back to the vanilla default rather than to 0 -
        a mob with no known category should still despawn, not become
        permanent because its radius multiplied out as zero.
        """
        category = self.category
        return category.despawn_distance if category else 128

    def spawn_blob(self) -> bytes:
        """The metadata the client needs at spawn, on top of add_entity.

        Health only, and it is load-bearing: see `metadata.mob_spawn_blob`.
        """
        return mob_spawn_blob(self.health)

    def panic(self, damage_type: str) -> None:
        """Make this mob run away - the entry point a damage system would call.

        Not called from anywhere yet, deliberately: `Mob.health` exists and
        nothing writes it, so wiring this to "health dropped" would be a
        half-damage path that damages nothing. When combat lands, one call
        here is all `EscapeDangerGoal` needs.
        """
        self.panic_source = damage_type

    def clear_panic(self) -> None:
        """Forget what hurt the mob. Vanilla drops `lastDamageType` on death
        and after a while; here it is cleared explicitly, because a mob that
        panicked once and never stopped would sprint off the edge of the world
        and there is nothing out there to catch it."""
        self.panic_source = None

    def check_spawn_rules(self, dimension, block_pos: Tuple[int, int, int]) -> bool:
        """Whether this mob's own rules admit a spawn at this block.

        The instance form, for a mob that exists. The spawner asks the same
        question about a *position* before anything is built, so the real
        work is in `spawn_rules_ok`, which needs no instance.
        """
        return type(self).spawn_rules_ok(dimension, block_pos)

    @classmethod
    def spawn_rules_ok(cls, dimension, block_pos: Tuple[int, int, int]) -> bool:
        """Whether this mob *class* admits a spawn at this block.

        A classmethod rather than an instance method so the spawner can ask
        the question before it has a mob to ask it of. Building a throwaway
        `Animal` to evaluate one boolean would allocate a UUID and a
        position, and the answer depends only on the class and the block.
        """
        return True

    def tick(self) -> None:
        """One game tick: the base behaviour, then the AI.

        `Dimension.tick` calls this for every non-player entity whose chunk is
        entity-ticking, on the dimension's actor thread. The AI writes
        velocity here and `physics.tick()` moves the body a few lines later in
        the same actor tick, which is the order `PHYSICS_PLAN.md` §4.6 chose:
        physics after entities, so a mob that wants to walk this tick does.
        """
        super().tick()
        if self.ai is not None:
            self.ai.tick()

    def __str__(self) -> str:
        type_name = self.entity_type.name if self.entity_type else "untyped"
        return f"{type(self).__name__}({type_name}, id={self.entity_id})"


class Animal(Mob):
    """A passive mob: the only kind this round spawns.

    Vanilla's brightness rule is the whole of the spawn rule for a plain
    animal - no light source, no game rule, no dimension tag. An `Animal`
    subclass that grows a block-light or weather rule overrides
    `spawn_rules_ok`.
    """

    @classmethod
    def spawn_rules_ok(cls, dimension, block_pos: Tuple[int, int, int]) -> bool:
        from entity.spawn_placement import is_bright_enough

        return is_bright_enough(dimension, *block_pos)


class Chicken(Animal):
    """4 HP, 0.4 x 0.7. The short one - fits a one-block gap a cow cannot.

    Carries the 0.25 attribute row (with pig and skeleton) and bolts at 1.4 x
    that, which is why a chicken is the hardest of the three to corner: the
    panic multiplier is a third lower than the cow's and the body it has to
    clear is a third smaller.
    """

    ai_attributes = CHICKEN_ATTRIBUTES
    ai_profile = CHICKEN_PROFILE


class Cow(Animal):
    """10 HP, 0.9 x 1.4. The tall one, and the reason height is checked at all.

    The slowest attribute of the three - the 0.2 row it shares with goat and
    mooshroom - but the biggest panic multiplier: `EscapeDangerGoal(2.0)`
    against the chicken's 1.4 and the sheep's 1.25.
    """

    ai_attributes = COW_ATTRIBUTES
    ai_profile = COW_PROFILE


class Sheep(Animal):
    """8 HP, 0.9 x 1.3. Renders white: there is no wool metadata yet.

    The 0.23 attribute row (with zombie), between the chicken's 0.25 and the
    cow's 0.2 - which is, as far as the AI is concerned, the whole of its
    difference from either of them.
    """

    ai_attributes = SHEEP_ATTRIBUTES
    ai_profile = SHEEP_PROFILE


class Pig(Animal):
    """10 HP, 0.9 x 0.9. Square in plan view, the only one of the four.

    Carries the 0.25 attribute row (with chicken) but bolts at 1.5 rather than
    1.4, which is why a startled pig closes a gap on a startled chicken that
    same row does not close in the other direction. `WanderAroundGoal`'s
    steady state here is the chicken's ~0.138 blocks per tick; the panic
    multiplier is what widens it. Saddle, breeding and the carrot-on-a-stick
    follow are not goals this round ships - see `MOB_AI_PLAN.md` §3.
    """

    ai_attributes = PIG_ATTRIBUTES
    ai_profile = PIG_PROFILE


class Rabbit(Animal):
    """3 HP, 0.4 x 0.5. The smallest passive mob, and the fastest at walking.

    0.3 movement_speed is the rabbit's alone - no other passive animal shares
    the row, and the panic multiplier is 1.7 (the sheep is 1.25, the pig is
    1.5). The reason it does not outrun everything in vanilla is that the
    rabbit's *visible* motion is hop-shaped: the rabbit's legs are short, so
    walking looks like running, and running looks like a series of jumps.
    That shape is `RabbitHopGoal`'s job - see `entity/ai/goals.py` for how it
    takes the `JUMP` control away from `WanderAroundGoal` briefly. Rabbit
    breeds (four variants), eats carrots, and hides in holes; none of those
    are goals this round ships.
    """

    ai_attributes = RABBIT_ATTRIBUTES
    ai_profile = RABBIT_PROFILE


#: Entity type name -> class. This set IS the feature boundary: the biome
#: spawner rows are filtered against it, so a type absent here cannot spawn,
#: cannot be summoned, and cannot be reached by any other path either.
MOB_CLASSES: Dict[str, Type[Mob]] = {
    "minecraft:chicken": Chicken,
    "minecraft:cow": Cow,
    "minecraft:pig": Pig,
    "minecraft:rabbit": Rabbit,
    "minecraft:sheep": Sheep,
}


def mob_class_for(type_name: str) -> Optional[Type[Mob]]:
    """The class for a canonical entity type name, or None if not built.

    None means "this server cannot make that mob", and the spawner treats it
    as a reason to skip the row - it never substitutes a base class.
    """
    return MOB_CLASSES.get(type_name)


def spawnable_type_names() -> Tuple[str, ...]:
    """The types this server can spawn, sorted, for the biome filter."""
    return tuple(sorted(MOB_CLASSES))


def create_mob(entity_type: EntityType, x: float, y: float, z: float,
               dimension: str = "minecraft:overworld",
               yaw: float = 0.0) -> Optional[Mob]:
    """Build the mob for a type, or None if this server has no class for it.

    The None return is load-bearing at the spawner: an unimplemented type is
    skipped, not approximated. Returning a base `Mob` here would be the one
    thing that makes the world lie - it would spawn, and it would be a
    creature with no behaviour and no way to tell from the wire.
    """
    cls = MOB_CLASSES.get(entity_type.name)
    if cls is None:
        return None
    return cls(entity_type, x, y, z, dimension=dimension, yaw=yaw)


def check_spawn_rules(entity_type: EntityType, dimension,
                      block_pos: Tuple[int, int, int]) -> bool:
    """Whether a *type*'s own spawn rules admit this block.

    The spawner's entry point, and the reason `spawn_rules_ok` is a class
    method: the rules have to be consulted before there is a mob, because a
    rejected position must not have allocated an entity id and a UUID.
    """
    cls = MOB_CLASSES.get(entity_type.name)
    if cls is None:
        return False
    return bool(cls.spawn_rules_ok(dimension, block_pos))


def mob_ai_context(dimension=None, physics=None, tracker=None) -> AIContext:
    """Build the world handle the spawner hands to every mob it places.

    A function rather than a module constant because the collaborators are the
    dimension actor's, and the actor is built after this module is imported. One
    context is built per spawner and shared by every mob in the dimension: it
    holds references, not per-mob state.
    """
    return AIContext(dimension, physics, tracker)


__all__ = [
    "MOB_CLASSES",
    "Animal",
    "Chicken",
    "Cow",
    "Mob",
    "Pig",
    "Rabbit",
    "Sheep",
    "check_spawn_rules",
    "create_mob",
    "mob_ai_context",
    "mob_class_for",
    "spawnable_type_names",
]
