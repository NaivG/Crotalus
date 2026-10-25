"""
Entity module - the vanilla entity layer.

Everything that exists in the world and is not a block is an `Entity`: the
players, the mobs, the dropped items, the boats. The layer is built on two
tables this server already loads (`assets/entities.json` for the 157 entity
types, `assets/entity_tracking.json` for their client-tracking distances) and
on one global id counter, all shared with the network layer through plain
integers: the runtime entity id *is* the id the client sees in every packet.

The class hierarchy mirrors Pumpkin's:

    Entity           <- position, runtime id, world, removed flag
      |
      LivingEntity   <- health, hurt cooldown, status effects, absorption
        |  \
        |   Player   <- name, food, gamemode, persisted profile
        Mob          <- AI: goals, controls, attributes
          |
          Animal     <- brightness-rule spawn rules
            ...

What lives where:

* `ids`          - the global entity-id allocator (one monotonic counter)
* `entity_type`  - the typed view over `minecraft:entity_type`
* `position`     - `EntityPosition`, the position/rotation state
* `entity`       - the `Entity` base class, which `LivingEntity` extends
* `living_entity`- health, hurt cooldown, status effects, death and the
                   small set of interaction entry points
* `metadata`     - the entity-metadata blob writer for Set Entity Data
* `entity_tracker` - who sees whom, and the delta codec that moves entities
* `mob_category` / `biome_spawns` / `spawn_placement` / `spawner` /
  `mob`          - natural mob spawning (first round: chicken, cow, sheep) --
                   see `MOB_SPAWNING_PLAN.md` for the vanilla departures and
                   the known gaps
* `ai`           - the mob brain (`MOB_AI_PLAN.md`): a goal selector over four
                   controls, vanilla's passive-animal goal list for the three
                   animals, a navigator that follows an A* path
                   (`PATHFINDER_PLAN.md`, `ai/pathfinding/`), and the
                   per-class profiles that make cow, sheep and chicken three
                   rows of one table

What this package deliberately does NOT contain: entity NBT persistence,
breeding, tempting, item entities. Mobs exist, spawn on vanilla's natural
cadence, and now walk - but they are not written to `entities/*.mca` (a
player who leaves and returns finds them re-spawned). Those layers land after
that is a consciously accepted gap rather than an oversight.
"""

from entity.ids import EntityIdAllocator, get_entity_id_allocator, allocate_entity_id
from entity.entity_type import (
    EntityType,
    entity_type_by_id,
    entity_type_by_name,
    entity_type_count,
    load_entity_types,
)
from entity.position import EntityPosition
from entity.entity import Entity, player_entity_type
from entity.living_entity import (
    DEFAULT_HURT_COOLDOWN,
    DEFAULT_KNOCKBACK_RESISTANCE,
    DEFAULT_MAX_HEALTH,
    LivingEntity,
)

__all__ = [
    "EntityIdAllocator",
    "get_entity_id_allocator",
    "allocate_entity_id",
    "EntityType",
    "entity_type_by_id",
    "entity_type_by_name",
    "entity_type_count",
    "load_entity_types",
    "EntityPosition",
    "Entity",
    "LivingEntity",
    "DEFAULT_HURT_COOLDOWN",
    "DEFAULT_KNOCKBACK_RESISTANCE",
    "DEFAULT_MAX_HEALTH",
    "player_entity_type",
]
