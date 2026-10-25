"""
Where a mob may stand: the block-level half of the spawner.

Vanilla validates a spawn position with four independent questions, and this
module is those four questions:

* is the block below a *spawn floor* - something with a sturdy top face?
* is the block the mob occupies, and the one above it, *empty*?
* is there room for the mob's actual bounding box?
* is it bright enough for this mob's rules?

Every one of them reads through `BlockFacts`, which answers from
`assets/blocks.json`, so "is this a full cube" is the report's own definition
rather than a list someone typed out.

**Unloaded chunks reject, they do not read as air.** `Dimension.get_block`
returns 0 for a coordinate whose chunk is not in memory, and air passes every
predicate here. A spawner that walked its +-5 block search into a neighbour it
had not loaded would therefore place mobs in the void. Every function below
goes through `chunk_at` and answers False when there is no chunk, so an
unloaded column is a rejected spawn rather than an empty one.

**Lighting reads the engine's own numbers.** `luminance_at` is
`max(sky light, block light)` out of the chunk's light arrays -- vanilla's
`getRawBrightness(pos, 0)` -- for a chunk the flood-fill engine has lit. The
earlier version answered 15 above the surface and 0 below, which reached
vanilla's first three conclusions (surface spawns, cave spawns do not, the
Nether and the End spawn no passive mobs) and missed the fourth: a deep cave
with a shaft to the sky. It also could not express block light, so a
torch-lit cave read as dark and a mob would not avoid a lit floor.

A chunk whose light was never populated (`Chunk.light_populated` False: built
by hand, or read from a region file that predates light persistence) falls
back to exactly the old answer, so the substitute is still in the code as a
fallback rather than deleted. It compares against
`top_motion_blocking_no_leaves`, not `Chunk.get_height_at`: the heightmap
counts tall grass, flowers and torches, and a cow standing in grass would be
judged underground. See `luminance_at`.

The remaining gap is a chunk that is *loaded but not yet lit* -- `luminance_at`
answers the fallback for it, which is the safe direction for a spawn rule
(refusing a spawn) and is resolved within a tick or two by the light pass.
"""

from __future__ import annotations

from typing import Optional, Tuple

from entity.entity_type import EntityType
from world.vanilla.features import block_facts

#: `heightmap` value in `assets/entities.json`'s `spawn_restriction`. All
#: three animals this round use it, and it is the heightmap that ignores
#: leaves, so a cow will not stand on an oak canopy.
HEIGHTMAP_NO_LEAVES = "motion_blocking_no_leaves"

#: Vanilla's `MAX_LIGHT_LEVEL` and the spawn floor's brightness ceiling.
MAX_LIGHT = 15
DARK_FLOOR_CEILING = 14

#: Vanilla's animal brightness rule: `getRawBrightness(pos, 0) > 8`, i.e. at
#: least 9. A full-lit (15) surface clears it; the substitute's 0 does not.
ANIMAL_MIN_LIGHT = 8

#: Blocks that hurt an entity that is not fire-immune, and the plants that
#: hurt everything. `is_burning_block` is the fire half, matched by block name
#: because there is no "is this block on fire" predicate that does not need a
#: light and fluid engine to evaluate.
BURNING_BLOCKS = frozenset({
    "minecraft:lava",
    "minecraft:magma_block",
    "minecraft:lava_cauldron",
    "minecraft:campfire",
    "minecraft:soul_campfire",
})
ALWAYS_DANGEROUS_BLOCKS = frozenset({
    "minecraft:wither_rose",
    "minecraft:sweet_berry_bush",
    "minecraft:cactus",
    "minecraft:powder_snow",
})


def chunk_at(dimension, x: int, z: int):
    """The loaded chunk holding a block column, or None.

    None means "no chunk", and every caller treats that as a rejection. This
    is the single place that knowledge lives, which is why the functions below
    take a `dimension` and coordinates instead of a chunk and local
    coordinates: the world-to-chunk conversion is exactly where the
    unloaded-column trap is.
    """
    return dimension.get_chunk(x >> 4, z >> 4)


def is_loaded(dimension, x: int, z: int) -> bool:
    """Whether a column's chunk is in memory. The spawner's residency test."""
    return chunk_at(dimension, x, z) is not None


def state_at(dimension, x: int, y: int, z: int) -> Optional[int]:
    """The block state id at a world position, or None if it is not loaded.

    The None is not decoration: `Chunk.get_block` answers AIR for coordinates
    outside the chunk, so a caller that treats a missing answer as air gets a
    world made of empty space.
    """
    chunk = chunk_at(dimension, x, z)
    if chunk is None:
        return None
    return chunk.get_block(x & 15, y, z & 15)


# ----- heightmap -----

def top_motion_blocking_no_leaves(dimension, x: int, z: int) -> Optional[int]:
    """The Y of the topmost block that blocks motion and is not a leaf.

    Vanilla reads this off a per-chunk `MOTION_BLOCKING_NO_LEAVES` heightmap.
    This server has no such table - `Chunk` keeps one `heightmap`, the
    topmost non-air block, which is the *WORLD_SURFACE* kind and counts
    leaves - so the answer is computed per column by scanning down from that
    height until the `motion_blocking_no_leaves` mask matches.

    Per-column rather than per-chunk on purpose: a spawn round asks a few
    dozen times for a few dozen chunks, so a precomputed table would cost more
    to build and store than the scan costs to run, and would need invalidating
    on every block change. When a real heightmap lands, this becomes one
    array index and the scan goes away; until then the scan is bounded by the
    column's own height and happens only where a spawn is already plausible.

    None means the column is not loaded, or is empty to the bottom of the
    world.
    """
    chunk = chunk_at(dimension, x, z)
    if chunk is None:
        return None
    mask = block_facts().heightmap_mask(HEIGHTMAP_NO_LEAVES)
    lx, lz = x & 15, z & 15
    for y in range(chunk.get_height_at(lx, lz), chunk.min_y - 1, -1):
        if mask[chunk.get_block(lx, y, lz)]:
            return y
    return None


# ----- lighting -----

def _light_at(dimension, x: int, y: int, z: int, sky: bool) -> int:
    """One packed light nibble at a world coordinate. 0 when out of range.

    The packing is `world/light_layout.py`'s: two cells per byte, cell
    `x + z*16 + y_local*256` at bit offset `(y_local % 16) * 256`, even cells in
    the low nibble. Spelled out here rather than calling `unpack_light` because
    unpacking 4096 cells to read one of them is the wrong shape for a query the
    spawner makes dozens of times a round.

    A chunk whose light was never populated (`light_populated` False: hand-built,
    or a region file from before light persistence) answers from the height
    approximation instead -- see `luminance_at`.
    """
    chunk = chunk_at(dimension, x, z)
    if chunk is None or not (chunk.min_y <= y < chunk.min_y + chunk.height):
        return 0
    if not chunk.light_populated:
        return -1  # "no real light here"; the caller has a fallback
    offset = y - chunk.min_y
    section = chunk.sections[offset // 16]
    cell = (x & 15) + (z & 15) * 16 + (offset % 16) * 256
    packed = section.sky_light if sky else section.block_light
    byte = int(packed[cell >> 1])
    return (byte >> 4) if (cell & 1) else (byte & 0x0F)


def sky_light_at(dimension, x: int, y: int, z: int) -> int:
    """Sky light at a block, or 0 where there is none to have.

    A dimension without a sky (`has_skylight` False -- the Nether and the End)
    answers 0 rather than reading the array, because that array is never
    written there: the light engine skips its sky passes entirely, and the
    `ChunkSection` default of 15 would report an open roof over a cave.
    """
    if not getattr(getattr(dimension, "settings", None), "has_skylight", True):
        return 0
    return max(0, _light_at(dimension, x, y, z, sky=True))


def block_light_at(dimension, x: int, y: int, z: int) -> int:
    """Block light at a block, 0 where nothing was ever lit."""
    return max(0, _light_at(dimension, x, y, z, sky=False))


def luminance_at(dimension, x: int, y: int, z: int) -> int:
    """Light level at a block: `max(sky light, block light)`.

    This is vanilla's `getRawBrightness(pos, 0)`, and it is now the engine's
    own numbers rather than a substitute. The earlier version returned 15 above
    the surface and 0 below, which got three of vanilla's four conclusions
    (surface spawns, cave spawns do not, the Nether and the End spawn no
    passive mobs) and missed the fourth: a deep cave with a shaft to the sky.
    It also could not express block light at all, so a torch-lit cave read as
    dark and a mob would not avoid a lit floor.

    The fallback for a chunk nobody has lit keeps the old answer exactly --
    15 above `top_motion_blocking_no_leaves`, 0 below. That is deliberately the
    same reference surface as before rather than `Chunk.get_height_at`: the
    heightmap counts tall grass, flowers and torches, and a cow standing in
    grass would be judged underground. See the module docstring.

    An unloaded chunk still refuses rather than answering bright; `chunk_at`
    is the gate, and it is the reason `is_loaded` is checked before this in
    every caller.
    """
    sky = _light_at(dimension, x, y, z, sky=True)
    block = _light_at(dimension, x, y, z, sky=False)

    if sky < 0 or block < 0:
        # Nobody lit this chunk. Answer the old way rather than mixing real
        # block light with a guessed sky, which would be a third behaviour
        # nobody has a rule for.
        top = top_motion_blocking_no_leaves(dimension, x, z)
        if top is None:
            return 0
        return MAX_LIGHT if y > top else 0

    if not getattr(getattr(dimension, "settings", None), "has_skylight", True):
        sky = 0
    return max(sky, block)


def is_bright_enough(dimension, x: int, y: int, z: int) -> bool:
    """Vanilla's animal spawn rule: brighter than 8, or not bright at all."""
    return luminance_at(dimension, x, y, z) > ANIMAL_MIN_LIGHT


# ----- the four validation questions -----

def is_burning_block(state_id: int) -> bool:
    """Fire, lava, magma, a cauldron or a campfire.

    `minecraft:fire` is matched through the block tag rather than by name
    because the tag is what vanilla tests and it covers every fire variant
    the report knows about; the rest are named individually because vanilla
    names them individually.
    """
    facts = block_facts()
    return (
        facts.in_tag(state_id, "minecraft:fire")
        or facts.block_of(state_id) in BURNING_BLOCKS
    )


def is_block_dangerous(state_id: int, fire_immune: bool) -> bool:
    """Whether a block may not be occupied by this mob.

    Two halves: anything that burns, which a fire-immune mob ignores, and
    four plants that hurt everything. Vanilla's `isBlockDangerous`, and the
    reason a cow will not appear standing in a sweet berry bush.
    """
    if not fire_immune and is_burning_block(state_id):
        return True
    return block_facts().block_of(state_id) in ALWAYS_DANGEROUS_BLOCKS


def is_valid_empty_spawn_block(state_id: int, fire_immune: bool) -> bool:
    """Whether a mob may *occupy* this block.

    Not a full cube, not a liquid, not one of the four blocks tagged
    `prevent_mob_spawning_inside` (rails, in vanilla), and not dangerous.
    """
    facts = block_facts()
    if facts.is_full_cube(state_id):
        return False
    if facts.is_liquid(state_id):
        return False
    if facts.in_tag(state_id, "minecraft:prevent_mob_spawning_inside"):
        return False
    return not is_block_dangerous(state_id, fire_immune)


#: `state id -> luminance`, filled on demand. See `_block_luminance`.
_LUMINANCE_CACHE: dict = {}


def _block_luminance(state_id: int) -> int:
    """How much light a block state emits, cached per state.

    `BlockFacts` carries the shape and tag tables but not this column, and
    `registry.blocks.light_emitters()` is the sparse form an engine wants
    ("which states emit") rather than the random-access one a predicate wants
    ("does state 1234 emit"). The cache is a dict instead of a 65536-entry
    table because the spawner asks about a handful of states per round, and it
    is built lazily so importing this module does not require the registry.

    An unknown state emits nothing, which is the safe answer: a state id
    outside the registry is a bug elsewhere, and saying "it glows" would make
    that bug look like a spawn-rule change.
    """
    cached = _LUMINANCE_CACHE.get(state_id)
    if cached is not None:
        return cached
    try:
        from registry import get_game_registry

        state = get_game_registry().blocks.state(state_id)
        value = int(state.luminance) if state is not None else 0
    except Exception:  # pragma: no cover - only a broken install or no registry
        value = 0
    _LUMINANCE_CACHE[state_id] = value
    return value


def is_valid_spawn_floor(state_id: int) -> bool:
    """Whether a mob may *stand on* this block.

    Vanilla's default floor predicate is "the top face is sturdy, and the
    block does not emit light at or above `DARK_FLOOR_CEILING`". All three
    animals use the default; the other five predicates (`Always`, `Never`,
    `OcelotOrParrot`, `PolarBear`, `FireImmune`) belong to entity types that
    are not in `MOB_CLASSES`, and the spawner cannot reach them until one is.

    The light half reads the **block's own luminance**, not the light level at
    the position. That distinction is the reason this predicate can stay a
    function of `state_id` while `luminance_at` needs a dimension: the rule is
    "do not spawn on a magma block or a lit campfire", which the block itself
    answers and which does not change when a torch is placed nearby. The
    earlier version always passed and its comment claimed the rule was about
    the position. Pumpkin's port of the same predicate
    (`natural_spawner.rs:1101`: `state.is_side_solid(Up) && state.luminance < 14`)
    has the same two terms, and it is where the constant comes from.
    """
    if not block_facts().face_sturdy(state_id, (0, 1, 0)):
        return False
    return _block_luminance(state_id) < DARK_FLOOR_CEILING


def is_spawn_position_ok(dimension, x: int, y: int, z: int,
                         entity_type: EntityType) -> bool:
    """The `ON_GROUND` placement check: a floor below, two empty blocks above.

    All three animals are `location: ON_GROUND` in
    `assets/entities.json`; `IN_WATER`, `IN_LAVA` and `UNRESTRICTED` belong
    to the water categories this round does not spawn, so an unexpected
    location is a rejection rather than a guess at which other rule to
    apply.

    Vanilla also tests the position against the world border here. This
    server has no world border, so that test is absent - a declared gap, not
    a rule that happens to pass.
    """
    if str(entity_type.spawn_restriction.get("location")) != "ON_GROUND":
        return False
    if not is_loaded(dimension, x, z):
        return False

    fire_immune = bool(entity_type.fire_immune)

    below = state_at(dimension, x, y - 1, z)
    if below is None or not is_valid_spawn_floor(below):
        return False

    here = state_at(dimension, x, y, z)
    if here is None or not is_valid_empty_spawn_block(here, fire_immune):
        return False

    above = state_at(dimension, x, y + 1, z)
    if above is None or not is_valid_empty_spawn_block(above, fire_immune):
        return False

    return True


def spawn_bounding_box(entity_type: EntityType, x: float, y: float,
                       z: float) -> Tuple[float, float, float,
                                          float, float, float]:
    """The box a mob of this type occupies when spawned at a block centre.

    `(x0, y0, z0, x1, y1, z1)`, from the type's own `width`/`height` in
    `assets/entities.json`, centred on the block: a 0.9-wide cow spans
    `x+0.05 .. x+0.95`, which floors to the one block it stands on, and
    `y .. y+1.4`, which covers two. A 0.7-tall chicken covers one. That
    difference is the whole point of the check - a cow does not fit where a
    chicken does.

    The type's spawn scale is not applied: `assets/entities.json` does not
    record it, and it is 1.0 for every animal.
    """
    half_width = entity_type.width / 2.0
    return (x - half_width, y, z - half_width,
            x + half_width, y + entity_type.height, z + half_width)


def is_space_empty(dimension, entity_type: EntityType, x: float, y: float,
                   z: float) -> bool:
    """Whether the mob's own box fits, block for block.

    Vanilla walks the bounding box and asks each block whether it collides.
    `is_valid_spawn_block` only ever looked at the mob's own block and the
    one above it, so without this a cow would happily stand inside a fence
    rail at head height with clear air at its feet. The block span here is
    `floor(min) .. floor(max)` inclusive on every axis, which is what makes a
    1.4-tall mob reach into the block above it.

    A block is in the way when it is a full cube or a liquid. Collision
    shapes narrower than a full block - slabs, stairs, fences - do not count,
    matching vanilla's spawn box test rather than its movement collision.
    """
    x0, y0, z0, x1, y1, z1 = spawn_bounding_box(entity_type, x, y, z)
    facts = block_facts()
    for by in range(int(y0 // 1), int(y1 // 1) + 1):
        for bx in range(int(x0 // 1), int(x1 // 1) + 1):
            for bz in range(int(z0 // 1), int(z1 // 1) + 1):
                if not is_loaded(dimension, bx, bz):
                    return False
                state = dimension.get_block(bx, by, bz)
                if facts.is_full_cube(state) or facts.is_liquid(state):
                    return False
    return True


__all__ = [
    "ALWAYS_DANGEROUS_BLOCKS",
    "ANIMAL_MIN_LIGHT",
    "BURNING_BLOCKS",
    "DARK_FLOOR_CEILING",
    "HEIGHTMAP_NO_LEAVES",
    "MAX_LIGHT",
    "block_light_at",
    "chunk_at",
    "is_block_dangerous",
    "is_bright_enough",
    "is_burning_block",
    "is_loaded",
    "is_space_empty",
    "is_spawn_position_ok",
    "is_valid_empty_spawn_block",
    "is_valid_spawn_floor",
    "luminance_at",
    "sky_light_at",
    "spawn_bounding_box",
    "state_at",
    "top_motion_blocking_no_leaves",
]
