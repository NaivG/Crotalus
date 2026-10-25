"""Mob spawning: the data it reads, the positions it accepts, the caps it obeys.

Three layers, checked bottom-up because a bug in any of them looks the same
on a client - "no animals" - and the fix is in a different file each time.

1. **The biome tables.** `assets/biome.json` lists a plains `creature` row
   per mob type with min/max counts and a weight; `BiomeSpawns` keeps only
   the rows whose type this server can build and re-normalises the weights
   over what is left. Also the spawn-cost table, and the assertion that the
   filtered list is what the weights are computed from.
2. **Placement.** Floor sturdy, space empty, bright enough, and the height
   discrimination that actually matters: a 1.4-tall cow does not fit in a
   one-block pocket where a 0.7-tall chicken does. Plus the unloaded-chunk
   rule, which is the one that decides whether a mob spawns in the void.
3. **The spawner.** The cap arithmetic, both distance guards, the same-type
   spacing rule, the weighted draw's distribution, `limit_per_chunk`, and a
   real round over real terrain followed by a real unload.

Run: py -3.14t tools/check_mob_spawning.py
"""

from __future__ import annotations

import math
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

failures: list[str] = []


def check(label: str, condition: bool, detail: str = "") -> None:
    if condition:
        print(f"  OK   {label}" + (f"  ({detail})" if detail else ""))
    else:
        failures.append(label)
        print(f"  FAIL {label}  ({detail})")


SEED = 0
#: A land chunk on seed 0, chosen because (0,0) is ocean. Chunk (0,0) top
#: block is water at y=63, which is a correct rejection and a useless test.
LAND_CHUNK = (16, -3)


# ---------------------------------------------------------------------------
# 1. The biome tables
# ---------------------------------------------------------------------------

def check_biome_tables() -> None:
    from entity.biome_spawns import biome_spawns
    from entity.mob import MOB_CLASSES
    from world.vanilla.biome import biome_table

    print("1. biome spawner rows and spawn costs")

    table = biome_table()
    plains = table.spawners_for("minecraft:plains", "creature")
    by_name = {row.type_name: row for row in plains}
    check("plains has six creature rows", len(plains) == 6, str(len(plains)))
    check("sheep weight 12, min 4 max 4",
          (by_name["minecraft:sheep"].weight,
           by_name["minecraft:sheep"].min_count,
           by_name["minecraft:sheep"].max_count) == (12, 4, 4),
          str(by_name["minecraft:sheep"]))
    check("chicken weight 10, min 4 max 4",
          (by_name["minecraft:chicken"].weight,
           by_name["minecraft:chicken"].min_count,
           by_name["minecraft:chicken"].max_count) == (10, 4, 4),
          str(by_name["minecraft:chicken"]))
    check("cow weight 8, min 2 max 4",
          (by_name["minecraft:cow"].weight,
           by_name["minecraft:cow"].max_count) == (8, 4),
          str(by_name["minecraft:cow"]))
    check("pig weight 10, min 4 max 4",
          (by_name["minecraft:pig"].weight,
           by_name["minecraft:pig"].min_count,
           by_name["minecraft:pig"].max_count) == (10, 4, 4),
          str(by_name["minecraft:pig"]))

    spawns = biome_spawns(tuple(MOB_CLASSES))
    rows = spawns.entries("minecraft:plains", "creature")
    names = {row.type_name for row in rows}
    check("every plains-surviving type is implemented",
          names <= set(MOB_CLASSES), str(sorted(names)))
    check("rabbit is implemented but plains doesn't list it",
          "minecraft:rabbit" in set(MOB_CLASSES)
          and "minecraft:rabbit" not in names,
          "rabbit habitat is desert / grove / snowy, not plains")
    check("horse and donkey are filtered away (still unimplemented)",
          not (names & {"minecraft:horse", "minecraft:donkey"}),
          str(sorted(names)))
    check("surviving weights are vanilla's 12/10/10/8 (sheep/pig/chicken/cow)",
          sorted((r.type_name, r.weight) for r in rows) == [
              ("minecraft:chicken", 10), ("minecraft:cow", 8),
              ("minecraft:pig", 10), ("minecraft:sheep", 12)],
          str(sorted((r.type_name, r.weight) for r in rows)))

    # Boundaries of the weighted draw. Total 40: sheep 0..12, pig 12..22,
    # chicken 22..32, cow 32..40. Pig landed between sheep and chicken -
    # 0.55 is the unit value where the draw flips from pig to chicken. A
    # uniform draw (Pumpkin's) would put the edge at 0.5, so a probe at
    # 0.55 separates the two implementations.
    total = 40.0
    check("unit 0.0 draws sheep",
          spawns.draw("minecraft:plains", "creature", 0.0).type_name
          == "minecraft:sheep")
    check("unit just under 0.3 draws sheep",
          spawns.draw("minecraft:plains", "creature", 0.3 - 1e-9).type_name
          == "minecraft:sheep")
    check("unit 0.3 draws pig",
          spawns.draw("minecraft:plains", "creature", 0.3).type_name
          == "minecraft:pig")
    check("unit just under 0.55 draws pig",
          spawns.draw("minecraft:plains", "creature", 0.55 - 1e-9
                      ).type_name == "minecraft:pig")
    check("unit 0.55 draws chicken",
          spawns.draw("minecraft:plains", "creature", 0.55).type_name
          == "minecraft:chicken")
    check("unit just under 0.8 draws chicken",
          spawns.draw("minecraft:plains", "creature", 32 / total - 1e-9
                      ).type_name == "minecraft:chicken")
    check("unit 0.8 draws cow",
          spawns.draw("minecraft:plains", "creature", 32 / total).type_name
          == "minecraft:cow")
    check("unit 0.999 draws cow",
          spawns.draw("minecraft:plains", "creature", 0.999).type_name
          == "minecraft:cow")
    check("a biome with no such category draws nothing",
          spawns.draw("minecraft:plains", "monster", 0.5) is None,
          "the filter left no monster rows")

    # Spawn costs. Only two biomes record any in this report, and neither
    # holds a type this server builds, so the energy test is unconditionally
    # satisfied for all three of ours - which is vanilla's own default.
    ghast = table.spawn_cost_of("minecraft:soul_sand_valley", "minecraft:ghast")
    check("soul_sand_valley ghast costs (0.15, 0.7)",
          (ghast.energy_budget, ghast.charge) == (0.15, 0.7), str(ghast))
    zero = table.spawn_cost_of("minecraft:plains", "minecraft:cow")
    check("a biome with no spawn_costs answers zero",
          (zero.energy_budget, zero.charge) == (0.0, 0.0), str(zero))
    stats = spawns.stats()
    check("the filter is a real reduction",
          stats["rows_before"] > stats["rows_after_filter"] > 0,
          f"{stats['rows_before']} -> {stats['rows_after_filter']}")
    check("only creature has surviving rows",
          stats["categories"] == ["creature"], str(stats["categories"]))
    print()


# ---------------------------------------------------------------------------
# 2. Placement
# ---------------------------------------------------------------------------

def make_dimension(seed: int = SEED):
    from world.dimension import Dimension, DimensionType
    from world.generator import GeneratorSettings, create_generator

    # The legacy backend, for the same reason every other lifecycle check
    # uses it: the vanilla backend costs about a second per chunk and these
    # checks load dozens. Nothing here depends on which generator produced
    # the terrain - only that it is real ground.
    return Dimension(
        DimensionType.OVERWORLD,
        world_seed=seed,
        generator=create_generator(GeneratorSettings(seed=seed, preset="overworld"),
                                   backend="legacy"),
    )


def install_chunk(dimension, cx: int, cz: int):
    """Generate one chunk straight into the manager, bypassing the lifecycle."""
    from world.chunk import Chunk, ChunkPos

    settings = dimension.settings
    chunk = Chunk(ChunkPos(cx, cz), min_y=settings.min_y, height=settings.height)
    dimension.generator.generate_chunk(chunk)
    chunk.is_generated = True
    dimension.chunk_manager.set_chunk(chunk)
    return chunk


def check_placement() -> None:
    from entity.entity_type import entity_type_by_name
    from entity.spawn_placement import (
        is_bright_enough, is_space_empty, is_spawn_position_ok,
        is_valid_empty_spawn_block, is_valid_spawn_floor, state_at,
        top_motion_blocking_no_leaves,
    )
    from world.vanilla.features import block_facts
    from world.block_states import state_id

    print("2. placement rules")

    dimension = make_dimension()
    cx, cz = LAND_CHUNK
    install_chunk(dimension, cx, cz)
    facts = block_facts()
    W = lambda lx, lz: (cx * 16 + lx, cz * 16 + lz)  # noqa: E731

    def floor_at(x, z):
        top = top_motion_blocking_no_leaves(dimension, x, z)
        return (state_at(dimension, x, top, z) if top is not None else None)

    # Find a real grass column, and count how many columns in this chunk can
    # host a spawn at all. Zero would mean the fixture is wrong, not the code.
    grass = None
    good = 0
    for lx in range(16):
        for lz in range(16):
            x, z = W(lx, lz)
            top = top_motion_blocking_no_leaves(dimension, x, z)
            if top is None or top + 1 >= dimension.settings.height - 1:
                continue
            if facts.block_of(state_at(dimension, x, top, z)) != "minecraft:grass_block":
                continue
            good += 1
            if grass is None:
                grass = (x, top + 1, z)
    check("the fixture chunk has spawnable ground", good > 0, f"{good} columns")
    check("a grass column is found", grass is not None, str(grass))

    if grass is not None:
        x, y, z = grass
        check("grass is a valid spawn floor",
              is_valid_spawn_floor(state_at(dimension, x, y - 1, z)))
        check("the two blocks above it are valid spawn blocks",
              is_valid_empty_spawn_block(state_at(dimension, x, y, z), False)
              and is_valid_empty_spawn_block(state_at(dimension, x, y + 1, z), False))
        check("a sky-exposed column is bright enough",
              is_bright_enough(dimension, x, y, z))
        check("a column 20 blocks down is not",
              not is_bright_enough(dimension, x, y - 20, z))
        for name in ("minecraft:chicken", "minecraft:cow", "minecraft:pig",
                     "minecraft:rabbit", "minecraft:sheep"):
            etype = entity_type_by_name(name)
            check(f"{etype.name} may spawn there",
                  is_spawn_position_ok(dimension, x, y, z, etype))

    # Liquids and fire.
    water = state_id("minecraft:water")
    lava = state_id("minecraft:lava")
    rails = state_id("minecraft:rail")
    rose = state_id("minecraft:wither_rose")
    check("water is not a valid spawn block",
          not is_valid_empty_spawn_block(water, False))
    check("lava is not a valid spawn block",
          not is_valid_empty_spawn_block(lava, False))
    check("a rail is not a valid spawn block "
          "(prevent_mob_spawning_inside)",
          not is_valid_empty_spawn_block(rails, False))
    check("a wither rose is dangerous, so not a valid spawn block",
          not is_valid_empty_spawn_block(rose, False))
    check("lava is still refused for a fire-immune mob",
          not is_valid_empty_spawn_block(lava, True))

    # The height discrimination. A pocket with a lid at y+1: the chicken
    # (0.7 tall) fits under it, the cow (1.4) does not. This is the check
    # that makes bounding boxes worth having over "is my own block empty".
    if grass is not None:
        x, y, z = grass
        dimension.set_block(x, y + 1, z, state_id("minecraft:oak_planks"))
        chicken = entity_type_by_name("minecraft:chicken")
        cow = entity_type_by_name("minecraft:cow")
        check("with a lid overhead the chicken's box still fits",
              is_space_empty(dimension, chicken, x + 0.5, float(y), z + 0.5))
        check("with the same lid the cow's box does not",
              not is_space_empty(dimension, cow, x + 0.5, float(y), z + 0.5))
        check("and the placement check refuses the cow at that spot too",
              not is_spawn_position_ok(dimension, x, y, z, cow))
        dimension.set_block(x, y + 1, z, 0)

        # A mob standing *in* tall grass. The world-surface heightmap counts
        # it, the no-leaves one does not; using the wrong one judges the cow
        # to be underground and refuses every animal on a flowery meadow.
        dimension.set_block(x, y, z, state_id("minecraft:tall_grass"))
        check("WORLD_SURFACE would call the grass block the top",
              dimension.get_chunk(cx, cz).get_height_at(x & 15, z & 15) == y,
              "the heightmap the wrong implementation uses")
        check("MOTION_BLOCKING_NO_LEAVES does not",
              top_motion_blocking_no_leaves(dimension, x, z) == y - 1)
        check("so a cow standing in it is still bright enough",
              is_bright_enough(dimension, x, y, z))
        check("and may still spawn there",
              is_spawn_position_ok(dimension, x, y, z, cow))
        dimension.set_block(x, y, z, 0)

    # The unloaded-chunk rule. Every predicate must reject rather than read
    # the 0 that Dimension.get_block returns for a coordinate with no chunk.
    far = (LAND_CHUNK[0] + 900, LAND_CHUNK[1])
    check("a column in an unloaded chunk is not loaded",
          state_at(dimension, far[0] * 16, 64, far[1] * 16) is None)
    check("and the placement check refuses it",
          not is_spawn_position_ok(dimension, far[0] * 16, 64, far[1] * 16,
                                   entity_type_by_name("minecraft:cow")))
    check("and the bounding box check refuses it",
          not is_space_empty(dimension, entity_type_by_name("minecraft:cow"),
                             far[0] * 16 + 0.5, 64.0, far[1] * 16 + 0.5))
    check("and the height scan says nothing about it",
          top_motion_blocking_no_leaves(dimension, far[0] * 16, far[1] * 16)
          is None)
    print()


# ---------------------------------------------------------------------------
# 3. The spawner
# ---------------------------------------------------------------------------

def check_caps() -> None:
    from entity.mob_category import CATEGORY_LIST, CREATURE, MISC, UNLIMITED
    from entity.spawner import MAGIC_NUMBER, SpawnState
    from world.vanilla.biome import SPAWN_CATEGORIES

    print("3. population caps")

    check("MAGIC_NUMBER is 289 = 17*17", MAGIC_NUMBER == 289, str(MAGIC_NUMBER))
    check("10*121//289 == 4", 10 * 121 // MAGIC_NUMBER == 4,
          str(10 * 121 // MAGIC_NUMBER))
    check("the same arithmetic in the category table",
          CREATURE.max_count * 121 // MAGIC_NUMBER == 4)

    state = SpawnState()
    state.spawnable_chunk_count = 121
    check("an empty world is under the cap", state.can_spawn_for_category_global(CREATURE))
    for _ in range(3):
        state.category_counts[1] += 1
    check("3 creatures still under the cap of 4",
          state.can_spawn_for_category_global(CREATURE))
    state.category_counts[1] = 4
    check("4 creatures is at the cap, not over it",
          not state.can_spawn_for_category_global(CREATURE))

    # The bug this test exists for. Pumpkin never calls
    # set_spawnable_chunk_count, so its value stays 0, `10*0//289` is 0, and
    # `counts[1] < 0` is false for a non-negative count: its natural spawner
    # can never place anything, ever.
    pumpkin = SpawnState()
    check("a zero spawnable_chunk_count caps at zero and spawns nothing "
          "(the Pumpkin bug)",
          not pumpkin.can_spawn_for_category_global(CREATURE),
          "computed here instead of inherited")

    small = SpawnState()
    small.spawnable_chunk_count = 9  # simulation radius 1
    check("a tiny view distance caps creature at 0",
          not small.can_spawn_for_category_global(CREATURE),
          f"10*9//289 = {10 * 9 // MAGIC_NUMBER}")

    check("MISC is the uncapped category", MISC.max_count == UNLIMITED)
    check("and it is never globally refused",
          SpawnState().can_spawn_for_category_global(MISC))
    check("the category table is 8 rows in vanilla's declaration order",
          [c.name for c in CATEGORY_LIST] == [
              "MONSTER", "CREATURE", "AMBIENT", "AXOLOTLS",
              "UNDERGROUND_WATER_CREATURE", "WATER_CREATURE", "WATER_AMBIENT",
              "MISC"],
          str([c.name for c in CATEGORY_LIST]))
    check("every category's biome_key is in the biome.json spelling",
          all(c.biome_key in SPAWN_CATEGORIES for c in CATEGORY_LIST),
          str([c.biome_key for c in CATEGORY_LIST]))
    print()


def check_energy() -> None:
    from entity.spawner import SpawnState

    print("4. spawn potential")

    state = SpawnState()
    check("no charges means no energy cost",
          state.potential_energy_change(0, 64, 0, 1.0) == 0.0)
    state.deposit_charge(0, 64, 0, 1.0)
    check("a charge at the same block is infinite (a stack self-limits)",
          state.potential_energy_change(0, 64, 0, 1.0) == math.inf)
    check("and 1/d falloff at 8 blocks, not 1/d^2",
          abs(state.potential_energy_change(8, 64, 0, 1.0) - 0.125) < 1e-9,
          str(state.potential_energy_change(8, 64, 0, 1.0)))
    check("the newcomer's own charge scales the field",
          abs(state.potential_energy_change(8, 64, 0, 0.5) - 0.0625) < 1e-9,
          str(state.potential_energy_change(8, 64, 0, 0.5)))
    two = SpawnState()
    two.deposit_charge(0, 64, 0, 1.0)
    two.deposit_charge(0, 72, 0, 1.0)
    check("two charges 8 and 16 blocks away sum to 1/8 + 1/16 = 0.1875",
          abs(two.potential_energy_change(0, 80, 0, 1.0) - 0.1875) < 1e-9,
          str(two.potential_energy_change(0, 80, 0, 1.0)))
    empty = SpawnState()
    empty.deposit_charge(0, 64, 0, 0.0)
    check("a zero charge deposits nothing, so the field stays empty",
          not empty.potential and empty.potential_energy_change(0, 64, 0, 1.0)
          == 0.0)
    check("an empty cost table passes every budget",
          SpawnState().can_spawn_by_energy(0, 64, 0, 0.0, 0.0))
    tight = SpawnState()
    tight.deposit_charge(0, 64, 0, 1.0)
    check("a nonzero budget is enforced against the field",
          not tight.can_spawn_by_energy(8, 64, 0, 0.1, 1.0))
    print()


def check_weighted_draw() -> None:
    from entity.biome_spawns import biome_spawns
    from entity.mob import MOB_CLASSES
    from world.vanilla.random_source import Xoroshiro

    print("5. the weighted draw")

    spawns = biome_spawns(tuple(MOB_CLASSES))
    # 20k draws, not 1000. At 1000 the standard error on a 0.4 share is
    # 1.5%, so a 4% tolerance is a 2.6-sigma test that fails often enough to
    # be noise; at 20k it is 0.35% and a 2% tolerance is nearly 6 sigma, so
    # a failure means the weights are actually wrong.
    draws = 20_000
    rng = Xoroshiro.from_seed(20261006)
    counts = Counter()
    for _ in range(draws):
        counts[spawns.draw("minecraft:plains", "creature",
                           rng.next_f32()).type_name] += 1
    # Vanilla's plains creature weights: sheep 12, pig 10, chicken 10, cow 8
    # over the surviving four. Total 40. The shares are 0.30 / 0.25 / 0.25 /
    # 0.20, and the same 2% tolerance from the three-row case still separates
    # weight-following from a uniform draw over four rows (which would be
    # 0.25 each).
    expected = {"minecraft:sheep": 12 / 40, "minecraft:pig": 10 / 40,
                "minecraft:chicken": 10 / 40, "minecraft:cow": 8 / 40}
    for name, share in expected.items():
        got = counts[name] / draws
        check(f"{name.split(':')[1]} frequency is within 2% of {share:.3f}",
              abs(got - share) < 0.02,
              f"{counts[name]}/{draws} = {got:.3f} (expected {share:.3f})")
    check("every plains draw lands on an implemented type",
          set(counts) <= set(MOB_CLASSES), str(sorted(counts)))
    check("the draw is not uniform over four rows",
          abs(counts["minecraft:sheep"] / draws - 1 / 4) > 0.05,
          "a uniform draw would give 0.25; sheep's weight gives 0.30")
    print()


def check_distance_and_spacing() -> None:
    from entity.spawner import MIN_SPAWN_DISTANCE_SQ, NaturalSpawner

    print("6. distance guards and same-type spacing")

    check("the guard is 24 blocks squared",
          MIN_SPAWN_DISTANCE_SQ == 576.0, str(MIN_SPAWN_DISTANCE_SQ))

    dimension = make_dimension()
    spawner = NaturalSpawner(dimension, seed=1)
    install_chunk(dimension, LAND_CHUNK[0], LAND_CHUNK[1])
    fx, fz = LAND_CHUNK[0] * 16 + 8, LAND_CHUNK[1] * 16 + 8
    # A lone player far to the north, and a candidate inside the loaded chunk.
    # The player is 200 blocks away so the local cap and the distance guard
    # are not decided by the fixture's geometry.
    spawner.state.players = [("p1", float(fx), 64.0, float(fz) - 200.0)]
    spawner.state.player_counts["p1"] = [0] * 8
    dimension.spawn_point = (0, 64, 0)

    near = 20
    far = 40
    check("20 blocks from a player is refused",
          not spawner._is_right_distance(near * near, fx, 64, fz, *LAND_CHUNK))
    check("40 blocks from a player is accepted",
          spawner._is_right_distance(far * far, fx, 64, fz, *LAND_CHUNK),
          f"d2={far * far:.0f} > 576")
    check("exactly 24 blocks is refused (the bound is strict)",
          not spawner._is_right_distance(576.0, fx, 64, fz, *LAND_CHUNK))
    check("no player at all means no distance (None is a refusal)",
          not spawner._is_right_distance(None, fx, 64, fz, *LAND_CHUNK))

    # The world-spawn guard, overworld only. The guard is about the world
    # spawn, so this needs loaded chunks *near the origin* - testing it out
    # at the land fixture would let the residency clause answer for it.
    # 20/24 blocks east of the origin is chunk (1,0); 40 is chunk (2,0).
    install_chunk(dimension, 1, 0)
    install_chunk(dimension, 2, 0)
    dimension.spawn_point = (0, 64, 0)
    check("20 blocks from the overworld world spawn is refused",
          not spawner._is_right_distance(far * far, 20, 64, 0, 1, 0),
          "chunk (1,0) is loaded, so this is the world-spawn clause")
    check("exactly 24 blocks is allowed there (the bound is strict, < 576)",
          spawner._is_right_distance(far * far, 24, 64, 0, 1, 0),
          "vanilla and Pumpkin both test `< 576` here, unlike the "
          "player guard's `<= 576`")
    check("40 blocks from it is accepted",
          spawner._is_right_distance(far * far, 40, 64, 0, 2, 0),
          "chunk (2,0) is loaded and past the guard")

    # The residency clause: the +-5 nudge walks into a neighbour, and a
    # neighbour that is not in memory must be a refusal, not air.
    check("a step into an unloaded neighbour chunk is refused",
          not spawner._is_right_distance(far * far, 900 * 16, 64, 0, 0, 0))
    check("a step inside the chunk being processed is allowed",
          spawner._is_right_distance(far * far, fx, 64, fz, *LAND_CHUNK))

    # Same-type spacing, 24 blocks. Same type refuses; a different type does
    # not, because a herd of four cows together is the point of the rule.
    from entity.entity_type import entity_type_by_name
    from entity.mob import create_mob

    base = (fx, 64, fz)

    def spacing_allows(existing_name, asking_name, offset):
        """Would `asking_name` be allowed at the base with `existing_name`?"""
        dimension.entities.clear()
        spawner.chunk_entities.clear()
        spawner.state.potential = []
        spawner.state.category_counts = [0] * 8
        etype = entity_type_by_name(existing_name)
        mob = create_mob(etype, base[0] + offset, float(base[1]), base[2],
                         dimension=dimension.name)
        dimension.spawn_entity(mob)
        asked = entity_type_by_name(asking_name)
        return spawner._passes_energy_and_spacing(asked, base[0], base[1],
                                                  base[2])

    check("a cow 10 blocks away refuses another cow",
          not spacing_allows("minecraft:cow", "minecraft:cow", 10))
    check("a cow 30 blocks away allows one",
          spacing_allows("minecraft:cow", "minecraft:cow", 30))
    check("exactly 24 blocks is allowed (the bound is strict, < 576)",
          spacing_allows("minecraft:cow", "minecraft:cow", 24),
          "same convention as the world-spawn clause")
    check("a cow 10 blocks away does not block a sheep",
          spacing_allows("minecraft:cow", "minecraft:sheep", 10),
          "the rule is same-type, so mixed herds are legal")
    check("a sheep 10 blocks away does block another sheep",
          not spacing_allows("minecraft:sheep", "minecraft:sheep", 10))
    print()


def check_limit_per_chunk() -> None:
    from entity.entity_type import entity_type_by_name

    print("7. limit_per_chunk")

    for name, limit in (("minecraft:chicken", 4), ("minecraft:cow", 4),
                        ("minecraft:pig", 4), ("minecraft:rabbit", 4),
                        ("minecraft:sheep", 4)):
        etype = entity_type_by_name(name)
        check(f"{name.split(':')[1]} limit_per_chunk is {limit}",
              etype.limit_per_chunk == limit, str(etype.limit_per_chunk))
    check("the spawner reads it off the type, not a constant",
          entity_type_by_name("minecraft:cow").limit_per_chunk == 4)
    print()


def check_real_round() -> None:
    from entity.mob import Mob
    from entity.spawner import NaturalSpawner
    from world.chunk_lifecycle import ChunkLifecycle

    print("8. a real round over real terrain, then a real unload")

    dimension = make_dimension()
    # Simulation radius 4 -> 81 entity-ticking chunks -> 10*81//289 = 2
    # creatures for the whole loaded world. A small view distance would cap
    # at 0 and the test would prove nothing.
    lifecycle = ChunkLifecycle(dimension, save_chunk=lambda chunk: True)
    spawner = NaturalSpawner(dimension, lifecycle=lifecycle, seed=7)

    # The player stands where the tickets are, on the surface of the land
    # chunk. The ticking square is centred on the player, so the chunks
    # within 24 blocks are refused by the distance guard and the ones in the
    # outer ring are not - which is the whole shape of a vanilla round. The
    # player must NOT be moved away from the tickets: `ticks_entities` is a
    # ticket query, so a player 200 blocks off would put every candidate
    # chunk outside the 128-block local cap and the round would place nothing.
    # The ticket is placed before the y is known because only x and z pick
    # the chunk; the real surface height is read back once the terrain is in.
    cx, cz = LAND_CHUNK
    world_x, world_z = cx * 16 + 8, cz * 16 + 8
    lifecycle.set_player_view("p1", float(world_x), 100.0, float(world_z), 4, 4)
    for _ in range(400):
        dimension.tick()
        lifecycle.tick()
        if not lifecycle._untracked and not lifecycle._requested:
            break

    from entity.spawn_placement import top_motion_blocking_no_leaves
    world_y = float(top_motion_blocking_no_leaves(dimension, world_x, world_z) + 1)
    land = dimension.get_chunk(cx, cz)
    check("the fixture's player stands on solid ground",
          land.get_block(world_x & 15, int(world_y) - 1, world_z & 15) != 0,
          f"y={world_y}")

    ticking = [p for p in lifecycle.owned_positions()
               if dimension.ticks_entities(p.x, p.z)]
    check("the lifecycle holds a square of entity-ticking chunks",
          len(ticking) == 81, f"{len(ticking)} ticking chunks")
    check("all of them are in memory",
          all(dimension.is_chunk_loaded(p.x, p.z) for p in ticking))
    check("they are all generated",
          all(dimension.get_chunk(p.x, p.z).is_generated for p in ticking))

    spawner.state.players = [("p1", float(world_x), world_y, float(world_z))]
    spawner.state.player_counts["p1"] = [0] * 8
    spawner._rebuild_state(len(ticking))
    check("spawnable_chunk_count is the ticking count, not zero",
          spawner.state.spawnable_chunk_count == len(ticking) == 81,
          str(spawner.state.spawnable_chunk_count))
    check("the global cap is 10*81//289 = 2",
          10 * len(ticking) // 289 == 2, str(10 * len(ticking) // 289))
    check("and the world is under it",
          spawner.state.can_spawn_for_category_global(_creature()))
    check("at least one candidate chunk is outside the 24-block player rule",
          any(spawner._nearest_player_distance_sq(
              (p.x << 4) + 8, int(world_y), (p.z << 4) + 8) > 576
              for p in ticking),
          "otherwise the round could not place anything by design")

    # Drive the sliced path: one full round is `interval_ticks` calls, and
    # `_begin_round` fires on the first of them.
    for t in range(spawner.interval_ticks):
        spawner.tick(t)
    stats = spawner.stats()
    check("one 20-tick round places mobs", stats["spawned"] > 0, str(stats))
    check("the round walked all 81 candidate chunks across its 20 slices",
          stats["ticks_spawned"] == spawner.interval_ticks,
          f"{stats['ticks_spawned']} spawning ticks")
    check("the cap held: at most 2 creatures in the whole world",
          stats["spawned"] <= 2, str(stats["spawned"]))
    check("every mob is in dimension.entities",
          len(dimension.entities) == stats["spawned"],
          f"{len(dimension.entities)} entities")
    check("every mob is in the residency index",
          stats["entities"] == stats["spawned"], str(stats))

    mobs = [e for e in dimension.entities.values() if isinstance(e, Mob)]
    check("all three types are reachable", len(mobs) == stats["spawned"])
    for mob in mobs:
        bx, by, bz = mob.block_pos
        check(f"{mob} stands on the surface (not floating)",
              by == top_motion_blocking_no_leaves(dimension, bx, bz) + 1,
              f"y={by}, no-leaves top="
              f"{top_motion_blocking_no_leaves(dimension, bx, bz)}")
        check(f"{mob} is in the chunk the index recorded",
              (bx >> 4, bz >> 4) in spawner.chunk_entities)
        check(f"{mob} is more than 24 blocks from the player",
              (bx - world_x) ** 2 + (by - world_y) ** 2 + (bz - world_z) ** 2 > 576,
              f"d2={(bx - world_x) ** 2 + (by - world_y) ** 2 + (bz - world_z) ** 2}")

    # A second round with the cap already reached places nothing. The round
    # boundary is a multiple of the interval, so 20 more ticks is one round.
    before = len(dimension.entities)
    for t in range(spawner.interval_ticks, 2 * spawner.interval_ticks):
        spawner.tick(t)
    check("a second round under the cap places nothing",
          len(dimension.entities) == before,
          f"{before} -> {len(dimension.entities)}")

    # The residency pass. Drop every chunk from memory and one tick later
    # both tables must be empty - the user's part of the spec.
    if mobs:
        for pos in list(lifecycle.owned_positions()):
            dimension.chunk_manager.remove(pos)
        spawner._reconcile_residency()
        check("dimension.entities is empty after every chunk leaves",
              not dimension.entities, f"{len(dimension.entities)} left")
        check("the residency index is empty too", not spawner.chunk_entities,
              str(spawner.chunk_entities))
        check("and the removals were counted",
              spawner.stats()["removed_on_unload"] >= len(mobs),
              str(spawner.stats()["removed_on_unload"]))
    print()


def _creature():
    from entity.mob_category import CREATURE

    return CREATURE


def _write_light_cell(section, cell: int, value: int, sky: bool = True) -> None:
    """One 4-bit cell of a packed light array, low nibble first.

    Written out here rather than imported from `world/light_layout.py` on
    purpose: this is the *reader's* copy of the format, so a bug in the layout
    module cannot make both sides agree.
    """
    packed = section.sky_light if sky else section.block_light
    byte = cell >> 1
    if cell & 1:
        packed[byte] = (int(packed[byte]) & 0x0F) | (value << 4)
    else:
        packed[byte] = (int(packed[byte]) & 0xF0) | value


def check_light_queries() -> None:
    """The luminance readers, on real terrain with hand-written light.

    The point of this section is the one conclusion the old 0/15 substitute
    could not reach: a cave with a shaft to the sky. Vanilla's answer there is
    "lit", and it is the difference between a deep cave never spawning a herd
    and a deep cave next to the surface sometimes doing so.
    """
    from entity.spawn_placement import (
        block_light_at, is_bright_enough, luminance_at, sky_light_at,
        top_motion_blocking_no_leaves,
    )

    print("8. light queries read the engine's arrays")

    overworld = make_dimension()
    cx, cz = LAND_CHUNK
    chunk = install_chunk(overworld, cx, cz)
    lx, lz = 8, 8
    x, z = cx * 16 + lx, cz * 16 + lz
    top = top_motion_blocking_no_leaves(overworld, x, z)
    check("the fixture has a surface to work from", top is not None, str(top))
    if top is None:
        return

    # Nobody has lit this chunk, so the fallback is what answers -- and it is
    # the old rule exactly, which is what makes "loaded but not yet lit" safe.
    check("an unlit chunk falls back to 15 above the surface",
          luminance_at(overworld, x, top + 1, z) == 15)
    check("and to 0 below it", luminance_at(overworld, x, top - 5, z) == 0)

    # Now punch a 15-block shaft straight down from the surface and light it
    # the way the engine would: 15 at the surface, one less per block down.
    # That gradient is what makes the animal rule legible: `> 8` means the
    # herd reaches seven blocks below the daylit surface and no further.
    sky_y = top + 1
    for step in range(15):
        depth = sky_y - step
        _write_light_cell(chunk.sections[(depth - chunk.min_y) // 16],
                          lx + lz * 16 + ((depth - chunk.min_y) % 16) * 256,
                          15 - step)
    # The block the lit part stops at is dark -- that contrast is the whole
    # point of the scenario. Without writing it, the cell above the surface
    # carries the `ChunkSection` default of 15 and the shaft would look like it
    # never ended.
    bottom = sky_y - 14
    below = bottom - 1
    _write_light_cell(chunk.sections[(below - chunk.min_y) // 16],
                      lx + lz * 16 + ((below - chunk.min_y) % 16) * 256, 0)
    chunk.light_populated = True

    check("the bottom of the shaft is lit (the fourth conclusion)",
          sky_light_at(overworld, x, bottom, z) == 1,
          str(sky_light_at(overworld, x, bottom, z)))
    check("the middle of the shaft is at its own level",
          sky_light_at(overworld, x, sky_y - 7, z) == 8,
          str(sky_light_at(overworld, x, sky_y - 7, z)))
    check("the surface itself is full daylight",
          sky_light_at(overworld, x, sky_y, z) == 15)
    check("seven blocks down is exactly the herd threshold: 8 is not enough",
          not is_bright_enough(overworld, x, sky_y - 7, z)
          and is_bright_enough(overworld, x, sky_y - 6, z),
          f"7 down = {sky_light_at(overworld, x, sky_y - 7, z)}, "
          f"6 down = {sky_light_at(overworld, x, sky_y - 6, z)}")
    check("so a herd can spawn in a shallow shaft but not a deep one",
          is_bright_enough(overworld, x, sky_y - 6, z)
          and not is_bright_enough(overworld, x, bottom, z))
    check("the block the shaft stops at is dark, and refused",
          sky_light_at(overworld, x, below, z) == 0
          and not is_bright_enough(overworld, x, below, z),
          str(sky_light_at(overworld, x, below, z)))
    check("an unlit chunk still answers the fallback, 15 above the surface",
          luminance_at(overworld, x, 319, z) == 15)

    # Block light is the half that had no representation at all before. The
    # torch goes next to the shaft bottom, in the dark block's column, so the
    # two sources are compared at the same place rather than in two stories.
    torch_section = chunk.sections[(below - chunk.min_y) // 16]
    _write_light_cell(torch_section,
                      (lx + 1) + lz * 16 + ((below - chunk.min_y) % 16) * 256,
                      12, sky=False)
    check("a torch lights its own cell", block_light_at(overworld, x + 1, below, z) == 12,
          str(block_light_at(overworld, x + 1, below, z)))
    check("a torch-lit cell is bright enough even where the sky is not",
          is_bright_enough(overworld, x + 1, below, z))
    check("luminance takes the brighter of the two",
          luminance_at(overworld, x, below, z) == max(
              sky_light_at(overworld, x, below, z),
              block_light_at(overworld, x, below, z)),
          str(luminance_at(overworld, x, below, z)))

    # An unloaded column still refuses rather than reading as dark or bright.
    far = (cx + 900, cz)
    check("an unloaded column answers 0, not a guess",
          luminance_at(overworld, far[0] * 16 + 8, 64, far[1] * 16 + 8) == 0)

    # A dimension with no sky must not report daylight from the array default.
    from world.dimension import Dimension, DimensionType
    from world.generator import GeneratorSettings, create_generator

    nether = Dimension(
        DimensionType.NETHER,
        world_seed=SEED,
        generator=create_generator(GeneratorSettings(seed=SEED, preset="nether"),
                                   backend="legacy"),
    )
    nether_chunk = install_chunk(nether, 0, 0)
    nether_chunk.light_populated = True
    check("the nether has no sky light to read",
          sky_light_at(nether, 8, 200, 8) == 0,
          str(sky_light_at(nether, 8, 200, 8)))
    check("so luminance there is block light alone",
          luminance_at(nether, 8, 200, 8) == block_light_at(nether, 8, 200, 8))

    # The floor predicate's light half is the block's own luminance. The three
    # magnitudes below are what `assets/blocks.json` reports for them, so the
    # test separates "refused for glowing" from "refused for shape": magma
    # block is a full cube that glows dimly and is *accepted*, which is what
    # makes the `< 14` threshold visible rather than incidental.
    from entity.spawn_placement import is_valid_spawn_floor
    from world.block_states import state_id

    check("a stone floor is valid",
          is_valid_spawn_floor(state_id("minecraft:stone")))
    check("glowstone (15) is refused for glowing",
          not is_valid_spawn_floor(state_id("minecraft:glowstone")))
    check("a sea lantern (15) is refused too",
          not is_valid_spawn_floor(state_id("minecraft:sea_lantern")))
    check("a magma block (3) is accepted: it is below the ceiling",
          is_valid_spawn_floor(state_id("minecraft:magma_block")))
    check("a floor that is not a full cube is refused for shape",
          not is_valid_spawn_floor(state_id("minecraft:oak_fence")))
    check("a redstone torch is refused for shape, not for its light",
          not is_valid_spawn_floor(state_id("minecraft:redstone_torch")))


def main() -> None:
    from registry.game_registry import init_registries, reset_registries

    init_registries()
    try:
        check_biome_tables()
        check_placement()
        check_caps()
        check_energy()
        check_weighted_draw()
        check_distance_and_spacing()
        check_limit_per_chunk()
        check_real_round()
        check_light_queries()
        print()
        if failures:
            print(f"{len(failures)} failure(s):")
            for f in failures:
                print(f"  - {f}")
            raise SystemExit(1)
        print("all checks passed")
    finally:
        reset_registries()


if __name__ == "__main__":
    main()
