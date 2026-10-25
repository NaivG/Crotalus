"""
Worldgen data check: the registry's worldgen views are complete, consistent,
and actually what the world generates from.

`registry.worldgen` is the bridge between `assets/` and the generator, and the
failure it guards against is quiet: a preset whose default block does not
resolve, a biome list pointing at a carver that is not in `carver.json`, a
dimension height the chunk codec disagrees with. None of those raise at boot;
they produce plausible terrain made of the wrong things. This tool drives the
real load path, asserts the cross-references and the numbers behind them, and
then builds a world to prove the generator is reading the same data.

Run: python tools/check_worldgen_data.py
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from loguru import logger  # noqa: E402

logger.remove()
logger.add(sys.stderr, level="ERROR")

failures: list[str] = []


def check(label: str, condition: bool, detail: str = "") -> None:
    marker = "OK  " if condition else "FAIL"
    print(f"  {marker} {label}" + (f"  ({detail})" if detail else ""))
    if not condition:
        failures.append(label)


def main() -> int:
    from registry import init_registries
    from registry.worldgen import (
        CLIMATE_PARAMETERS,
        DENSITY_FUNCTION_CLASSES,
        NOISE_ROUTER_SLOTS,
        WorldgenData,
        iter_density_nodes,
    )

    print("1. The loader builds the views and they validate")
    registry = init_registries(log_summary=False)
    worldgen = registry.worldgen
    check("registry.worldgen is a WorldgenData", isinstance(worldgen, WorldgenData), repr(worldgen))

    start = time.perf_counter()
    problems = worldgen.validate()
    elapsed = (time.perf_counter() - start) * 1000
    check("no worldgen validation problems", not problems, "; ".join(problems[:3]))
    check("validation is cheap enough for boot", elapsed < 1000, f"{elapsed:.0f} ms")

    summary = worldgen.summary()
    check(
        "expected counts",
        summary["presets"] == 7
        and summary["dimensions"] == 4
        and summary["biomes"] == 65
        and summary["noises"] == 60
        and summary["carvers"] == 4
        and summary["chunk_statuses"] == 12,
        str(summary),
    )

    print("\n2. Generator presets, straight out of chunk_gen_settings.json")
    overworld = worldgen.settings("overworld")
    check("overworld settings exist", overworld is not None)
    if overworld is None:
        return 1
    check("overworld sea level is 63", overworld.sea_level == 63, str(overworld.sea_level))
    check(
        "overworld shape is -64..319",
        (overworld.shape.min_y, overworld.shape.height) == (-64, 384),
        f"{overworld.shape.min_y} +{overworld.shape.height}",
    )
    check("overworld default block is stone", overworld.default_block.name == "minecraft:stone",
          overworld.default_block.name)
    check("overworld default fluid is water", overworld.default_fluid.name == "minecraft:water",
          overworld.default_fluid.name)
    check("default states resolve", overworld.default_block.state_id == 1
          and overworld.default_fluid.state_id == 86,
          f"{overworld.default_block.state_id}/{overworld.default_fluid.state_id}")

    nether = worldgen.settings("nether")
    end = worldgen.settings("end")
    check("nether terrain shape is 128 tall",
          nether is not None and nether.shape.height == 128,
          str(nether.shape) if nether else "missing")
    check("nether default block is netherrack",
          nether is not None and nether.default_block.name == "minecraft:netherrack",
          nether.default_block.name if nether else "missing")
    check("nether default fluid is lava",
          nether is not None and nether.default_fluid.name == "minecraft:lava",
          nether.default_fluid.name if nether else "missing")
    check("end default block is end stone",
          end is not None and end.default_block.name == "minecraft:end_stone",
          end.default_block.name if end else "missing")
    check("end default fluid is air",
          end is not None and end.default_fluid.name == "minecraft:air",
          end.default_fluid.name if end else "missing")

    check(
        "every preset resolves all 15 router slots",
        all(
            len(settings.noise_router) == len(NOISE_ROUTER_SLOTS)
            for settings in worldgen.settings_by_preset.values()
        ),
        ", ".join(
            f"{name}={len(s.noise_router)}"
            for name, s in sorted(worldgen.settings_by_preset.items())
        ),
    )
    check(
        "vanilla slot names are used on the settings side",
        set(overworld.raw_noise_router) == {slot for slot, _ in NOISE_ROUTER_SLOTS},
        f"{len(overworld.raw_noise_router)} slots",
    )
    check(
        "the final_density slot is a resolved tree, not a name",
        isinstance(overworld.router("final_density"), dict)
        and "_class" in overworld.router("final_density"),
        str(overworld.router("final_density").get("_class")),
    )
    unknown_slot = False
    try:
        overworld.router("nope")
    except KeyError:
        unknown_slot = True
    check("an unknown slot raises instead of returning None", unknown_slot)

    print("\n3. Density-function trees")
    classes: set = set()
    for graph in worldgen.density_graphs.values():
        for node in graph.values():
            classes |= {name for name, _ in iter_density_nodes(node)}
    check(
        "every node class is one this module knows",
        classes <= set(DENSITY_FUNCTION_CLASSES),
        str(sorted(classes - set(DENSITY_FUNCTION_CLASSES))),
    )
    check(
        "the schema covers all 19 classes the dump uses",
        len(DENSITY_FUNCTION_CLASSES) == 19 and len(classes) == 19,
        f"declared={len(DENSITY_FUNCTION_CLASSES)} seen={len(classes)}",
    )
    check(
        "overworld is the richest graph (18 classes)",
        len({
            name
            for node in worldgen.density_graphs["overworld"].values()
            for name, _ in iter_density_nodes(node)
        }) == 18,
    )

    print("\n4. Biomes, noise parameters, multi-noise tree")
    planes = worldgen.biome("minecraft:plains")
    check("plains exists", planes is not None)
    check("plains is biome id 40", worldgen.biome_id("minecraft:plains") == 40,
          str(worldgen.biome_id("minecraft:plains")))
    check(
        "plains runs three carvers",
        planes is not None and planes.carvers == (
            "minecraft:cave", "minecraft:cave_extra_underground", "minecraft:canyon"
        ),
        str(planes.carvers) if planes else "missing",
    )
    check(
        "plains has 11 decoration steps",
        planes is not None and len(planes.features) == 11,
        str(len(planes.features)) if planes else "missing",
    )
    check(
        "every biome has a synchronized id",
        all(biome.numeric_id is not None for biome in worldgen.biomes.values()),
        f"{sum(1 for b in worldgen.biomes.values() if b.numeric_id is None)} without",
    )
    check(
        "every biome's features resolve",
        all(
            worldgen.placed_feature(feature) is not None
            for biome in worldgen.biomes.values()
            for step in biome.features
            for feature in step
        ),
    )

    temperature = worldgen.noise("temperature")
    check(
        "temperature noise parameters",
        temperature is not None and temperature.first_octave == -10
        and temperature.amplitudes[:3] == (1.5, 0.0, 1.0),
        str(temperature),
    )

    overworld_tree = worldgen.multi_noise("overworld")
    nether_tree = worldgen.multi_noise("nether")
    check("overworld multi-noise source exists", overworld_tree is not None)
    check("nether multi-noise source exists", nether_tree is not None)
    if overworld_tree is not None:
        names = overworld_tree.biome_names()
        check(
            "overworld tree leaves are packed and biome-valid",
            overworld_tree.leaf_count() > 1000 and len(names) > 40
            and names <= set(worldgen.biomes),
            f"{overworld_tree.leaf_count()} leaves, {len(names)} distinct biomes",
        )
        check(
            "climate parameters are seven",
            len(CLIMATE_PARAMETERS) == 7 and len(overworld_tree.parameters) == 7,
            str(len(overworld_tree.parameters)),
        )

    print("\n5. Dimension types are the world's dimensions")
    for dimension_name, min_y, height, logical in (
        ("minecraft:overworld", -64, 384, 384),
        ("minecraft:the_nether", 0, 256, 128),
        ("minecraft:the_end", 0, 256, 256),
    ):
        data = worldgen.dimension(dimension_name)
        check(
            f"{dimension_name} is {height} tall from y={min_y} (logical {logical})",
            data is not None
            and (data.min_y, data.height, data.logical_height) == (min_y, height, logical),
            str(data),
        )

    check(
        "the nether's dimension is twice its terrain shape",
        worldgen.dimension("minecraft:the_nether").height == 256
        and worldgen.settings("nether").shape.height == 128,
    )
    check(
        "the end uses the end preset",
        worldgen.dimension("minecraft:the_end").preset == "end",
        worldgen.dimension("minecraft:the_end").preset,
    )

    from world.dimension import DimensionSettings, DimensionType

    for dimension_type, source in (
        (DimensionType.OVERWORLD, "registry"),
        (DimensionType.NETHER, "registry"),
        (DimensionType.END, "registry"),
    ):
        settings = DimensionSettings.for_dimension(dimension_type)
        data = worldgen.dimension(dimension_type.value)
        check(
            f"DimensionSettings({dimension_type.name}) comes from the registry",
            settings.source == source
            and (settings.min_y, settings.height, settings.logical_height)
            == (data.min_y, data.height, data.logical_height),
            f"source={settings.source} y={settings.min_y} h={settings.height} lh={settings.logical_height}",
        )
        check(
            f"DimensionSettings({dimension_type.name}) sea level matches the preset",
            settings.sea_level == worldgen.settings(data.preset).sea_level,
            f"{settings.sea_level} vs {worldgen.settings(data.preset).sea_level}",
        )

    # The built-in fallback table is what a custom dimension (or a tool without
    # a registry) still gets, so it must not drift away from the data.
    for dimension_type in (DimensionType.OVERWORLD, DimensionType.NETHER, DimensionType.END):
        data = worldgen.dimension(dimension_type.value)
        builtin = DimensionSettings.builtin(dimension_type)
        check(
            f"the built-in table agrees with {dimension_type.name} on height",
            (builtin.min_y, builtin.height, builtin.logical_height)
            == (data.min_y, data.height, data.logical_height),
            f"builtin={builtin.min_y}/{builtin.height}/{builtin.logical_height}",
        )

    print("\n6. The world generates from it")
    workdir = tempfile.mkdtemp(prefix="crotalus-worldgen-")
    previous = os.getcwd()
    os.chdir(workdir)
    try:
        from world import World
        from world.chunk_codec import BIOME_PLAINS, ChunkCodec, default_biome_id

        world = World(name="worldgen", seed=1234, world_dir=Path("worldgen"))
        overworld_gen = world.overworld.generator
        check(
            "the overworld generator took the overworld preset",
            overworld_gen.settings.preset == "overworld"
            and overworld_gen.settings.sea_level == 63
            and overworld_gen.shape is not None and overworld_gen.shape.min_y == -64,
            f"preset={overworld_gen.settings.preset} sea={overworld_gen.settings.sea_level}",
        )
        check(
            "the overworld generator fills with the registry's default block/fluid",
            overworld_gen.settings.default_block == registry.blocks.block("minecraft:stone").default_state_id
            and overworld_gen.settings.default_fluid == registry.blocks.block("minecraft:water").default_state_id,
            f"{overworld_gen.settings.default_block}/{overworld_gen.settings.default_fluid}",
        )
        nether_gen = world.nether.generator
        check(
            "the nether generator took the nether preset",
            nether_gen.settings.preset == "nether"
            and nether_gen.settings.sea_level == 32
            and nether_gen.settings.default_block == registry.blocks.block("minecraft:netherrack").default_state_id
            and nether_gen.settings.default_fluid == registry.blocks.block("minecraft:lava").default_state_id,
            f"preset={nether_gen.settings.preset} sea={nether_gen.settings.sea_level} "
            f"block={nether_gen.settings.default_block} fluid={nether_gen.settings.default_fluid}",
        )
        check(
            "the end generator took the end preset, and air is its fluid",
            world.end.generator.settings.preset == "end"
            and world.end.generator.settings.sea_level == 0
            and world.end.generator.settings.default_block
            == registry.blocks.block("minecraft:end_stone").default_state_id
            and world.end.generator.settings.default_fluid == 0,
            f"preset={world.end.generator.settings.preset} "
            f"block={world.end.generator.settings.default_block} "
            f"fluid={world.end.generator.settings.default_fluid}",
        )

        # The codec walks the dimension's height. The nether's used to be 128
        # (the terrain shape), so half of every nether chunk was dropped.
        for dimension, sections in (
            (world.overworld, 24),
            (world.nether, 16),
            (world.end, 16),
        ):
            codec = ChunkCodec(
                min_y=dimension.settings.min_y,
                height=dimension.settings.height,
                default_biome=default_biome_id(),
            )
            check(
                f"{dimension.name} encodes {sections} sections",
                codec.section_count == sections,
                f"{codec.section_count} (min_y={codec.min_y}, height={codec.height})",
            )

        check(
            "the chunk codec's default biome is the registry's plains",
            default_biome_id() == 40 == BIOME_PLAINS,
            f"{default_biome_id()} vs {BIOME_PLAINS}",
        )
        check(
            "the biome id matches the synced registry",
            registry.synced.numeric_id("minecraft:worldgen/biome", "minecraft:plains") == 40,
        )

        chunk = world.load_chunk(0, 0)
        from world import block_states

        unknown = {
            int(value)
            for section in chunk.sections
            for value in set(section.blocks.tolist())
            if value != 0 and block_states.state(int(value)) is None
        }
        check("every block the generator writes is a registry state", not unknown, str(sorted(unknown)[:5]))

        # The old storage table bound 32 blocks, so the nether and the end were
        # generated out of stone. They now name their own blocks.
        nether = world.load_chunk(0, 0, dimension=DimensionType.NETHER)
        nether_names = {
            block_states.name_of(int(value)) for value in set(nether.sections[0].blocks.tolist())
        }
        check(
            "the nether generates netherrack, not stone",
            "minecraft:netherrack" in nether_names,
            str(sorted(nether_names)),
        )
    finally:
        os.chdir(previous)
        shutil.rmtree(workdir, ignore_errors=True)

    print("\n7. The sample chunk is a fixture, not an input")
    sample = worldgen.chunk_sample()
    check("chunk.json has 98,304 entries", len(sample) == 16 * 16 * 384, str(len(sample)))
    values = sorted(set(sample))
    check(
        "its values are registry state ids for air/stone/water",
        values == [0, 1, 86]
        and all(registry.blocks.state(value) is not None for value in values),
        str(values),
    )

    print("\n8. The console view")
    described = registry.describe()
    check("describe() includes the worldgen summary", "worldgen:" in described)
    check(
        "describe() lists every preset",
        all(preset in described for preset in worldgen.settings_by_preset),
    )

    print()
    if failures:
        print(f"{len(failures)} worldgen check(s) FAILED:")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print("All worldgen data checks passed.")
    return 0


if __name__ == "__main__":
    _registry = None
    raise SystemExit(main())
