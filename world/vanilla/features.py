"""
The decorated-feature stage: a configured feature, placed by an ordered list of
placement modifiers, once per chunk per decoration step.

    from world.vanilla.features import decorate, load_placed_features

    stats = decorate(chunk_x, chunk_z, blocks, biomes, preset="overworld",
                     seed=1234, min_y=-64, height=384,
                     biome_table=BiomeTable.from_assets(),
                     placed_features=load_placed_features(),
                     configured_features=load_configured_features())
    stats.features_placed, stats.writes_dropped, stats.unimplemented

This is the stage after terrain and surface rules: `world/feature/features.py`
hands in a chunk of block-state ids and a biome volume, and this module writes
features into it in place. Nothing here is approximate in the ways that matter
-- the RNG
is reseeded per feature exactly as vanilla does, every placement modifier is a
separate step of a nested pipeline, and every f32 is rounded where vanilla
rounds it.

Four things that are easy to get wrong
--------------------------------------

**The random source is reseeded per placed feature, per chunk.** Vanilla builds
one `WorldgenRandom` per chunk, calls `setDecorationSeed(levelSeed, minX, minZ)`
once, and then `setFeatureSeed(decorationSeed, index, step)` before every placed
feature. `setFeatureSeed` reseeds the underlying Xoroshiro with
`decorationSeed + index + 10000 * step`, so features do not share a stream and
the order they run in cannot change their positions. `index` is the feature's
position in the *global* per-step list (`FeatureSorter.buildFeaturesPerStep`),
not its position in this chunk's filtered list -- see `feature_order`.

**`setDecorationSeed` consumes two `nextLong()` draws, and `WorldgenRandom`'s
`nextLong` is two 32-bit draws, not one 64-bit one.** `WorldgenRandom extends
LegacyRandomSource`; its `next(bits)` forwards to the Xoroshiro source but
`nextLong()` is `BitRandomSource`'s default, so it costs two Xoroshiro draws.
Using `Xoroshiro.next_raw()` in its place consumes half as many draws and
derives a different decoration seed for every chunk.

**Positions are a nested pipeline, not a list of filters.** Modifiers run in
order and each one maps one position to zero or more; the configured feature
runs at the end of every surviving path. `count` multiplies the number of
downstream calls, `block_predicate_filter` and `rarity_filter` divide it, and
the RNG is shared between the modifiers *and* the feature. Running the modifiers
as independent filters in one flat pass produces plausible terrain and the wrong
positions, because the draws land in a different order.

**A chunk is not a world.** The seam hands in 16x16 columns, and features are
routinely larger than that: a lake is 16 blocks wide by design, a geode is 32,
and a tree one block inside the border is half outside it. Writes outside the
footprint are dropped and counted (`writes_dropped`), never silently truncated,
because the caller streams chunks independently and cannot complete an
overhanging feature later. Reads outside the footprint have no neighbour to
read, so they see `border_block` (air) and are counted in `reads_outside`:
border behaviour is therefore *not* identical to vanilla's, and the counters are
how you find out how much of a chunk was affected.

Heightmaps
----------
`heightmap` placement modifiers name a `Heightmap.Types`. The worldgen variants
(`*_WG`) are the ones features mean: they are maintained during generation,
while the non-WG types are the client's. The three predicates differ in what
counts as "top":

===========================  ==================================================
`WORLD_SURFACE(_WG)`         topmost non-air block
`OCEAN_FLOOR(_WG)`           topmost non-air block that is not a fluid
`MOTION_BLOCKING(_NO_LEAVES)`topmost block that blocks motion or holds a fluid
===========================  ==================================================

`HeightmapPlacementModifier` then places at `getHeight(type, x, z)`, which is the
block *after* that top block -- the first air above the surface -- and only when
that Y is *strictly above* `min_y`. Reading `getHeight` as "the top block's own
Y" is the expensive mistake here: a tree placed one block into the ground fails
every free-space check, so all 108 placed features with a `heightmap` modifier
quietly stop placing anything. `ore`, `seagrass` and `freeze_top_layer` want the
surface *block*, so they subtract one from the same number.

The `heightmap` argument is one `(16, 16)` array of top-block Y values, read as
the world-surface column; the fluid and motion variants are derived from `blocks`
here, so a caller that passes the array does not have to know which of the five a
modifier asked for.

Where this stops
----------------
`implemented_types()` lists the configured-feature types this module can place;
everything else is recorded in `FeatureStats.unimplemented` and in
`missing_types(...)`, which `tools/check_features.py` prints with the counts the
biome data actually references. There is no silent no-op path.
"""

from __future__ import annotations

import json
import math
import os
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np

try:  # pragma: no cover - numba is a declared dependency
    from numba import njit

    _HAS_NUMBA = True
except ImportError:  # pragma: no cover
    _HAS_NUMBA = False

    def njit(*args, **kwargs):
        def decorator(func):
            return func

        return decorator if not args else decorator(args[0])


from world.vanilla import kernel_rng
from world.vanilla.random_source import (
    LegacyRandom,
    MASK64,
    WorldgenRandom,
    Xoroshiro,
    XoroshiroSplitter,
)
from world.vanilla.simplex import OctaveSimplexNoiseSampler

#: Vanilla's `GenerationStep.Decoration` ordinals, in declaration order. The
#: biome JSON's `features` list is indexed by these: entry 0 is raw generation,
#: entry 9 is vegetal decoration, and a biome that stops early simply has no
#: features in the later steps. The step number is part of the feature seed, so
#: inserting a step anywhere but the end moves every feature in the world.
STEP_NAMES: Tuple[str, ...] = (
    "raw_generation",
    "lakes",
    "local_modifications",
    "underground_structures",
    "surface_structures",
    "strongholds",
    "underground_ores",
    "underground_decoration",
    "fluid_springs",
    "vegetal_decoration",
    "top_layer_modification",
)

#: Biome order used to break ties in `feature_order`. Vanilla feeds
#: `FeatureSorter` the biome source's `possibleBiomes()`: for the overworld that
#: is `OverworldBiomeBuilder`'s insertion order, which is a *code* constant and
#: not in any datapack file. This list is that order, cross-checked against
#: Pumpkin's `generation/feature_order.rs` (whose `OVERWORLD_BIOMES` array is a
#: port of it). A biome missing from the list is ordered after all of these, in
#: `assets/biome.json` order.
OVERWORLD_BIOME_ORDER: Tuple[str, ...] = (
    "minecraft:mushroom_fields",
    "minecraft:deep_frozen_ocean",
    "minecraft:frozen_ocean",
    "minecraft:deep_cold_ocean",
    "minecraft:cold_ocean",
    "minecraft:deep_ocean",
    "minecraft:ocean",
    "minecraft:deep_lukewarm_ocean",
    "minecraft:lukewarm_ocean",
    "minecraft:warm_ocean",
    "minecraft:stony_shore",
    "minecraft:swamp",
    "minecraft:mangrove_swamp",
    "minecraft:snowy_slopes",
    "minecraft:snowy_plains",
    "minecraft:snowy_beach",
    "minecraft:windswept_gravelly_hills",
    "minecraft:grove",
    "minecraft:windswept_hills",
    "minecraft:snowy_taiga",
    "minecraft:windswept_forest",
    "minecraft:taiga",
    "minecraft:plains",
    "minecraft:meadow",
    "minecraft:beach",
    "minecraft:forest",
    "minecraft:old_growth_spruce_taiga",
    "minecraft:flower_forest",
    "minecraft:birch_forest",
    "minecraft:dark_forest",
    "minecraft:pale_garden",
    "minecraft:savanna_plateau",
    "minecraft:savanna",
    "minecraft:jungle",
    "minecraft:badlands",
    "minecraft:desert",
    "minecraft:wooded_badlands",
    "minecraft:jagged_peaks",
    "minecraft:stony_peaks",
    "minecraft:frozen_river",
    "minecraft:river",
    "minecraft:ice_spikes",
    "minecraft:old_growth_pine_taiga",
    "minecraft:sunflower_plains",
    "minecraft:old_growth_birch_forest",
    "minecraft:sparse_jungle",
    "minecraft:bamboo_jungle",
    "minecraft:eroded_badlands",
    "minecraft:windswept_savanna",
    "minecraft:cherry_grove",
    "minecraft:frozen_peaks",
    "minecraft:dripstone_caves",
    "minecraft:lush_caves",
    "minecraft:sulfur_caves",
    "minecraft:deep_dark",
    "minecraft:dappled_forest",
)

#: `state_flags` bit layout from `assets/blocks.json` (Pumpkin's
#: `pumpkin-data/src/block_state.rs`). `IS_SOLID` is vanilla's legacy
#: `BlockState.isSolid()`; `REPLACEABLE` is `canBeReplaced()`.
FLAG_IS_AIR = 1 << 0
FLAG_REPLACEABLE = 1 << 4
FLAG_IS_LIQUID = 1 << 5
FLAG_IS_SOLID = 1 << 6
FLAG_IS_FULL_CUBE = 1 << 7

#: `side_flags` bits: `isFaceSturdy(direction)`.
SIDE_UP = 1 << 1
SIDE_DOWN = 1 << 0

#: A state id is a `uint16`, so this is every id a block volume can hold. The
#: heightmap tables are this long rather than `len(names)` so that an id the flag
#: arrays do not cover reads as air -- which is what `is_air` already answers for
#: one -- instead of raising.
STATE_ID_SPACE = 1 << 16

#: `Direction.values()` order. Several vanilla loops count neighbours in this
#: order and short-circuit, so it is part of the RNG contract.
DIRECTIONS: Tuple[Tuple[int, int, int], ...] = (
    (0, -1, 0),
    (0, 1, 0),
    (0, 0, -1),
    (0, 0, 1),
    (-1, 0, 0),
    (1, 0, 0),
)
HORIZONTAL_DIRECTIONS: Tuple[Tuple[int, int, int], ...] = (
    (0, 0, -1),
    (0, 0, 1),
    (-1, 0, 0),
    (1, 0, 0),
)
#: `Direction.Plane.HORIZONTAL` in its natural order: north, east, south, west.
HORIZONTAL_PLANE: Tuple[Tuple[int, int, int], ...] = (
    (0, 0, -1),
    (1, 0, 0),
    (0, 0, 1),
    (-1, 0, 0),
)

F32 = np.float32


def _assets_path(name: str) -> str:
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return os.path.join(root, "assets", name)


def _canonical(name: str) -> str:
    """`acacia` -> `minecraft:acacia`; already-qualified names pass through.

    The two feature tables disagree about this: `placed_feature.json` is keyed by
    bare names while every cross-reference in `biome.json` is qualified. Both
    spellings resolve to the same entry here.
    """
    return name if ":" in name else f"minecraft:{name}"


def _load_json(name: str, path: Optional[str]) -> Any:
    with open(path or _assets_path(name), encoding="utf-8") as handle:
        return json.load(handle)


# ==================== configured and placed features ====================


class ConfiguredFeature:
    """One entry of `configured_features.json`: a type and its config.

    The config is kept as the raw JSON object rather than a typed tree. The 55
    types have 55 shapes, several of them recursive (a `random_selector` holds
    placed features, which hold configured features), and the datapack is the
    only authority on those shapes -- re-encoding them in Python would be a
    second place for the numbers to drift.
    """

    __slots__ = ("name", "type", "config", "raw")

    def __init__(self, name: str, payload: dict) -> None:
        self.name = name
        self.type = payload.get("type", "").removeprefix("minecraft:")
        self.config = payload.get("config") or {}
        self.raw = payload

    def __repr__(self) -> str:
        return f"<ConfiguredFeature {self.name} type={self.type}>"


class PlacedFeature:
    """One entry of `placed_feature.json`: a configured feature plus placement.

    `feature` is a configured-feature name, or an inline `ConfiguredFeature`
    when the JSON nests one (the selectors and `random_patch` all do). The same
    is true of `placement`: each modifier may itself be an inline `PlacedFeature`
    reference with its own placement list, which is why modifiers are stored raw
    and parsed at run time.
    """

    __slots__ = ("name", "feature", "feature_name", "placement", "raw")

    def __init__(self, name: Optional[str], payload: dict) -> None:
        self.name = name
        self.raw = payload
        feature = payload.get("feature")
        if isinstance(feature, str):
            self.feature = None
            self.feature_name = _canonical(feature)
        elif isinstance(feature, dict):
            self.feature = ConfiguredFeature(name or "<inline>", feature)
            self.feature_name = None
        else:
            raise ValueError(f"placed feature {name!r} has no feature")
        self.placement: Tuple[dict, ...] = tuple(payload.get("placement") or ())

    def __repr__(self) -> str:
        target = self.feature_name or (self.feature.name if self.feature else "?")
        return f"<PlacedFeature {self.name} -> {target} x{len(self.placement)}>"


def load_configured_features(path: Optional[str] = None) -> Dict[str, ConfiguredFeature]:
    """Every configured feature, keyed by its canonical `minecraft:` name."""
    payload = _load_json("configured_features.json", path)
    return {_canonical(name): ConfiguredFeature(_canonical(name), entry)
            for name, entry in payload.items()}


def load_placed_features(path: Optional[str] = None) -> Dict[str, PlacedFeature]:
    """Every placed feature, keyed by its canonical `minecraft:` name."""
    payload = _load_json("placed_feature.json", path)
    return {_canonical(name): PlacedFeature(_canonical(name), entry)
            for name, entry in payload.items()}


# ==================== block facts ====================


class BlockFacts:
    """The four block-level questions features ask, answered from `assets/`.

    Features need more than "what state id is this": they need the block a state
    belongs to (ore targets match blocks, not states), the `state_flags` bits
    (`isAir`, `canBeReplaced`, `isSolid`, `isLiquid`), the fluid of a state
    (including waterlogged blocks, whose fluid is water), and the ability to
    rewrite one property of a state (`axis` on a fallen log, `distance` on a
    leaf, `snowy` on grass). All of it comes from `assets/blocks.json` and
    `assets/properties.json`; nothing is duplicated in this file.

    Loaded lazily and cached. The payload is 14 MB, so this is once per process,
    and `assets/blocks.json` is a *report of one protocol version* -- the note in
    `world/block_states.py` about state ids only meaning anything for the file
    they came from applies here too.
    """

    __slots__ = ("names", "_state_block", "state_flags", "side_flags", "fluid",
                 "_collision", "_props", "_strides", "_by_name", "_default",
                 "_first", "_tags", "_tag_states", "_tag_cache",
                 "_heightmap_masks", "_is_air_mask",
                 "shape_min", "shape_max", "shape_offsets", "shape_ids",
                 "block_slipperiness", "block_velocity_multiplier",
                 "block_jump_velocity_multiplier")

    def __init__(self, blocks_payload: dict, properties_payload: list,
                 fluids_payload: list) -> None:
        records = blocks_payload["blocks"]
        by_hash = {int(record["hash_key"]): record for record in properties_payload}

        total = sum(len(record["states"]) for record in records)
        self.names: List[str] = [""] * total
        self._state_block = np.zeros(total, dtype=np.int32)
        self.state_flags = np.zeros(total, dtype=np.int32)
        self.side_flags = np.zeros(total, dtype=np.int32)
        self._collision = np.zeros(total, dtype=bool)
        # The boxes behind `_collision`, kept instead of collapsed into the bool.
        # Vanilla's `blocksMotion` is "has any collision shape at all", which is
        # everything the feature stage needs and nothing the physics stage needs:
        # stairs, slabs and fences are decided by *where* their boxes are, not by
        # whether they exist. `shape_refs[state]` holds the `shapes` table
        # indices of one state's boxes, in the order the payload lists them.
        shape_refs: List[Tuple[int, ...]] = [()] * total
        # The three per-block material values, expanded to per-state. Physics
        # does not read them this round (no friction, no jumping), but the
        # expansion lives here so the round that does add friction does not have
        # to open `assets/blocks.json` a second time.
        self.block_slipperiness = np.zeros(total, dtype=np.float64)
        self.block_velocity_multiplier = np.ones(total, dtype=np.float64)
        self.block_jump_velocity_multiplier = np.ones(total, dtype=np.float64)
        self._props: List[Tuple[Tuple[str, Tuple[str, ...]], ...]] = []
        self._strides: List[Tuple[int, ...]] = []
        self._by_name: Dict[str, int] = {}
        self._default: List[int] = []
        self._first: List[int] = []

        for index, record in enumerate(records):
            name = _canonical(record["name"])
            states = sorted(record["states"], key=lambda state: state["id"])
            first = states[0]["id"]
            properties = []
            for hash_key in record.get("properties") or ():
                prop = by_hash[int(hash_key)]
                kind = prop["type"]
                if kind == "boolean":
                    values = ("true", "false")
                elif kind == "int":
                    values = tuple(str(v) for v in range(int(prop["min"]), int(prop["max"]) + 1))
                else:
                    values = tuple(prop["values"])
                properties.append((prop["serialized_name"], values))
            strides = [1] * len(properties)
            stride = 1
            for position in range(len(properties) - 1, -1, -1):
                strides[position] = stride
                stride *= len(properties[position][1])

            self._by_name[name] = index
            self._props.append(tuple(properties))
            self._strides.append(tuple(strides))
            self._default.append(int(record["default_state_id"]))
            self._first.append(first)
            slipperiness = float(record.get("slipperiness", 0.6))
            velocity_multiplier = float(record.get("velocity_multiplier", 1.0))
            jump_multiplier = float(record.get("jump_velocity_multiplier", 1.0))
            for offset, state in enumerate(states):
                state_id = first + offset
                self.names[state_id] = name
                self._state_block[state_id] = index
                self.state_flags[state_id] = int(state["state_flags"])
                self.side_flags[state_id] = int(state["side_flags"])
                refs = tuple(int(shape) for shape in state.get("collision_shapes") or ())
                shape_refs[state_id] = refs
                self._collision[state_id] = bool(refs)
                self.block_slipperiness[state_id] = slipperiness
                self.block_velocity_multiplier[state_id] = velocity_multiplier
                self.block_jump_velocity_multiplier[state_id] = jump_multiplier

        # Flat `(offsets, ids)` over `shape_refs`, which is what an njit kernel
        # can index: `shape_ids[shape_offsets[s]:shape_offsets[s + 1]]` are the
        # shapes of state `s`. `offsets` carries a sentinel so the count is a
        # subtraction and never a special case. Out-of-range state ids (a plugin
        # block added after this table was built) have no entry and read as air.
        offsets = np.zeros(total + 1, dtype=np.int64)
        np.cumsum(np.fromiter((len(refs) for refs in shape_refs), dtype=np.int64,
                              count=total), out=offsets[1:])
        self.shape_offsets = offsets
        self.shape_ids = np.fromiter(
            (shape for refs in shape_refs for shape in refs),
            dtype=np.int32, count=int(offsets[-1]),
        )

        # The AABB table itself: `shapes` in `assets/blocks.json`, one row per
        # entry, `min`/`max` split into two `(n, 3)` arrays because that is the
        # layout the collision sweep wants. Shape 0 is the full cube and the
        # only shape at index 0; every other index is a partial box.
        shapes_payload = blocks_payload.get("shapes") or ()
        self.shape_min = np.array([shape["min"] for shape in shapes_payload],
                                  dtype=np.float64).reshape(-1, 3)
        self.shape_max = np.array([shape["max"] for shape in shapes_payload],
                                  dtype=np.float64).reshape(-1, 3)

        # Fluid of a state, inverted from `fluids.json`. Inverting (rather than
        # asking "is this water") is what makes waterlogged blocks work: a
        # waterlogged state appears in the water fluid's state list.
        fluid_of: Dict[int, str] = {}
        for fluid in fluids_payload:
            name = _canonical(fluid["name"])
            for state in fluid.get("states") or ():
                fluid_of[int(state["block_state_id"])] = name
        # `fluids.json` only lists the fluid *blocks*; a waterlogged block holds
        # water without appearing there, so the property has to be consulted.
        # `matching_fluids: minecraft:water` gates every seagrass and kelp in the
        # datapack, and it fails on a waterlogged state without this.
        for index, properties in enumerate(self._props):
            slot = next((position for position, entry in enumerate(properties)
                         if entry[0] == "waterlogged"), None)
            if slot is None:
                continue
            values = properties[slot][1]
            stride = self._strides[index][slot]
            # `first`, not `default`: a block's ids are a contiguous run from its
            # lowest id, and the default state is not that id for 626 of 1166
            # blocks, so a `default` base marks a neighbour's states as water.
            start = self._first[index]
            count = 1
            for entry in properties:
                count *= len(entry[1])
            for state_id in range(start, start + count):
                if (state_id - start) // stride % len(values) == values.index("true"):
                    fluid_of.setdefault(state_id, "minecraft:water")
        self.fluid: Dict[int, str] = fluid_of

        self._tags: Optional[Dict[str, FrozenSet[str]]] = None
        self._tag_states: Dict[str, FrozenSet[int]] = {}
        self._tag_cache: Dict[str, FrozenSet[int]] = {}
        self._heightmap_masks: Dict[str, np.ndarray] = {}
        self._is_air_mask: Optional[np.ndarray] = None

    # ----- lookups -----

    def block_of(self, state_id: int) -> str:
        """Block name of a state id; out-of-range reads are air."""
        if 0 <= state_id < len(self.names):
            return self.names[state_id]
        return "minecraft:air"

    def block_index(self, state_id: int) -> int:
        if 0 <= state_id < len(self._state_block):
            return int(self._state_block[state_id])
        return 0

    def default_state(self, name: str) -> int:
        index = self._by_name.get(_canonical(name))
        return 0 if index is None else self._default[index]

    def state_id(self, codec) -> int:
        """`{Name, Properties}` -> global state id. Unknown blocks read as air.

        A codec with no `Properties` selects the block's *default* state, which
        is what a bare `"minecraft:water"` in a predicate means -- not "any water
        state". Vanilla's `BlockStateCodec` also accepts a plain block name
        wherever it accepts the object form, and spells the fields `id` /
        `properties` in some data, so all four shapes resolve here.
        """
        if isinstance(codec, str):
            codec = {"Name": codec}
        name = codec.get("Name") or codec.get("id") or "minecraft:air"
        if isinstance(name, dict):
            name = name.get("Name") or "minecraft:air"
        name = _canonical(str(name))
        index = self._by_name.get(name)
        if index is None:
            return 0
        properties = codec.get("Properties") or codec.get("properties") or {}
        if not properties:
            return self._default[index]
        position = 0
        for name_i, values in self._props[index]:
            value = properties.get(name_i)
            if value is None:
                continue
            position += values.index(str(value)) * self._strides[index][
                [p[0] for p in self._props[index]].index(name_i)]
        # A block's ids are a contiguous run from its *first* id, and `position`
        # indexes that run -- not the default state. `default_state_id` is not
        # the lowest id for 626 of the 1166 blocks (`oak_log` defaults to 137
        # while its run starts at 136), so adding to it lands on a neighbour's
        # state.
        return self._first[index] + position

    def with_property(self, state_id: int, name: str, value: Any) -> int:
        """Rewrite one property of a state, keeping the others.

        Used all over the feature code and easy to get wrong: a state id is a
        mixed-radix index over the block's properties *in declaration order*,
        with the last property varying fastest, so the change has to be computed
        as a delta on that index rather than on a value.
        """
        index = self.block_index(state_id)
        properties = self._props[index]
        strides = self._strides[index]
        position = state_id - self._first[index]
        for slot, (prop_name, values) in enumerate(properties):
            if prop_name != name:
                continue
            old = (position // strides[slot]) % len(values)
            new = values.index(str(value))
            return state_id + (new - old) * strides[slot]
        return state_id

    def property(self, state_id: int, name: str) -> Optional[str]:
        """Read one property back. Offsets the *block's* run, like `state_id`."""
        index = self.block_index(state_id)
        properties = self._props[index]
        strides = self._strides[index]
        position = state_id - self._first[index]
        for slot, (prop_name, values) in enumerate(properties):
            if prop_name == name:
                return values[(position // strides[slot]) % len(values)]
        return None

    def has_property(self, state_id: int, name: str) -> bool:
        return any(prop == name for prop, _ in self._props[self.block_index(state_id)])

    # ----- flags -----

    def is_air(self, state_id: int) -> bool:
        return not (0 <= state_id < len(self.names)) or bool(
            int(self.state_flags[state_id]) & FLAG_IS_AIR)

    def is_air_mask(self) -> np.ndarray:
        """`is_air` as a `STATE_ID_SPACE` lookup table, for the ore kernel.

        Built by asking `is_air` itself -- 65k calls once, so the table cannot
        drift from the predicate it is derived from. The out-of-range tail is
        air, like the predicate's first clause.
        """
        cached = self._is_air_mask
        if cached is None:
            mask = np.ones(STATE_ID_SPACE, dtype=np.uint8)
            for state in range(len(self.names)):
                mask[state] = 1 if self.is_air(state) else 0
            self._is_air_mask = mask
            cached = mask
        return cached

    def is_solid(self, state_id: int) -> bool:
        """Vanilla's `BlockState.isSolid()` -- the legacy solidity flag."""
        return bool(int(self.state_flags[state_id]) & FLAG_IS_SOLID)

    def is_replaceable(self, state_id: int) -> bool:
        return bool(int(self.state_flags[state_id]) & FLAG_REPLACEABLE)

    def is_liquid(self, state_id: int) -> bool:
        return bool(int(self.state_flags[state_id]) & FLAG_IS_LIQUID)

    def is_full_cube(self, state_id: int) -> bool:
        return bool(int(self.state_flags[state_id]) & FLAG_IS_FULL_CUBE)

    def blocks_motion(self, state_id: int) -> bool:
        """Vanilla's `blocksMotion()`, i.e. the block has collision at all.

        `collision_shapes` non-empty is the same predicate the client uses to
        decide whether an entity can walk through: cobweb and grass have none.

        A state outside this table (a plugin block added after the table was
        built) reads as air, like `is_air` and for the same reason: the shape
        tables in `physics/` have no row for it either.
        """
        if not (0 <= state_id < self._collision.shape[0]):
            return False
        return bool(self._collision[state_id])

    def collision_shape_ids(self, state_id: int) -> Tuple[int, ...]:
        """Shape-table indices of a state's collision boxes; `()` for none.

        This is the raw `collision_shapes` of the state, in payload order, with
        the same meaning `registry.blocks` gives it. Out-of-range states are air.
        """
        if not (0 <= state_id < self._collision.shape[0]):
            return ()
        start = int(self.shape_offsets[state_id])
        end = int(self.shape_offsets[state_id + 1])
        return tuple(int(shape) for shape in self.shape_ids[start:end])

    def collision_shapes(self, state_id: int) -> Tuple[Tuple[float, ...], ...]:
        """A state's collision boxes as `(min_x, min_y, min_z, max_x, max_y, max_z)`.

        The boxes are in block-local coordinates -- `(0,0,0)` to `(1,1,1)` for a
        full cube -- and a block position is added to them by the caller.
        """
        boxes = []
        for shape in self.collision_shape_ids(state_id):
            boxes.append((
                float(self.shape_min[shape, 0]), float(self.shape_min[shape, 1]),
                float(self.shape_min[shape, 2]), float(self.shape_max[shape, 0]),
                float(self.shape_max[shape, 1]), float(self.shape_max[shape, 2]),
            ))
        return tuple(boxes)

    def face_sturdy(self, state_id: int, offset: Tuple[int, int, int]) -> bool:
        """`isFaceSturdy(direction)` for one of the six `DIRECTIONS` offsets."""
        if offset == (0, 1, 0):
            bit = SIDE_UP
        elif offset == (0, -1, 0):
            bit = SIDE_DOWN
        else:
            bit = {
                (0, 0, -1): 1 << 2,
                (0, 0, 1): 1 << 3,
                (-1, 0, 0): 1 << 4,
                (1, 0, 0): 1 << 5,
            }[offset]
        return bool(int(self.side_flags[state_id]) & bit)

    def fluid_of(self, state_id: int) -> str:
        """Fluid name of a state; `minecraft:empty` when it holds none."""
        return self.fluid.get(state_id, "minecraft:empty")

    # ----- heightmaps -----

    def heightmap_mask(self, kind: str) -> np.ndarray:
        """`[state id] -> counts for this heightmap type`, built once per type.

        The three heightmap predicates are functions of the state id alone, so
        asking them per cell -- once per `y`, for every column, on a hot path
        that `height_at` hits 79k times a chunk -- is the same answer as indexing
        one table by the whole column at once. `Surface.top` needs the *topmost*
        match, so it scans this mask rather than the states.

        Built from the same sources the per-cell version read: the `isAir`
        flag, whether a state holds a fluid (which covers waterlogged blocks,
        not just fluid blocks), whether it has a collision shape, and -- for
        the no-leaves variant only -- membership of `minecraft:leaves`.
        Out-of-range ids read as air, matching `is_air`.

        "Holds a fluid" is the fluid's *name* not being `minecraft:empty`, not
        membership of `self.fluid`: `fluids.json` also lists states against the
        empty fluid, and `fluid_of` reports those as `minecraft:empty`.
        """
        cached = self._heightmap_masks.get(kind)
        if cached is not None:
            return cached

        total = len(self.names)
        air = np.ones(STATE_ID_SPACE, dtype=bool)
        air[:total] = (self.state_flags & FLAG_IS_AIR) != 0
        motion = np.zeros(STATE_ID_SPACE, dtype=bool)
        motion[:total] = self._collision
        wet = np.zeros(STATE_ID_SPACE, dtype=bool)
        wet[np.array([s for s, fluid in self.fluid.items()
                      if fluid != "minecraft:empty" and 0 <= s < STATE_ID_SPACE],
                     dtype=np.intp)] = True

        if kind in ("world_surface_wg", "world_surface"):
            mask = ~air
        elif kind in ("ocean_floor_wg", "ocean_floor"):
            mask = ~air & ~wet
        elif kind in ("motion_blocking", "motion_blocking_no_leaves"):
            mask = motion | wet
            if kind == "motion_blocking_no_leaves":
                # Vanilla's predicate is `(isSolid || hasFluid) && !is(LEAVES)`.
                # A leaf block is a full collision cube, so without the tag
                # term the mask calls a canopy ground: the top of a tree
                # becomes "the surface", and every reader of this heightmap
                # answers 4-6 blocks above the dirt it means. For the
                # `motion_blocking` kind the leaves *are* ground -- that is
                # the whole difference between the two types.
                leaves = np.zeros(STATE_ID_SPACE, dtype=bool)
                leaves[np.array(
                    [s for s in self.tag_states("minecraft:leaves")
                     if 0 <= s < STATE_ID_SPACE], dtype=np.intp)] = True
                mask = mask & ~leaves
        else:
            raise NotImplementedError(f"unknown heightmap type {kind!r}")

        self._heightmap_masks[kind] = mask
        return mask

    # ----- tags -----

    def tag_blocks(self, tag_name: str) -> FrozenSet[str]:
        if self._tags is None:
            self._tags = load_block_tags()
        return self._tags.get(_canonical(tag_name), frozenset())

    def tag_states(self, tag_name: str) -> FrozenSet[int]:
        """State ids of every block in a tag. Cached; the expander is not cheap."""
        name = _canonical(tag_name)
        cached = self._tag_cache.get(name)
        if cached is not None:
            return cached
        resolved: set = set()
        for block_name in self.tag_blocks(name):
            index = self._by_name.get(block_name)
            if index is None:
                continue
            start = self._default[index]
            resolved.update(range(start, start + 1))
        # Blocks own contiguous runs, so `range` above is a single state; widen
        # it by walking the block names again through the state table.
        wanted = self.tag_blocks(name)
        for state_id, block_name in enumerate(self.names):
            if block_name in wanted:
                resolved.add(state_id)
        frozen = frozenset(resolved)
        self._tag_cache[name] = frozen
        return frozen

    def in_tag(self, state_id: int, tag_name: str) -> bool:
        return state_id in self.tag_states(tag_name)


_BLOCK_FACTS: Optional[BlockFacts] = None


def block_facts(refresh: bool = False) -> BlockFacts:
    """The shared `BlockFacts`. Building it reads 14 MB of JSON, once."""
    global _BLOCK_FACTS
    if _BLOCK_FACTS is None or refresh:
        _BLOCK_FACTS = BlockFacts(
            _load_json("blocks.json", None),
            _load_json("properties.json", None),
            _load_json("fluids.json", None),
        )
    return _BLOCK_FACTS


def load_block_tags(path: Optional[str] = None) -> Dict[str, FrozenSet[str]]:
    """`assets/tags.json`'s `block` registry, flattened to block names.

    Nested `#tag` references are expanded. A tag that references itself (or a
    cycle) stops rather than recursing forever: the datapack has no such cycle,
    and a plugin that introduces one should not hang the generator.
    """
    payload = _load_json("tags.json", path)
    raw = payload.get("block") or {}
    resolved: Dict[str, FrozenSet[str]] = {}

    def expand(name: str, seen: FrozenSet[str]) -> FrozenSet[str]:
        name = _canonical(name)
        cached = resolved.get(name)
        if cached is not None:
            return cached
        if name in seen:
            return frozenset()
        entry = raw.get(name) or raw.get(name.removeprefix("minecraft:"))
        values: List[str] = []
        if isinstance(entry, dict):
            values = [str(value) for value in entry.get("values") or ()]
        elif isinstance(entry, list):
            values = [str(value) for value in entry]
        members: set = set()
        for value in values:
            if value.startswith("#"):
                members.update(expand(value[1:], seen | {name}))
            else:
                members.add(_canonical(value))
        frozen = frozenset(members)
        resolved[name] = frozen
        return frozen

    for key in raw:
        expand(key, frozenset())
    return resolved


# ==================== block predicates ====================


class BlockPredicate:
    """One `BlockPredicate` codec, callable against a `FeatureContext`.

    Subclasses implement `test(ctx, x, y, z)`. The offset is already folded into
    the coordinates by the caller, which is what vanilla does: every simple
    predicate wraps an `OffsetPredicate` and applies the offset before testing.
    """

    __slots__ = ("offset",)

    def __init__(self, offset: Optional[Sequence[int]] = None) -> None:
        self.offset = tuple(offset) if offset else (0, 0, 0)

    def test(self, ctx: "FeatureContext", x: int, y: int, z: int) -> bool:  # pragma: no cover
        raise NotImplementedError

    def _at(self, x: int, y: int, z: int) -> Tuple[int, int, int]:
        dx, dy, dz = self.offset
        return x + dx, y + dy, z + dz


class AlwaysTrue(BlockPredicate):
    __slots__ = ()

    def test(self, ctx, x, y, z) -> bool:
        return True


class MatchingBlocks(BlockPredicate):
    __slots__ = ("states",)

    def __init__(self, states: Sequence[int], offset=None) -> None:
        super().__init__(offset)
        self.states = tuple(states)

    def test(self, ctx, x, y, z) -> bool:
        return ctx.get(*self._at(x, y, z)) in self.states


class MatchingBlockTag(BlockPredicate):
    __slots__ = ("tag",)

    def __init__(self, tag: str, offset=None) -> None:
        super().__init__(offset)
        # The datapack spells tag references in predicates and in `replaceable`
        # with a leading `#`; `tags.json` is keyed without it.
        self.tag = _canonical(tag.lstrip("#"))

    def test(self, ctx, x, y, z) -> bool:
        return ctx.facts.in_tag(ctx.get(*self._at(x, y, z)), self.tag)


class MatchingFluids(BlockPredicate):
    __slots__ = ("fluids",)

    def __init__(self, fluids: Sequence[str], offset=None) -> None:
        super().__init__(offset)
        self.fluids = tuple(_canonical(fluid) for fluid in fluids)

    def test(self, ctx, x, y, z) -> bool:
        return ctx.facts.fluid_of(ctx.get(*self._at(x, y, z))) in self.fluids


class HasSturdyFace(BlockPredicate):
    __slots__ = ("direction",)

    def __init__(self, direction: Sequence[int], offset=None) -> None:
        super().__init__(offset)
        self.direction = tuple(direction)

    def test(self, ctx, x, y, z) -> bool:
        return ctx.facts.face_sturdy(ctx.get(*self._at(x, y, z)), self.direction)


class Solid(BlockPredicate):
    __slots__ = ()

    def test(self, ctx, x, y, z) -> bool:
        return ctx.facts.is_solid(ctx.get(*self._at(x, y, z)))


class Replaceable(BlockPredicate):
    __slots__ = ()

    def test(self, ctx, x, y, z) -> bool:
        return ctx.facts.is_replaceable(ctx.get(*self._at(x, y, z)))


class InsideWorldBounds(BlockPredicate):
    __slots__ = ()

    def test(self, ctx, x, y, z) -> bool:
        _, yy, _ = self._at(x, y, z)
        return ctx.min_y <= yy < ctx.min_y + ctx.height


class Unobstructed(BlockPredicate):
    """Vanilla's `unobstructed` predicate: always true in worldgen.

    It exists for the live-world path, where it checks that no entity occupies
    the position. During generation there are none, so the codec never refuses.
    """

    __slots__ = ()

    def test(self, ctx, x, y, z) -> bool:
        return True


class AnyOf(BlockPredicate):
    __slots__ = ("predicates",)

    def __init__(self, predicates: Sequence[BlockPredicate], offset=None) -> None:
        super().__init__(offset)
        self.predicates = tuple(predicates)

    def test(self, ctx, x, y, z) -> bool:
        return any(predicate.test(ctx, *self._at(x, y, z)) for predicate in self.predicates)


class AllOf(BlockPredicate):
    __slots__ = ("predicates",)

    def __init__(self, predicates: Sequence[BlockPredicate], offset=None) -> None:
        super().__init__(offset)
        self.predicates = tuple(predicates)

    def test(self, ctx, x, y, z) -> bool:
        return all(predicate.test(ctx, *self._at(x, y, z)) for predicate in self.predicates)


class Not(BlockPredicate):
    __slots__ = ("predicate",)

    def __init__(self, predicate: BlockPredicate, offset=None) -> None:
        super().__init__(offset)
        self.predicate = predicate

    def test(self, ctx, x, y, z) -> bool:
        return not self.predicate.test(ctx, *self._at(x, y, z))


class WouldSurvive(BlockPredicate):
    """`would_survive`, approximated from the tags the datapack ships.

    Vanilla asks the block's own `canSurvive`. Block behaviour is Java code and
    is not in any asset file, so the rules are reconstructed per family in
    `_plant_rule` from the tags the data does ship (plus Pumpkin's port of the
    vanilla classes for the ones that are shape tests rather than substrate
    tests). The 1.21.11 data gates saplings, sugar cane, cactus and firefly
    bushes through this predicate.

    A block the rule set does not model is *recorded*, not guessed:
    `stats.unimplemented` gets a `would_survive:<block>` entry and the predicate
    answers False, so a new datapack use shows up in the check tool instead of
    quietly changing terrain.
    """

    __slots__ = ("state",)

    #: `VegetationBlock.mayPlaceOn` is `#minecraft:dirt`; `SaplingBlock` widens
    #: it to farmland, which `assets/tags.json` ships as a block rather than a
    #: tag -- see `_is_substrate`.
    _SOIL_TAGS = ("minecraft:dirt",)
    _EXTRA_SOIL = ("minecraft:farmland",)
    #: `SugarCaneBlock.canSurvive`: sand or dirt below, water beside.
    _CANE_TAGS = ("minecraft:sand", "minecraft:dirt")
    #: Plants that are plain `VegetationBlock`s: dirt below, nothing else. A
    #: *placement* may widen that to clay (see `_plant_rule`), but `canSurvive`
    #: itself asks only about `#minecraft:dirt`.
    _VEGETATION = ("minecraft:firefly_bush",)

    def __init__(self, state: dict, offset=None) -> None:
        super().__init__(offset)
        self.state = state

    def test(self, ctx, x, y, z) -> bool:
        """`WouldSurviveBlockPredicate`: ask the state's own `canSurvive`.

        `state` is a `BlockStateCodec`, which is either a bare block name or a
        `{Name, Properties}` object -- `BlockFacts.state_id` accepts both.
        """
        block = ctx.facts.block_of(ctx.facts.state_id(self.state))
        if _survives(ctx, block, self._at(x, y, z)):
            return True
        # Only an *unmodelled* family is recorded. A modelled family answering
        # False is the predicate working: it means the position has no valid
        # substrate, not that this module is missing a rule. Recording those
        # would drown the real gaps in noise.
        if _plant_rule(block) is None:
            ctx.stats.unimplemented[f"would_survive:{block}"] += 1
        return False


#: `VegetationBlock` and its subclasses whose `canSurvive` is `#minecraft:dirt`.
#: Vanilla asks the block's own `mayPlaceOn`, which block behaviour is Java code
#: for; the datapack ships no table of it. These are the families the worldgen
#: data actually places through `simple_block`, each mirrored from Pumpkin's
#: port of the vanilla class (see `_plant_rule`).
_FAMILY_FLOWERS = (
    "minecraft:allium", "minecraft:azure_bluet", "minecraft:blue_orchid",
    "minecraft:cornflower", "minecraft:dandelion", "minecraft:lily_of_the_valley",
    "minecraft:orange_tulip", "minecraft:oxeye_daisy", "minecraft:pink_tulip",
    "minecraft:poppy", "minecraft:red_tulip", "minecraft:white_tulip",
    "minecraft:closed_eyeblossom", "minecraft:open_eyeblossom",
)
#: `DoublePlantBlock`: lilac/peony/rose_bush/sunflower/tall grass/large fern.
#: `#dirt` below *and* air in the upper half.
_FAMILY_DOUBLE_PLANTS = (
    "minecraft:lilac", "minecraft:peony", "minecraft:rose_bush",
    "minecraft:sunflower", "minecraft:tall_grass", "minecraft:large_fern",
)
_FAMILY_DRY_VEGETATION = (
    "minecraft:dead_bush", "minecraft:short_dry_grass", "minecraft:tall_dry_grass",
)
_FAMILY_FLOWERBED = ("minecraft:pink_petals", "minecraft:wildflowers")

#: `FireflyBushBlock` and `MossyCarpetBlock` are `VegetationBlock`s, and
#: `VegetationBlock.mayPlaceOn` is `#minecraft:dirt`. The worldgen data also
#: lands them on clay, which `assets/tags.json` ships as a block, not a tag.
_VEGETATION_SOIL_EXTRA = ("minecraft:clay",)


def _plant_rule(block: str):
    """`canSurvive` for one block, or None when this stage does not model it.

    Keyed by the vanilla class that implements `canSurvive`, because that is what
    the rule is: the substrate tags a family accepts, plus the shape checks that
    are not substrate tests at all (the double plant's head room).
    """
    if block.endswith("_sapling") or block == "minecraft:mangrove_propagule":
        # SaplingBlock widens `#dirt` with farmland.
        return ("soil", WouldSurvive._SOIL_TAGS + WouldSurvive._EXTRA_SOIL)
    if block == "minecraft:sugar_cane":
        return ("sugar_cane", ())
    if block == "minecraft:cactus":
        return ("cactus", ())
    if block in _FAMILY_DRY_VEGETATION:
        # DryVegetationBlock: `#dry_vegetation_may_place_on` (sand, terracotta,
        # dirt and friends).
        return ("soil", ("minecraft:dry_vegetation_may_place_on",))
    if block in _FAMILY_FLOWERBED:
        # FlowerbedBlock: `#dirt` or farmland.
        return ("soil", WouldSurvive._SOIL_TAGS + WouldSurvive._EXTRA_SOIL)
    if block == "minecraft:leaf_litter":
        # LeafLitterBlock: the block below has a sturdy upward face.
        return ("sturdy_up", ())
    if block == "minecraft:lily_pad":
        # LilyPadBlock: water or ice below, and this cell is not blocked.
        return ("lily_pad", ())
    if block == "minecraft:small_dripleaf":
        # SmallDripleafBlock: `#small_dripleaf_placeable` is clay and moss.
        return ("soil", ("minecraft:small_dripleaf_placeable",))
    if block == "minecraft:spore_blossom":
        # SporeBlossomBlock hangs: it asks about the block *above*.
        return ("spore_blossom", ())
    if block in ("minecraft:crimson_roots", "minecraft:warped_roots",
                 "minecraft:nether_sprouts"):
        # RootsBlock: `#dirt`, `#nylium` or soul soil (a block here, not a tag).
        return ("soil", ("minecraft:dirt", "minecraft:nylium",
                         "minecraft:soul_soil"))
    if block in ("minecraft:moss_carpet", "minecraft:pale_moss_carpet"):
        # CarpetBlock: anything that is not air.
        return ("not_air", ())
    if block in ("minecraft:azalea", "minecraft:flowering_azalea"):
        # AzaleaBlock accepts more than `#dirt`: `#azalea_grows_on` also carries
        # sand, red sand, terracotta, snow and moss.
        return ("soil", ("minecraft:azalea_grows_on",))
    if block in ("minecraft:short_grass", "minecraft:fern", "minecraft:bush",
                 "minecraft:firefly_bush", "minecraft:sweet_berry_bush",
                 "minecraft:moss_block", "minecraft:pale_moss_block") \
            or block in _FAMILY_FLOWERS:
        # VegetationBlock: `#dirt`, widened by the placement path with clay.
        return ("soil", WouldSurvive._SOIL_TAGS + _VEGETATION_SOIL_EXTRA)
    if block in _FAMILY_DOUBLE_PLANTS:
        return ("double_plant", ())
    if block == "minecraft:bamboo":
        # `BambooBlock` extends `PlantBlockBase`, whose `canPlantOnTop` is
        # `#minecraft:supports_vegetation`; `#bamboo_plantable_on` is the tag the
        # data carries for bamboo, and it also admits an existing stalk, which a
        # stalk continuing upward needs.
        return ("soil", ("minecraft:bamboo_plantable_on",))
    if block in ("minecraft:brown_mushroom", "minecraft:red_mushroom"):
        # MushroomBlock.canSurvive also wants light below 13, which this stage
        # cannot answer (it has no light channel), so only the substrate half is
        # enforced. `mayPlaceOn` is `is_solid`, which is the same flag the data
        # carries for every solid-render block; the mushroom data lands on water
        # and on open air without it.
        return ("mushroom", ())


def _is_substrate(facts, state: int, entry: str) -> bool:
    """Is `state` the block, or a member of the tag, named by `entry`?

    Vanilla spells every one of these as a tag reference, but `assets/tags.json`
    does not ship a tag for a set of one: there is no `minecraft:farmland`,
    `minecraft:clay` or `minecraft:soul_soil` key, and `in_tag` answers False for
    a missing tag rather than raising. A bare block name is therefore accepted
    here as a one-member set, which is what vanilla's `BlockTags.FARMLAND` and
    friends amount to.
    """
    if facts.in_tag(state, entry):
        return True
    bare = entry.removeprefix("minecraft:")
    return facts.block_of(state) == f"minecraft:{bare}"


def _survives(ctx, block: str, pos) -> bool:
    """Vanilla's `BlockState.canSurvive` at `pos`, for the families modelled.

    Answers False for a family this stage does not model *and* records it in
    `stats.unimplemented`, so an unmodelled plant shows up in the check tool
    instead of quietly changing terrain. `_can_place_at` keeps its old
    permissive answer for those, which is where the pre-existing gaps are.
    """
    facts = ctx.facts
    rule = _plant_rule(block)
    if rule is None:
        return False
    kind, tags = rule
    x, y, z = pos
    # Every modelled family but the spore blossom hangs off the block below; a
    # column with nothing under it can never satisfy any of them.
    if kind != "spore_blossom":
        below = ctx.get(x, y - 1, z)
        if facts.is_air(below):
            return False
    if kind == "soil":
        below = ctx.get(x, y - 1, z)
        return any(_is_substrate(facts, below, tag) for tag in tags)
    if kind == "sugar_cane":
        below = ctx.get(x, y - 1, z)
        if facts.block_of(below) != "minecraft:sugar_cane" and \
                not any(_is_substrate(facts, below, tag)
                        for tag in WouldSurvive._CANE_TAGS):
            return False
        # Vanilla scans the ring *below* the cane's own cell, not the cane's cell.
        for offset in HORIZONTAL_PLANE:
            neighbour = ctx.get(x + offset[0], y - 1, z + offset[2])
            if facts.fluid_of(neighbour) == "minecraft:water" or \
                    facts.block_of(neighbour) == "minecraft:frosted_ice":
                return True
        return False
    if kind == "cactus":
        below = ctx.get(x, y - 1, z)
        if facts.block_of(below) != "minecraft:cactus" and \
                not facts.in_tag(below, "minecraft:sand"):
            return False
        for offset in HORIZONTAL_PLANE:
            neighbour = ctx.get(x + offset[0], y, z + offset[2])
            if facts.is_solid(neighbour) or \
                    facts.block_of(neighbour) == "minecraft:lava":
                return False
        return True
    if kind == "double_plant":
        below = ctx.get(x, y - 1, z)
        if not any(facts.in_tag(below, tag) for tag in WouldSurvive._SOIL_TAGS):
            return False
        return facts.is_air(ctx.get(x, y + 1, z))
    if kind == "sturdy_up":
        return facts.face_sturdy(ctx.get(x, y - 1, z), (0, 1, 0))
    if kind == "lily_pad":
        below = ctx.get(x, y - 1, z)
        if facts.fluid_of(below) != "minecraft:water" and \
                facts.block_of(below) not in ("minecraft:ice",
                                              "minecraft:frosted_ice"):
            return False
        return not facts.blocks_motion(ctx.get(x, y, z))
    if kind == "spore_blossom":
        above = ctx.get(x, y + 1, z)
        return facts.is_solid(above) and \
            not facts.in_tag(above, "minecraft:leaves")
    if kind == "not_air":
        return not facts.is_air(ctx.get(x, y - 1, z))
    if kind == "mushroom":
        # `MushroomBlock.mayPlaceOn` is `isSolidRender`; the light arm is not
        # modelled, so this is the substrate half of `canSurvive` only.
        return facts.is_solid(ctx.get(x, y - 1, z))
    return False


def parse_block_predicate(payload: dict) -> BlockPredicate:
    """Parse one `BlockPredicate` codec object."""
    kind = str(payload.get("type", "")).removeprefix("minecraft:")
    offset = payload.get("offset")
    facts = block_facts()

    if kind in ("always_true", ""):
        return AlwaysTrue(offset)
    if kind == "matching_blocks":
        blocks = payload.get("blocks")
        if isinstance(blocks, str):
            states = [facts.state_id({"Name": blocks})]
        else:
            states = [facts.state_id({"Name": name}) for name in blocks]
        return MatchingBlocks(states, offset)
    if kind == "matching_block_tag":
        return MatchingBlockTag(str(payload["tag"]), offset)
    if kind == "matching_fluids":
        fluids = payload.get("fluids")
        if isinstance(fluids, str):
            fluids = [fluids]
        return MatchingFluids(fluids, offset)
    if kind == "has_sturdy_face":
        return HasSturdyFace(_direction_offset(str(payload["direction"])), offset)
    if kind == "solid":
        return Solid(offset)
    if kind == "replaceable":
        return Replaceable(offset)
    if kind == "would_survive":
        return WouldSurvive(payload["state"], offset)
    if kind == "inside_world_bounds":
        return InsideWorldBounds(offset)
    if kind == "unobstructed":
        return Unobstructed(offset)
    if kind == "any_of":
        return AnyOf([parse_block_predicate(p) for p in payload["predicates"]], offset)
    if kind == "all_of":
        return AllOf([parse_block_predicate(p) for p in payload["predicates"]], offset)
    if kind == "not":
        return Not(parse_block_predicate(payload["predicate"]), offset)
    raise NotImplementedError(f"unsupported block predicate {kind!r}")


def _direction_offset(name: str) -> Tuple[int, int, int]:
    return {
        "down": (0, -1, 0),
        "up": (0, 1, 0),
        "north": (0, 0, -1),
        "south": (0, 0, 1),
        "west": (-1, 0, 0),
        "east": (1, 0, 0),
    }[name.removeprefix("minecraft:")]


# ==================== rule tests (ore targets) ====================


class RuleTest:
    """One ore `RuleTest`: which block a vein is allowed to replace."""

    __slots__ = ("kind", "tag", "block", "probability")

    def __init__(self, kind: str, tag: Optional[str] = None,
                 block: Optional[str] = None, probability: float = 0.0) -> None:
        self.kind = kind
        self.tag = tag
        self.block = block
        self.probability = probability

    def test(self, ctx: "FeatureContext", state_id: int, rng) -> bool:
        facts = ctx.facts
        if self.kind == "tag_match":
            return facts.in_tag(state_id, self.tag or "")
        if self.kind == "block_match":
            return facts.block_of(state_id) == self.block
        if self.kind == "random_block_match":
            # The draw happens whether or not the block matches, because
            # `RandomBlockMatchRuleTest` tests the block first and only then
            # rolls -- getting the order wrong desynchronises every vein.
            if facts.block_of(state_id) != self.block:
                return False
            return rng.next_f32() < F32(self.probability)
        if self.kind == "always_true":
            return True
        raise NotImplementedError(f"unsupported rule test {self.kind!r}")


def parse_rule_test(payload: dict) -> RuleTest:
    kind = str(payload.get("predicate_type", payload.get("type", ""))).removeprefix("minecraft:")
    if kind == "tag_match":
        return RuleTest(kind, tag=_canonical(str(payload["tag"]).lstrip("#")))
    if kind in ("block_match", "random_block_match"):
        return RuleTest(kind, block=_canonical(str(payload["block"])),
                        probability=float(payload.get("probability", 0.0)))
    if kind == "always_true":
        return RuleTest(kind)
    raise NotImplementedError(f"unsupported rule test {kind!r}")


# ==================== block state providers ====================


class StateProvider:
    """A `BlockStateProvider` codec. `get` may consume RNG draws."""

    __slots__ = ()

    def get(self, ctx: "FeatureContext", rng, x: int, y: int, z: int) -> int:  # pragma: no cover
        raise NotImplementedError


class SimpleStateProvider(StateProvider):
    __slots__ = ("state",)

    def __init__(self, state: int) -> None:
        self.state = state

    def get(self, ctx, rng, x, y, z) -> int:
        return self.state


class WeightedStateProvider(StateProvider):
    __slots__ = ("entries", "total")

    def __init__(self, entries: Sequence[Tuple[int, int]]) -> None:
        self.entries = tuple(entries)
        self.total = sum(weight for _, weight in entries)

    def get(self, ctx, rng, x, y, z) -> int:
        if self.total <= 0:
            return 0
        roll = rng.next_bounded_i32(self.total)
        for state, weight in self.entries:
            roll -= weight
            if roll < 0:
                return state
        return self.entries[-1][0]


class RotatedStateProvider(StateProvider):
    """`rotated_block_provider`: a random rotation of one state.

    Only the two shapes the datapack uses are handled: an `axis` pillar picks
    one of x/y/z, a horizontal `facing` picks one of the four horizontals. The
    draw is `nextInt(3)` / `nextInt(4)` in both cases, which is what makes
    `pile_hay` reproducible.
    """

    __slots__ = ("state", "kind")

    def __init__(self, state: int, facts: BlockFacts) -> None:
        self.state = state
        if facts.has_property(state, "axis"):
            self.kind = "axis"
        elif facts.has_property(state, "facing"):
            self.kind = "facing"
        else:
            self.kind = ""

    def get(self, ctx, rng, x, y, z) -> int:
        if self.kind == "axis":
            return ctx.facts.with_property(
                self.state, "axis", ("x", "y", "z")[rng.next_bounded_i32(3)])
        if self.kind == "facing":
            return ctx.facts.with_property(
                self.state, "facing",
                ("north", "east", "south", "west")[rng.next_bounded_i32(4)])
        return self.state


class RandomizedIntStateProvider(StateProvider):
    __slots__ = ("source", "property", "values")

    def __init__(self, source: StateProvider, property_name: str, values) -> None:
        self.source = source
        self.property = property_name
        self.values = values

    def get(self, ctx, rng, x, y, z) -> int:
        state = self.source.get(ctx, rng, x, y, z)
        return ctx.facts.with_property(state, self.property, str(self.values.get(rng)))


class NoiseStateProvider(StateProvider):
    """`noise_provider`: pick a state by a `NormalNoise` value at the position.

    The noise is seeded from the config's own `seed` through a *legacy* LCG, and
    the parameters come from the config (they are inlined in the datapack). The
    sampler is built once per provider.
    """

    __slots__ = ("states", "_noise", "scale")

    def __init__(self, payload: dict, states: Sequence[int]) -> None:
        from world.vanilla.perlin import NormalNoise

        seed = int(payload["seed"])
        noise = payload["noise"]
        self.states = tuple(states)
        self.scale = float(payload.get("scale", 1.0))
        self._noise = NormalNoise(
            _LegacySplitter(LegacyRandom(seed)),
             int(noise.get("firstOctave", 0)),
            tuple(float(a) for a in noise.get("amplitudes") or (1.0,)),
        )

    def _value(self, x: int, y: int, z: int) -> float:
        return float(self._noise.get(x * self.scale, y * self.scale, z * self.scale))

    def get(self, ctx, rng, x, y, z) -> int:
        value = self._value(x, y, z)
        # `NoiseBlockStateProvider.getStateByValue`: map [-1, 1] onto the state
        # list and clamp *below* 1.0, because the index is a truncated product.
        scaled = min(max((1.0 + value) / 2.0, 0.0), 0.9999)
        return self.states[int(scaled * len(self.states))]


class NoiseThresholdStateProvider(StateProvider):
    __slots__ = ("default_state", "low_states", "high_states", "threshold",
                 "high_chance", "_noise", "scale")

    def __init__(self, payload: dict) -> None:
        from world.vanilla.perlin import NormalNoise

        facts = block_facts()
        seed = int(payload["seed"])
        noise = payload["noise"]
        self.default_state = facts.state_id(payload["default_state"])
        self.low_states = tuple(facts.state_id(s) for s in payload["low_states"])
        self.high_states = tuple(facts.state_id(s) for s in payload["high_states"])
        self.threshold = float(payload["threshold"])
        self.high_chance = F32(payload["high_chance"])
        self.scale = float(payload.get("scale", 1.0))
        self._noise = NormalNoise(
            _LegacySplitter(LegacyRandom(seed)),
            int(noise.get("firstOctave", 0)),
            tuple(float(a) for a in noise.get("amplitudes") or (1.0,)),
        )

    def get(self, ctx, rng, x, y, z) -> int:
        value = float(self._noise.get(x * self.scale, y * self.scale, z * self.scale))
        if value < self.threshold:
            return self.low_states[rng.next_bounded_i32(len(self.low_states))]
        if rng.next_f32() < self.high_chance:
            return self.high_states[rng.next_bounded_i32(len(self.high_states))]
        return self.default_state


class DualNoiseStateProvider(StateProvider):
    """`dual_noise_provider`: a per-column variety list chosen by a slow noise.

    The slow noise indexes into a list built at *relative* positions
    (`i * 54545, 0, i * 34234`), not at the queried position -- that is the
    trick that makes the variety coherent per column, and sampling at the query
    position instead produces a state list that changes every block.
    """

    __slots__ = ("base_states", "variety", "slow_scale", "_noise", "_slow_noise")

    def __init__(self, payload: dict) -> None:
        from world.vanilla.perlin import NormalNoise

        self.base_states = tuple(block_facts().state_id(s) for s in payload["states"])
        self.variety = (float(payload["variety"][0]), float(payload["variety"][1]))
        seed = int(payload["seed"])
        noise = payload["noise"]
        slow = payload["slow_noise"]
        self.slow_scale = float(payload.get("slow_scale", 1.0))
        self._noise = NormalNoise(
            _LegacySplitter(LegacyRandom(seed)),
            int(noise.get("firstOctave", 0)),
            tuple(float(a) for a in noise.get("amplitudes") or (1.0,)),
        )
        self._slow_noise = NormalNoise(
            _LegacySplitter(LegacyRandom(seed)),
            int(slow.get("firstOctave", 0)),
            tuple(float(a) for a in slow.get("amplitudes") or (1.0,)),
        )

    def _slow(self, x: float, y: float, z: float) -> float:
        return float(self._slow_noise.get(x * self.slow_scale, y * self.slow_scale,
                                          z * self.slow_scale))

    def get(self, ctx, rng, x, y, z) -> int:
        raise NotImplementedError(
            "dual_noise_provider is not implemented: its state list changes with "
            "height, and the position it is sampled at is the block, not the origin"
        )


class RuleBasedStateProvider(StateProvider):
    __slots__ = ("rules", "fallback")

    def __init__(self, rules, fallback: Optional[StateProvider]) -> None:
        self.rules = tuple(rules)
        self.fallback = fallback

    def get(self, ctx, rng, x, y, z) -> int:
        for predicate, provider in self.rules:
            if predicate.test(ctx, x, y, z):
                return provider.get(ctx, rng, x, y, z)
        if self.fallback is not None:
            return self.fallback.get(ctx, rng, x, y, z)
        # No rule matched and no fallback: vanilla keeps the existing block.
        return ctx.get(x, y, z)


class _LegacySplitter:
    """Adapter giving a `LegacyRandom` the `next_splitter` API `NormalNoise` wants.

    Vanilla's `NormalNoise` constructor draws two longs from the source and
    builds a Xoroshiro from them, so a legacy-seeded noise provider still ends up
    on a Xoroshiro. Two `nextLong()`s on an LCG are *four* 32-bit draws; taking
    one raw 64-bit draw instead changes every state the provider picks.
    """

    __slots__ = ("_legacy",)

    def __init__(self, legacy: LegacyRandom) -> None:
        self._legacy = legacy

    def next_splitter(self) -> XoroshiroSplitter:
        return XoroshiroSplitter(self._legacy.next_i64(), self._legacy.next_i64())


def parse_state_provider(payload: dict) -> StateProvider:
    """Parse one `BlockStateProvider` codec object."""
    kind = str(payload.get("type", "")).removeprefix("minecraft:")
    facts = block_facts()
    if kind == "simple_state_provider":
        return SimpleStateProvider(facts.state_id(payload["state"]))
    if kind == "weighted_state_provider":
        return WeightedStateProvider([
            (facts.state_id(entry["data"]), int(entry["weight"]))
            for entry in payload["entries"]
        ])
    if kind == "rotated_block_provider":
        return RotatedStateProvider(facts.state_id(payload["state"]), facts)
    if kind == "randomized_int_state_provider":
        return RandomizedIntStateProvider(
            parse_state_provider(payload["source"]),
            str(payload["property"]).removeprefix("minecraft:"),
            parse_int_provider(payload["values"]),
        )
    if kind == "noise_provider":
        return NoiseStateProvider(payload, [facts.state_id(s) for s in payload["states"]])
    if kind == "noise_threshold_provider":
        return NoiseThresholdStateProvider(payload)
    if kind == "dual_noise_provider":
        return DualNoiseStateProvider(payload)
    if kind == "rule_based_state_provider":
        rules = [(parse_block_predicate(rule["if_true"]),
                  parse_state_provider(rule["then"])) for rule in payload["rules"]]
        fallback = payload.get("fallback")
        return RuleBasedStateProvider(
            rules, parse_state_provider(fallback) if fallback else None)
    raise NotImplementedError(f"unsupported block state provider {kind!r}")


# ==================== int and height providers ====================


class IntProvider:
    """An `IntProvider` codec: `get(rng) -> int`."""

    __slots__ = ()

    def get(self, rng) -> int:  # pragma: no cover
        raise NotImplementedError


class ConstantInt(IntProvider):
    __slots__ = ("value",)

    def __init__(self, value: int) -> None:
        self.value = int(value)

    def get(self, rng) -> int:
        return self.value


class UniformInt(IntProvider):
    __slots__ = ("minimum", "maximum")

    def __init__(self, minimum: int, maximum: int) -> None:
        self.minimum = int(minimum)
        self.maximum = int(maximum)

    def get(self, rng) -> int:
        # Vanilla's `nextInt(max - min + 1) + min`: inclusive at both ends.
        if self.maximum <= self.minimum:
            return self.minimum
        return self.minimum + rng.next_bounded_i32(self.maximum - self.minimum + 1)


class BiasedToBottomInt(IntProvider):
    slots__ = ("minimum", "maximum")

    def __init__(self, minimum: int, maximum: int) -> None:
        self.minimum = int(minimum)
        self.maximum = int(maximum)

    def get(self, rng) -> int:
        # Two nested bounded draws, not one triangular one. The inner bound is
        # itself random, which is what biases the result toward the minimum.
        high = UniformInt(self.minimum, self.maximum).get(rng)
        return UniformInt(self.minimum, high).get(rng)


class ClampedNormalInt(IntProvider):
    __slots__ = ("mean", "deviation", "minimum", "maximum")

    def __init__(self, mean: float, deviation: float, minimum: int, maximum: int) -> None:
        self.mean = mean
        self.deviation = deviation
        self.minimum = int(minimum)
        self.maximum = int(maximum)

    def get(self, rng) -> int:
        value = F32(self.mean) + F32(rng.next_gaussian()) * F32(self.deviation)
        clamped = min(max(float(value), float(self.minimum)), float(self.maximum))
        return int(round(clamped))


class WeightedListInt(IntProvider):
    __slots__ = ("entries", "total")

    def __init__(self, entries: Sequence[Tuple[IntProvider, int]]) -> None:
        self.entries = tuple(entries)
        self.total = sum(weight for _, weight in entries)

    def get(self, rng) -> int:
        if self.total <= 0:
            return 0
        roll = rng.next_bounded_i32(self.total)
        for provider, weight in self.entries:
            roll -= weight
            if roll < 0:
                return provider.get(rng)
        return self.entries[-1][0].get(rng)


class ClampedInt(IntProvider):
    """`clamped`: a source provider with its result forced into a range.

    It still consumes the source's draws when the value is out of range, so the
    clamp is not free: skipping the draw for an out-of-range value desynchronises
    every following modifier.
    """

    __slots__ = ("source", "minimum", "maximum")

    def __init__(self, source: IntProvider, minimum: int, maximum: int) -> None:
        self.source = source
        self.minimum = int(minimum)
        self.maximum = int(maximum)

    def get(self, rng) -> int:
        return min(max(self.source.get(rng), self.minimum), self.maximum)


def parse_int_provider(payload) -> IntProvider:
    """Parse an `IntProvider`, including the bare-integer form.

    `"count": 2` is a constant provider in vanilla's codec, and it is the most
    common form in the data (58 of the 126 `count` modifiers); treating a bare
    int as an error is the fastest way to implement almost nothing.
    """
    if isinstance(payload, int):
        return ConstantInt(payload)
    if isinstance(payload, float) and float(payload).is_integer():
        return ConstantInt(int(payload))
    kind = str(payload.get("type", "minecraft:constant")).removeprefix("minecraft:")
    if kind == "constant":
        return ConstantInt(int(payload["value"]))
    if kind == "uniform":
        return UniformInt(int(payload["min_inclusive"]), int(payload["max_inclusive"]))
    if kind == "biased_to_bottom":
        return BiasedToBottomInt(int(payload["min_inclusive"]), int(payload["max_inclusive"]))
    if kind == "clamped_normal":
        return ClampedNormalInt(float(payload["mean"]), float(payload["deviation"]),
                                int(payload["min_inclusive"]), int(payload["max_inclusive"]))
    if kind == "clamped":
        return ClampedInt(parse_int_provider(payload["source"]),
                          int(payload["min_inclusive"]), int(payload["max_inclusive"]))
    if kind == "weighted_list":
        return WeightedListInt([
            (parse_int_provider(entry["data"]), int(entry["weight"]))
            for entry in payload["distribution"]
        ])
    raise NotImplementedError(f"unsupported int provider {kind!r}")


class HeightProvider:
    """A `HeightProvider`: `get(rng, min_y, height) -> y`."""

    __slots__ = ()

    def get(self, rng, min_y: int, height: int) -> int:  # pragma: no cover
        raise NotImplementedError


class UniformHeight(HeightProvider):
    __slots__ = ("minimum", "maximum")

    def __init__(self, minimum, maximum) -> None:
        self.minimum = minimum
        self.maximum = maximum

    def get(self, rng, min_y, height) -> int:
        low = resolve_y_offset(self.minimum, min_y, height)
        high = resolve_y_offset(self.maximum, min_y, height)
        if high <= low:
            return low
        return low + rng.next_bounded_i32(high - low + 1)


class TrapezoidHeight(HeightProvider):
    __slots__ = ("minimum", "maximum", "plateau")

    def __init__(self, minimum, maximum, plateau: int) -> None:
        self.minimum = minimum
        self.maximum = maximum
        self.plateau = int(plateau)

    def get(self, rng, min_y, height) -> int:
        low = resolve_y_offset(self.minimum, min_y, height)
        high = resolve_y_offset(self.maximum, min_y, height)
        if high <= low:
            return low
        span = high - low
        if self.plateau >= span:
            return low + rng.next_bounded_i32(span + 1)
        left = (span - self.plateau) // 2
        right = span - left
        return low + rng.next_bounded_i32(right + 1) + rng.next_bounded_i32(left + 1)


class VeryBiasedToBottomHeight(HeightProvider):
    """Three nested bounded draws; the middle one is exclusive at the top.

    Vanilla's third draw is `nextInt(l - i) + i` (note: no `+ 1`), so the
    intermediate value is *excluded*. Making it inclusive widens the
    distribution by one block and moves every geode and amethyst cluster.
    """

    __slots__ = ("minimum", "maximum", "inner")

    def __init__(self, minimum, maximum, inner: int) -> None:
        self.minimum = minimum
        self.maximum = maximum
        self.inner = max(1, int(inner))

    def get(self, rng, min_y, height) -> int:
        low = resolve_y_offset(self.minimum, min_y, height)
        high = resolve_y_offset(self.maximum, min_y, height)
        if high - low - self.inner < 0:
            return low
        middle = low + self.inner + rng.next_bounded_i32(high - low - self.inner + 1)
        if middle - low <= 0:
            return low
        upper = low + rng.next_bounded_i32(middle - low)
        return low + rng.next_bounded_i32(upper - low + 1)


def resolve_y_offset(offset, min_y: int, height: int) -> int:
    """`VerticalAnchor.resolveY`: absolute, above_bottom, or below_top.

    `below_top` counts *down from the top block*, i.e. `min_y + height - 1 - n`,
    so `below_top: 0` is the highest block in the world and not one above it.
    """
    if isinstance(offset, int):
        return offset
    if "absolute" in offset:
        return int(offset["absolute"])
    if "above_bottom" in offset:
        return min_y + int(offset["above_bottom"])
    if "below_top" in offset:
        return min_y + height - 1 - int(offset["below_top"])
    raise NotImplementedError(f"unknown vertical anchor {offset!r}")


def parse_height_provider(payload) -> HeightProvider:
    if isinstance(payload, int):
        return UniformHeight({"absolute": payload}, {"absolute": payload})
    kind = str(payload.get("type", "")).removeprefix("minecraft:")
    if kind == "uniform":
        return UniformHeight(payload["min_inclusive"], payload["max_inclusive"])
    if kind == "trapezoid":
        return TrapezoidHeight(payload["min_inclusive"], payload["max_inclusive"],
                               int(payload.get("plateau", 0)))
    if kind == "very_biased_to_bottom":
        return VeryBiasedToBottomHeight(payload["min_inclusive"], payload["max_inclusive"],
                                        int(payload.get("inner", 1)))
    raise NotImplementedError(f"unsupported height provider {kind!r}")


# ==================== the world view features write into ====================


@dataclass
class FeatureStats:
    """What one `decorate` call did, including what it could not do.

    `writes_dropped` is the counter that matters most: a feature that spans a
    chunk border cannot be completed by this call, so the writes that fall
    outside are counted and discarded rather than pretended away.
    """

    steps_run: int = 0
    features_attempted: int = 0
    features_placed: int = 0
    writes_dropped: int = 0
    features_run: int = 0
    placements_nested: int = 0
    writes_outside_xz: int = 0
    writes_outside_y: int = 0
    reads_outside: int = 0
    unimplemented: Counter = field(default_factory=Counter)
    attempts_by_feature: Counter = field(default_factory=Counter)
    runs_by_feature: Counter = field(default_factory=Counter)
    by_step: Counter = field(default_factory=Counter)

    def merge(self, other: "FeatureStats") -> None:
        self.steps_run += other.steps_run
        self.features_attempted += other.features_attempted
        self.features_placed += other.features_placed
        self.writes_dropped += other.writes_dropped
        self.features_run += other.features_run
        self.placements_nested += other.placements_nested
        self.writes_outside_xz += other.writes_outside_xz
        self.writes_outside_y += other.writes_outside_y
        self.reads_outside += other.reads_outside
        self.unimplemented.update(other.unimplemented)
        self.attempts_by_feature.update(other.attempts_by_feature)
        self.runs_by_feature.update(other.runs_by_feature)
        self.by_step.update(other.by_step)

    def summary(self) -> Dict[str, Any]:
        return {
            "steps_run": self.steps_run,
            "features_run": self.features_run,
            "features_attempted": self.features_attempted,
            "features_placed": self.features_placed,
            "placements_nested": self.placements_nested,
            "writes_dropped": self.writes_dropped,
            "reads_outside": self.reads_outside,
            "unimplemented": dict(self.unimplemented),
        }


class FeatureContext:
    """The chunk a feature is being placed into, plus its counters.

    Reads outside the 16x16 footprint have no neighbour to consult and return
    `border_block` (air) -- vanilla reads a 3x3 chunk region instead, so a
    feature straddling a border sees a slightly different world here. That is a
    deliberate seam limitation, not an oversight, and `reads_outside` says how
    often it happened.
    """

    __slots__ = ("chunk_x", "chunk_z", "blocks", "biomes", "min_y", "height",
                 "biome_table", "facts", "stats", "border_block", "surface",
                 "sea_level", "rng", "feature_name", "preset", "configured",
                 "_biome_quart_base", "_biome_rows")

    def __init__(self, chunk_x: int, chunk_z: int, blocks: np.ndarray,
                 biomes: np.ndarray, min_y: int, height: int, biome_table,
                 stats: FeatureStats, surface: "Surface", border_block: int = 0,
                 preset: str = "overworld") -> None:
        self.chunk_x = chunk_x
        self.chunk_z = chunk_z
        self.blocks = blocks
        self.biomes = biomes
        self.min_y = min_y
        self.height = height
        self.biome_table = biome_table
        self.facts = block_facts()
        self.stats = stats
        self.border_block = border_block
        self.surface = surface
        self.preset = preset
        self.sea_level = sea_level_for(preset)
        self.rng: Optional[WorldgenRandom] = None
        self.feature_name: Optional[str] = None
        self.configured: Dict[str, ConfiguredFeature] = {}
        self._biome_quart_base = min_y >> 2
        self._biome_rows = biomes.shape[0]

    # ----- blocks -----

    def get(self, x: int, y: int, z: int) -> int:
        lx = x - (self.chunk_x << 4)
        lz = z - (self.chunk_z << 4)
        if not (0 <= lx < 16 and 0 <= lz < 16):
            self.stats.reads_outside += 1
            return self.border_block
        row = y - self.min_y
        if not (0 <= row < self.height):
            self.stats.reads_outside += 1
            return self.border_block
        return int(self.blocks[row, lz, lx])

    def set(self, x: int, y: int, z: int, state: int) -> bool:
        """Write a state, or count the write as dropped. Never raises."""
        lx = x - (self.chunk_x << 4)
        lz = z - (self.chunk_z << 4)
        if not (0 <= lx < 16 and 0 <= lz < 16):
            self.stats.writes_dropped += 1
            self.stats.writes_outside_xz += 1
            return False
        row = y - self.min_y
        if not (0 <= row < self.height):
            self.stats.writes_dropped += 1
            self.stats.writes_outside_y += 1
            return False
        self.blocks[row, lz, lx] = state
        return True

    def is_air(self, x: int, y: int, z: int) -> bool:
        return self.facts.is_air(self.get(x, y, z))

    def in_height(self, y: int) -> bool:
        return self.min_y <= y < self.min_y + self.height

    # ----- biomes -----

    def biome_at(self, x: int, y: int, z: int) -> str:
        """Biome name at a block position, via its quart cell.

        Quart cells are 4x4x4 and the volume is indexed `[y, z, x]`; asking for a
        position outside the chunk clamps to the edge cell (there is no
        neighbour), which only matters for features that read across a border.
        """
        qy = (y >> 2) - self._biome_quart_base
        qx = ((x - (self.chunk_x << 4)) >> 2)
        qz = ((z - (self.chunk_z << 4)) >> 2)
        qy = 0 if qy < 0 else (self._biome_rows - 1 if qy >= self._biome_rows else qy)
        qx = 0 if qx < 0 else (3 if qx > 3 else qx)
        qz = 0 if qz < 0 else (3 if qz > 3 else qz)
        numeric = int(self.biomes[qy, qz, qx])
        return self.biome_table.names.get(numeric, "minecraft:plains")

    def biome_has_feature(self, biome_name: str, feature_name: str) -> bool:
        for step in self.biome_table.features.get(biome_name, ()):
            if feature_name in step:
                return True
        return False

    # ----- heightmaps -----

    def height_at(self, kind: str, x: int, z: int) -> int:
        """Vanilla's `ChunkAccess.getHeight(type, x, z)`: the first Y *above* the surface.

        It is the block *after* the topmost matching one, not the top block
        itself. That is the whole reason a `heightmap` placement modifier puts a
        tree in the air above the grass rather than inside it, and why
        `ore`/`seagrass`/`freeze_top_layer` subtract one when they want the
        surface block. An empty column answers `min_y`, which the `heightmap`
        modifier rejects (it needs a Y strictly above `min_y`).
        """
        return self.surface.top(self, kind, x, z)

    def temperature(self, biome_name: str, x: int, y: int, z: int) -> float:
        return biome_temperature(self.biome_table, biome_name, x, y, z, self.sea_level)


class Surface:
    """The four heightmap predicates, computed per column from `blocks`.

    `top` answers vanilla's `getHeight`: the first Y *above* the topmost matching
    block, `min_y` for a column with no match. Computed live, like vanilla's
    chunk heightmaps, which `setBlock` keeps up to date as features write -- a
    tree's leaves count for `ocean_floor_wg`, and a later filter in the same
    feature has to see them. Caching per column instead froze whatever the first
    query saw, so two call orders (a heightmap probe before the trees, say)
    answered different values for the same chunk state.

    The caller's `heightmap` argument, when given, is the world-surface column of
    *top block* Y values (what a heightmap array means elsewhere in the repo), so
    it is used for `world_surface(_wg)` only. The fluid and motion variants
    cannot be derived from it: a "top non-air Y" array has already thrown away
    whether that block was water.
    """

    __slots__ = ("_given", "_ctx", "_arrays")

    def __init__(self, given: Optional[np.ndarray]) -> None:
        self._given = given
        self._ctx: Optional[FeatureContext] = None
        self._arrays: Dict[str, np.ndarray] = {}

    def bind(self, ctx: FeatureContext) -> None:
        self._ctx = ctx

    def columns(self, ctx: FeatureContext, kind: str) -> np.ndarray:
        """The 16x16 grid of `top` answers, computed in one sweep, then frozen.

        `_place_ore`'s heightmap probe asks about up to 27x27 columns per call
        and ~240 probes run per chunk, so it reads a grid rather than walking
        `top` per column. The grid freezes at first call, which is the right
        semantics for that probe and not a general promise: the ore step's
        features replace solid with solid and cannot move a
        `world_surface`/`ocean_floor` row, so for every probe the frozen grid
        equals the live answer. Both the interpreted probe and the kernel read
        this same grid, which is what keeps them identical. Out-of-footprint
        columns are not in the grid; callers answer `min_y` for them, like
        `top`'s outside branch.
        """
        cached = self._arrays.get(kind)
        if cached is not None:
            return cached
        if kind in ("world_surface_wg", "world_surface") and self._given is not None:
            values = (self._given.astype(np.int64) + 1).astype(np.int32)
        else:
            mask = ctx.facts.heightmap_mask(kind)[ctx.blocks]
            reversed_mask = mask[::-1]
            has = reversed_mask.any(axis=0)
            first = reversed_mask.argmax(axis=0)
            values = np.where(has, ctx.min_y + ctx.height - first,
                              ctx.min_y).astype(np.int32)
        self._arrays[kind] = values
        return values

    def top(self, ctx: FeatureContext, kind: str, x: int, z: int) -> int:
        lx = x - (ctx.chunk_x << 4)
        lz = z - (ctx.chunk_z << 4)
        if not (0 <= lx < 16 and 0 <= lz < 16):
            ctx.stats.reads_outside += 1
            return ctx.min_y
        if kind in ("world_surface_wg", "world_surface") and self._given is not None:
            # The caller's array is the world-surface column: the *top block*,
            # which is what a heightmap array means everywhere else in the repo.
            return int(self._given[lz, lx]) + 1
        # The topmost matching row is the reversed mask's first True. The walk
        # this replaces started at the top of the world (y=383) and stepped down
        # to the terrain, ~530 predicate calls for a column whose top block sits
        # near y=64.
        mask = ctx.facts.heightmap_mask(kind)[ctx.blocks[:, lz, lx]]
        if not mask.any():
            return ctx.min_y
        return ctx.min_y + ctx.height - int(mask[::-1].argmax())


_SEA_LEVELS: Optional[Dict[str, int]] = None


def sea_level_for(preset: str) -> int:
    """The preset's sea level, from `chunk_gen_settings.json`.

    `freeze_top_layer` and `Biome.getTemperature` both need it, and it is a
    datapack number (63 in the overworld, 32 in the nether): hardcoding 63 would
    silently change snow in any pack that moves the sea.
    """
    global _SEA_LEVELS
    if _SEA_LEVELS is None:
        payload = _load_json("chunk_gen_settings.json", None)
        levels: Dict[str, int] = {}
        for name, entry in payload.items():
            if isinstance(entry, dict):
                levels[_canonical(name)] = int(entry.get("sea_level", 63))
        _SEA_LEVELS = levels
    return _SEA_LEVELS.get(_canonical(preset), 63)


_TEMPERATURE_NOISE: Optional[OctaveSimplexNoiseSampler] = None
_FROZEN_OCEAN_NOISE: Optional[OctaveSimplexNoiseSampler] = None
_FOLIAGE_NOISE: Optional[OctaveSimplexNoiseSampler] = None


def temperature_noise() -> OctaveSimplexNoiseSampler:
    """`Biome.TEMPERATURE_NOISE`: octave simplex, legacy LCG seed 1234."""
    global _TEMPERATURE_NOISE
    if _TEMPERATURE_NOISE is None:
        _TEMPERATURE_NOISE = OctaveSimplexNoiseSampler(LegacyRandom(1234), (0,))
    return _TEMPERATURE_NOISE


def frozen_ocean_noise() -> OctaveSimplexNoiseSampler:
    """`Biome.FROZEN_OCEAN_NOISE`: octaves -2, -1, 0, legacy LCG seed 3456."""
    global _FROZEN_OCEAN_NOISE
    if _FROZEN_OCEAN_NOISE is None:
        _FROZEN_OCEAN_NOISE = OctaveSimplexNoiseSampler(LegacyRandom(3456), (-2, -1, 0))
    return _FROZEN_OCEAN_NOISE


def foliage_noise() -> OctaveSimplexNoiseSampler:
    """`Biome.FOLIAGE_NOISE`: octave simplex, legacy LCG seed 2345.

    `noise_based_count` and `noise_threshold_count` sample this at
    `(x / factor, z / factor)`; it is also the second half of the frozen-ocean
    temperature modifier. Vanilla's field is an `OctaveSimplexNoiseSampler`
    built from `new LegacyRandomSource(2345L)`, i.e. the legacy LCG draws each
    octave's simplex permutation directly -- not a `NormalNoise`, which would
    hash octave names off a Xoroshiro instead.
    """
    global _FOLIAGE_NOISE
    if _FOLIAGE_NOISE is None:
        _FOLIAGE_NOISE = OctaveSimplexNoiseSampler(LegacyRandom(2345), (0,))
    return _FOLIAGE_NOISE


def biome_temperature(biome_table, biome_name: str, x: int, y: int, z: int,
                      sea_level: int) -> float:
    """`Biome.getTemperature(pos, seaLevel)`, base temperature included.

    Above sea level the temperature falls with height, which is why a snowy peak
    does not freeze a lake at y=200 in a temperate biome. The modifier is the
    frozen-ocean one, and it is only applied by biomes whose JSON asks for it.
    """
    base = float(biome_table.temperature.get(biome_name, 0.5))
    entry = getattr(biome_table, "raw", {}).get(biome_name, {}) if hasattr(biome_table, "raw") else {}
    modifier = str(entry.get("temperature_modifier", "none")).removeprefix("minecraft:")
    if y > sea_level:
        noise = temperature_noise().sample(x / 8.0, z / 8.0, False) * 8.0
        base -= (noise + y - sea_level) * 0.05 / 40.0
    if modifier == "frozen":
        return biome_table.frozen_temperature(x, z, base)
    return base


# ==================== placement modifiers ====================


class PlacementModifier:
    """One placement modifier: `positions(ctx, rng, pos) -> Iterator[position]`."""

    __slots__ = ("config",)

    def __init__(self, config: dict) -> None:
        self.config = config

    def positions(self, ctx: FeatureContext, rng: WorldgenRandom,
                  pos: Tuple[int, int, int]) -> Iterator[Tuple[int, int, int]]:  # pragma: no cover
        raise NotImplementedError


class BiomeModifier(PlacementModifier):
    """`biome`: keep the position only if this biome lists this placed feature.

    Vanilla implements this through `PlacementContext.getTopFeature()`: the
    modifier asks whether the biome at the position has the *currently placing*
    feature in any of its decoration steps. That is the second half of biome
    gating -- the first half is that only biomes present in the chunk contribute
    features at all -- and it is why a feature shared by two biomes still only
    runs where the chunk's own biome volume agrees.
    """

    __slots__ = ()

    def positions(self, ctx, rng, pos):
        if ctx.feature_name is None:
            yield pos
            return
        if ctx.biome_has_feature(ctx.biome_at(*pos), ctx.feature_name):
            yield pos


class InSquareModifier(PlacementModifier):
    """`in_square`: pick a column uniformly inside the chunk's 16x16 footprint.

    The offsets are added to `pos`, which is a *world* position -- the chunk
    origin here. Yielding the raw 0..15 offsets instead works for chunk (0, 0)
    and silently places everything in chunk (0, 0) for every other chunk, which
    is the kind of bug that looks like "features only exist near spawn".
    """

    __slots__ = ()

    def positions(self, ctx, rng, pos):
        yield (pos[0] + rng.next_bounded_i32(16), pos[1],
               pos[2] + rng.next_bounded_i32(16))


class CountModifier(PlacementModifier):
    __slots__ = ("count",)

    def __init__(self, config: dict) -> None:
        super().__init__(config)
        self.count = parse_int_provider(config["count"])

    def positions(self, ctx, rng, pos):
        for _ in range(max(0, self.count.get(rng))):
            yield pos


class HeightmapModifier(PlacementModifier):
    __slots__ = ("kind",)

    def __init__(self, config: dict) -> None:
        super().__init__(config)
        self.kind = str(config["heightmap"]).lower()

    def positions(self, ctx, rng, pos):
        y = ctx.height_at(self.kind, pos[0], pos[2])
        if y > ctx.min_y:
            yield (pos[0], y, pos[2])


class HeightRangeModifier(PlacementModifier):
    __slots__ = ("provider",)

    def __init__(self, config: dict) -> None:
        super().__init__(config)
        self.provider = parse_height_provider(config["height"])

    def positions(self, ctx, rng, pos):
        yield (pos[0], self.provider.get(rng, ctx.min_y, ctx.height), pos[2])


class BlockPredicateFilterModifier(PlacementModifier):
    __slots__ = ("predicate",)

    def __init__(self, config: dict) -> None:
        super().__init__(config)
        self.predicate = parse_block_predicate(config["predicate"])

    def positions(self, ctx, rng, pos):
        if self.predicate.test(ctx, *pos):
            yield pos


class RarityFilterModifier(PlacementModifier):
    """`rarity_filter`: one `nextFloat()` against `1 / chance`."""

    __slots__ = ("chance",)

    def __init__(self, config: dict) -> None:
        super().__init__(config)
        self.chance = int(config["chance"])

    def positions(self, ctx, rng, pos):
        if rng.next_f32() < F32(1.0) / F32(self.chance):
            yield pos


class SurfaceWaterDepthModifier(PlacementModifier):
    __slots__ = ("max_depth",)

    def __init__(self, config: dict) -> None:
        super().__init__(config)
        self.max_depth = int(config["max_water_depth"])

    def positions(self, ctx, rng, pos):
        world_top = ctx.height_at("world_surface_wg", pos[0], pos[2])
        ocean_floor = ctx.height_at("ocean_floor_wg", pos[0], pos[2])
        if world_top - ocean_floor <= self.max_depth:
            yield pos


class RandomOffsetModifier(PlacementModifier):
    """`random_offset`: *both* x and z come from the xz spread provider.

    Two separate draws, in x-then-z order, and no `nextInt` for a sign: the
    provider already carries the range. Reusing one draw for both axes makes
    every offset diagonal.
    """

    __slots__ = ("xz_spread", "y_spread")

    def __init__(self, config: dict) -> None:
        super().__init__(config)
        self.xz_spread = parse_int_provider(config["xz_spread"])
        self.y_spread = parse_int_provider(config["y_spread"])

    def positions(self, ctx, rng, pos):
        x = pos[0] + self.xz_spread.get(rng)
        y = pos[1] + self.y_spread.get(rng)
        z = pos[2] + self.xz_spread.get(rng)
        yield (x, y, z)


class EnvironmentScanModifier(PlacementModifier):
    """`environment_scan`: walk in one direction to the first matching block.

    Three traps. The search condition is tested *before* the first step, the
    height bound is checked after each step (so a scan can leave the world and
    stop), and the final target test after the loop is what lets a scan that
    exhausted `max_steps` still succeed. Dropping any of the three moves ocean
    floors and cave ceilings by a block.
    """

    __slots__ = ("direction", "target", "allowed", "max_steps")

    def __init__(self, config: dict) -> None:
        super().__init__(config)
        self.direction = _direction_offset(str(config["direction_of_search"]))
        self.target = parse_block_predicate(config["target_condition"])
        allowed = config.get("allowed_search_condition")
        self.allowed = parse_block_predicate(allowed) if allowed else AlwaysTrue()
        self.max_steps = int(config["max_steps"])

    def positions(self, ctx, rng, pos):
        if not self.allowed.test(ctx, *pos):
            return
        x, y, z = pos
        dx, dy, dz = self.direction
        for _ in range(self.max_steps):
            if self.target.test(ctx, x, y, z):
                yield (x, y, z)
                return
            x += dx
            y += dy
            z += dz
            if not ctx.in_height(y):
                return
            if not self.allowed.test(ctx, x, y, z):
                break
        if self.target.test(ctx, x, y, z):
            yield (x, y, z)


class CountOnEveryLayerModifier(PlacementModifier):
    """`count_on_every_layer`: `count` columns per "layer" of cave openings.

    The layer index is how many spawnable-to-solid transitions below the column
    are being skipped; the loop ends when a whole round finds nothing, which is
    what makes the layer count data-dependent rather than `max_steps`.
    """

    __slots__ = ("count",)

    def __init__(self, config: dict) -> None:
        super().__init__(config)
        self.count = parse_int_provider(config["count"])

    def positions(self, ctx, rng, pos):
        layer = 0
        while True:
            found = False
            for _ in range(max(0, self.count.get(rng))):
                x = pos[0] + rng.next_bounded_i32(16)
                z = pos[2] + rng.next_bounded_i32(16)
                top = ctx.height_at("motion_blocking", x, z)
                y = self._find_pos(ctx, x, top, z, layer)
                if y == 2**31 - 1:
                    continue
                found = True
                yield (x, y, z)
            layer += 1
            if not found:
                return

    @staticmethod
    def _find_pos(ctx: FeatureContext, x: int, y: int, z: int, target_layer: int) -> int:
        facts = ctx.facts
        found = 0
        current = ctx.get(x, y, z)
        for row in range(y, ctx.min_y, -1):
            next_state = ctx.get(x, row - 1, z)
            if (not _blocks_spawn(facts, next_state) and _blocks_spawn(facts, current)
                    and facts.block_of(next_state) != "minecraft:bedrock"):
                if found == target_layer:
                    return row
                found += 1
            current = next_state
        return 2**31 - 1


def _blocks_spawn(facts: BlockFacts, state: int) -> bool:
    block = facts.block_of(state)
    return facts.is_air(state) or block in ("minecraft:water", "minecraft:lava")


class NoiseThresholdCountModifier(PlacementModifier):
    """`noise_threshold_count`: two constant counts, chosen by foliage noise."""

    __slots__ = ("noise_level", "below", "above")

    def __init__(self, config: dict) -> None:
        super().__init__(config)
        self.noise_level = float(config["noise_level"])
        self.below = int(config["below_noise"])
        self.above = int(config["above_noise"])

    def positions(self, ctx, rng, pos):
        noise = foliage_noise().sample(pos[0] / 200.0, pos[2] / 200.0, False)
        count = self.below if noise < self.noise_level else self.above
        for _ in range(max(0, count)):
            yield pos


class NoiseBasedCountModifier(PlacementModifier):
    """`noise_based_count`: `ceil((noise + offset) * ratio)` copies.

    The ceil is on the *product*, in f64, and the noise is sampled at
    `(x / factor, z / factor)` -- dividing by the factor, not multiplying.
    """

    __slots__ = ("ratio", "factor", "offset")

    def __init__(self, config: dict) -> None:
        super().__init__(config)
        self.ratio = int(config["noise_to_count_ratio"])
        self.factor = float(config["noise_factor"])
        self.offset = float(config["noise_offset"])

    def positions(self, ctx, rng, pos):
        noise = foliage_noise().sample(pos[0] / self.factor, pos[2] / self.factor, False)
        count = math.ceil((noise + self.offset) * self.ratio)
        for _ in range(max(0, count)):
            yield pos


class SurfaceRelativeThresholdModifier(PlacementModifier):
    __slots__ = ("kind", "minimum", "maximum")

    def __init__(self, config: dict) -> None:
        super().__init__(config)
        self.kind = str(config["heightmap"]).lower()
        self.minimum = config.get("min_inclusive")
        self.maximum = config.get("max_inclusive")

    def positions(self, ctx, rng, pos):
        y = ctx.height_at(self.kind, pos[0], pos[2])
        low = y + (int(self.minimum) if self.minimum is not None else -(2**31))
        high = y + (int(self.maximum) if self.maximum is not None else 2**31)
        if low <= pos[1] <= high:
            yield pos


class FixedPlacementModifier(PlacementModifier):
    """`fixed_placement`: absolute positions, in the order the data gives them."""

    __slots__ = ("positions_list",)

    def __init__(self, config: dict) -> None:
        super().__init__(config)
        self.positions_list = tuple(tuple(int(v) for v in entry)
                                    for entry in config["positions"])

    def positions(self, ctx, rng, pos):
        yield from self.positions_list


_PLACEMENT_MODIFIERS = {
    "biome": BiomeModifier,
    "in_square": InSquareModifier,
    "count": CountModifier,
    "heightmap": HeightmapModifier,
    "height_range": HeightRangeModifier,
    "block_predicate_filter": BlockPredicateFilterModifier,
    "rarity_filter": RarityFilterModifier,
    "surface_water_depth_filter": SurfaceWaterDepthModifier,
    "random_offset": RandomOffsetModifier,
    "environment_scan": EnvironmentScanModifier,
    "count_on_every_layer": CountOnEveryLayerModifier,
    "noise_threshold_count": NoiseThresholdCountModifier,
    "noise_based_count": NoiseBasedCountModifier,
    "surface_relative_threshold_filter": SurfaceRelativeThresholdModifier,
    "fixed_placement": FixedPlacementModifier,
}


def parse_placement_modifier(payload: dict) -> PlacementModifier:
    kind = str(payload.get("type", "")).removeprefix("minecraft:")
    factory = _PLACEMENT_MODIFIERS.get(kind)
    if factory is None:
        raise NotImplementedError(f"unsupported placement modifier {kind!r}")
    return factory(payload)


# ==================== configured features ====================


def _pos_add(pos, dx, dy, dz):
    return (pos[0] + dx, pos[1] + dy, pos[2] + dz)


def _iter_box(low, high):
    """`BlockPos.betweenClosed` order: x, then y, then z."""
    for x in range(low[0], high[0] + 1):
        for y in range(low[1], high[1] + 1):
            for z in range(low[2], high[2] + 1):
                yield (x, y, z)


# ----- feature-size and tree helpers -----


class FeatureSize:
    """`FeatureSize`: how wide the free space around a trunk must be at height y."""

    __slots__ = ("kind", "limit", "lower", "upper", "middle", "upper_limit",
                 "min_clipped_height")

    def __init__(self, payload: dict) -> None:
        self.kind = str(payload["type"]).removeprefix("minecraft:")
        self.limit = int(payload.get("limit", 1))
        self.lower = int(payload.get("lower_size", 0))
        self.upper = int(payload.get("upper_size", 0))
        self.middle = int(payload.get("middle_size", 0))
        self.upper_limit = int(payload.get("upper_limit", 0))
        self.min_clipped_height = payload.get("min_clipped_height")

    def radius(self, height: int, y: int) -> int:
        if y < self.limit:
            return self.lower
        if self.kind == "three_layers_feature_size":
            if y >= height - self.upper_limit:
                return self.upper
            return self.middle
        return self.upper


class TreeNode:
    __slots__ = ("center", "foliage_radius", "giant_trunk")

    def __init__(self, center, foliage_radius: int = 0, giant_trunk: bool = False) -> None:
        self.center = center
        self.foliage_radius = foliage_radius
        self.giant_trunk = giant_trunk


# ----- configured feature implementations -----
#
# Each takes `(ctx, config, rng, pos)` and returns whether it changed anything.
# The signatures are uniform so the dispatcher at the bottom can be a dict.


def _place_ore_python(ctx, config, rng, pos) -> bool:
    """`ore`: an ellipsoid vein, walked as `size` interpolated spheres.

    The vein is a chain of spheres whose centres and radii are laid out first
    (consuming `size` `nextFloat`/`nextDouble` draws), thinned by a
    sphere-occlusion sweep, and only then rasterised. Rasterising as you lay out
    the chain gives the same blocks in a different RNG order, so the *next*
    feature in the chunk moves.

    This is the interpreted reference: `_place_ore` runs the same thing through
    `_ore_kernel` when numba is installed, and `tools/check_ore_veins.py`
    compares the two paths block for block and draw for draw.
    """
    size = int(config["size"])
    targets = [(parse_rule_test(entry["target"]), block_facts().state_id(entry["state"]))
               for entry in config["targets"]]
    discard = F32(config.get("discard_chance_on_air_exposure", 0.0))

    angle = F32(rng.next_f32() * F32(math.pi))
    spread = F32(F32(size) / F32(8.0))
    # Vanilla: `Mth.ceil((size / 16.0F * 2.0F + 1.0F) / 2.0F)`. The `+ 1` is
    # *inside* the ceil, so the half-thickness is not `ceil(size / 16)`.
    widened = F32(F32(size) / F32(16.0) * F32(2.0))
    thickness = int(math.ceil(float(F32((widened + F32(1.0)) / F32(2.0)))))

    dx1 = pos[0] + math.sin(float(angle)) * float(spread)
    dx2 = pos[0] - math.sin(float(angle)) * float(spread)
    dz1 = pos[2] + math.cos(float(angle)) * float(spread)
    dz2 = pos[2] - math.cos(float(angle)) * float(spread)
    y1 = pos[1] + rng.next_bounded_i32(3) - 2
    y2 = pos[1] + rng.next_bounded_i32(3) - 2

    x_bound = pos[0] - math.ceil(float(spread)) - thickness
    y_bound = pos[1] - 2 - thickness
    z_bound = pos[2] - math.ceil(float(spread)) - thickness
    box = 2 * (math.ceil(float(spread)) + thickness)
    vertical = 2 * (2 + thickness)

    # The probe reads `Surface.columns`' frozen grid, exactly like the kernel:
    # `top` is live now, and reading it here would make the answer depend on how
    # many trees have already written leaves. Out-of-footprint columns answer
    # `min_y` and count in `reads_outside`, one per examined column up to the
    # first hit -- the same accounting the kernel does.
    heights = ctx.surface.columns(ctx, "ocean_floor_wg")
    chunk_min_x = ctx.chunk_x << 4
    chunk_min_z = ctx.chunk_z << 4
    for probe_x in range(box + 1):
        for probe_z in range(box + 1):
            lx = x_bound + probe_x - chunk_min_x
            lz = z_bound + probe_z - chunk_min_z
            if 0 <= lx < 16 and 0 <= lz < 16:
                top = int(heights[lz, lx])
            else:
                ctx.stats.reads_outside += 1
                top = ctx.min_y
            if y_bound <= top:
                return _ore_vein_python(ctx, rng, targets, discard, size,
                                        (dx1, y1, dz1), (dx2, y2, dz2),
                                        x_bound, y_bound, z_bound, box + 1, vertical + 1)
    return False


def _ore_vein_python(ctx, rng, targets, discard, size, start, end,
                     x_bound, y_bound, z_bound, horizontal, vertical) -> bool:
    samples = []
    for index in range(size):
        fraction = F32(F32(index) / F32(size))
        centre_x = _lerp_f64(float(fraction), start[0], end[0])
        centre_y = _lerp_f64(float(fraction), start[1], end[1])
        centre_z = _lerp_f64(float(fraction), start[2], end[2])
        raw = F32(rng.next_f64() * size / 16.0)
        # `f32::midpoint((sin(PI * f) + 1) * raw, 1.0)`: the sine is of the f32
        # angle, and the `+ 1` happens before the halving.
        sine = F32(math.sin(float(F32(F32(math.pi) * fraction))))
        radius = float(F32((sine + F32(1.0)) * raw + F32(1.0))) / 2.0
        samples.append([centre_x, centre_y, centre_z, radius])

    # Sphere occlusion: a sample swallowed by a later, larger one is dropped.
    # The squares are multiplies, like the Java/Rust reference -- `x ** 2` goes
    # through libm `pow` and rounds up to one ULP off `x * x`, which can flip
    # the drop decision when the two sides nearly tie.
    for first in range(size - 1):
        if samples[first][3] <= 0.0:
            continue
        for second in range(first + 1, size):
            if samples[second][3] <= 0.0:
                continue
            delta_r = samples[first][3] - samples[second][3]
            delta_x = samples[first][0] - samples[second][0]
            delta_y = samples[first][1] - samples[second][1]
            delta_z = samples[first][2] - samples[second][2]
            if delta_r * delta_r <= (delta_x * delta_x
                                     + delta_y * delta_y
                                     + delta_z * delta_z):
                continue
            if delta_r > 0.0:
                samples[second][3] = -1.0
            else:
                samples[first][3] = -1.0

    visited = np.zeros(horizontal * vertical * horizontal, dtype=bool)
    facts = ctx.facts
    placed = 0
    for sample in samples:
        radius = sample[3]
        if radius < 0.0:
            continue
        low_x = max(math.floor(sample[0] - radius), x_bound)
        low_y = max(math.floor(sample[1] - radius), y_bound)
        low_z = max(math.floor(sample[2] - radius), z_bound)
        high_x = max(math.floor(sample[0] + radius), low_x)
        high_y = max(math.floor(sample[1] + radius), low_y)
        high_z = max(math.floor(sample[2] + radius), low_z)
        for x in range(low_x, high_x + 1):
            fx = (x + 0.5 - sample[0]) / radius
            if fx * fx >= 1.0:
                continue
            for y in range(low_y, high_y + 1):
                fy = (y + 0.5 - sample[1]) / radius
                if fx * fx + fy * fy >= 1.0:
                    continue
                if not ctx.in_height(y):
                    continue
                for z in range(low_z, high_z + 1):
                    fz = (z + 0.5 - sample[2]) / radius
                    if fx * fx + fy * fy + fz * fz >= 1.0:
                        continue
                    index = ((x - x_bound)
                             + (y - y_bound) * horizontal
                             + (z - z_bound) * horizontal * vertical)
                    if visited[index]:
                        continue
                    visited[index] = True
                    state = ctx.get(x, y, z)
                    for rule, replacement in targets:
                        if _ore_should_place(ctx, rule, discard, state, rng, (x, y, z)):
                            ctx.set(x, y, z, replacement)
                            placed += 1
                            break
    return placed > 0


def _lerp_f64(delta: float, start: float, end: float) -> float:
    return start + delta * (end - start)


def _ore_should_place(ctx, rule, discard, state, rng, pos) -> bool:
    """Pumpkin's `OreFeature::should_place`, in its three-branch shape.

    `should_not_discard || !is_exposed_to_air`: a `discard` of 1.0 means the
    ore is discarded *when it is exposed to air* -- buried lapis, buried
    diamond and ancient debris are all `discard = 1.0` and still place, as long
    as no neighbour is air. The roll, when there is one, happens before the
    exposure scan and regardless of it, so the draw stream matches vanilla's
    `shouldNotDiscard` short-circuit order.
    """
    if not rule.test(ctx, state, rng):
        return False
    if discard <= 0.0:
        return True
    if discard < 1.0 and rng.next_f32() >= discard:
        return True
    for dx, dy, dz in DIRECTIONS:
        if ctx.facts.is_air(ctx.get(pos[0] + dx, pos[1] + dy, pos[2] + dz)):
            return False
    return True


# ----- the compiled twin -----
#
# `_place_ore_python` is the slowest thing in the features pass -- three nested
# Python loops per sphere sample -- so the same walk runs through `_ore_kernel`
# when numba is installed. The contract is the one `carver.py` uses for its
# ellipsoids: the interpreted function is the reference, the kernel keeps every
# f32 rounding point and the exact draw order, and `tools/check_ore_veins.py`
# compares the two block for block, draw for draw, counter for counter.
#
# The kernel owns the draw sequence end to end (the setup draws, `size` layout
# draws, and every roll `_ore_should_place` makes), so the Python random source
# is snapshotted with `kernel_rng.state_from` before the call and written back
# after it -- the modifier pipeline keeps drawing from the same object, and a
# kernel that forgets the write-back loses every draw it made.

#: Set to False to force the interpreted path; the check tool does exactly that.
_USE_NUMBA = _HAS_NUMBA

_ORE_DIRECTIONS = np.array(DIRECTIONS, dtype=np.int64)

_rng_next_f32 = kernel_rng.next_f32
_rng_next_f64 = kernel_rng.next_f64
_rng_next_bounded_i32 = kernel_rng.next_bounded_i32


class _OrePrep:
    """What `_ore_kernel` needs from an `ore` config, parsed once.

    The kernel wants flat arrays -- a `STATE_ID_SPACE` match mask per target --
    and `_place_ore_python` reparses the config on every call, which is right
    for the reference and far too slow for the hot path.
    """

    __slots__ = ("size", "discard", "masks", "replacements", "kinds", "probs")

    def __init__(self, size: int, discard, masks, replacements, kinds, probs) -> None:
        self.size = size
        self.discard = discard
        self.masks = masks
        self.replacements = replacements
        self.kinds = kinds
        self.probs = probs


def _rule_match_mask(rule: RuleTest, facts: BlockFacts) -> np.ndarray:
    """Where `rule.test` would answer True without an RNG roll.

    `random_block_match` gets the block-match half only; its probability roll is
    the kernel's job and only fires on a match, exactly as
    `RandomBlockMatchRuleTest` short-circuits.
    """
    mask = np.zeros(STATE_ID_SPACE, dtype=np.uint8)
    if rule.kind == "always_true":
        mask[:] = 1
    elif rule.kind == "tag_match":
        for state in facts.tag_states(rule.tag or ""):
            mask[state] = 1
    elif rule.kind in ("block_match", "random_block_match"):
        block = rule.block or ""
        names = facts.names
        for state, name in enumerate(names):
            if name == block:
                mask[state] = 1
        # `block_of` reads out-of-range ids as air.
        if block == "minecraft:air":
            mask[len(names):] = 1
    else:
        raise NotImplementedError(f"unsupported rule test {rule.kind!r}")
    return mask


_ORE_PREP_CACHE: Dict[int, Tuple[dict, _OrePrep]] = {}


def _ore_prep_for(config: dict) -> _OrePrep:
    """The kernel's view of one `ore` config, cached on the config dict.

    Keyed on `id(config)` with the config itself pinned in the cache entry, so
    a live id can never be reused under a dead config while the cache holds it.
    """
    key = id(config)
    hit = _ORE_PREP_CACHE.get(key)
    if hit is not None and hit[0] is config:
        return hit[1]
    facts = block_facts()
    entries = config["targets"]
    count = len(entries)
    masks = np.zeros((count, STATE_ID_SPACE), dtype=np.uint8)
    replacements = np.zeros(count, dtype=np.int32)
    kinds = np.zeros(count, dtype=np.uint8)
    probs = np.zeros(count, dtype=np.float32)
    for index, entry in enumerate(entries):
        rule = parse_rule_test(entry["target"])
        masks[index] = _rule_match_mask(rule, facts)
        replacements[index] = facts.state_id(entry["state"])
        if rule.kind == "random_block_match":
            kinds[index] = 1
            probs[index] = F32(rule.probability)
    prep = _OrePrep(int(config["size"]),
                    F32(config.get("discard_chance_on_air_exposure", 0.0)),
                    np.ascontiguousarray(masks), replacements, kinds, probs)
    _ORE_PREP_CACHE[key] = (config, prep)
    return prep


@njit(cache=True, inline="always")
def _ore_get(blocks, x, y, z, chunk_min_x, chunk_min_z, min_y, height,
             border, counters):
    """`FeatureContext.get` for the kernel: out-of-footprint reads see `border`
    and count in `counters[0]`."""
    lx = x - chunk_min_x
    lz = z - chunk_min_z
    if lx < 0 or lx >= 16 or lz < 0 or lz >= 16:
        counters[0] += 1
        return border
    row = y - min_y
    if row < 0 or row >= height:
        counters[0] += 1
        return border
    return blocks[row, lz, lx]


@njit(cache=True, inline="always")
def _ore_set(blocks, x, y, z, new_state, chunk_min_x, chunk_min_z, min_y,
             height, counters):
    """`FeatureContext.set` for the kernel: out-of-footprint writes are dropped
    and counted in `counters[1]`, with the xz/y split in `[2]`/`[3]`."""
    lx = x - chunk_min_x
    lz = z - chunk_min_z
    if lx < 0 or lx >= 16 or lz < 0 or lz >= 16:
        counters[1] += 1
        counters[2] += 1
        return
    row = y - min_y
    if row < 0 or row >= height:
        counters[1] += 1
        counters[3] += 1
        return
    blocks[row, lz, lx] = new_state


@njit(cache=True, nogil=True)
def _ore_kernel(blocks, heights, masks, replacements, kinds, probs, is_air,
                discard, size, pos_x, pos_y, pos_z, chunk_min_x, chunk_min_z,
                min_y, height, border, directions, state, counters):
    """`ore` end to end in one draw sequence: probe, vein walk, rasteriser.

    Mirrors `_place_ore_python` + `_ore_vein_python` operation for operation,
    including where the f32 rounds and which draws happen when. Returns the
    number of blocks placed. `heights` is `Surface.columns`' frozen
    `ocean_floor_wg` grid -- the same grid the interpreted probe reads -- and
    columns outside the footprint answer `min_y` like `Surface.top`'s outside
    branch, counting one `reads_outside` per examined column up to the first hit.
    """
    # -- setup draws and geometry, `_place_ore_python`'s head --
    angle = np.float32(_rng_next_f32(state) * np.float32(math.pi))
    size32 = np.float32(size)
    spread = np.float32(size32 / np.float32(8.0))
    widened = np.float32(np.float32(size32 / np.float32(16.0)) * np.float32(2.0))
    thickness = int(math.ceil(np.float64(np.float32(
        np.float32(widened + np.float32(1.0)) / np.float32(2.0)))))
    sin_a = math.sin(np.float64(angle))
    cos_a = math.cos(np.float64(angle))
    start_x = np.float64(pos_x) + sin_a * np.float64(spread)
    end_x = np.float64(pos_x) - sin_a * np.float64(spread)
    start_z = np.float64(pos_z) + cos_a * np.float64(spread)
    end_z = np.float64(pos_z) - cos_a * np.float64(spread)
    start_y = pos_y + np.int64(_rng_next_bounded_i32(state, 3)) - 2
    end_y = pos_y + np.int64(_rng_next_bounded_i32(state, 3)) - 2

    spread_ceil = int(math.ceil(np.float64(spread)))
    x_bound = pos_x - spread_ceil - thickness
    y_bound = pos_y - 2 - thickness
    z_bound = pos_z - spread_ceil - thickness
    horizontal = 2 * (spread_ceil + thickness) + 1
    vertical_rows = 2 * (2 + thickness) + 1

    # -- the heightmap probe: place only where the box meets the ocean floor --
    place = False
    for probe_x in range(horizontal):
        for probe_z in range(horizontal):
            px = x_bound + probe_x
            pz = z_bound + probe_z
            lx = px - chunk_min_x
            lz = pz - chunk_min_z
            if (0 <= lx < 16) and (0 <= lz < 16):
                top = heights[lz, lx]
            else:
                counters[0] += 1
                top = min_y
            if y_bound <= top:
                place = True
                break
        if place:
            break
    if not place:
        return 0

    # -- `size` samples along the vein, `_ore_vein_python`'s layout --
    samples = np.zeros((size, 4), dtype=np.float64)
    pi32 = np.float32(math.pi)
    for index in range(size):
        fraction = np.float32(np.float32(index) / size32)
        frac = np.float64(fraction)
        samples[index, 0] = start_x + frac * (end_x - start_x)
        samples[index, 1] = (np.float64(start_y)
                             + frac * (np.float64(end_y) - np.float64(start_y)))
        samples[index, 2] = start_z + frac * (end_z - start_z)
        raw = np.float32(_rng_next_f64(state) * np.float64(size) / 16.0)
        sine = np.float32(math.sin(np.float64(np.float32(pi32 * fraction))))
        samples[index, 3] = np.float64(np.float32(
            (sine + np.float32(1.0)) * raw + np.float32(1.0))) / 2.0

    # -- sphere occlusion: a sample swallowed by a later, larger one is dropped --
    for first in range(size - 1):
        if samples[first, 3] <= 0.0:
            continue
        for second in range(first + 1, size):
            if samples[second, 3] <= 0.0:
                continue
            delta_r = samples[first, 3] - samples[second, 3]
            dx = samples[first, 0] - samples[second, 0]
            dy = samples[first, 1] - samples[second, 1]
            dz = samples[first, 2] - samples[second, 2]
            if delta_r * delta_r <= dx * dx + dy * dy + dz * dz:
                continue
            if delta_r > 0.0:
                samples[second, 3] = -1.0
            else:
                samples[first, 3] = -1.0

    # -- the rasteriser --
    visited = np.zeros(horizontal * vertical_rows * horizontal, dtype=np.uint8)
    n_targets = masks.shape[0]
    placed = 0
    for sample_index in range(size):
        radius = samples[sample_index, 3]
        if radius < 0.0:
            continue
        centre_x = samples[sample_index, 0]
        centre_y = samples[sample_index, 1]
        centre_z = samples[sample_index, 2]
        low_x = int(math.floor(centre_x - radius))
        if low_x < x_bound:
            low_x = x_bound
        low_y = int(math.floor(centre_y - radius))
        if low_y < y_bound:
            low_y = y_bound
        low_z = int(math.floor(centre_z - radius))
        if low_z < z_bound:
            low_z = z_bound
        high_x = int(math.floor(centre_x + radius))
        if high_x < low_x:
            high_x = low_x
        high_y = int(math.floor(centre_y + radius))
        if high_y < low_y:
            high_y = low_y
        high_z = int(math.floor(centre_z + radius))
        if high_z < low_z:
            high_z = low_z
        for x in range(low_x, high_x + 1):
            fx = (x + 0.5 - centre_x) / radius
            if fx * fx >= 1.0:
                continue
            for y in range(low_y, high_y + 1):
                fy = (y + 0.5 - centre_y) / radius
                if fx * fx + fy * fy >= 1.0:
                    continue
                if y < min_y or y >= min_y + height:
                    continue
                for z in range(low_z, high_z + 1):
                    fz = (z + 0.5 - centre_z) / radius
                    if fx * fx + fy * fy + fz * fz >= 1.0:
                        continue
                    index = ((x - x_bound)
                             + (y - y_bound) * horizontal
                             + (z - z_bound) * horizontal * vertical_rows)
                    if visited[index]:
                        continue
                    visited[index] = 1
                    current = _ore_get(blocks, x, y, z, chunk_min_x, chunk_min_z,
                                       min_y, height, border, counters)
                    for target in range(n_targets):
                        # `_ore_should_place`, inlined: rule test (which rolls
                        # only for `random_block_match`, and only on a block
                        # match), then `should_not_discard || !is_exposed_to_air`
                        # with the short-circuit order the reference keeps.
                        passed = masks[target, current] != 0
                        if passed and kinds[target] == 1:
                            passed = _rng_next_f32(state) < probs[target]
                        if not passed:
                            continue
                        keep = True
                        if discard > 0.0:
                            if discard >= 1.0:
                                keep = False
                            else:
                                keep = _rng_next_f32(state) >= discard
                        if not keep:
                            exposed = False
                            for d in range(directions.shape[0]):
                                neighbour = _ore_get(
                                    blocks,
                                    x + directions[d, 0],
                                    y + directions[d, 1],
                                    z + directions[d, 2],
                                    chunk_min_x, chunk_min_z,
                                    min_y, height, border, counters)
                                if is_air[neighbour] != 0:
                                    exposed = True
                                    break
                            if exposed:
                                continue
                        _ore_set(blocks, x, y, z, replacements[target],
                                 chunk_min_x, chunk_min_z, min_y, height,
                                 counters)
                        placed += 1
                        break
    return placed


def _place_ore_compiled(ctx, config, rng, pos) -> bool:
    """`_place_ore_python`'s body through `_ore_kernel`."""
    prep = _ore_prep_for(config)
    state = kernel_rng.state_from(rng)
    counters = np.zeros(4, dtype=np.int64)
    placed = _ore_kernel(
        ctx.blocks, ctx.surface.columns(ctx, "ocean_floor_wg"),
        prep.masks, prep.replacements, prep.kinds, prep.probs,
        ctx.facts.is_air_mask(),
        prep.discard, prep.size, pos[0], pos[1], pos[2],
        ctx.chunk_x << 4, ctx.chunk_z << 4, ctx.min_y, ctx.height,
        ctx.border_block, _ORE_DIRECTIONS, state, counters)
    ctx.stats.reads_outside += int(counters[0])
    ctx.stats.writes_dropped += int(counters[1])
    ctx.stats.writes_outside_xz += int(counters[2])
    ctx.stats.writes_outside_y += int(counters[3])
    kernel_rng.state_to(rng, state)
    return placed > 0


def _place_ore(ctx, config, rng, pos) -> bool:
    """`ore`: the compiled twin when numba is installed, the reference otherwise."""
    if _USE_NUMBA:
        return _place_ore_compiled(ctx, config, rng, pos)
    return _place_ore_python(ctx, config, rng, pos)


def _place_scattered_ore(ctx, config, rng, pos) -> bool:
    """`scattered_ore`: `size` independent blobs within a growing radius.

    The radius grows with the blob index and the offset is
    `round((f1 - f2) * maxDist)` -- a triangular distribution, not a uniform
    one, and both floats are consumed even when the result rounds to zero.
    """
    size = int(config["size"])
    targets = [(parse_rule_test(entry["target"]), block_facts().state_id(entry["state"]))
               for entry in config["targets"]]
    discard = F32(config.get("discard_chance_on_air_exposure", 0.0))
    count = rng.next_bounded_i32(size + 1)
    for index in range(count):
        max_dist = min(index, 7)
        offset = (
            _scattered_offset(rng, max_dist),
            _scattered_offset(rng, max_dist),
            _scattered_offset(rng, max_dist),
        )
        target = _pos_add(pos, *offset)
        if not ctx.in_height(target[1]):
            continue
        state = ctx.get(*target)
        for rule, replacement in targets:
            if _ore_should_place(ctx, rule, discard, state, rng, target):
                ctx.set(*target, replacement)
                break
    return True


def _scattered_offset(rng, max_dist: int) -> int:
    first = F32(rng.next_f32())
    second = F32(rng.next_f32())
    return int(round(float(F32(first - second) * F32(max_dist))))


def _place_replace_single_block(ctx, config, rng, pos) -> bool:
    """`replace_single_block`: first matching target wins, always "placed"."""
    state = ctx.get(*pos)
    for entry in config["targets"]:
        rule = parse_rule_test(entry["target"])
        if rule.test(ctx, state, rng):
            ctx.set(*pos, block_facts().state_id(entry["state"]))
            break
    return True


def _place_simple_block(ctx, config, rng, pos) -> bool:
    """`simple_block`: a state provider, then a `canPlaceAt` check.

    `canPlaceAt` is block behaviour, so this uses the same approximation as the
    `would_survive` predicate -- see `WouldSurvive`. Nothing here requires the
    target to be replaceable: vanilla's `SimpleBlockFeature` only asks whether
    the chosen state can survive where it landed, and the datapack guards the
    rest with a `block_predicate_filter` in front of it.
    """
    provider = parse_state_provider(config["to_place"])
    state = provider.get(ctx, rng, *pos)
    if not _can_place_at(ctx, ctx.facts.block_of(state), pos):
        return False
    ctx.set(*pos, state)
    return True


def _can_place_at(ctx, block: str, pos) -> bool:
    """The `canSurvive` question, for the blocks the worldgen data places.

    This is the gate that keeps `minecraft:grass` off the top of the ocean. A
    `simple_block` placement chain lands wherever its modifiers put it, and the
    1.21.11 data relies on `canSurvive` -- not on a predicate in the chain -- to
    reject a position with no soil under it. Returning True for an unmodelled
    family would put the plant back in the air, so an unmodelled family is
    *named* here rather than waved through: only blocks with no substrate rule at
    all (`melon`, `pumpkin`, `fire`, `soul_fire`) keep the permissive answer.
    """
    rule = _plant_rule(block)
    if rule is None:
        return True
    return _survives(ctx, block, pos)


def _place_random_patch(ctx, config, rng, pos) -> bool:
    """`random_patch`: `tries` attempts in a box, each delegated to a feature.

    Every try consumes four bounded draws *before* the nested feature runs, even
    when the offset lands outside the chunk, so the nested feature's stream
    depends on the try index.
    """
    tries = int(config["tries"])
    xz = int(config["xz_spread"]) + 1
    y = int(config["y_spread"]) + 1
    nested = _inline_placed(config["feature"])
    placed = False
    for _ in range(tries):
        candidate = (
            pos[0] + rng.next_bounded_i32(xz) - rng.next_bounded_i32(xz),
            pos[1] + rng.next_bounded_i32(y) - rng.next_bounded_i32(y),
            pos[2] + rng.next_bounded_i32(xz) - rng.next_bounded_i32(xz),
        )
        if _run_placed(ctx, nested, rng, candidate):
            placed = True
    return placed


def _inline_placed(payload, lookup: Optional[Dict[str, PlacedFeature]] = None) -> PlacedFeature:
    """A nested `{feature, placement}` object, or a named reference, as a `PlacedFeature`."""
    if isinstance(payload, str):
        table = _PLACED_LOOKUP if lookup is None else lookup
        key = _canonical(payload)
        entry = table.get(key)
        if entry is None:
            raise NotImplementedError(f"placed feature {key!r} is not in the table")
        return entry
    return PlacedFeature(None, payload)


_PLACED_LOOKUP: Dict[str, PlacedFeature] = {}


def _place_random_selector(ctx, config, rng, pos) -> bool:
    """`random_selector`: first entry whose `nextFloat()` beats its chance.

    The draws are sequential and each entry consumes one *whether or not* it is
    taken, so a selector with three entries always consumes at least one draw
    and at most three.
    """
    for entry in config["features"]:
        if rng.next_f32() >= F32(entry["chance"]):
            continue
        return _run_placed(ctx, _inline_placed(entry["feature"]), rng, pos)
    default = config.get("default")
    if default is None:
        return False
    return _run_placed(ctx, _inline_placed(default), rng, pos)


def _place_simple_random_selector(ctx, config, rng, pos) -> bool:
    features = [_inline_placed(entry) for entry in config["features"]]
    if not features:
        return False
    return _run_placed(ctx, features[rng.next_bounded_i32(len(features))], rng, pos)


def _place_random_boolean_selector(ctx, config, rng, pos) -> bool:
    chosen = config["feature_true"] if rng.next_bool() else config["feature_false"]
    return _run_placed(ctx, _inline_placed(chosen), rng, pos)


def _place_weighted_random_selector(ctx, config, rng, pos) -> bool:
    entries = [(entry, int(entry.get("weight", 1))) for entry in config["features"]]
    total = sum(weight for _, weight in entries)
    if total <= 0:
        return False
    roll = rng.next_bounded_i32(total)
    for entry, weight in entries:
        roll -= weight
        if roll < 0:
            return _run_placed(ctx, _inline_placed(entry["feature"]), rng, pos)
    return False


def _place_block_pile(ctx, config, rng, pos) -> bool:
    """`block_pile`: blocks scattered on a surface, biased to the centre.

    Note the short circuit in the acceptance test: the radius test consumes two
    floats and the `0.031` test consumes a third only when the first fails.
    Rewriting the condition as two independent `if`s changes the draws.
    """
    if pos[1] < ctx.min_y + 5:
        return False
    provider = parse_state_provider(config["state_provider"])
    facts = ctx.facts
    xr = 2 + rng.next_bounded_i32(2)
    zr = 2 + rng.next_bounded_i32(2)
    for x, y, z in _iter_box(_pos_add(pos, -xr, 0, -zr), _pos_add(pos, xr, 1, zr)):
        dx = pos[0] - x
        dz = pos[2] - z
        accepted = F32(dx * dx + dz * dz) <= F32(rng.next_f32() * F32(10.0)
                                                 - rng.next_f32() * F32(6.0))
        if not accepted and rng.next_f32() >= F32(0.031):
            continue
        if not ctx.is_air(x, y, z):
            continue
        below = ctx.get(x, y - 1, z)
        if facts.block_of(below) == "minecraft:dirt_path":
            if not rng.next_bool():
                continue
        elif not facts.face_sturdy(below, (0, 1, 0)):
            continue
        ctx.set(x, y, z, provider.get(ctx, rng, x, y, z))
    return True


def _place_disk(ctx, config, rng, pos) -> bool:
    """`disk`: a filled circle of blocks, one column deep plus `half_height`."""
    provider = parse_state_provider(config["state_provider"])
    target = parse_block_predicate(config["target"])
    radius = parse_int_provider(config["radius"]).get(rng)
    half_height = int(config["half_height"])
    top = pos[1] + half_height
    bottom = pos[1] - half_height - 1
    placed = False
    for x in range(pos[0] - radius, pos[0] + radius + 1):
        for z in range(pos[2] - radius, pos[2] + radius + 1):
            dx = x - pos[0]
            dz = z - pos[2]
            if dx * dx + dz * dz > radius * radius:
                continue
            for y in range(top, bottom, -1):
                if target.test(ctx, x, y, z):
                    ctx.set(x, y, z, provider.get(ctx, rng, x, y, z))
                    placed = True
    return placed


def _place_spring(ctx, config, rng, pos) -> bool:
    """`spring_feature`: a fluid source in a pocket of exactly the right shape.

    `rock_count + hole_count` covers the five directions *excluding up*, which
    is checked separately as the cap. Counting all six neighbours makes
    `rock_count: 4` impossible to satisfy in stone and the feature never fires.
    """
    facts = ctx.facts
    state = facts.state_id(config["state"])
    valid = {_canonical(name) for name in
             (config["valid_blocks"] if isinstance(config["valid_blocks"], list)
              else [config["valid_blocks"]])}
    requires_below = bool(config.get("requires_block_below", True))
    rock_count = int(config["rock_count"])
    hole_count = int(config["hole_count"])

    above = ctx.get(pos[0], pos[1] + 1, pos[2])
    if facts.block_of(above) not in valid:
        return False
    below = ctx.get(pos[0], pos[1] - 1, pos[2])
    if requires_below and facts.block_of(below) not in valid:
        return False
    here = ctx.get(*pos)
    if not facts.is_air(here) and facts.block_of(here) not in valid:
        return False

    rocks = 0
    holes = 0
    for offset in HORIZONTAL_DIRECTIONS + ((0, -1, 0),):
        neighbour = ctx.get(pos[0] + offset[0], pos[1] + offset[1], pos[2] + offset[2])
        if facts.block_of(neighbour) in valid:
            rocks += 1
        if facts.is_air(neighbour):
            holes += 1
    if rocks == rock_count and holes == hole_count:
        ctx.set(*pos, state)
        return True
    return False


def _place_seagrass(ctx, config, rng, pos) -> bool:
    """`seagrass`: on the ocean floor, sometimes two blocks tall.

    The Y comes from the *exclusive* ocean-floor height, i.e. the first block
    above the floor, which is the water the seagrass grows in. Using the floor's
    own Y makes the `is(WATER)` test fail and the feature never places anything.
    """
    probability = F32(config.get("probability", 0.0))
    facts = ctx.facts
    x = pos[0] + rng.next_bounded_i32(8) - rng.next_bounded_i32(8)
    z = pos[2] + rng.next_bounded_i32(8) - rng.next_bounded_i32(8)
    y = ctx.height_at("ocean_floor_wg", x, z)
    if facts.block_of(ctx.get(x, y, z)) != "minecraft:water":
        return False
    below = ctx.get(x, y - 1, z)
    if facts.in_tag(below, "minecraft:cannot_support_seagrass"):
        return False
    if not facts.face_sturdy(below, (0, 1, 0)):
        return False
    if rng.next_f64() < float(probability):
        if facts.block_of(ctx.get(x, y + 1, z)) != "minecraft:water":
            return False
        lower = facts.default_state("minecraft:tall_seagrass")
        upper = facts.with_property(lower, "half", "upper")
        ctx.set(x, y, z, lower)
        ctx.set(x, y + 1, z, upper)
        return True
    ctx.set(x, y, z, facts.default_state("minecraft:seagrass"))
    return True


def _place_freezing(ctx, config, rng, pos) -> bool:
    """`freeze_top_layer`: ice on water, snow on ground, per column.

    Runs over the whole chunk from the origin, not from the placement position:
    the one `freeze_top_layer` placed feature in the overworld carries a single
    step-10 entry, and the modifier chain only decides *which* chunk it runs in.
    """
    facts = ctx.facts
    origin_x, origin_z = pos[0], pos[2]
    for dx in range(16):
        for dz in range(16):
            x = origin_x + dx
            z = origin_z + dz
            top = ctx.height_at("motion_blocking", x, z)
            below_y = top - 1
            below = ctx.get(x, below_y, z)
            biome = ctx.biome_at(x, top, z)
            if ctx.temperature(biome, x, top, z) <= 0.15 and \
                    facts.block_of(below) == "minecraft:water":
                ctx.set(x, below_y, z, facts.default_state("minecraft:ice"))
                continue
            if ctx.temperature(biome, x, top, z) < 0.15:
                if ctx.is_air(x, top, z) and not facts.is_air(below) and \
                        not facts.in_tag(below, "minecraft:cannot_support_snow_layer"):
                    ctx.set(x, top, z, facts.default_state("minecraft:snow"))
                    if facts.has_property(below, "snowy"):
                        ctx.set(x, below_y, z, facts.with_property(below, "snowy", "true"))
    return True


def _place_vegetation_patch(ctx, config, rng, pos) -> bool:
    """`vegetation_patch`: a disc of ground replacement plus a sprinkled feature.

    The columns are walked *into* the surface and then back out, so the patch
    follows the terrain instead of cutting a flat disc. The corner columns are
    skipped and the edge columns are probabilistic, which is what makes the
    patch round rather than square.
    """
    facts = ctx.facts
    replaceable = parse_block_predicate(_tag_predicate(config["replaceable"]))
    ground = parse_state_provider(config["ground_state"])
    nested = _inline_placed(config["vegetation_feature"])
    surface = str(config["surface"]).removeprefix("minecraft:")
    depth_provider = parse_int_provider(config["depth"])
    extra_bottom = F32(config.get("extra_bottom_block_chance", 0.0))
    vertical_range = int(config["vertical_range"])
    vegetation_chance = F32(config.get("vegetation_chance", 0.0))
    xz_radius = parse_int_provider(config["xz_radius"])
    extra_edge = F32(config.get("extra_edge_column_chance", 0.0))

    inwards = (0, -1, 0) if surface == "floor" else (0, 1, 0)
    outwards = (0, 1, 0) if surface == "floor" else (0, -1, 0)
    x_radius = xz_radius.get(rng) + 1
    z_radius = xz_radius.get(rng) + 1

    touched = []
    for dx in range(-x_radius, x_radius + 1):
        x_edge = dx in (-x_radius, x_radius)
        for dz in range(-z_radius, z_radius + 1):
            z_edge = dz in (-z_radius, z_radius)
            if x_edge and z_edge:
                continue
            if (x_edge or z_edge) and (extra_edge == 0.0 or rng.next_f32() > extra_edge):
                continue
            x = pos[0] + dx
            z = pos[2] + dz
            y = pos[1]
            for _ in range(vertical_range):
                if not ctx.is_air(x, y, z):
                    break
                y += inwards[1]
            for _ in range(vertical_range):
                if ctx.is_air(x, y, z):
                    break
                y += outwards[1]
            below = (x, y + inwards[1], z)
            if not ctx.is_air(x, y, z):
                continue
            if not facts.face_sturdy(ctx.get(*below), outwards):
                continue
            depth = depth_provider.get(rng)
            if extra_bottom > 0.0 and rng.next_f32() < extra_bottom:
                depth += 1
            if _vegetation_ground(ctx, ground, replaceable, rng, below, depth, inwards):
                touched.append(below)

    for ground_pos in touched:
        if vegetation_chance > 0.0 and rng.next_f32() < vegetation_chance:
            _run_placed(ctx, nested, rng, (ground_pos[0], ground_pos[1] + outwards[1], ground_pos[2]))
    return bool(touched)


def _vegetation_ground(ctx, ground, replaceable, rng, below, depth, inwards) -> bool:
    position = below
    for index in range(depth):
        state = ground.get(ctx, rng, *position)
        if ctx.facts.block_of(state) != ctx.facts.block_of(ctx.get(*position)):
            if not replaceable.test(ctx, *position):
                return index != 0
            ctx.set(*position, state)
            position = (position[0] + inwards[0], position[1] + inwards[1],
                        position[2] + inwards[2])
    return True


def _tag_predicate(tag: str) -> dict:
    return {"type": "minecraft:matching_block_tag", "tag": tag.lstrip("#")}


def _place_waterlogged_vegetation_patch(ctx, config, rng, pos) -> bool:
    """`waterlogged_vegetation_patch`: the patch, then its unexposed part floods.

    "Unexposed" means all four sides and the bottom are face-sturdy; those
    columns become water and the vegetation goes *in* the water block rather
    than above it.
    """
    facts = ctx.facts
    replaceable = parse_block_predicate(_tag_predicate(config["replaceable"]))
    ground = parse_state_provider(config["ground_state"])
    nested = _inline_placed(config["vegetation_feature"])
    surface = str(config["surface"]).removeprefix("minecraft:")
    depth_provider = parse_int_provider(config["depth"])
    extra_bottom = F32(config.get("extra_bottom_block_chance", 0.0))
    vertical_range = int(config["vertical_range"])
    vegetation_chance = F32(config.get("vegetation_chance", 0.0))
    xz_radius = parse_int_provider(config["xz_radius"])
    extra_edge = F32(config.get("extra_edge_column_chance", 0.0))

    inwards = (0, -1, 0) if surface == "floor" else (0, 1, 0)
    outwards = (0, 1, 0) if surface == "floor" else (0, -1, 0)
    x_radius = xz_radius.get(rng) + 1
    z_radius = xz_radius.get(rng) + 1

    touched = []
    for dx in range(-x_radius, x_radius + 1):
        x_edge = dx in (-x_radius, x_radius)
        for dz in range(-z_radius, z_radius + 1):
            z_edge = dz in (-z_radius, z_radius)
            if x_edge and z_edge:
                continue
            if (x_edge or z_edge) and (extra_edge == 0.0 or rng.next_f32() > extra_edge):
                continue
            x = pos[0] + dx
            z = pos[2] + dz
            y = pos[1]
            for _ in range(vertical_range):
                if not ctx.is_air(x, y, z):
                    break
                y += inwards[1]
            for _ in range(vertical_range):
                if ctx.is_air(x, y, z):
                    break
                y += outwards[1]
            below = (x, y + inwards[1], z)
            if not ctx.is_air(x, y, z):
                continue
            if not facts.face_sturdy(ctx.get(*below), outwards):
                continue
            depth = depth_provider.get(rng)
            if extra_bottom > 0.0 and rng.next_f32() < extra_bottom:
                depth += 1
            if _vegetation_ground(ctx, ground, replaceable, rng, below, depth, inwards):
                touched.append(below)

    water_surface = [position for position in touched
                     if not _is_exposed(ctx, position)]
    for position in water_surface:
        ctx.set(*position, facts.default_state("minecraft:water"))
    for position in water_surface:
        if vegetation_chance > 0.0 and rng.next_f32() < vegetation_chance:
            _run_placed(ctx, nested, rng, position)
    return bool(water_surface)


def _is_exposed(ctx, pos) -> bool:
    """`WaterloggedVegetationPatchFeature.isExposed`: no open side or bottom."""
    for offset in HORIZONTAL_PLANE + ((0, -1, 0),):
        neighbour = ctx.get(pos[0] + offset[0], pos[1] + offset[1], pos[2] + offset[2])
        opposite = (-offset[0], -offset[1], -offset[2])
        if not ctx.facts.face_sturdy(neighbour, opposite):
            return True
    return False


def _place_block_column(ctx, config, rng, pos) -> bool:
    """`block_column`: stacked layers growing in one direction.

    The heights are drawn first, then the column is probed for its allowed
    placement, and `adjust_layer_heights` trims from the tip (or the base) when
    the probe stops early. The probe always starts one block *past* the origin.
    """
    facts = ctx.facts
    layers = [(parse_int_provider(layer["height"]), parse_state_provider(layer["provider"]))
              for layer in config["layers"]]
    direction = _direction_offset(str(config["direction"]))
    allowed = parse_block_predicate(config["allowed_placement"])
    prioritize_tip = bool(config.get("prioritize_tip", False))

    heights = [provider.get(rng) for provider, _ in layers]
    total = sum(heights)
    if total == 0:
        return False

    cursor = pos
    probe = _pos_add(pos, *direction)
    progressed = 0
    while progressed < total:
        if not allowed.test(ctx, *probe):
            _adjust_layer_heights(heights, total, progressed, prioritize_tip)
            break
        probe = _pos_add(probe, *direction)
        progressed += 1

    cursor = pos
    for index, height in enumerate(heights):
        if height == 0:
            continue
        provider = layers[index][1]
        for _ in range(height):
            ctx.set(*cursor, provider.get(ctx, rng, *cursor))
            cursor = _pos_add(cursor, *direction)
    return True


def _adjust_layer_heights(heights: List[int], expected: int, actual: int,
                          prioritize_tip: bool) -> None:
    surplus = expected - actual
    order = range(len(heights)) if prioritize_tip else range(len(heights) - 1, -1, -1)
    for index in order:
        if surplus <= 0:
            return
        taken = min(heights[index], surplus)
        heights[index] -= taken
        surplus -= taken


def _place_lake(ctx, config, rng, pos) -> bool:
    """`lake`: a fluid blob with a barrier shell, in a 16x8x16 box.

    The blob is four to seven ellipsoids stamped into a `16 x 8 x 16` mask; the
    mask is then validated against the terrain before anything is written (a
    "lake" cut into a hillside would spill), and only then is the fluid poured.
    The fluid/barrier providers are sampled once, at the *origin*, not per block.
    """
    facts = ctx.facts
    if pos[1] <= ctx.min_y + 4:
        return False
    fluid = parse_state_provider(config["fluid"])
    barrier = parse_state_provider(config["barrier"])
    origin = _pos_add(pos, -8, -4, -8)
    mask = np.zeros((16, 16, 8), dtype=bool)
    spots = rng.next_bounded_i32(4) + 4
    for _ in range(spots):
        xr = rng.next_f64() * 6.0 + 3.0
        yr = rng.next_f64() * 4.0 + 2.0
        zr = rng.next_f64() * 6.0 + 3.0
        xp = rng.next_f64() * (16.0 - xr - 2.0) + 1.0 + xr / 2.0
        yp = rng.next_f64() * (8.0 - yr - 4.0) + 2.0 + yr / 2.0
        zp = rng.next_f64() * (16.0 - zr - 2.0) + 1.0 + zr / 2.0
        for x in range(1, 15):
            for z in range(1, 15):
                for y in range(1, 7):
                    dx = (x - xp) / (xr / 2.0)
                    dy = (y - yp) / (yr / 2.0)
                    dz = (z - zp) / (zr / 2.0)
                    if dx * dx + dy * dy + dz * dz < 1.0:
                        mask[x, z, y] = True

    def shell(x: int, z: int, y: int) -> bool:
        if mask[x, z, y]:
            return False
        if x < 15 and mask[x + 1, z, y]:
            return True
        if x > 0 and mask[x - 1, z, y]:
            return True
        if z < 15 and mask[x, z + 1, y]:
            return True
        if z > 0 and mask[x, z - 1, y]:
            return True
        if y < 7 and mask[x, z, y + 1]:
            return True
        return y > 0 and mask[x, z, y - 1]

    fluid_state = fluid.get(ctx, rng, *origin)
    for x in range(16):
        for z in range(16):
            for y in range(8):
                if not shell(x, z, y):
                    continue
                state = ctx.get(origin[0] + x, origin[1] + y, origin[2] + z)
                if y >= 4 and facts.is_liquid(state):
                    return False
                if y < 4 and not facts.is_solid(state) and state != fluid_state:
                    return False

    air_state = facts.default_state("minecraft:cave_air")
    for x in range(16):
        for z in range(16):
            for y in range(8):
                if not mask[x, z, y]:
                    continue
                place = (origin[0] + x, origin[1] + y, origin[2] + z)
                current = ctx.get(*place)
                if facts.in_tag(current, "minecraft:features_cannot_replace"):
                    continue
                ctx.set(*place, air_state if y >= 4 else fluid_state)

    barrier_state = barrier.get(ctx, rng, *origin)
    if not facts.is_air(barrier_state):
        for x in range(16):
            for z in range(16):
                for y in range(8):
                    if not shell(x, z, y):
                        continue
                    if y >= 4 and rng.next_bounded_i32(2) == 0:
                        continue
                    place = (origin[0] + x, origin[1] + y, origin[2] + z)
                    current = ctx.get(*place)
                    if not facts.is_solid(current):
                        continue
                    if facts.in_tag(current, "minecraft:lava_pool_stone_cannot_replace"):
                        continue
                    ctx.set(*place, barrier_state)

    if facts.block_of(fluid_state) == "minecraft:water":
        for x in range(16):
            for z in range(16):
                place = (origin[0] + x, origin[1] + 4, origin[2] + z)
                biome = ctx.biome_at(*place)
                if ctx.temperature(biome, *place) < 0.15:
                    current = ctx.get(*place)
                    if not facts.in_tag(current, "minecraft:features_cannot_replace"):
                        ctx.set(*place, facts.default_state("minecraft:ice"))
    return True


def _place_monster_room(ctx, config, rng, pos) -> bool:
    """`monster_room`: a cobblestone box carved out of solid rock.

    Block *entities* (the chest's loot and the spawner's mob list) are not part
    of this seam -- `decorate` gets a block array with no block-entity channel --
    so the chest and spawner are placed as blocks and their NBT is not. The
    chest `facing` is still derived from its single solid wall, because that is a
    block state and it is visible.
    """
    facts = ctx.facts
    xr = rng.next_bounded_i32(2) + 2
    zr = rng.next_bounded_i32(2) + 2
    min_x, max_x = -xr - 1, xr + 1
    min_z, max_z = -zr - 1, zr + 1

    holes = 0
    for dx in range(min_x, max_x + 1):
        for dy in range(-1, 5):
            for dz in range(min_z, max_z + 1):
                solid = facts.is_solid(ctx.get(pos[0] + dx, pos[1] + dy, pos[2] + dz))
                if dy == -1 and not solid:
                    return False
                if dy == 4 and not solid:
                    return False
                on_wall = dx in (min_x, max_x) or dz in (min_z, max_z)
                if on_wall and dy == 0 and \
                        ctx.is_air(pos[0] + dx, pos[1], pos[2] + dz) and \
                        ctx.is_air(pos[0] + dx, pos[1] + 1, pos[2] + dz):
                    holes += 1
    if not 1 <= holes <= 5:
        return False

    cave_air = facts.default_state("minecraft:cave_air")
    cobble = facts.default_state("minecraft:cobblestone")
    mossy = facts.default_state("minecraft:mossy_cobblestone")
    for dx in range(min_x, max_x + 1):
        for dy in range(3, -2, -1):
            for dz in range(min_z, max_z + 1):
                x, y, z = pos[0] + dx, pos[1] + dy, pos[2] + dz
                on_boundary = dx in (min_x, max_x) or dz in (min_z, max_z) or dy in (-1, 4)
                if on_boundary:
                    below_solid = facts.is_solid(ctx.get(x, y - 1, z))
                    current = ctx.get(x, y, z)
                    if y >= ctx.min_y and not below_solid:
                        ctx.set(x, y, z, cave_air)
                    elif facts.is_solid(current) and \
                            facts.block_of(current) != "minecraft:chest":
                        if dy == -1 and rng.next_bounded_i32(4) != 0:
                            ctx.set(x, y, z, mossy)
                        else:
                            ctx.set(x, y, z, cobble)
                else:
                    current = ctx.get(x, y, z)
                    block = facts.block_of(current)
                    if block not in ("minecraft:chest", "minecraft:spawner"):
                        ctx.set(x, y, z, cave_air)

    for _ in range(2):
        for _ in range(3):
            x = pos[0] + rng.next_bounded_i32(xr * 2 + 1) - xr
            z = pos[2] + rng.next_bounded_i32(zr * 2 + 1) - zr
            y = pos[1]
            if not ctx.is_air(x, y, z):
                continue
            solid_walls = [offset for offset in HORIZONTAL_PLANE
                           if facts.is_solid(ctx.get(x + offset[0], y, z + offset[2]))]
            if len(solid_walls) != 1:
                continue
            wall = solid_walls[0]
            facing = {(-1, 0): "east", (1, 0): "west", (0, -1): "south", (0, 1): "north"}[
                (wall[0], wall[2])]
            chest = facts.with_property(facts.default_state("minecraft:chest"), "facing", facing)
            ctx.set(x, y, z, chest)
            break

    ctx.set(*pos, facts.default_state("minecraft:spawner"))
    return True


def _place_fallen_tree(ctx, config, rng, pos) -> bool:
    """`fallen_tree`: a stump, then a log lying along a horizontal direction."""
    facts = ctx.facts
    trunk = parse_state_provider(config["trunk_provider"])
    length_provider = parse_int_provider(config["log_length"])
    stump_decorators = [parse_decorator(entry) for entry in config.get("stump_decorators") or ()]
    log_decorators = [parse_decorator(entry) for entry in config.get("log_decorators") or ()]

    stump = _fallen_log_block(ctx, trunk, rng, pos, None)
    _decorate(ctx, rng, stump_decorators, [], [stump], [])

    direction = HORIZONTAL_PLANE[rng.next_bounded_i32(4)]
    log_length = length_provider.get(rng) - 2
    start = _pos_add(pos, direction[0] * (2 + rng.next_bounded_i32(2)), 0,
                     direction[2] * (2 + rng.next_bounded_i32(2)))
    # `set_ground_height_for_fallen_log_start_pos` climbs from one above the
    # start and walks down six blocks looking for ground.
    start = (start[0], start[1] + 1, start[2])
    for _ in range(6):
        if _can_replace_or_log(ctx, start) and _over_solid_ground(ctx, start):
            break
        start = (start[0], start[1] - 1, start[2])

    if log_length <= 0:
        return True
    cursor = start
    for _ in range(log_length):
        state = ctx.get(*cursor)
        if not _can_replace(ctx, cursor):
            return True
        cursor = _pos_add(cursor, *direction)
    logs = []
    cursor = start
    gap = 0
    ok = True
    for _ in range(log_length):
        if _over_solid_ground(ctx, cursor):
            gap = 0
        else:
            gap += 1
            if gap > 2:
                ok = False
                break
        logs.append(_fallen_log_block(ctx, trunk, rng, cursor, direction))
        cursor = _pos_add(cursor, *direction)
    if ok:
        _decorate(ctx, rng, log_decorators, [], logs, [])
    return True


def _fallen_log_block(ctx, trunk, rng, pos, direction) -> Tuple[int, int, int]:
    state = trunk.get(ctx, rng, *pos)
    if direction is not None and ctx.facts.has_property(state, "axis"):
        axis = "x" if direction[0] else ("z" if direction[2] else "y")
        state = ctx.facts.with_property(state, "axis", axis)
    ctx.set(*pos, state)
    return pos


def _can_replace(ctx, pos) -> bool:
    facts = ctx.facts
    state = ctx.get(*pos)
    return facts.is_air(state) or facts.in_tag(state, "minecraft:replaceable_by_trees")


def _can_replace_or_log(ctx, pos) -> bool:
    return _can_replace(ctx, pos) or ctx.facts.in_tag(ctx.get(*pos), "minecraft:logs")


def _over_solid_ground(ctx, pos) -> bool:
    return ctx.facts.face_sturdy(ctx.get(pos[0], pos[1] - 1, pos[2]), (0, 1, 0))


def _place_bamboo(ctx, config, rng, pos) -> bool:
    """`bamboo`: a stalk that grows upward until it hits something."""
    facts = ctx.facts
    probability = F32(config.get("probability", 0.0))
    # `BambooFeature` places through the block's own `canSurvive`, so the stalk
    # refuses a position with no substrate. Without this the feature roots stalks
    # in open water wherever the `heightmap` modifier put them.
    if not _survives(ctx, "minecraft:bamboo", pos):
        return False
    height = rng.next_bounded_i32(12) + 5
    if rng.next_f32() < probability:
        radius = rng.next_bounded_i32(4) + 1
        for x in range(pos[0] - radius, pos[0] + radius + 1):
            for z in range(pos[2] - radius, pos[2] + radius + 1):
                dx = x - pos[0]
                dz = z - pos[2]
                if dx * dx + dz * dz > radius * radius:
                    continue
                podzol_y = ctx.height_at("world_surface_wg", x, z) - 1
                state = ctx.get(x, podzol_y, z)
                if facts.in_tag(state, "minecraft:beneath_bamboo_podzol_replaceable"):
                    ctx.set(x, podzol_y, z, facts.default_state("minecraft:podzol"))

    trunk = facts.default_state("minecraft:bamboo")
    trunk = facts.with_property(trunk, "age", "1")
    cursor = pos
    grown = 0
    for _ in range(height):
        if not ctx.is_air(*cursor):
            break
        ctx.set(*cursor, trunk)
        cursor = (cursor[0], cursor[1] + 1, cursor[2])
        grown += 1
    if grown >= 3:
        large_top = facts.with_property(
            facts.with_property(trunk, "leaves", "large"), "stage", "1")
        large = facts.with_property(trunk, "leaves", "large")
        small = facts.with_property(trunk, "leaves", "small")
        ctx.set(cursor[0], cursor[1] - 1, cursor[2], large_top)
        ctx.set(cursor[0], cursor[1] - 2, cursor[2], large)
        ctx.set(cursor[0], cursor[1] - 3, cursor[2], small)
    return True


def _place_netherrack_replace_blobs(ctx, config, rng, pos) -> bool:
    facts = ctx.facts
    target = facts.state_id(config["target"])
    replacement = facts.state_id(config["state"])
    radius = parse_int_provider(config["radius"])
    cursor = pos
    while cursor[1] > ctx.min_y + 1:
        if facts.block_of(ctx.get(*cursor)) == facts.block_of(target):
            break
        cursor = (cursor[0], cursor[1] - 1, cursor[2])
    else:
        return False
    xr = radius.get(rng)
    yr = radius.get(rng)
    zr = radius.get(rng)
    distance = max(xr, yr, zr)
    placed = False
    for x, y, z in _iter_outwards(cursor, xr, yr, zr):
        if abs(x - cursor[0]) + abs(y - cursor[1]) + abs(z - cursor[2]) > distance:
            break
        if facts.block_of(ctx.get(x, y, z)) != facts.block_of(target):
            continue
        ctx.set(x, y, z, replacement)
        placed = True
    return placed


def _iter_outwards(origin, xr: int, yr: int, zr: int):
    """`BlockPos.iterateOutwards`: shells of increasing Chebyshev radius.

    It is a *Manhattan* bound applied to a shell walk, so the loop's `break` is
    load-bearing: the walk overshoots and relies on the caller stopping it.
    """
    for radius in range(max(xr, yr, zr) + 1):
        for x in range(-radius, radius + 1):
            for y in range(-radius, radius + 1):
                for z in range(-radius, radius + 1):
                    if max(abs(x), abs(y), abs(z)) != radius:
                        continue
                    yield (origin[0] + x, origin[1] + y, origin[2] + z)


def _place_glowstone_blob(ctx, config, rng, pos) -> bool:
    facts = ctx.facts
    if not ctx.is_air(*pos):
        return False
    above = facts.block_of(ctx.get(pos[0], pos[1] + 1, pos[2]))
    if above not in ("minecraft:netherrack", "minecraft:basalt", "minecraft:blackstone"):
        return False
    glowstone = facts.default_state("minecraft:glowstone")
    ctx.set(*pos, glowstone)
    for _ in range(1500):
        candidate = (pos[0] + rng.next_bounded_i32(8) - rng.next_bounded_i32(8),
                     pos[1] - rng.next_bounded_i32(12),
                     pos[2] + rng.next_bounded_i32(8) - rng.next_bounded_i32(8))
        if not ctx.is_air(*candidate):
            continue
        neighbours = 0
        for offset in DIRECTIONS:
            if facts.block_of(ctx.get(candidate[0] + offset[0], candidate[1] + offset[1],
                                      candidate[2] + offset[2])) == "minecraft:glowstone":
                neighbours += 1
                if neighbours > 1:
                    break
        if neighbours == 1:
            ctx.set(*candidate, glowstone)
    return True


def _place_fill_layer(ctx, config, rng, pos) -> bool:
    y = ctx.min_y + int(config["height"])
    state = block_facts().state_id(config["state"])
    for dx in range(16):
        for dz in range(16):
            x = pos[0] + dx
            z = pos[2] + dz
            if ctx.is_air(x, y, z):
                ctx.set(x, y, z, state)
    return True


def _place_tree(ctx, config, rng, pos) -> bool:
    """`tree`: trunk placer, foliage placer, decorators, then leaf distances.

    The order is not negotiable: the trunk height is drawn first, then the free
    height is clipped by `minimum_size`, then the trunk places, then the foliage
    radius and height are drawn, then the foliage, then decorators, and finally
    every leaf's `distance` property is recomputed by a breadth-first sweep from
    the logs. Drawing the foliage radius before the trunk has placed gives the
    same tree shape and a different RNG stream.
    """
    placer = TrunkPlacer(config)
    height = placer.height(rng)

    clip = config["minimum_size"].get("min_clipped_height")
    top = _tree_free_height(ctx, FeatureSize(config["minimum_size"]), height, pos,
                            bool(config.get("ignore_vines", False)))
    if top < height and (clip is None or top < int(clip)):
        return False

    dirt = parse_state_provider(config["dirt_provider"])
    trunk_state = parse_state_provider(config["trunk_provider"]).get(ctx, rng, *pos)
    nodes, logs = placer.generate(ctx, height, pos, rng, trunk_state, dirt,
                                  bool(config.get("force_dirt", False)))

    foliage = FoliagePlacer(config["foliage_placer"])
    foliage_height = foliage.height(rng, height)
    base_height = height - foliage_height
    radius = foliage.radius(rng, base_height)
    foliage_state = parse_state_provider(config["foliage_provider"]).get(ctx, rng, *pos)
    foliage_positions: List[Tuple[int, int, int]] = []
    for node in nodes:
        foliage_positions.extend(
            foliage.generate(ctx, rng, node, foliage_height, radius, foliage_state))

    decorators = [parse_decorator(entry) for entry in config.get("decorators") or ()]
    _decorate(ctx, rng, decorators, [], logs, foliage_positions)
    _update_leaf_distances(ctx, logs, foliage_positions)
    if not logs and not foliage_positions:
        return False
    return True


def _tree_free_height(ctx, size: FeatureSize, height: int, pos,
                      ignore_vines: bool) -> int:
    """`getMaxFreeTreeHeight`: how tall the tree can be before something blocks it.

    The scan starts at the *origin* and widens with the feature size, so a tree
    next to a wall is clipped by two blocks (`i - 2`) rather than rejected.
    """
    for y in range(0, height + 2):
        radius = size.radius(height, y)
        for x in range(-radius, radius + 1):
            for z in range(-radius, radius + 1):
                probe = (pos[0] + x, pos[1] + y, pos[2] + z)
                if _tree_position_free(ctx, probe, ignore_vines):
                    continue
                return max(y - 2, 0)
    return height


def _tree_position_free(ctx, pos, ignore_vines: bool) -> bool:
    """`validTreePos`: air or `#replaceable_by_trees`, plus the vine rule.

    `ignore_vines: true` means *vines do not block the tree*, which is vanilla's
    `validTreePos(...) || (ignoreVines && isVine(...))`. Pumpkin's port has this
    inverted for the vine case; vanilla is what the datapack's configs mean.
    """
    facts = ctx.facts
    state = ctx.get(*pos)
    if facts.in_tag(state, "minecraft:logs"):
        return True
    if facts.is_air(state) or facts.in_tag(state, "minecraft:replaceable_by_trees"):
        return True
    if ignore_vines and facts.block_of(state) == "minecraft:vine":
        return True
    return False


class TrunkPlacer:
    """The six trunk placers the datapack uses here.

    Height is `base + nextInt(a + 1) + nextInt(b + 1)` -- two draws, each with an
    inclusive bound, which is why `height_rand_b: 0` still costs one.
    """

    __slots__ = ("config", "kind")

    def __init__(self, tree_config: dict) -> None:
        self.config = tree_config["trunk_placer"]
        self.kind = str(self.config["type"]).removeprefix("minecraft:")

    def height(self, rng) -> int:
        base = int(self.config["base_height"])
        first = int(self.config.get("height_rand_a", 0))
        second = int(self.config.get("height_rand_b", 0))
        return base + rng.next_bounded_i32(first + 1) + rng.next_bounded_i32(second + 1)

    def generate(self, ctx, height: int, pos, rng, trunk_state, dirt, force_dirt: bool):
        facts = ctx.facts
        below = (pos[0], pos[1] - 1, pos[2])

        def set_dirt(target):
            # `setDirt` writes the provider's state when the block differs; with
            # `force_dirt` it writes unconditionally, which also means the
            # provider's own RNG draw still happens either way.
            state = dirt.get(ctx, rng, *target)
            if force_dirt or facts.block_of(ctx.get(*target)) != facts.block_of(state):
                ctx.set(*target, state)

        if self.kind in ("straight_trunk_placer", "forking_trunk_placer"):
            set_dirt(below)
        elif self.kind in ("fancy_trunk_placer",):
            set_dirt(below)
        elif self.kind in ("giant_trunk_placer", "mega_jungle_trunk_placer",
                           "dark_oak_trunk_placer"):
            set_dirt(below)
            set_dirt((below[0] + 1, below[1], below[2]))
            set_dirt((below[0], below[1], below[2] + 1))
            set_dirt((below[0] + 1, below[1], below[2] + 1))

        if self.kind == "straight_trunk_placer":
            return self._straight(ctx, height, pos, trunk_state)
        if self.kind == "forking_trunk_placer":
            return self._forking(ctx, height, pos, rng, trunk_state)
        if self.kind == "giant_trunk_placer":
            return self._giant(ctx, height, pos, trunk_state)
        if self.kind == "mega_jungle_trunk_placer":
            return self._mega_jungle(ctx, height, pos, rng, trunk_state)
        if self.kind == "dark_oak_trunk_placer":
            return self._dark_oak(ctx, height, pos, rng, trunk_state)
        if self.kind == "fancy_trunk_placer":
            return self._fancy(ctx, height, pos, rng, trunk_state)
        raise NotImplementedError(f"trunk placer {self.kind!r}")

    # The placers themselves. Each returns `(nodes, logs)`.

    def _straight(self, ctx, height, pos, trunk_state):
        logs = []
        for index in range(height):
            target = (pos[0], pos[1] + index, pos[2])
            if _trunk_place(ctx, target, trunk_state):
                logs.append(target)
        return [TreeNode((pos[0], pos[1] + height, pos[2]))], logs

    def _forking(self, ctx, height, pos, rng, trunk_state):
        nodes = []
        logs = []
        direction = HORIZONTAL_PLANE[rng.next_bounded_i32(4)]
        lean_height = height - rng.next_bounded_i32(4) - 1
        lean_steps = 3 - rng.next_bounded_i32(3)
        tx, tz = pos[0], pos[2]
        last_y = None
        for index in range(height):
            if index >= lean_height and lean_steps > 0:
                tx += direction[0]
                tz += direction[2]
                lean_steps -= 1
            target = (tx, pos[1] + index, tz)
            if _trunk_place(ctx, target, trunk_state):
                logs.append(target)
                last_y = pos[1] + index + 1
        if last_y is not None:
            nodes.append(TreeNode((tx, last_y, tz), 1))
        branch_direction = HORIZONTAL_PLANE[rng.next_bounded_i32(4)]
        if branch_direction != direction:
            branch_pos = lean_height - rng.next_bounded_i32(2) - 1
            branch_steps = 1 + rng.next_bounded_i32(3)
            bx, bz = pos[0], pos[2]
            branch_last = None
            cursor = branch_pos
            while cursor < height and branch_steps > 0:
                if cursor >= 1:
                    bx += branch_direction[0]
                    bz += branch_direction[2]
                    target = (bx, pos[1] + cursor, bz)
                    if _trunk_place(ctx, target, trunk_state):
                        logs.append(target)
                        branch_last = pos[1] + cursor + 1
                cursor += 1
                branch_steps -= 1
            if branch_last is not None:
                nodes.append(TreeNode((bx, branch_last, bz), 0))
        return nodes, logs

    def _giant(self, ctx, height, pos, trunk_state):
        logs = []
        for index in range(height):
            target = (pos[0], pos[1] + index, pos[2])
            if _trunk_try_place(ctx, target, trunk_state):
                logs.append(target)
            if index >= height - 1:
                continue
            for offset in ((1, 0), (1, 1), (0, 1)):
                other = (pos[0] + offset[0], pos[1] + index, pos[2] + offset[1])
                if _trunk_try_place(ctx, other, trunk_state):
                    logs.append(other)
        return [TreeNode((pos[0], pos[1] + height, pos[2]), 0, True)], logs

    def _mega_jungle(self, ctx, height, pos, rng, trunk_state):
        nodes, logs = self._giant(ctx, height, pos, trunk_state)
        cursor = height - 2 - rng.next_bounded_i32(4)
        j_val = 0
        k_val = 0
        while cursor > height // 2:
            angle = F32(rng.next_f32() * F32(2.0) * F32(math.pi))
            for index in range(5):
                j_val = int(F32(1.5) + F32(math.cos(float(angle))) * F32(index))
                k_val = int(F32(1.5) + F32(math.sin(float(angle))) * F32(index))
                target = (pos[0] + j_val, pos[1] + cursor - 3 + index // 2, pos[2] + k_val)
                if _trunk_try_place(ctx, target, trunk_state):
                    logs.append(target)
            nodes.append(TreeNode((pos[0] + j_val, pos[1] + cursor, pos[2] + k_val), -2))
            cursor -= 2 + rng.next_bounded_i32(4)
        return nodes, logs

    def _dark_oak(self, ctx, height, pos, rng, trunk_state):
        logs = []
        direction = HORIZONTAL_PLANE[rng.next_bounded_i32(4)]
        lean_height = height - rng.next_bounded_i32(4)
        lean_steps = 2 - rng.next_bounded_i32(3)
        tx, tz = pos[0], pos[2]
        end_y = pos[1] + height - 1
        for index in range(height):
            if index >= lean_height and lean_steps > 0:
                tx += direction[0]
                tz += direction[2]
                lean_steps -= 1
            target = (tx, pos[1] + index, tz)
            if _tree_is_air_or_leaves(ctx, target):
                for offset in ((0, 0), (1, 0), (0, 1), (1, 1)):
                    candidate = (target[0] + offset[0], target[1], target[2] + offset[1])
                    if _trunk_try_place(ctx, candidate, trunk_state):
                        logs.append(candidate)
        nodes = [TreeNode((tx, end_y, tz), 0, True)]
        for ox in range(-1, 3):
            for oz in range(-1, 3):
                if (0 <= ox <= 1 and 0 <= oz <= 1) or rng.next_bounded_i32(3) > 0:
                    continue
                length = rng.next_bounded_i32(3) + 2
                for branch_y in range(length):
                    target = (pos[0] + ox, end_y - branch_y - 1, pos[2] + oz)
                    if _trunk_try_place(ctx, target, trunk_state):
                        logs.append(target)
                nodes.append(TreeNode((pos[0] + ox, end_y, pos[2] + oz)))
        return nodes, logs

    def _fancy(self, ctx, height, pos, rng, trunk_state):
        """`fancy_trunk_placer`: a curved trunk with branches on a 5-block cap.

        `l` -- the number of branches attempted per level -- is
        `min(floor(1.382 + (treeHeight / 13)^2), 1)`, i.e. it is *clamped down to
        1* for every tree in the datapack. That looks like a bug and is vanilla's
        actual expression, so the branch fan-out is one per level here too.
        """
        total = height + 2
        mid = int(math.floor(total * 0.618))
        length = min(int(math.floor(1.382 + (1.0 * total / 13.0) ** 2)), 1)
        base_y = pos[1] + mid
        branches = [(TreeNode((pos[0], base_y, pos[2])), base_y)]
        logs: List[Tuple[int, int, int]] = []
        for level in range(total - 5, -1, -1):
            weight = _fancy_branch_weight(total, level)
            if weight < 0.0:
                continue
            for _ in range(length):
                magnitude = float(F32(F32(weight) * F32(rng.next_f32() + F32(0.328))))
                angle = float(F32(rng.next_f32() * F32(2.0) * F32(math.pi)))
                px = math.floor(magnitude * math.sin(angle) + 0.5)
                pz = math.floor(magnitude * math.cos(angle) + 0.5)
                end = (pos[0] + int(px), pos[1] + level - 1, pos[2] + int(pz))
                top_end = (end[0], end[1] + 5, end[2])
                ok, new_logs = _fancy_branch(ctx, (pos[0], pos[1], pos[2]), end,
                                             trunk_state, False)
                logs.extend(new_logs)
                if not ok:
                    continue
                dx = pos[0] - end[0]
                dz = pos[2] - end[2]
                candidate_y = end[1] - int(math.sqrt(dx * dx + dz * dz) * 0.381)
                node_y = base_y if candidate_y > base_y else candidate_y
                node = (pos[0], node_y, pos[2])
                ok, new_logs = _fancy_branch(ctx, node, end, trunk_state, False)
                logs.extend(new_logs)
                if not ok:
                    continue
                branches.append((TreeNode(end), node[1]))
        ok, new_logs = _fancy_branch(ctx, (pos[0], pos[1], pos[2]),
                                     (pos[0], pos[1] + mid, pos[2]), trunk_state, True)
        logs.extend(new_logs)
        for node, end_y in branches:
            if node.center == (pos[0], end_y, pos[2]):
                continue
            if not _fancy_high_enough(total, end_y - pos[1]):
                continue
            target = (pos[0], end_y, pos[2])
            ok, new_logs = _fancy_branch(ctx, target, node.center, trunk_state, True)
            logs.extend(new_logs)

        nodes = [node for node, end_y in branches
                 if _fancy_high_enough(total, end_y - pos[1])]
        return nodes, logs


def _fancy_branch_weight(tree_height: int, height: int) -> float:
    if F32(height) < F32(tree_height) * F32(0.3):
        return -1.0
    half = F32(tree_height) / F32(2.0)
    delta = half - F32(height)
    if delta == 0.0:
        return float(F32(math.sqrt(float(F32(half * half)))))
    if abs(float(delta)) >= float(half):
        return 0.0
    inner = F32(half * half - delta * delta)
    return float(F32(math.sqrt(float(inner))) * F32(0.5))


def _fancy_high_enough(tree_height: int, height: int) -> bool:
    return float(height) >= float(tree_height) * 0.2


def _fancy_branch(ctx, start, end, trunk_state, make: bool):
    """A straight run of logs between two positions, axis-aligned per block."""
    if not make and start == end:
        return True, []
    offset = (end[0] - start[0], end[1] - start[1], end[2] - start[2])
    longest = max(abs(offset[0]), abs(offset[1]), abs(offset[2]))
    if longest == 0:
        return True, []
    fx = F32(offset[0]) / F32(longest)
    fy = F32(offset[1]) / F32(longest)
    fz = F32(offset[2]) / F32(longest)
    logs = []
    for step in range(longest + 1):
        target = (start[0] + int(math.floor(float(F32(0.5) + F32(step) * fx))),
                  start[1] + int(math.floor(float(F32(0.5) + F32(step) * fy))),
                  start[2] + int(math.floor(float(F32(0.5) + F32(step) * fz))))
        state = ctx.get(*target)
        if make:
            if _can_replace(ctx, target):
                axis = "y"
                dx = abs(target[0] - start[0])
                dz = abs(target[2] - start[2])
                if max(dx, dz) > 0:
                    axis = "x" if dx == max(dx, dz) else "z"
                ctx.set(*target, ctx.facts.with_property(trunk_state, "axis", axis))
                logs.append(target)
                continue
        if _can_replace_or_log(ctx, target):
            continue
        return False, logs
    return True, logs


def _trunk_place(ctx, pos, trunk_state) -> bool:
    """`TrunkPlacer.place`: only into replaceable blocks, *not* into logs.

    The distinction from `try_place` matters: a straight trunk that hits an
    existing log stops growing, while a giant trunk overwrites it.
    """
    if not _can_replace(ctx, pos):
        return False
    ctx.set(*pos, trunk_state)
    return True


def _trunk_try_place(ctx, pos, trunk_state) -> bool:
    if not _can_replace_or_log(ctx, pos):
        return False
    return _trunk_place(ctx, pos, trunk_state)


def _tree_is_air_or_leaves(ctx, pos) -> bool:
    facts = ctx.facts
    state = ctx.get(*pos)
    return facts.is_air(state) or facts.in_tag(state, "minecraft:leaves")


class FoliagePlacer:
    """The eleven foliage placers the datapack uses.

    `place_square` walks `-radius..=radius` (plus one on each side for a giant
    trunk), rejects some positions through a validator, and only records the
    ones that were actually written. The validator runs *before* the write and
    may consume RNG draws, which is why the rejection test is part of the shape
    rather than a filter applied afterwards.
    """

    __slots__ = ("config", "kind", "offset_provider", "radius_provider")

    def __init__(self, payload: dict) -> None:
        self.config = payload
        self.kind = str(payload["type"]).removeprefix("minecraft:")
        self.offset_provider = parse_int_provider(payload.get("offset", 0))
        self.radius_provider = parse_int_provider(payload.get("radius", 0))

    def height(self, rng, trunk_height: int) -> int:
        if self.kind == "blob_foliage_placer":
            return int(self.config["height"])
        if self.kind in ("bush_foliage_placer",):
            return int(self.config["height"])
        if self.kind == "fancy_foliage_placer":
            return int(self.config["height"])
        if self.kind == "jungle_foliage_placer":
            return int(self.config["height"])
        if self.kind == "dark_oak_foliage_placer":
            return 4
        if self.kind == "acacia_foliage_placer":
            return 0
        if self.kind == "spruce_foliage_placer":
            trunk = parse_int_provider(self.config["trunk_height"]).get(rng)
            return max(trunk_height - trunk, 4)
        if self.kind == "pine_foliage_placer":
            return parse_int_provider(self.config["height"]).get(rng)
        if self.kind == "mega_pine_foliage_placer":
            return parse_int_provider(self.config["crown_height"]).get(rng)
        if self.kind == "random_spread_foliage_placer":
            return parse_int_provider(self.config["foliage_height"]).get(rng)
        raise NotImplementedError(f"foliage placer {self.kind!r}")

    def radius(self, rng, base_height: int) -> int:
        if self.kind == "pine_foliage_placer":
            return self.radius_provider.get(rng) + rng.next_bounded_i32(max(base_height + 1, 1))
        return self.radius_provider.get(rng)

    def generate(self, ctx, rng, node: TreeNode, foliage_height: int, radius: int,
                 foliage_state: int) -> List[Tuple[int, int, int]]:
        offset = self.offset_provider.get(rng)
        out: List[Tuple[int, int, int]] = []
        if self.kind == "blob_foliage_placer":
            for y in range(offset, offset - foliage_height - 1, -1):
                row_radius = max(radius + node.foliage_radius - 1 - y // 2, 0)
                self._square(ctx, rng, out, node, row_radius, y, foliage_state,
                             lambda dx, dz, r, yy: dx == r and dz == r
                             and (rng.next_bounded_i32(2) == 0 or yy == 0))
        elif self.kind in ("fancy_foliage_placer",):
            for y in range(offset, offset - foliage_height - 1, -1):
                row_radius = radius + (0 if y in (offset, offset - foliage_height) else 1)
                self._square(ctx, rng, out, node, row_radius, y, foliage_state,
                             lambda dx, dz, r, yy: F32(dx + 0.5) ** 2 + F32(dz + 0.5) ** 2
                             > F32(r * r))
        elif self.kind == "bush_foliage_placer":
            for y in range(offset, offset - foliage_height - 1, -1):
                row_radius = radius + node.foliage_radius - 1 - y
                self._square(ctx, rng, out, node, row_radius, y, foliage_state,
                             lambda dx, dz, r, yy: dx == r and dz == r
                             and rng.next_bounded_i32(2) == 0)
        elif self.kind == "jungle_foliage_placer":
            height = foliage_height if node.giant_trunk else 1 + rng.next_bounded_i32(2)
            for y in range(offset, offset - height - 1, -1):
                row_radius = radius + node.foliage_radius + 1 - y
                self._square(ctx, rng, out, node, row_radius, y, foliage_state,
                             lambda dx, dz, r, yy: dx + dz >= 7 or dx * dx + dz * dz > r * r)
        elif self.kind == "acacia_foliage_placer":
            self._square(ctx, rng, out, node, radius + node.foliage_radius, -1, foliage_state,
                         lambda dx, dz, r, yy: dx == r and dz == r and r > 0)
            self._square(ctx, rng, out, node, radius - 1, -foliage_height, foliage_state,
                         lambda dx, dz, r, yy: dx == r and dz == r and r > 0)
            self._square(ctx, rng, out, node, radius + node.foliage_radius - 1, 0, foliage_state,
                         lambda dx, dz, r, yy: (dx > 1 or dz > 1) and dx != 0 and dz != 0)
        elif self.kind == "dark_oak_foliage_placer":
            center = (node.center[0], node.center[1] + offset, node.center[2])
            save = node.center
            node.center = center
            self._square(ctx, rng, out, node, radius + 2, -1, foliage_state,
                         _dark_oak_validator)
            if node.giant_trunk:
                self._square(ctx, rng, out, node, radius + 3, 0, foliage_state,
                             _dark_oak_validator)
                self._square(ctx, rng, out, node, radius + 2, 1, foliage_state,
                             _dark_oak_validator)
                if rng.next_bool():
                    self._square(ctx, rng, out, node, radius, 2, foliage_state,
                                 _dark_oak_validator)
            else:
                self._square(ctx, rng, out, node, radius + 1, 0, foliage_state,
                             _dark_oak_validator)
            node.center = save
        elif self.kind == "spruce_foliage_placer":
            row_radius = rng.next_bounded_i32(2)
            maximum = 1
            nxt = 0
            for y in range(offset, -foliage_height - 1, -1):
                self._square(ctx, rng, out, node, row_radius, y, foliage_state,
                             lambda dx, dz, r, yy: dx == r and dz == r and r > 0)
                if row_radius >= maximum:
                    row_radius = nxt
                    nxt = 1
                    maximum = min(radius + node.foliage_radius, maximum + 1)
                    continue
                row_radius += 1
        elif self.kind == "pine_foliage_placer":
            row_radius = 0
            for y in range(offset, offset - foliage_height - 1, -1):
                self._square(ctx, rng, out, node, row_radius, y, foliage_state,
                             lambda dx, dz, r, yy: dx == r and dz == r and r > 0)
                if row_radius >= 1 and y == offset - foliage_height + 1:
                    row_radius -= 1
                elif row_radius < radius + node.foliage_radius:
                    row_radius += 1
        elif self.kind == "mega_pine_foliage_placer":
            current = 0
            for y in range(node.center[1] - foliage_height + offset,
                           node.center[1] + offset + 1):
                delta = node.center[1] - y
                computed = radius + node.foliage_radius + int(
                    math.floor((delta / foliage_height) * 3.5))
                row_radius = computed + 1 if (delta > 0 and computed == current
                                              and (y & 1) == 0) else computed
                # The rows move with `y`, so the node is re-centred per row; the
                # row itself is placed at offset 0 from that centre.
                row_node = TreeNode((node.center[0], y, node.center[2]),
                                    node.foliage_radius, node.giant_trunk)
                self._square(ctx, rng, out, row_node, row_radius, 0, foliage_state,
                             lambda dx, dz, r, yy: dx + dz >= 7 or dx * dx + dz * dz > r * r)
                current = computed
        elif self.kind == "random_spread_foliage_placer":
            attempts = int(self.config["leaf_placement_attempts"])
            for _ in range(attempts):
                target = (node.center[0] + rng.next_bounded_i32(radius)
                          - rng.next_bounded_i32(radius),
                          node.center[1] + rng.next_bounded_i32(foliage_height)
                          - rng.next_bounded_i32(foliage_height),
                          node.center[2] + rng.next_bounded_i32(radius)
                          - rng.next_bounded_i32(radius))
                if _place_foliage(ctx, target, foliage_state):
                    out.append(target)
        else:
            raise NotImplementedError(f"foliage placer {self.kind!r}")
        return out

    def _square(self, ctx, rng, out, node: TreeNode, row_radius: int, y: int,
                foliage_state: int, invalid) -> None:
        """`FoliagePlacer.generateSquare`, validator first and draw included.

        The giant-trunk flag folds the x/z offsets by one, which is what turns a
        four-trunk dark oak canopy into a rounded one instead of a square.
        """
        extra = 1 if node.giant_trunk else 0
        for dx in range(-row_radius, row_radius + extra + 1):
            for dz in range(-row_radius, row_radius + extra + 1):
                fx = min(abs(dx), abs(dx - 1)) if node.giant_trunk else abs(dx)
                fz = min(abs(dz), abs(dz - 1)) if node.giant_trunk else abs(dz)
                if invalid(fx, fz, row_radius, y):
                    continue
                target = (node.center[0] + dx, node.center[1] + y, node.center[2] + dz)
                if _place_foliage(ctx, target, foliage_state):
                    out.append(target)


def _place_foliage(ctx, pos, foliage_state) -> bool:
    if not _can_replace(ctx, pos):
        return False
    ctx.set(*pos, foliage_state)
    return True


def _dark_oak_validator(dx: int, dz: int, radius: int, y: int) -> bool:
    if y == -1:
        return dx == radius and dz == radius
    if y == 1:
        return dx + dz > radius * 2 - 2
    return False


def _update_leaf_distances(ctx, logs, leaves) -> None:
    """Recompute every leaf's `distance` property (vanilla's 7-bucket BFS).

    Leaves decay when they are more than 7 blocks from a log, so the property is
    not cosmetic: it is what the client uses to tick decay. The sweep runs over
    the bounding box of logs *and* leaves, treats `#minecraft:logs` as distance
    0, and rewrites existing leaves as well as new ones.
    """
    facts = ctx.facts
    if not logs and not leaves:
        return
    positions = list(logs) + list(leaves)
    min_x = min(p[0] for p in positions)
    min_y = min(p[1] for p in positions)
    min_z = min(p[2] for p in positions)
    max_x = max(p[0] for p in positions)
    max_y = max(p[1] for p in positions)
    max_z = max(p[2] for p in positions)

    def index(pos):
        x, y, z = pos
        if not (min_x <= x <= max_x and min_y <= y <= max_y and min_z <= z <= max_z):
            return None
        return ((x - min_x) * (max_y - min_y + 1) + (y - min_y)) * (max_z - min_z + 1) \
            + (z - min_z)

    visited = [False] * ((max_x - min_x + 1) * (max_y - min_y + 1) * (max_z - min_z + 1))
    buckets: List[List[Tuple[int, int, int]]] = [[] for _ in range(7)]
    for pos in logs:
        idx = index(pos)
        if idx is not None:
            buckets[0].append(pos)
    smallest = 0
    while smallest < 7:
        while smallest < 7 and buckets[smallest]:
            pos = buckets[smallest].pop(0)
            idx = index(pos)
            if idx is None:
                continue
            if smallest != 0:
                state = ctx.get(*pos)
                if facts.has_property(state, "distance"):
                    ctx.set(*pos, facts.with_property(state, "distance", str(smallest)))
            visited[idx] = True
            for offset in DIRECTIONS:
                neighbour = (pos[0] + offset[0], pos[1] + offset[1], pos[2] + offset[2])
                n_idx = index(neighbour)
                if n_idx is None or visited[n_idx]:
                    continue
                n_state = ctx.get(*neighbour)
                if facts.in_tag(n_state, "minecraft:prevents_nearby_leaf_decay"):
                    distance: Optional[int] = 0
                elif facts.has_property(n_state, "distance"):
                    distance = int(facts.property(n_state, "distance") or 7)
                else:
                    distance = None
                if distance is None:
                    continue
                new_distance = min(distance, smallest + 1)
                if new_distance < 7:
                    buckets[new_distance].append(neighbour)
                    smallest = min(smallest, new_distance)
        smallest += 1


def _place_multiface_growth(ctx, config, rng, pos) -> bool:
    """`multiface_growth`: glow lichen and sculk veins on the faces of a cave.

    Two traps. The candidate directions are built in a fixed order (ceiling,
    floor, then the four walls) and *shuffled with one draw per element*, so the
    order the faces are tried in is part of the RNG contract. And the search
    walks outward in each shuffled direction, testing a freshly shuffled
    direction list at every step -- the shuffles are what consume the stream, so
    "optimising" them into one list per direction changes the result.
    """
    facts = ctx.facts
    block = _canonical(str(config["block"]))
    place_on = set()
    for entry in config["can_be_placed_on"]:
        entry = str(entry)
        if entry.startswith("#"):
            place_on.update(facts.tag_blocks(entry))
        else:
            place_on.add(_canonical(entry))
    search_range = int(config["search_range"])

    def valid_directions() -> List[Tuple[int, int, int]]:
        directions = []
        if config.get("can_place_on_ceiling"):
            directions.append((0, 1, 0))
        if config.get("can_place_on_floor"):
            directions.append((0, -1, 0))
        if config.get("can_place_on_wall"):
            directions.extend(HORIZONTAL_PLANE)
        return directions

    def shuffle(directions: List[Tuple[int, int, int]]) -> List[Tuple[int, int, int]]:
        for index in range(len(directions) - 1, 0, -1):
            other = rng.next_bounded_i32(index + 1)
            directions[index], directions[other] = directions[other], directions[index]
        return directions

    def air_or_water(target) -> bool:
        return facts.block_of(ctx.get(*target)) in ("minecraft:air", "minecraft:water")

    def place_if_possible(target, directions) -> bool:
        for direction in directions:
            neighbour = (target[0] + direction[0], target[1] + direction[1],
                         target[2] + direction[2])
            if facts.block_of(ctx.get(*neighbour)) not in place_on:
                continue
            waterlogged = facts.block_of(ctx.get(*target)) == "minecraft:water"
            state = facts.default_state(block)
            for name, offset in (("down", (0, -1, 0)), ("up", (0, 1, 0)),
                                 ("north", (0, 0, -1)), ("south", (0, 0, 1)),
                                 ("west", (-1, 0, 0)), ("east", (1, 0, 0))):
                state = facts.with_property(state, name,
                                            "true" if direction == offset else "false")
            if facts.has_property(state, "waterlogged"):
                state = facts.with_property(state, "waterlogged",
                                            "true" if waterlogged else "false")
            current = ctx.get(*target)
            if facts.block_of(current) == block:
                for name in ("down", "up", "north", "south", "west", "east"):
                    if facts.has_property(current, name) and \
                            facts.property(current, name) == "true":
                        state = facts.with_property(state, name, "true")
            ctx.set(*target, state)
            return True
        return False

    if not air_or_water(pos):
        return False
    directions = shuffle(valid_directions())
    if place_if_possible(pos, directions):
        return True
    for search in directions:
        opposite = (-search[0], -search[1], -search[2])
        placement = shuffle([d for d in valid_directions() if d != opposite])
        cursor = pos
        for _ in range(search_range):
            cursor = (cursor[0] + search[0], cursor[1] + search[1], cursor[2] + search[2])
            if not air_or_water(cursor) and facts.block_of(ctx.get(*cursor)) != block:
                break
            if place_if_possible(cursor, placement):
                return True
    return False


def _place_underwater_magma(ctx, config, rng, pos) -> bool:
    """`underwater_magma`: magma on the floor under a water column.

    The `nextFloat()` is drawn for every cell of the cube *before* the validity
    test, so the number of draws is `(2r+1)^3` regardless of what is there;
    testing first and drawing after makes the feature cheap and the world wrong.
    """
    facts = ctx.facts
    search_range = int(config["floor_search_range"])
    radius = int(config["placement_radius_around_floor"])
    probability = F32(config["placement_probability_per_valid_position"])

    def is_water(target) -> bool:
        return facts.block_of(ctx.get(*target)) == "minecraft:water"

    floor_y = None
    for dy in range(search_range + 1):
        check_y = pos[1] - dy
        if is_water((pos[0], check_y, pos[2])):
            candidate = check_y - 1
            while True:
                if not is_water((pos[0], candidate, pos[2])):
                    floor_y = candidate
                    break
                if check_y - candidate > search_range:
                    break
                candidate -= 1
            if floor_y is not None:
                break
    if floor_y is None:
        return False

    def valid(target) -> bool:
        if facts.block_of(ctx.get(*target)) in ("minecraft:water", "minecraft:air"):
            return False
        below = (target[0], target[1] - 1, target[2])
        if facts.block_of(ctx.get(*below)) in ("minecraft:water", "minecraft:air"):
            return False
        for offset in HORIZONTAL_PLANE:
            neighbour = (target[0] + offset[0], target[1], target[2] + offset[2])
            if facts.block_of(ctx.get(*neighbour)) in ("minecraft:water", "minecraft:air"):
                return False
        return True

    magma = facts.default_state("minecraft:magma_block")
    placed = 0
    for dx in range(-radius, radius + 1):
        for dy in range(-radius, radius + 1):
            for dz in range(-radius, radius + 1):
                if rng.next_f32() >= probability:
                    continue
                target = (pos[0] + dx, floor_y + dy, pos[2] + dz)
                if not valid(target):
                    continue
                ctx.set(*target, magma)
                placed += 1
    return placed > 0


def _place_geode(ctx, config, rng, pos) -> bool:
    """`geode`: nested shells of amethyst around scattered distribution points.

    The surface is an implicit field, not a sphere: each block's value is
    `sum(1 / sqrt(dist^2 + point_offset))` over the distribution points, plus a
    noise offset, and the shell it lands in is decided by comparing that against
    four thresholds derived from the layer distances. Every threshold is a
    *reciprocal square root* of a layer distance plus a crack-size adjustment,
    which is why changing the point count slightly moves every shell.

    Crystal placements are collected while the shells are written and only
    placed afterwards, one per budding block, which is what keeps the interior
    from growing a crystal for every air pocket it passes.
    """
    facts = ctx.facts
    blocks = config["blocks"]
    filling = parse_state_provider(blocks["filling_provider"])
    inner = parse_state_provider(blocks["inner_layer_provider"])
    alternate = parse_state_provider(blocks["alternate_inner_layer_provider"])
    middle = parse_state_provider(blocks["middle_layer_provider"])
    outer = parse_state_provider(blocks["outer_layer_provider"])
    inner_placements = [facts.state_id(state) for state in blocks["inner_placements"]]
    cannot_replace = set()
    for entry in _block_set(blocks["cannot_replace"]):
        cannot_replace.update(facts.tag_blocks(entry[1:]) if entry.startswith("#")
                              else {_canonical(entry)})
    invalid_blocks = set()
    for entry in _block_set(blocks["invalid_blocks"]):
        invalid_blocks.update(facts.tag_blocks(entry[1:]) if entry.startswith("#")
                              else {_canonical(entry)})

    layers = config["layers"]
    crack = config["crack"]
    outer_wall = parse_int_provider(config["outer_wall_distance"])
    point_offset = parse_int_provider(config["point_offset"])

    def can_replace(target) -> bool:
        state = ctx.get(*target)
        return facts.is_air(state) or facts.block_of(state) not in cannot_replace

    def safe_set(target, state) -> None:
        if can_replace(target):
            ctx.set(*target, state)

    num_points = parse_int_provider(config["distribution_points"]).get(rng)
    from world.vanilla.perlin import NormalNoise

    noise = NormalNoise(_WorldgenSplitter(rng), -4, (1.0,))
    crack_adjustment = num_points / float(
        _provider_max(outer_wall, config["outer_wall_distance"]))

    inner_air = 1.0 / math.sqrt(float(layers["filling"]))
    innermost = 1.0 / math.sqrt(float(layers["inner_layer"]) + crack_adjustment)
    inner_crust = 1.0 / math.sqrt(float(layers["middle_layer"]) + crack_adjustment)
    outer_crust = 1.0 / math.sqrt(float(layers["outer_layer"]) + crack_adjustment)
    crack_size = 1.0 / math.sqrt(float(crack["base_crack_size"])
                                 + rng.next_f64() / 2.0
                                 + (crack_adjustment if num_points > 3 else 0.0))
    should_crack = rng.next_f32() < F32(crack["generate_crack_chance"])

    points: List[Tuple[Tuple[int, int, int], int]] = []
    invalid = 0
    threshold = int(config["invalid_blocks_threshold"])
    for _ in range(num_points):
        point = (pos[0] + outer_wall.get(rng), pos[1] + outer_wall.get(rng),
                 pos[2] + outer_wall.get(rng))
        state = ctx.get(*point)
        if facts.is_air(state) or facts.block_of(state) in invalid_blocks:
            invalid += 1
            if invalid > threshold:
                return False
        points.append((point, point_offset.get(rng)))

    crack_points: List[Tuple[int, int, int]] = []
    if should_crack:
        index = rng.next_bounded_i32(4)
        offset = num_points * 2 + 1
        for height in (7, 5, 1):
            if index == 0:
                crack_points.append((pos[0] + offset, pos[1] + height, pos[2]))
            elif index == 1:
                crack_points.append((pos[0], pos[1] + height, pos[2] + offset))
            elif index == 2:
                crack_points.append((pos[0] + offset, pos[1] + height, pos[2] + offset))
            else:
                crack_points.append((pos[0], pos[1] + height, pos[2]))

    potential: List[Tuple[int, int, int]] = []
    low = config["min_gen_offset"]
    high = config["max_gen_offset"]
    multiplier = float(config["noise_multiplier"])
    crawl = float(crack["crack_point_offset"])
    alternate_chance = F32(config["use_alternate_layer0_chance"])
    potential_chance = F32(config["use_potential_placements_chance"])
    require_alternate = bool(config["placements_require_layer0_alternate"])

    for target in _iter_box((pos[0] + low, pos[1] + low, pos[2] + low),
                            (pos[0] + high, pos[1] + high, pos[2] + high)):
        noise_offset = float(noise.get(target[0], target[1], target[2])) * multiplier
        shell = 0.0
        for point, offset in points:
            dx = target[0] - point[0]
            dy = target[1] - point[1]
            dz = target[2] - point[2]
            shell += 1.0 / math.sqrt(dx * dx + dy * dy + dz * dz + offset) + noise_offset
        if shell < outer_crust:
            continue
        crack_sum = 0.0
        for point in crack_points:
            dx = target[0] - point[0]
            dy = target[1] - point[1]
            dz = target[2] - point[2]
            crack_sum += 1.0 / math.sqrt(dx * dx + dy * dy + dz * dz + crawl) + noise_offset
        if should_crack and crack_sum >= crack_size and shell < inner_air:
            safe_set(target, 0)
        elif shell >= inner_air:
            safe_set(target, filling.get(ctx, rng, *target))
        elif shell >= innermost:
            used_alternate = rng.next_f32() < alternate_chance
            provider = alternate if used_alternate else inner
            safe_set(target, provider.get(ctx, rng, *target))
            if (not require_alternate or used_alternate) and rng.next_f32() < potential_chance:
                potential.append(target)
        elif shell >= inner_crust:
            safe_set(target, middle.get(ctx, rng, *target))
        elif shell >= outer_crust:
            safe_set(target, outer.get(ctx, rng, *target))

    for crystal in potential:
        base = inner_placements[rng.next_bounded_i32(len(inner_placements))]
        for direction in DIRECTIONS:
            target = (crystal[0] + direction[0], crystal[1] + direction[1],
                      crystal[2] + direction[2])
            state = ctx.get(*target)
            if not facts.is_air(state) and facts.block_of(state) != "minecraft:water":
                continue
            if facts.has_property(base, "facing"):
                base = facts.with_property(base, "facing", _direction_name(direction))
            if facts.has_property(base, "waterlogged"):
                base = facts.with_property(
                    base, "waterlogged",
                    "true" if facts.block_of(state) == "minecraft:water" else "false")
            safe_set(target, base)
            break
    return True


class _WorldgenSplitter:
    """Gives the feature's `WorldgenRandom` the `next_splitter` API.

    Vanilla's `NormalNoise` constructor draws two longs from whatever source it
    is handed and builds a Xoroshiro from them -- even when that source is the
    feature's own `WorldgenRandom`, whose `nextLong` is two 32-bit draws. The
    geode's noise therefore advances the feature RNG by four draws, and skipping
    that moves every subsequent block the feature places.
    """

    __slots__ = ("_random",)

    def __init__(self, random: WorldgenRandom) -> None:
        self._random = random

    def next_splitter(self) -> XoroshiroSplitter:
        return XoroshiroSplitter(self._random.next_i64(), self._random.next_i64())


def _direction_name(offset: Tuple[int, int, int]) -> str:
    return {
        (0, 1, 0): "up", (0, -1, 0): "down", (0, 0, -1): "north",
        (0, 0, 1): "south", (-1, 0, 0): "west", (1, 0, 0): "east",
    }[offset]


def _block_set(entry) -> List[str]:
    if isinstance(entry, str):
        return [entry]
    return [str(value) for value in entry]


def _provider_max(provider, payload) -> int:
    """The *upper bound* of an int provider, for the crack-size adjustment.

    `getMaxValue()` is the provider's maximum, not a sample: it must not consume
    a draw, and a `uniform` provider's maximum is `max_inclusive`.
    """
    if isinstance(provider, UniformInt):
        return provider.maximum
    return int(payload.get("max_inclusive", 1))


def _place_huge_fungus(ctx, config, rng, pos) -> bool:
    """Not implemented: the nether's fungi.

    Listed by `missing_types` and counted in `FeatureStats.unimplemented`; there
    is no path through `decorate` that reaches this function, because a type is
    only dispatched when it is in `_FEATURE_PLACERS`.
    """
    raise NotImplementedError("huge_fungus")


# ----- tree decorators -----


class Decorator:
    __slots__ = ("config", "kind")

    def __init__(self, payload: dict) -> None:
        self.config = payload
        self.kind = str(payload["type"]).removeprefix("minecraft:")


def parse_decorator(payload: dict) -> Decorator:
    decorator = Decorator(payload)
    if decorator.kind not in _DECORATORS:
        raise NotImplementedError(f"tree decorator {decorator.kind!r}")
    return decorator


def _decorate(ctx, rng, decorators, roots, logs, foliage) -> None:
    for decorator in decorators:
        _DECORATORS[decorator.kind](ctx, decorator.config, rng, roots, logs, foliage)


def _decorate_beehive(ctx, config, rng, roots, logs, foliage) -> None:
    """`beehive`: a bee nest on the first log row below the canopy, facing south.

    The candidate list is built in log order and then shuffled with a
    `nextInt(i + 1)` swap loop running *down* from the end -- the standard
    Fisher-Yates, which consumes one draw per element and is the usual place a
    "shuffle" gets replaced by something cheaper and wrong.
    """
    if not logs or rng.next_f32() >= F32(config["probability"]):
        return
    if foliage:
        hive_y = max(foliage[0][1] - 1, logs[0][1] + 1)
    else:
        last_y = logs[-1][1]
        hive_y = min(logs[0][1] + 1 + rng.next_bounded_i32(3), last_y)
    candidates = []
    for log in logs:
        if log[1] != hive_y:
            continue
        candidates.extend([(log[0] + 1, log[1], log[2]),
                           (log[0] - 1, log[1], log[2]),
                           (log[0], log[1], log[2] + 1)])
    if not candidates:
        return
    for index in range(len(candidates) - 1, 0, -1):
        other = rng.next_bounded_i32(index + 1)
        candidates[index], candidates[other] = candidates[other], candidates[index]
    facts = ctx.facts
    for candidate in candidates:
        front = (candidate[0], candidate[1], candidate[2] + 1)
        if ctx.is_air(*candidate) and ctx.is_air(*front):
            nest = facts.with_property(facts.default_state("minecraft:bee_nest"),
                                       "facing", "south")
            ctx.set(*candidate, nest)
            return


def _decorate_alter_ground(ctx, config, rng, roots, logs, foliage) -> None:
    provider = parse_state_provider(config["provider"])
    reference = roots if roots else logs
    if not reference:
        return
    min_y = min(pos[1] for pos in reference)
    for pos in [entry for entry in reference if entry[1] == min_y]:
        _alter_circle(ctx, provider, rng, (pos[0] - 1, pos[1], pos[2] - 1))
        _alter_circle(ctx, provider, rng, (pos[0] + 2, pos[1], pos[2] - 1))
        _alter_circle(ctx, provider, rng, (pos[0] - 1, pos[1], pos[2] + 2))
        _alter_circle(ctx, provider, rng, (pos[0] + 2, pos[1], pos[2] + 2))
        for _ in range(5):
            placement = rng.next_bounded_i32(64)
            xx = placement % 8
            zz = placement // 8
            if xx in (0, 7) or zz in (0, 7):
                _alter_circle(ctx, provider, rng, (pos[0] - 3 + xx, pos[1], pos[2] - 3 + zz))


def _alter_circle(ctx, provider, rng, center) -> None:
    for xx in range(-2, 3):
        for zz in range(-2, 3):
            if abs(xx) == 2 and abs(zz) == 2:
                continue
            _alter_block(ctx, provider, rng, (center[0] + xx, center[1], center[2] + zz))


def _alter_block(ctx, provider, rng, pos) -> None:
    for dy in range(2, -4, -1):
        cursor = (pos[0], pos[1] + dy, pos[2])
        state = provider.get(ctx, rng, *cursor)
        if ctx.facts.is_air(state) or ctx.is_air(*cursor) or \
                ctx.facts.is_replaceable(ctx.get(*cursor)):
            ctx.set(*cursor, state)
            return
        if not ctx.is_air(*cursor) and dy < 0:
            return


def _decorate_place_on_ground(ctx, config, rng, roots, logs, foliage) -> None:
    provider = parse_state_provider(config["block_state_provider"])
    reference = roots if roots else logs
    if not reference:
        return
    base_y = reference[0][1]
    min_x = min((p[0] for p in reference if p[1] == base_y), default=reference[0][0])
    max_x = max((p[0] for p in reference if p[1] == base_y), default=reference[0][0])
    min_z = min((p[2] for p in reference if p[1] == base_y), default=reference[0][2])
    max_z = max((p[2] for p in reference if p[1] == base_y), default=reference[0][2])
    radius = int(config["radius"])
    height = int(config["height"])
    tries = int(config["tries"])
    for _ in range(tries):
        x = min_x - radius + rng.next_bounded_i32((max_x + radius) - (min_x - radius) + 1)
        y = base_y - height + rng.next_bounded_i32((base_y + height) - (base_y - height) + 1)
        z = min_z - radius + rng.next_bounded_i32((max_z + radius) - (min_z - radius) + 1)
        state = ctx.get(x, y, z)
        below = ctx.get(x, y - 1, z)
        if (ctx.facts.is_air(state) or ctx.facts.block_of(state) == "minecraft:vine") \
                and ctx.facts.is_full_cube(below) \
                and not ctx.facts.in_tag(below, "minecraft:leaves"):
            ctx.set(x, y, z, provider.get(ctx, rng, x, y, z))


def _decorate_trunk_vine(ctx, config, rng, roots, logs, foliage) -> None:
    facts = ctx.facts
    for pos in logs:
        for offset, side in (((-1, 0, 0), "east"), ((1, 0, 0), "west"),
                             ((0, 0, -1), "south"), ((0, 0, 1), "north")):
            if rng.next_bounded_i32(3) <= 0:
                continue
            target = (pos[0] + offset[0], pos[1], pos[2] + offset[2])
            if ctx.is_air(*target):
                state = facts.with_property(facts.default_state("minecraft:vine"), side, "true")
                ctx.set(*target, state)


def _decorate_leave_vine(ctx, config, rng, roots, logs, foliage) -> None:
    facts = ctx.facts
    probability = F32(config["probability"])
    for pos in foliage:
        for offset, side in (((-1, 0, 0), "east"), ((1, 0, 0), "west"),
                             ((0, 0, -1), "south"), ((0, 0, 1), "north")):
            if rng.next_f32() >= probability:
                continue
            target = (pos[0] + offset[0], pos[1], pos[2] + offset[2])
            if not ctx.is_air(*target):
                continue
            state = facts.with_property(facts.default_state("minecraft:vine"), side, "true")
            ctx.set(*target, state)
            cursor = (target[0], target[1] - 1, target[2])
            for _ in range(4):
                if not ctx.is_air(*cursor):
                    break
                ctx.set(*cursor, state)
                cursor = (cursor[0], cursor[1] - 1, cursor[2])


def _decorate_attached_to_logs(ctx, config, rng, roots, logs, foliage) -> None:
    directions = [_direction_offset(str(entry)) for entry in config["directions"]]
    if not directions or not logs:
        return
    provider = parse_state_provider(config["block_provider"])
    probability = F32(config["probability"])
    shuffled = list(logs)
    for index in range(len(shuffled) - 1, 0, -1):
        other = rng.next_bounded_i32(index + 1)
        shuffled[index], shuffled[other] = shuffled[other], shuffled[index]
    for log in shuffled:
        offset = directions[rng.next_bounded_i32(len(directions))]
        target = (log[0] + offset[0], log[1] + offset[1], log[2] + offset[2])
        if rng.next_f32() <= probability and ctx.is_air(*target):
            ctx.set(*target, provider.get(ctx, rng, *target))


def _decorate_attached_to_leaves(ctx, config, rng, roots, logs, foliage) -> None:
    directions = [_direction_offset(str(entry)) for entry in config["directions"]]
    if not directions or not foliage:
        return
    provider = parse_state_provider(config["block_provider"])
    probability = F32(config["probability"])
    exclusion_xz = int(config["exclusion_radius_xz"])
    exclusion_y = int(config["exclusion_radius_y"])
    required = int(config["required_empty_blocks"])
    blacklist: set = set()
    shuffled = list(foliage)
    for index in range(len(shuffled) - 1, 0, -1):
        other = rng.next_bounded_i32(index + 1)
        shuffled[index], shuffled[other] = shuffled[other], shuffled[index]
    for leaf in shuffled:
        offset = directions[rng.next_bounded_i32(len(directions))]
        target = (leaf[0] + offset[0], leaf[1] + offset[1], leaf[2] + offset[2])
        if target in blacklist or rng.next_f32() >= probability:
            continue
        if not all(ctx.is_air(leaf[0] + offset[0] * step, leaf[1] + offset[1] * step,
                              leaf[2] + offset[2] * step)
                   for step in range(1, required + 1)):
            continue
        for dx in range(-exclusion_xz, exclusion_xz + 1):
            for dy in range(-exclusion_y, exclusion_y + 1):
                for dz in range(-exclusion_xz, exclusion_xz + 1):
                    blacklist.add((target[0] + dx, target[1] + dy, target[2] + dz))
        ctx.set(*target, provider.get(ctx, rng, *target))


def _decorate_cocoa(ctx, config, rng, roots, logs, foliage) -> None:
    facts = ctx.facts
    if not logs or rng.next_f32() >= F32(config["probability"]):
        return
    base_y = logs[0][1]
    for pos in logs:
        if pos[1] - base_y > 2:
            continue
        for facing in HORIZONTAL_PLANE:
            if rng.next_f32() > F32(0.25):
                continue
            target = (pos[0] - facing[0], pos[1], pos[2] - facing[2])
            if not ctx.is_air(*target):
                continue
            name = {(-1, 0): "west", (1, 0): "east", (0, -1): "north", (0, 1): "south"}[
                (facing[0], facing[2])]
            state = facts.default_state("minecraft:cocoa")
            state = facts.with_property(state, "facing", name)
            state = facts.with_property(state, "age", str(rng.next_bounded_i32(3)))
            ctx.set(*target, state)


def _decorate_pale_moss(ctx, config, rng, roots, logs, foliage) -> None:
    if not logs:
        return
    for pos in logs:
        if rng.next_f32() < F32(config["trunk_probability"]) and ctx.is_air(pos[0], pos[1] - 1, pos[2]):
            _place_moss_hanger(ctx, rng, (pos[0], pos[1] - 1, pos[2]))
    for pos in foliage:
        if rng.next_f32() < F32(config["leaves_probability"]) and ctx.is_air(pos[0], pos[1] - 1, pos[2]):
            _place_moss_hanger(ctx, rng, (pos[0], pos[1] - 1, pos[2]))


def _place_moss_hanger(ctx, rng, pos) -> None:
    facts = ctx.facts
    cursor = pos
    while ctx.is_air(cursor[0], cursor[1] - 1, cursor[2]) and rng.next_f32() >= F32(0.5):
        state = facts.with_property(facts.default_state("minecraft:pale_hanging_moss"),
                                    "tip", "false")
        ctx.set(*cursor, state)
        cursor = (cursor[0], cursor[1] - 1, cursor[2])
    state = facts.with_property(facts.default_state("minecraft:pale_hanging_moss"),
                                "tip", "true")
    ctx.set(*cursor, state)


def _decorate_creaking_heart(ctx, config, rng, roots, logs, foliage) -> None:
    if not logs or rng.next_f32() >= F32(config["probability"]):
        return
    facts = ctx.facts
    shuffled = list(logs)
    for index in range(len(shuffled) - 1, 0, -1):
        other = rng.next_bounded_i32(index + 1)
        shuffled[index], shuffled[other] = shuffled[other], shuffled[index]
    for pos in shuffled:
        if all(facts.in_tag(ctx.get(pos[0] + o[0], pos[1] + o[1], pos[2] + o[2]),
                            "minecraft:logs") for o in DIRECTIONS):
            state = facts.default_state("minecraft:creaking_heart")
            state = facts.with_property(state, "axis", "y")
            ctx.set(*pos, state)
            return


_DECORATORS = {
    "beehive": _decorate_beehive,
    "alter_ground": _decorate_alter_ground,
    "place_on_ground": _decorate_place_on_ground,
    "trunk_vine": _decorate_trunk_vine,
    "leave_vine": _decorate_leave_vine,
    "attached_to_logs": _decorate_attached_to_logs,
    "attached_to_leaves": _decorate_attached_to_leaves,
    "cocoa": _decorate_cocoa,
    "pale_moss": _decorate_pale_moss,
    "creaking_heart": _decorate_creaking_heart,
}


# ----- dispatch -----

_FEATURE_PLACERS: Dict[str, Any] = {}


def _register():
    _FEATURE_PLACERS.update({
        "ore": _place_ore,
        "scattered_ore": _place_scattered_ore,
        "replace_single_block": _place_replace_single_block,
        "simple_block": _place_simple_block,
        "random_patch": _place_random_patch,
        "flower": _place_random_patch,
        "no_bonemeal_flower": _place_random_patch,
        "random_selector": _place_random_selector,
        "simple_random_selector": _place_simple_random_selector,
        "random_boolean_selector": _place_random_boolean_selector,
        "weighted_random_selector": _place_weighted_random_selector,
        "block_pile": _place_block_pile,
        "disk": _place_disk,
        "spring_feature": _place_spring,
        "seagrass": _place_seagrass,
        "freeze_top_layer": _place_freezing,
        "vegetation_patch": _place_vegetation_patch,
        "waterlogged_vegetation_patch": _place_waterlogged_vegetation_patch,
        "block_column": _place_block_column,
        "lake": _place_lake,
        "monster_room": _place_monster_room,
        "fallen_tree": _place_fallen_tree,
        "bamboo": _place_bamboo,
        "netherrack_replace_blobs": _place_netherrack_replace_blobs,
        "glowstone_blob": _place_glowstone_blob,
        "fill_layer": _place_fill_layer,
        "multiface_growth": _place_multiface_growth,
        "underwater_magma": _place_underwater_magma,
        "geode": _place_geode,
        "tree": _place_tree,
    })


_register()


def implemented_types() -> FrozenSet[str]:
    """Configured-feature type names this module can place."""
    return frozenset(_FEATURE_PLACERS)


def missing_types(biome_table, placed_features: Dict[str, PlacedFeature],
                  configured_features: Dict[str, ConfiguredFeature]) -> Counter:
    """Configured-feature types the biome data references but this module cannot place.

    The count is *per biome step reference*: a type used by six biomes in one
    step counts six, which is the number that says how much of the world is
    affected, not just how many types are missing.
    """
    missing: Counter = Counter()
    seen: set = set()
    for biome, step, feature_name in biome_table.iter_features():
        if (biome, feature_name) in seen:
            continue
        seen.add((biome, feature_name))
        placed = placed_features.get(feature_name)
        if placed is None:
            missing[f"<missing placed feature {feature_name}>"] += 1
            continue
        seen_types = set()
        for configured in _referenced_configured(placed, configured_features,
                                                 lookup=placed_features):
            # A set, not a count: a feature that wraps two nested features of the
            # same type would otherwise count twice and disagree with the check
            # tool's independent scan.
            if configured.type in _FEATURE_PLACERS or configured.type in seen_types:
                continue
            seen_types.add(configured.type)
            missing[configured.type] += 1
    return missing


#: Configured-feature types whose config nests other *placed* features. The mode
#: says how: `placed` is one inline (or named) placed feature, `entry` is a list
#: of `{feature, chance|weight}` wrappers, `list` is a list of placed features
#: with no wrapper. `simple_random_selector` is the one that differs from
#: `random_selector` here, and reading a configured feature as if it were a
#: placed one is the easy mistake: the two shapes both have a `feature` key.
_NESTED_PLACED_KEYS: Dict[str, Tuple[Tuple[str, str], ...]] = {
    "random_patch": (("feature", "placed"),),
    "flower": (("feature", "placed"),),
    "no_bonemeal_flower": (("feature", "placed"),),
    "random_selector": (("features", "entry"), ("default", "placed")),
    "weighted_random_selector": (("features", "entry"),),
    "simple_random_selector": (("features", "list"),),
    "random_boolean_selector": (("feature_true", "placed"), ("feature_false", "placed")),
    "vegetation_patch": (("vegetation_feature", "placed"),),
    "waterlogged_vegetation_patch": (("vegetation_feature", "placed"),),
    "root_system": (("feature", "placed"),),
}


def _referenced_configured(placed: PlacedFeature, configured_features,
                           depth: int = 0, lookup=None) -> Iterator[ConfiguredFeature]:
    """Every configured feature this placed feature can reach, transitively."""
    if depth > 8:
        return
    feature = placed.feature
    if feature is None:
        feature = configured_features.get(placed.feature_name or "")
    if feature is None:
        return
    yield feature
    for nested in _nested_placed(feature, lookup):
        yield from _referenced_configured(nested, configured_features, depth + 1, lookup)


def _nested_placed(feature: ConfiguredFeature, lookup=None) -> Iterator[PlacedFeature]:
    """The placed features a configured feature wraps, if any.

    Only the whitelist above is walked: `geode` and `dripstone_cluster` carry
    block-state providers that *look* like feature configs, and a blanket
    recursive search would report their inner types as separately missing.
    """
    config = feature.config
    for key, mode in _NESTED_PLACED_KEYS.get(feature.type, ()):
        value = config.get(key)
        if mode == "placed":
            if isinstance(value, (str, dict)):
                yield _inline_placed(value, lookup)
        elif mode == "entry":
            for entry in value or ():
                if isinstance(entry, dict) and "feature" in entry:
                    yield _inline_placed(entry["feature"], lookup)
        elif mode == "list":
            for entry in value or ():
                yield _inline_placed(entry, lookup)


def _run_placed(ctx: FeatureContext, placed: PlacedFeature, rng, pos,
                top: bool = False) -> bool:
    """Run a placed feature's modifier pipeline at `pos`.

    `top` marks the biome entry the chunk loop started from. Nested features
    (`random_patch`, the selectors) run through here too, and the counters keep
    the two apart: `attempts_by_feature` answers "how many placements did this
    biome entry make", which is the number the placement modifiers decide.
    """
    if placed.feature_name is not None:
        feature = ctx.configured.get(placed.feature_name)
    else:
        feature = placed.feature
    if feature is None:
        ctx.stats.unimplemented[f"<missing configured feature {placed.feature_name}>"] += 1
        return False
    return _run_pipeline(ctx, placed, feature, rng, pos, 0, top)


def _run_pipeline(ctx: FeatureContext, placed: PlacedFeature, feature: ConfiguredFeature,
                  rng, pos, index: int, top: bool) -> bool:
    if index == len(placed.placement):
        ctx.stats.features_attempted += 1
        if top and ctx.feature_name is not None:
            ctx.stats.attempts_by_feature[ctx.feature_name] += 1
        else:
            ctx.stats.placements_nested += 1
        placer = _FEATURE_PLACERS.get(feature.type)
        if placer is None:
            ctx.stats.unimplemented[feature.type] += 1
            return False
        try:
            placed_ok = bool(placer(ctx, feature.config, rng, pos))
        except NotImplementedError as error:
            # A type that is implemented but whose config asks for a sub-part
            # this module does not have yet (a trunk placer, a state provider).
            # Recorded under the type, so the gap list names what is missing.
            _record_gap(ctx, feature, error)
            return False
        except KeyError as error:
            # A config shape this module did not anticipate. It is a bug here,
            # not in the data, and the check tool's layer 7 fails on it rather
            # than letting it take chunk generation down.
            _record_gap(ctx, feature, error)
            return False
        if placed_ok:
            ctx.stats.features_placed += 1
        return placed_ok
    try:
        modifier = parse_placement_modifier(placed.placement[index])
    except (NotImplementedError, KeyError) as error:
        # A modifier type or provider the module does not have. Recorded against
        # the feature rather than raised: an unhandled modifier would otherwise
        # take chunk generation down with it.
        _record_gap(ctx, feature, error)
        return False
    result = False
    try:
        for next_pos in modifier.positions(ctx, rng, pos):
            if _run_pipeline(ctx, placed, feature, rng, next_pos, index + 1, top):
                result = True
    except (NotImplementedError, KeyError) as error:
        _record_gap(ctx, feature, error)
    return result


def _record_gap(ctx: FeatureContext, feature: ConfiguredFeature, error: Exception) -> None:
    if isinstance(error, KeyError):
        ctx.stats.unimplemented[
            f"{feature.type}: unexpected config key {error}"] += 1
    else:
        ctx.stats.unimplemented[f"{feature.type}: {error}"] += 1


# ==================== feature order and selection ====================


class FeatureOrder:
    """`FeatureSorter.buildFeaturesPerStep`: the global per-step feature order.

    Vanilla sorts features so that a feature shared by two biomes runs once, at
    one deterministic index: for each biome, the features of a step form a
    chain, and the whole graph is topologically sorted with
    `(step, first-encounter index)` as the tie-break. The position in the result
    is the `index` in `setFeatureSeed(decorationSeed, index, step)`, so two
    implementations that order the same features differently place every tree in
    the world one chunk over.
    """

    __slots__ = ("per_step", "order", "table")

    def __init__(self, biome_table) -> None:
        self.table = biome_table
        self.per_step = build_feature_order(biome_table)
        self.order = {name: index
                      for index, names in enumerate(self.per_step)
                      for name in names}

    def select(self, biome_names: Iterable[str], step: int) -> List[Tuple[int, str]]:
        """`(global index, feature)` pairs for the biomes present, in order.

        The enumerate happens *before* the filter, so the index stays the global
        one -- this is the whole reason `FeatureSorter.StepFeatureData` carries
        an `indexMapping` function instead of the caller re-deriving an index.
        """
        wanted: set = set()
        for biome in biome_names:
            steps = self.table.features.get(biome, ())
            if step < len(steps):
                wanted.update(steps[step])
        if not wanted:
            return []
        names = self.per_step[step] if step < len(self.per_step) else ()
        return [(index, name) for index, name in enumerate(names) if name in wanted]

    def steps(self) -> int:
        return len(self.per_step)


def biome_order(biome_table) -> List[str]:
    """The biome order `FeatureSorter` sees: overworld order, then the rest.

    Vanilla's order is the biome source's `possibleBiomes()` insertion order,
    which for the overworld is a code constant (see `OVERWORLD_BIOME_ORDER`).
    Anything not in that list -- the nether and end biomes -- keeps
    `assets/biome.json` order, which is *not* vanilla's: it changes the global
    feature index of every nether and end feature, and therefore its seed. The
    gap is reported rather than hidden.
    """
    known = set(OVERWORLD_BIOME_ORDER)
    remaining = [name for name in biome_table.features if name not in known]
    return [name for name in OVERWORLD_BIOME_ORDER if name in biome_table.features] + remaining


def build_feature_order(biome_table) -> List[Tuple[str, ...]]:
    """Topologically sort each biome's step chains into one global list per step.

    A depth-first post-order traversal reversed at the end is vanilla's
    `FeatureSorter`; the `visiting` set turns a cycle in the datapack into an
    error instead of a hang. Both parts matter: features are nodes, "A comes
    before B in some biome's step" is an edge, and the tie-break between
    independent features is the order they were first encountered.
    """
    encounter: Dict[str, int] = {}
    edges: Dict[Tuple[int, str], List[Tuple[int, str]]] = {}
    max_steps = 0
    biome_features: Dict[str, List[Tuple[int, str]]] = {}

    for biome in biome_order(biome_table):
        steps = biome_table.features.get(biome, ())
        max_steps = max(max_steps, len(steps))
        chain: List[Tuple[int, str]] = []
        for step, entries in enumerate(steps):
            for name in entries:
                node = (step, name)
                if name not in encounter:
                    encounter[name] = len(encounter)
                edges.setdefault(node, [])
                chain.append(node)
        biome_features[biome] = chain
        for first, second in zip(chain, chain[1:]):
            edges.setdefault(first, [])
            if second not in edges[first]:
                edges[first].append(second)

    # Deterministic tie-break: (step, first-encounter index).
    ordered_nodes = sorted(edges, key=lambda node: (node[0], encounter[node[1]]))

    sorted_nodes: List[Tuple[int, str]] = []
    discovered: set = set()
    visiting: set = set()

    def visit(node) -> None:
        if node in discovered:
            return
        if node in visiting:
            raise ValueError(f"feature order cycle at {node}")
        visiting.add(node)
        for following in edges.get(node, ()):  # already in insertion order
            visit(following)
        visiting.discard(node)
        discovered.add(node)
        sorted_nodes.append(node)

    for node in ordered_nodes:
        visit(node)
    sorted_nodes.reverse()

    per_step: List[List[str]] = [[] for _ in range(max_steps)]
    for step, name in sorted_nodes:
        per_step[step].append(name)
    return [tuple(step) for step in per_step]


# ==================== the public entry point ====================


def decoration_seed(seed: int, chunk_x: int, chunk_z: int) -> int:
    """`WorldgenRandom.setDecorationSeed(levelSeed, minBlockX, minBlockZ)`.

    Two `nextLong()`s off a Xoroshiro seeded with the world seed, each forced odd,
    then the chunk's minimum block coordinates are mixed in with those as
    multipliers and the world seed is XORed back on top.
    """
    rng = WorldgenRandom(Xoroshiro.from_seed(seed))
    first = (rng.next_i64() | 1) & MASK64
    second = (rng.next_i64() | 1) & MASK64
    x = (chunk_x * 16) & MASK64
    z = (chunk_z * 16) & MASK64
    mixed = (((x * first) & MASK64) + ((z * second) & MASK64)) & MASK64
    return (mixed ^ (seed & MASK64)) & MASK64


def feature_random(seed: int, chunk_x: int, chunk_z: int, index: int, step: int,
                   decoration: Optional[int] = None) -> WorldgenRandom:
    """`setFeatureSeed`: a fresh Xoroshiro per placed feature, per chunk.

    The `+ index + 10000 * step` salt is small on purpose (vanilla's own comment
    calls it out); reproducing it exactly is what ties a feature's positions to
    its index in the *global* per-step list.
    """
    base = decoration if decoration is not None else decoration_seed(seed, chunk_x, chunk_z)
    return WorldgenRandom(Xoroshiro.from_seed((base + index + 10000 * step) & MASK64))


def decorate(chunk_x: int, chunk_z: int, blocks: np.ndarray, biomes: np.ndarray, *,
             preset: str, seed: int, min_y: int, height: int, biome_table,
             placed_features: Dict[str, PlacedFeature],
             configured_features: Dict[str, ConfiguredFeature],
             heightmap: Optional[np.ndarray] = None) -> FeatureStats:
    """Run the biome decoration steps for one chunk, mutating `blocks` in place.

    `blocks` is `(height, 16, 16)` `uint16` in `[y, z, x]`; `biomes` is
    `(height // 4, 4, 4)` `int32` in `[y, z, x]` quart cells. Everything a
    feature writes goes through `FeatureContext.set`, so a write that leaves the
    footprint is counted in `writes_dropped` rather than lost.
    """
    global _PLACED_LOOKUP
    _PLACED_LOOKUP = placed_features

    stats = FeatureStats()
    surface = Surface(heightmap)
    ctx = FeatureContext(chunk_x, chunk_z, blocks, biomes, min_y, height, biome_table,
                         stats, surface, preset=preset)
    ctx.configured = configured_features
    surface.bind(ctx)

    order = _feature_order_for(biome_table)
    present = []
    for row in range(biomes.shape[0]):
        for z in range(4):
            for x in range(4):
                name = biome_table.names.get(int(biomes[row, z, x]))
                if name is not None and name not in present:
                    present.append(name)

    steps = max((len(biome_table.features.get(name, ())) for name in present), default=0)
    decoration = decoration_seed(seed, chunk_x, chunk_z)
    for step in range(steps):
        selected = order.select(present, step)
        if not selected:
            continue
        stats.steps_run += 1
        for index, feature_name in selected:
            placed = placed_features.get(feature_name)
            if placed is None:
                stats.unimplemented[f"<missing placed feature {feature_name}>"] += 1
                continue
            stats.features_run += 1
            stats.by_step[step] += 1
            stats.runs_by_feature[feature_name] += 1
            rng = feature_random(seed, chunk_x, chunk_z, index, step, decoration)
            ctx.rng = rng
            ctx.feature_name = feature_name
            # The origin is the chunk's own origin, not a placement position:
            # every modifier chain in the data starts with `in_square`, `count`
            # or a `heightmap`, and those all work from `pos` as given.
            _run_placed(ctx, placed, rng, (chunk_x << 4, min_y, chunk_z << 4), top=True)
    return stats


_FEATURE_ORDERS: Dict[int, Tuple[Any, FeatureOrder]] = {}


def _feature_order_for(biome_table) -> FeatureOrder:
    """Cache one `FeatureOrder` per biome table.

    Keyed by `id()` because `BiomeTable` has `__slots__` and cannot carry the
    cache itself; the stored table reference is compared for identity so a
    garbage-collected table cannot hand its order to a new one at the same
    address.
    """
    cached = _FEATURE_ORDERS.get(id(biome_table))
    if cached is not None and cached[0] is biome_table:
        return cached[1]
    order = FeatureOrder(biome_table)
    _FEATURE_ORDERS[id(biome_table)] = (biome_table, order)
    return order


__all__ = [
    "BlockFacts",
    "ConfiguredFeature",
    "FeatureContext",
    "FeatureOrder",
    "FeatureStats",
    "F32",
    "OVERWORLD_BIOME_ORDER",
    "PlacedFeature",
    "STEP_NAMES",
    "Surface",
    "biome_temperature",
    "block_facts",
    "build_feature_order",
    "decorate",
    "decoration_seed",
    "feature_random",
    "foliage_noise",
    "implemented_types",
    "load_block_tags",
    "load_configured_features",
    "load_placed_features",
    "missing_types",
    "parse_block_predicate",
    "parse_height_provider",
    "parse_int_provider",
    "parse_placement_modifier",
    "parse_state_provider",
    "sea_level_for",
]
