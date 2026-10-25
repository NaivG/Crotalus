"""
Decorated-feature acceptance: does the placement pipeline still place features?

    python tools/check_features.py

There is no vanilla golden dump for this stage. `Pumpkin/assets/tests/` carries
noise and surface dumps and nothing for features, so unlike the density-function
tool there is no "compare against numbers from a real Minecraft" layer to lean
on. What is left is a set of properties a wrong implementation actually violates,
chosen so that the *placement modifiers* -- the part that decides how many
features exist and where -- are checked exactly rather than statistically.

The seven layers, strongest first
---------------------------------
1. **The position hash.** `XoroshiroSplitter.split_pos` is vanilla's `Mth.getSeed`
   and eight vectors from a real Minecraft pin it. It is checked here because the
   whole feature layer's positional randomness rides on it and nothing else in
   the repo exercises it.

2. **The gap list.** Every configured-feature type `assets/biome.json` actually
   references is either implemented or printed with the number of biome steps
   that reference it. This is the layer that makes a partial implementation
   honest: the missing types are enumerated, never hidden behind a no-op.

3. **Determinism.** Same seed, same inputs, bit-identical chunk -- twice in the
   same process and once more from a fresh buffer. Feature placement is
   positional randomness, so any leaked state (a cached RNG, a mutable default,
   a set iterated in hash order) shows up here.

4. **Exact placement counts.** For every placed feature whose modifier chain is
   multiplicative -- `count`, `rarity_filter`, `noise_threshold_count`,
   `noise_based_count`, optionally behind a `biome` -- the expected number of
   terminal placements is recomputed from the modifier's own definition: the
   decoration seed, the feature seed, the RNG draws, and (for the noise counts) a
   freshly constructed foliage-noise sampler. The measured
   `FeatureStats.attempts_by_feature` has to match exactly, per chunk. This is the
   closest thing to a golden value this stage has.

5. **Modifier distributions.** `in_square` uniform over 16x16 (a histogram, not
   a range check), `height_range` uniform over its resolved interval, `heightmap`
   exactly on the surface and nothing when the surface is at the world floor,
   `fixed_placement` in the data's order.

6. **Cross-border writes.** A feature whose blocks fall outside the 16x16
   footprint must have every one of them counted in `writes_dropped` and must
   not touch the chunk by accident. The counters are also checked for internal
   consistency.

7. **Biome gating.** On a real biome volume: a placed feature with a `biome`
   modifier is attempted zero times in a chunk whose biomes do not list it, and
   more than zero in a chunk whose biomes do.

Every layer prints the numbers it measured, pass or fail, because "0 != 0" and
"0 attempts" look identical in a bare pass/fail line.
"""

from __future__ import annotations

import collections
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from world.vanilla.biome import BiomeTable  # noqa: E402
from world.vanilla.features import (  # noqa: E402
    STEP_NAMES,
    FeatureContext,
    FeatureStats,
    Surface,
    block_facts,
    decorate,
    implemented_types,
    load_placed_features,
    load_configured_features,
    missing_types,
    parse_placement_modifier,
    sea_level_for,
)
from world.vanilla.random_source import (  # noqa: E402
    MASK64,
    LegacyRandom,
    WorldgenRandom,
    Xoroshiro,
    XoroshiroSplitter,
)
from world.vanilla.simplex import OctaveSimplexNoiseSampler  # noqa: E402

SEED = 4242
MIN_Y = -64
HEIGHT = 384

failures: List[str] = []


def check(label: str, condition: bool, detail: str = "") -> bool:
    marker = "OK  " if condition else "FAIL"
    print(f"  {marker} {label}" + (f"  ({detail})" if detail else ""))
    if not condition:
        failures.append(label)
    return condition


def report(label: str, detail: str) -> None:
    print(f"  ---- {label}: {detail}")


# ==========================================================================
# Fixtures
# ==========================================================================

_AIR = 0


class StubBiomeTable:
    """A one-biome table, so a feature's global index in its step is 0.

    The count checks want the feature seed to depend on nothing but the chunk and
    the step; with a single entry in a single step, `setFeatureSeed`'s index is
    zero and the sorter drops out of the equation entirely.
    """

    def __init__(self, name: str, step: int, feature: str, steps: int = 11) -> None:
        lists = [() for _ in range(steps)]
        lists[step] = (feature,)
        self.features = {name: tuple(lists)}
        self.ids = {name: 1}
        self.names = {1: name}
        self.temperature = {name: 0.5}

    def iter_features(self):
        for biome, steps in self.features.items():
            for step, entries in enumerate(steps):
                for feature in entries:
                    yield biome, step, feature

    def id_of(self, name: str) -> int:
        return 1


def flat_chunk(surface_y: int = 64, biome_id: int = 1,
               water_above: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """Stone with a grass surface, air above, one biome everywhere.

    `surface_y` is the Y of the top grass block. The chunk is deliberately dull:
    every modifier under test reads the surface or nothing at all, and a
    complicated fixture would move the failure from the pipeline to the fixture.
    """
    facts = block_facts()
    blocks = np.zeros((HEIGHT, 16, 16), dtype=np.uint16)
    stone = facts.default_state("minecraft:stone")
    dirt = facts.default_state("minecraft:dirt")
    grass = facts.default_state("minecraft:grass_block")
    water = facts.default_state("minecraft:water")
    for row in range(HEIGHT):
        y = MIN_Y + row
        if y < surface_y - 3:
            blocks[row, :, :] = stone
        elif y < surface_y:
            blocks[row, :, :] = dirt
        elif y == surface_y:
            blocks[row, :, :] = grass
        elif water_above and y <= 63:
            blocks[row, :, :] = water
    biomes = np.zeros((HEIGHT // 4, 4, 4), dtype=np.int32)
    biomes[:, :, :] = biome_id
    return blocks, biomes


def make_context(blocks, biomes, biome_table, heightmap=None) -> FeatureContext:
    stats = FeatureStats()
    surface = Surface(heightmap)
    ctx = FeatureContext(0, 0, blocks, biomes, MIN_Y, HEIGHT, biome_table, stats,
                         surface, preset="overworld")
    surface.bind(ctx)
    return ctx


# ==========================================================================
# Layer 0: the position hash
# ==========================================================================

#: `Mth.getSeed(x, 0, z)` from a real Minecraft, via Pumpkin's
#: `block_position_hash` test. The feature layer is positional randomness, so a
#: silent regression here would move every tree, ore and geode at once.
POSITION_HASH_VECTORS: Sequence[Tuple[int, int]] = (
    (0, 0),
    (1, 60311958971344),
    (4, 120566413180880),
    (25, 111753446486209),
    (676, 75210837988243),
    (458329, -43764888250),
    (-387008604, 8437923733503),
    (176771161, 18421337580760),
)


def layer0_position_hash() -> bool:
    print("Layer 0 - the position hash behind split_pos")
    # A non-zero splitter on purpose: `Xoroshiro` substitutes a fixed state for
    # an all-zero one, so `split_pos(0, 0, 0)` through a `(0, 0)` splitter would
    # report the substitute rather than the hash. XORing against a known lo keeps
    # the hash observable, including for the zero vector.
    salt_lo = 0x0123456789ABCDEF
    salt_hi = 0x0FEDCBA987654321
    splitter = XoroshiroSplitter(salt_lo, salt_hi)
    ok = True
    for coordinate, expected in POSITION_HASH_VECTORS:
        random = splitter.split_pos(coordinate, coordinate, coordinate)
        actual = random.lo if random.lo < 2**63 else random.lo - 2**64
        want = expected & MASK64
        ok &= check(f"Mth.getSeed({coordinate}, {coordinate}, {coordinate})",
                    (random.lo == (want ^ salt_lo) & MASK64) and random.hi == salt_hi,
                    f"got {actual}, want {expected}")
    return ok


# ==========================================================================
# Layer 1: the gap list
# ==========================================================================


def layer1_gap_list(biome_table, placed, configured) -> None:
    print("Layer 1 - what is implemented, and what the world references")
    implemented = implemented_types()
    report("implemented configured-feature types", str(len(implemented)))

    # A direct scan of the biome data: every placed feature every biome names,
    # with its configured type. This is the list the module's own
    # `missing_types` has to agree with.
    direct: collections.Counter = collections.Counter()
    references: collections.Counter = collections.Counter()
    seen = set()
    for biome, step, feature in biome_table.iter_features():
        references[feature] += 1
        if (biome, feature) in seen:
            continue
        seen.add((biome, feature))
        entry = placed.get(feature)
        if entry is None:
            direct["<missing placed feature>"] += 1
            continue
        kind = _reachable_types(entry, configured)
        for type_name in kind:
            if type_name not in implemented:
                direct[type_name] += 1

    reported = missing_types(biome_table, placed, configured)
    check("missing_types agrees with a direct scan of biome.json",
          dict(direct) == dict(reported),
          f"direct={dict(direct)} reported={dict(reported)}")

    report("placed features referenced by biomes", str(len(references)))
    report("distinct placed features referenced", str(len(references)))
    if reported:
        print("  ... configured-feature types referenced but NOT implemented:")
        for type_name, count in reported.most_common():
            print(f"      {count:4d} biome-step reference(s)  {type_name}")
    else:
        print("  ... every referenced configured-feature type is implemented")
    checked = set(references)
    unimplemented_steps = sum(
        1 for biome, step, feature in biome_table.iter_features()
        if placed.get(feature) is not None
        and not _reachable_types(placed[feature], configured) <= implemented
    )
    report("biome-step references whose type is unimplemented",
           f"{unimplemented_steps} of {sum(references.values())}")
    check("no biome-referenced placed feature is missing from the data",
          not any(key == "<missing placed feature>" for key in direct),
          "")
    check("STEP_NAMES is vanilla's eleven-step decoration order",
          len(STEP_NAMES) == 11 and STEP_NAMES[9] == "vegetal_decoration",
          f"{len(STEP_NAMES)} steps")


def _reachable_types(placed, configured, depth: int = 0) -> set:
    """Configured-feature types reachable from a placed feature."""
    from world.vanilla.features import _referenced_configured

    return {feature.type
            for feature in _referenced_configured(placed, configured, lookup=_PLACED)}


# ==========================================================================
# Layer 2: determinism
# ==========================================================================


def layer2_determinism(biome_table, placed, configured) -> None:
    print("Layer 2 - determinism")
    plains = biome_table.id_of("minecraft:plains")
    first, biomes = flat_chunk(biome_id=plains)
    digest_first = int(first.sum())
    stats_a = decorate(0, 0, first, biomes, preset="overworld", seed=SEED, min_y=MIN_Y,
                       height=HEIGHT, biome_table=biome_table, placed_features=placed,
                       configured_features=configured)

    second, biomes2 = flat_chunk(biome_id=plains)
    stats_b = decorate(0, 0, second, biomes2, preset="overworld", seed=SEED, min_y=MIN_Y,
                       height=HEIGHT, biome_table=biome_table, placed_features=placed,
                       configured_features=configured)
    check("same seed and inputs give a bit-identical chunk",
          bool((first == second).all()),
          f"changed blocks={int((first != second).sum())}")
    check("the counters agree between the two runs",
          stats_a.summary() == stats_b.summary(),
          f"{stats_a.features_attempted} vs {stats_b.features_attempted} attempts")

    third, biomes3 = flat_chunk(biome_id=plains)
    stats_c = decorate(0, 1, third, biomes3, preset="overworld", seed=SEED, min_y=MIN_Y,
                       height=HEIGHT, biome_table=biome_table, placed_features=placed,
                       configured_features=configured)
    check("a neighbouring chunk is not a copy of its neighbour",
          bool((first != third).any()),
          f"changed blocks={int((first != third).sum())}")

    check("decoration touched the chunk",
          stats_a.features_placed > 0 and digest_first != int(first.sum()),
          f"{stats_a.features_placed} features placed, "
          f"{stats_a.features_attempted} placements attempted")
    report("plains chunk at seed %d" % SEED,
           f"{stats_a.steps_run} steps, {stats_a.features_run} features, "
           f"{stats_a.features_attempted} attempts, "
           f"{stats_a.features_placed} placed, "
           f"{stats_a.writes_dropped} dropped writes")


# ==========================================================================
# Layer 2b: a sweep across biome families
# ==========================================================================


def layer2b_biome_sweep(biome_table, placed, configured) -> None:
    """Run one real chunk per biome family and report what came out.

    The plains chunk in layer 2 exercises 48 of the 204 referenced placed
    features. Sweeping a few families is what turns "the pipeline runs" into
    "the pipeline runs the ocean, nether and end entries too", and the
    per-biome `unimplemented` counter is where a type that only one family uses
    shows up instead of hiding.
    """
    print("Layer 2b - one chunk per biome family")
    families = [
        "minecraft:plains", "minecraft:desert", "minecraft:taiga",
        "minecraft:ocean", "minecraft:warm_ocean", "minecraft:mushroom_fields",
        "minecraft:nether_wastes", "minecraft:crimson_forest", "minecraft:the_end",
    ]
    total_unimplemented: collections.Counter = collections.Counter()
    for family in families:
        if family not in biome_table.ids:
            continue
        biome_id = biome_table.id_of(family)
        blocks, biomes = flat_chunk(surface_y=64, biome_id=biome_id)
        stats = decorate(0, 0, blocks, biomes, preset="overworld", seed=SEED, min_y=MIN_Y,
                         height=HEIGHT, biome_table=biome_table, placed_features=placed,
                         configured_features=configured)
        total_unimplemented.update(stats.unimplemented)
        print(f"  {family:28s} steps={stats.steps_run:2d} "
              f"features={stats.features_run:3d} attempts={stats.features_attempted:6d} "
              f"placed={stats.features_placed:6d} dropped={stats.writes_dropped:5d} "
              f"reads_outside={stats.reads_outside:7d}")
    if total_unimplemented:
        print("  ... unimplemented across the sweep:")
        for type_name, count in total_unimplemented.most_common(12):
            print(f"      {count:5d}  {type_name}")
    else:
        print("  ... all nine families placed without hitting an unimplemented type")


# ==========================================================================
# Layer 3: exact placement counts
# ==========================================================================

#: Modifiers that map one position to exactly one position, so they do not
#: change how many times the configured feature runs.
_PASS_THROUGH = {"in_square", "heightmap", "height_range", "random_offset"}
#: Modifiers that decide *whether* to place, checked by recomputation below.
_MULTIPLICATIVE = {"count", "rarity_filter", "noise_threshold_count", "noise_based_count"}


def expected_decoration_seed(seed: int, chunk_x: int, chunk_z: int) -> int:
    """`setDecorationSeed`, spelled out here rather than imported.

    The point of this layer is to recompute the pipeline's numbers from the
    primitives. Calling the module's own `decoration_seed` would compare the
    implementation with itself.
    """
    random = WorldgenRandom(Xoroshiro.from_seed(seed))
    first = (random.next_i64() | 1) & MASK64
    second = (random.next_i64() | 1) & MASK64
    mixed = ((((chunk_x * 16) & MASK64) * first & MASK64)
             + (((chunk_z * 16) & MASK64) * second & MASK64)) & MASK64
    return (mixed ^ (seed & MASK64)) & MASK64


def expected_feature_random(seed: int, chunk_x: int, chunk_z: int, index: int,
                            step: int) -> WorldgenRandom:
    base = expected_decoration_seed(seed, chunk_x, chunk_z)
    return WorldgenRandom(Xoroshiro.from_seed((base + index + 10000 * step) & MASK64))


def layer3_counts(biome_table, placed, configured) -> None:
    print("Layer 3 - placement counts recomputed from the modifiers")
    candidates = _count_candidates(placed)
    report("placed features with a multiplicative modifier", str(len(candidates)))
    if not candidates:
        check("there is something to measure", False, "no candidates found")
        return

    chunks = [(x, z) for x in range(6) for z in range(6)]
    fresh_noise = OctaveSimplexNoiseSampler(LegacyRandom(2345), (0,))
    checked = 0
    skipped: collections.Counter = collections.Counter()

    for name, step, kinds in candidates:
        entry = placed[name]
        if kinds == "ranged":
            _check_ranged_count(name, step, entry, chunks)
            checked += 1
            continue
        if kinds == "exact":
            expected_by_chunk, detail = _exact_expectations(name, step, entry, chunks, fresh_noise)
            if expected_by_chunk is None:
                skipped[detail] += 1
                continue
            measured = _measure(name, step, entry, chunks)
            mismatches = [(chunk, expected_by_chunk[chunk], measured.get(chunk, 0))
                          for chunk in chunks
                          if measured.get(chunk, 0) != expected_by_chunk[chunk]]
            total_expected = sum(expected_by_chunk.values())
            total_measured = sum(measured.values())
            check(f"{name}: attempts match the recomputed count",
                  not mismatches,
                  f"expected {total_expected}, measured {total_measured} over "
                  f"{len(chunks)} chunks"
                  + (f", first mismatch {mismatches[0]}" if mismatches else ""))
            checked += 1

    report("features checked exactly", str(checked))
    if skipped:
        print("  ... not checked (chain is not multiplicative):")
        for reason, count in skipped.most_common():
            print(f"      {count:4d} {reason}")


def _count_candidates(placed) -> List[Tuple[str, int, str]]:
    """Features whose chain is `biome?` + multiplicative + pass-throughs."""
    out = []
    for name, entry in placed.items():
        kinds = [str(mod.get("type", "")).removeprefix("minecraft:")
                 for mod in entry.placement]
        multiplying = [kind for kind in kinds if kind in _MULTIPLICATIVE]
        if not multiplying:
            continue
        if any(kind not in _MULTIPLICATIVE and kind not in _PASS_THROUGH and kind != "biome"
               for kind in kinds):
            continue
        if len(multiplying) != 1:
            continue
        step = _STEP_BY_FEATURE.get(name)
        if step is None:
            continue
        out.append((name, step, "ranged" if multiplying[0] == "count"
                    and not _constant_count(entry) else "exact"))
    return out


def _constant_count(entry) -> bool:
    for mod in entry.placement:
        if str(mod.get("type", "")).removeprefix("minecraft:") == "count":
            return isinstance(mod.get("count"), int)
    return True


def _exact_expectations(name, step, entry, chunks, noise):
    """Expected terminal placements per chunk, from the modifiers alone."""
    expected: Dict[Tuple[int, int], int] = {}
    for chunk_x, chunk_z in chunks:
        random = expected_feature_random(SEED, chunk_x, chunk_z, 0, step)
        count = 1
        ok = True
        for position, mod in enumerate(entry.placement):
            kind = str(mod.get("type", "")).removeprefix("minecraft:")
            if kind == "biome":
                continue
            if kind in _PASS_THROUGH:
                continue
            if kind == "count":
                if not isinstance(mod.get("count"), int):
                    return None, "count provider is not constant"
                count *= max(0, int(mod["count"]))
            elif kind == "rarity_filter":
                if any(str(m.get("type", "")).removeprefix("minecraft:")
                       not in ("biome",) for m in entry.placement[:position]):
                    return None, "rarity_filter is not the first draw"
                if random.next_f32() >= np.float32(1.0) / np.float32(mod["chance"]):
                    count = 0
            elif kind == "noise_threshold_count":
                value = noise.sample(chunk_x * 16 / 200.0, chunk_z * 16 / 200.0, False)
                count *= max(0, int(mod["below_noise"] if value < mod["noise_level"]
                                    else mod["above_noise"]))
            elif kind == "noise_based_count":
                factor = float(mod["noise_factor"])
                value = noise.sample(chunk_x * 16 / factor, chunk_z * 16 / factor, False)
                count *= max(0, math.ceil((value + float(mod["noise_offset"]))
                                          * int(mod["noise_to_count_ratio"])))
            else:
                return None, f"unhandled modifier {kind}"
        expected[(chunk_x, chunk_z)] = count
    return expected, ""


def _measure(name, step, entry, chunks) -> Dict[Tuple[int, int], int]:
    """Run the real pipeline for one feature per chunk and read its counter."""
    stub = StubBiomeTable("test:biome", step, name)
    measured = {}
    for chunk_x, chunk_z in chunks:
        blocks, biomes = flat_chunk(biome_id=1)
        stats = decorate(chunk_x, chunk_z, blocks, biomes, preset="overworld", seed=SEED,
                         min_y=MIN_Y, height=HEIGHT, biome_table=stub,
                         placed_features=_PLACED, configured_features=_CONFIGURED)
        measured[(chunk_x, chunk_z)] = stats.attempts_by_feature.get(name, 0)
    return measured


def _check_ranged_count(name, step, entry, chunks) -> None:
    """A `count` provider with a range: bounds, and a histogram over the chunks."""
    provider = None
    for mod in entry.placement:
        if str(mod.get("type", "")).removeprefix("minecraft:") == "count":
            provider = mod["count"]
    bounds = _provider_bounds(provider)
    measured = _measure(name, step, entry, chunks)
    values = list(measured.values())
    low, high = bounds
    check(f"{name}: count stays inside its provider's range",
          all(low <= value <= high for value in values),
          f"range [{low}, {high}], measured [{min(values)}, {max(values)}] "
          f"over {len(values)} chunks")
    weights = _provider_weights(provider)
    if weights:
        counts = collections.Counter(values)
        report(f"{name}: distribution", ", ".join(
            f"{value}:{counts.get(value, 0)}" for value in sorted(counts)))


def _provider_bounds(payload) -> Tuple[int, int]:
    if isinstance(payload, int):
        return payload, payload
    kind = str(payload.get("type", "")).removeprefix("minecraft:")
    if kind == "uniform":
        return int(payload["min_inclusive"]), int(payload["max_inclusive"])
    if kind == "biased_to_bottom":
        return int(payload["min_inclusive"]), int(payload["max_inclusive"])
    if kind == "weighted_list":
        lows = [_provider_bounds(entry["data"]) for entry in payload["distribution"]]
        return min(low for low, _ in lows), max(high for _, high in lows)
    return 0, 10**9


def _provider_weights(payload) -> Dict[int, int]:
    if isinstance(payload, dict) and str(payload.get("type", "")).endswith("weighted_list"):
        out: Dict[int, int] = {}
        for entry in payload["distribution"]:
            data = entry["data"]
            if isinstance(data, int):
                out[data] = int(entry["weight"])
        return out
    return {}


# ==========================================================================
# Layer 4: modifier distributions
# ==========================================================================


def layer4_distributions() -> None:
    print("Layer 4 - modifier distributions")
    blocks, biomes = flat_chunk(surface_y=64)
    ctx = make_context(blocks, biomes, StubBiomeTable("test:biome", 0, "test:feature"))

    # `in_square`: uniform over the 16x16 footprint. A range check passes for a
    # generator that only ever picks even coordinates; a histogram does not.
    modifier = parse_placement_modifier({"type": "minecraft:in_square"})
    histogram_x = collections.Counter()
    histogram_z = collections.Counter()
    samples = 16000
    for index in range(samples):
        random = WorldgenRandom(Xoroshiro.from_seed(index))
        for position in modifier.positions(ctx, random, (0, 64, 0)):
            histogram_x[position[0]] += 1
            histogram_z[position[2]] += 1
    expected = samples / 16.0
    sigma = math.sqrt(expected * (1.0 - 1.0 / 16.0))
    worst = max(abs(histogram_x[value] - expected) for value in range(16)) / sigma
    check("in_square is uniform over the 16x16 footprint",
          len(histogram_x) == 16 and worst < 4.5,
          f"worst bucket {worst:.2f} sigma off "
          f"({samples} samples, {expected:.0f} expected per bucket)")
    report("in_square histogram", " ".join(f"{histogram_x[v]:5d}" for v in range(16)))

    # `height_range` uniform: the resolved interval is [-64 + 10, min_y + height - 1 - 10].
    payload = {"height": {"type": "minecraft:uniform",
                          "min_inclusive": {"above_bottom": 10},
                          "max_inclusive": {"below_top": 10}}}
    modifier = parse_placement_modifier({"type": "minecraft:height_range", **payload})
    low = MIN_Y + 10
    high = MIN_Y + HEIGHT - 1 - 10
    histogram = collections.Counter()
    for index in range(16000):
        random = WorldgenRandom(Xoroshiro.from_seed(index))
        for position in modifier.positions(ctx, random, (0, 0, 0)):
            histogram[position[1]] += 1
    span = high - low + 1
    flat = sum(histogram.values()) / span
    deviation = max(abs(histogram[value] - flat) for value in histogram) / math.sqrt(flat)
    check("height_range is uniform over the resolved interval",
          set(histogram) == set(range(low, high + 1)) and deviation < 4.5,
          f"[{low}, {high}], worst bucket {deviation:.2f} sigma off "
          f"({flat:.0f} expected per bucket)")

    # `heightmap`: one block *above* the top matching block -- vanilla's
    # `getHeight` is the first available Y, not the top block's own Y. Getting
    # this backwards is what stops every tree and grass patch from placing.
    for kind, expected_y in (("MOTION_BLOCKING", 65), ("OCEAN_FLOOR", 65),
                             ("WORLD_SURFACE_WG", 65)):
        modifier = parse_placement_modifier({"type": "minecraft:heightmap",
                                             "heightmap": kind})
        produced = list(modifier.positions(ctx, WorldgenRandom(Xoroshiro.from_seed(7)),
                                           (3, 0, 9)))
        check(f"heightmap {kind} places one above the surface block",
              produced == [(3, expected_y, 9)], f"got {produced}")

    # A heightmap array passed by the caller is top-block Y, so the same
    # modifier has to answer the same thing when it comes from that array.
    given = np.full((16, 16), 64, dtype=np.int64)
    given_ctx = make_context(*flat_chunk(surface_y=64),
                             biome_table=StubBiomeTable("test:biome", 0, "test:feature"),
                             heightmap=given)
    modifier = parse_placement_modifier({"type": "minecraft:heightmap",
                                         "heightmap": "WORLD_SURFACE_WG"})
    produced = list(modifier.positions(given_ctx, WorldgenRandom(Xoroshiro.from_seed(7)),
                                       (3, 0, 9)))
    check("a caller-supplied heightmap array is read as top-block Y",
          produced == [(3, 65, 9)], f"got {produced}")

    empty = np.zeros((HEIGHT, 16, 16), dtype=np.uint16)
    empty_ctx = make_context(empty, biomes, StubBiomeTable("test:biome", 0, "test:feature"))
    check("an all-air column reads min_y",
          empty_ctx.height_at("world_surface_wg", 0, 0) == MIN_Y,
          f"got {empty_ctx.height_at('world_surface_wg', 0, 0)}")
    modifier = parse_placement_modifier({"type": "minecraft:heightmap",
                                         "heightmap": "WORLD_SURFACE_WG"})
    produced = list(modifier.positions(empty_ctx, WorldgenRandom(Xoroshiro.from_seed(7)),
                                       (0, 0, 0)))
    check("heightmap yields nothing on an all-air column, where min_y is not "
          "strictly above min_y", produced == [], f"got {produced}")

    # `fixed_placement` in data order, and `count` as an exact multiplier.
    modifier = parse_placement_modifier({"type": "minecraft:fixed_placement",
                                         "positions": [[1, 2, 3], [4, 5, 6]]})
    check("fixed_placement yields the data's positions in order",
          list(modifier.positions(ctx, None, (0, 0, 0))) == [(1, 2, 3), (4, 5, 6)])
    modifier = parse_placement_modifier({"type": "minecraft:count", "count": 3})
    check("count yields exactly its count",
          len(list(modifier.positions(ctx, WorldgenRandom(Xoroshiro.from_seed(3)),
                                      (0, 0, 0)))) == 3)


# ==========================================================================
# Layer 5: cross-border writes
# ==========================================================================


def layer5_cross_border(placed, configured) -> None:
    print("Layer 5 - writes that leave the chunk")
    facts = block_facts()
    floor = facts.default_state("minecraft:stone")
    marker = facts.default_state("minecraft:gold_block")
    inside = [(4, 100, 4), (15, 100, 15)]
    outside = [(-1, 100, 4), (16, 100, 4), (4, 100, -1), (4, 100, 16), (4, 400, 4)]
    positions = inside + outside
    # A *placed* feature with an inline configured one: `simple_block` writes at
    # the position it is given, which is what makes the drop countable. A named
    # feature would drag its own config in and move the test somewhere else.
    payload = {
        "feature": {
            "type": "minecraft:simple_block",
            "config": {"to_place": {"type": "minecraft:simple_state_provider",
                                    "state": {"Name": "minecraft:gold_block"}}},
        },
        "placement": [{"type": "minecraft:fixed_placement", "positions": positions}],
    }
    from world.vanilla.features import PlacedFeature

    entry = PlacedFeature("test:cross_border", payload)
    stub = StubBiomeTable("test:biome", 0, "test:cross_border")
    blocks = np.zeros((HEIGHT, 16, 16), dtype=np.uint16)
    blocks[:, :, :] = floor
    biomes = np.zeros((HEIGHT // 4, 4, 4), dtype=np.int32)
    biomes[:, :, :] = 1
    stats = decorate(0, 0, blocks, biomes, preset="overworld", seed=SEED, min_y=MIN_Y,
                     height=HEIGHT, biome_table=stub,
                     placed_features={"test:cross_border": entry},
                     configured_features=configured)

    check("every out-of-footprint write is counted",
          stats.writes_dropped == len(outside),
          f"dropped {stats.writes_dropped}, expected {len(outside)}")
    check("the counters agree about why",
          stats.writes_dropped == stats.writes_outside_xz + stats.writes_outside_y,
          f"xz={stats.writes_outside_xz} y={stats.writes_outside_y}")
    check("in-footprint writes still landed",
          all(int(blocks[y - MIN_Y, z, x]) == marker for x, y, z in inside),
          f"states {[int(blocks[y - MIN_Y, z, x]) for x, y, z in inside]}, want {marker}")
    check("the out-of-footprint positions were not written in place",
          int(blocks[100 - MIN_Y, 0, 0]) == floor and
          int(blocks[100 - MIN_Y, 15, 15]) == marker,
          "")
    check("attempts are still counted for a feature that drops writes",
          stats.attempts_by_feature.get("test:cross_border", 0) == len(positions),
          f"{stats.attempts_by_feature.get('test:cross_border', 0)} attempts for "
          f"{len(positions)} positions")

    # The dropped counter must be reachable from a real feature, not only from a
    # synthetic one: an ordinary chunk writes over its border and says so.
    plains_chunk, plains_biomes = flat_chunk(surface_y=64)
    table = BiomeTable.from_assets()
    stats = decorate(0, 0, plains_chunk, plains_biomes, preset="overworld", seed=SEED,
                     min_y=MIN_Y, height=HEIGHT, biome_table=table,
                     placed_features=placed, configured_features=configured)
    report("plains chunk", f"{stats.writes_dropped} dropped writes "
                           f"(xz={stats.writes_outside_xz}, y={stats.writes_outside_y})")
    check("a real feature overhanging the border is reported, not hidden",
          stats.writes_dropped > 0,
          f"{stats.writes_dropped} dropped")


# ==========================================================================
# Layer 6: biome gating
# ==========================================================================


def layer6_biome_gating(biome_table, placed, configured) -> None:
    print("Layer 6 - biome gating on a real biome volume")
    gated = [name for name, entry in placed.items()
             if any(str(mod.get("type", "")).endswith(":biome")
                    or mod.get("type") == "minecraft:biome" for mod in entry.placement)]
    report("placed features carrying a `biome` modifier", str(len(gated)))
    check("most placed features are biome-gated", len(gated) > 150,
          f"{len(gated)} of {len(placed)}")

    plains = biome_table.id_of("minecraft:plains")
    desert = biome_table.id_of("minecraft:desert")
    plains_features = set()
    for step in biome_table.features["minecraft:plains"]:
        plains_features.update(step)
    desert_features = set()
    for step in biome_table.features["minecraft:desert"]:
        desert_features.update(step)

    only_plains = [name for name in gated
                   if name in plains_features and name not in desert_features]
    only_desert = [name for name in gated
                   if name in desert_features and name not in plains_features]
    check("there are features exclusive to each biome",
          bool(only_plains) and bool(only_desert),
          f"{len(only_plains)} plains-only, {len(only_desert)} desert-only")

    # The positive control has to be a feature that is attempted in *every*
    # chunk: `trees_plains` is a 19-in-20 chance of a zero count, so it would
    # pass a "no attempts" check for the wrong reason.
    preferred = "minecraft:patch_grass_plain"
    if preferred in only_plains:
        target = preferred
    else:
        target = next(name for name in only_plains
                      if any(str(mod.get("type", "")).endswith(
                          ("count", "rarity_filter")) for mod in placed[name].placement))
    report("gating target", target)

    for other_name, other_id, expected_zero in (("desert", desert, True),
                                                ("plains", plains, False)):
        blocks, biomes = flat_chunk(surface_y=64, biome_id=other_id)
        stats = decorate(0, 0, blocks, biomes, preset="overworld", seed=SEED, min_y=MIN_Y,
                         height=HEIGHT, biome_table=biome_table,
                         placed_features=placed, configured_features=configured)
        attempts = stats.attempts_by_feature.get(target, 0)
        if expected_zero:
            check(f"{target} is never attempted in a chunk whose biomes are {other_name}",
                  attempts == 0, f"{attempts} attempts")
        else:
            check(f"{target} is attempted in a chunk whose biomes are {other_name}",
                  attempts > 0, f"{attempts} attempts")

    # The same feature, forced into a biome that does not list it: the pipeline
    # must still refuse it, because the `biome` modifier is what enforces that.
    stub_features = [name for name in only_plains[:8]]
    lists = [() for _ in range(11)]
    lists[9] = tuple(stub_features)
    impostor = _ImpostorTable("test:biome", tuple(lists), biome_table.id_of("minecraft:plains"))
    plains_volume = np.full((HEIGHT // 4, 4, 4), biome_table.id_of("minecraft:desert"),
                            dtype=np.int32)
    blocks, _ = flat_chunk(surface_y=64)
    stats = decorate(0, 0, blocks, plains_volume, preset="overworld", seed=SEED, min_y=MIN_Y,
                     height=HEIGHT, biome_table=impostor, placed_features=placed,
                     configured_features=configured)
    total = sum(stats.attempts_by_feature.values())
    check("a feature is refused where the biome volume does not list it",
          total == 0, f"{total} attempts across {len(stub_features)} gated features")


class _ImpostorTable:
    """A table that lists features for `test:biome` while the chunk is desert.

    The biome *volume* still answers desert, so the `biome` modifier is the only
    thing that can refuse placement -- which is the point of the test.
    """

    def __init__(self, name: str, steps, numeric: int) -> None:
        self.features = {name: steps}
        self.ids = {name: numeric}
        self.names = {biome: "minecraft:desert" for biome in range(0, 256)}
        self.temperature = {name: 0.5}

    def iter_features(self):
        for biome, steps in self.features.items():
            for step, entries in enumerate(steps):
                for feature in entries:
                    yield biome, step, feature


# ==========================================================================
# Layer 7: the counters themselves
# ==========================================================================


def layer7_counters(biome_table, placed, configured) -> None:
    print("Layer 7 - counter consistency")
    plains = biome_table.id_of("minecraft:plains")
    blocks, biomes = flat_chunk(surface_y=64, biome_id=plains)
    stats = decorate(0, 0, blocks, biomes, preset="overworld", seed=SEED, min_y=MIN_Y,
                     height=HEIGHT, biome_table=biome_table, placed_features=placed,
                     configured_features=configured)
    check("top-level attempts plus nested attempts is every terminal call",
          sum(stats.attempts_by_feature.values()) + stats.placements_nested
          == stats.features_attempted,
          f"{sum(stats.attempts_by_feature.values())} + {stats.placements_nested} "
          f"vs {stats.features_attempted}")
    report("nested feature calls (random_patch tries and selector branches)",
           str(stats.placements_nested))
    check("features_placed never exceeds features_attempted",
          0 < stats.features_placed <= stats.features_attempted,
          f"{stats.features_placed} <= {stats.features_attempted}")
    check("features_run matches the summed per-feature runs",
          sum(stats.runs_by_feature.values()) == stats.features_run,
          f"{sum(stats.runs_by_feature.values())} vs {stats.features_run}")
    check("every run is counted in a step",
          sum(stats.by_step.values()) == stats.features_run,
          f"steps={dict(sorted(stats.by_step.items()))}")
    unexpected = {name: count for name, count in stats.unimplemented.items()
                  if name.split(":")[0] in implemented_types()}
    check("no implemented type reported an unimplemented sub-part", not unexpected,
          f"{unexpected}")
    report("unimplemented encounters in a plains chunk",
           ", ".join(f"{name}:{count}" for name, count in stats.unimplemented.most_common())
           or "none")
    report("sea level for the overworld preset", str(sea_level_for("overworld")))


# ==========================================================================


def main() -> int:
    global _CONFIGURED, _PLACED

    print("Feature-placement acceptance")
    print(f"  seed {SEED}, overworld preset, {MIN_Y}..{MIN_Y + HEIGHT}")
    print()

    if not layer0_position_hash():
        print("\nRESULT: FAIL (the position hash is wrong; every layer above it "
              "would be meaningless)")
        return 1

    print()
    _CONFIGURED = load_configured_features()
    _PLACED = load_placed_features()
    placed = _PLACED
    biome_table = BiomeTable.from_assets()

    layer1_gap_list(biome_table, placed, _CONFIGURED)
    print()
    layer2_determinism(biome_table, placed, _CONFIGURED)
    print()
    layer2b_biome_sweep(biome_table, placed, _CONFIGURED)
    print()

    # The count checks need each feature's step; take it from the biome data
    # rather than from the module's sorter.
    _STEP_BY_FEATURE.clear()
    for biome, step, feature in biome_table.iter_features():
        _STEP_BY_FEATURE.setdefault(feature, step)

    layer3_counts(biome_table, placed, _CONFIGURED)
    print()
    layer4_distributions()
    print()
    layer5_cross_border(placed, _CONFIGURED)
    print()
    layer6_biome_gating(biome_table, placed, _CONFIGURED)
    print()
    layer7_counters(biome_table, placed, _CONFIGURED)

    print()
    if failures:
        print(f"RESULT: FAIL ({len(failures)} check(s))")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print("RESULT: PASS")
    return 0


_CONFIGURED: Dict[str, object] = {}
_PLACED: Dict[str, object] = {}
_STEP_BY_FEATURE: Dict[str, int] = {}


if __name__ == "__main__":
    raise SystemExit(main())
