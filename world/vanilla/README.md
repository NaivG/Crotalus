# `world/vanilla` — vanilla-compatible worldgen primitives

Terrain generation that reproduces Minecraft's numbers rather than merely
resembling them. `world/noise.py`'s own Perlin field is fine for plausible
terrain; nothing in this package is fine for that, because the requirement here
is that the same seed gives the same coastline.

    from world.vanilla import NoiseRouter

    router = NoiseRouter.from_assets("overworld", seed=1234)
    router.begin_chunk(0, 0)
    router.surface_height(x, z, -64, 320)   # highest Y where final_density > 0

    python tools/check_density_functions.py

## Layout

A chunk is generated in the order the modules are listed. Everything after
`router` consumes the one before it.

| module | what it holds |
|---|---|
| `random_source.py` | Xoroshiro128+, the splitter, `WorldgenRandom`, the 48-bit LCG, `Mth.getSeed` |
| `perlin.py` | Perlin, the Y-quantised variant, and the octave stack |
| `simplex.py` | the 2D/3D simplex sampler the biome temperature reads |
| `density.py` | the density-function interpreter: 16 node types, 4 wrappers |
| `router.py` | the fifteen-slot noise router, assembled and sampleable |
| `biome.py` | the multi-noise biome source: six climate values, quantised, resolved through the parameter tree |
| `surface.py` | the surface-rule interpreter, plus badlands and iceberg column surgery |
| `carver.py` | Minecraft's cave and canyon carvers (not re-exported from `__init__`) |
| `features.py` | the decorated-feature stage: 15 placement modifiers, 30 configured-feature types (not re-exported) |
| `terrain.py` | **the driver**: turns `router`'s `finalDensity` into a chunk's stone (not re-exported — see below) |

`terrain.py` is the only module here that writes to a `Chunk`; every other one
returns arrays and knows nothing about storage. It is also deliberately *not*
re-exported from `__init__.py`, because it subclasses `BaseGenerator` from
`world/generator.py`, and every other tool imports `world.vanilla.something`
directly — re-exporting it would drag the generator factory into all of them.
Reach it as `world.vanilla.terrain`, or better, let
`world.generator.create_generator()` pick it from `[world] generator`.

## The four things that are easy to get wrong

**Noise samplers come from names, not positions.** Each `Noise` node derives its
sampler from `md5("minecraft:<name>")`, XORed into the base deriver. Nothing
consumes a shared RNG stream, so a node's value does not depend on where it sits
in the tree. That is what makes `density_function_tests.json` evaluable on its
own, and it is why reordering the tree cannot change the terrain.

**Everything is f32 except the coordinates.** Vanilla's `DensityFunction` returns
`float`; the noise it samples is addressed by `double`. Python floats are all
`double`, so every node rounds its result to `np.float32` on the way out. Skip
that and the low bits drift and the surface moves.

**The wrappers split into two unrelated groups.** `CacheOnce`, `Cache2D` and
`FlatCache` are memoisation — they cannot change a value, only its cost.
`Interpolated` genuinely does change the value: it samples only the eight
corners of a cell and trilinearly blends, so a 4×8×4 cell costs 8 evaluations
instead of 128, and the field inside a cell is a straight line between corners
rather than the wrapped function. Treating the two groups alike is the easiest
way to get quietly wrong terrain.

**There are two phases, and golden values belong to one of them.** Vanilla
builds a *proto* router (wrappers erased) and then a *chunk* router (caching
and interpolation added). Pumpkin's test suite evaluates the proto phase, so
those reference values are pass-through evaluations. Comparing them against a
chunk-phase evaluation makes every `Interpolated` interior point disagree,
which reads like a broken interpolator when the interpolator is fine.
`parse(..., pass_through_wrappers=True)` selects the proto phase.

> **Where the phases actually differ, measured.** `Interpolated` appears in this
> dump under exactly three slots — `finalDensity`, `veinToggle`, `veinRidged` —
> and nowhere else. The six climate slots and the surface estimator contain
> none, so for them the two phases are the same function.
> `tools/check_biomes.py` and `tools/check_surface_rules.py` assert that, so a
> future dump that gains a wrapper fails a check instead of quietly moving biome
> borders.

## Biomes: the six climate numbers

`biome.py` samples six router slots at a **quart** position, quantises each to an
integer (`f32 * 10000`, truncated toward zero), and walks an R-tree of parameter
boxes for the nearest leaf. Four things about it:

* **The sampler takes quart coordinates and shifts them left by two.** Passing
  block coordinates raises nothing and spreads every biome four times as far.
* **The tree search is stateful.** Vanilla caches the last leaf and seeds the
  next search's distance bound with it, so a tie goes to whichever leaf was
  found first *in that history*. `fill_volume` is one batch call per chunk for
  exactly this reason: row order is part of the answer.
* **The biome ids are the `worldgen/biome` registry order**, which
  `assets/biome.json` records and the wire uses. They are not an index into the
  file's own ordering.
* **The overworld, `large_biomes` and `amplified` presets share one biome tree.**
  Only the terrain splines differ.

`tools/check_biomes.py` compares against 24576 quantised climate values, every
quart cell of a vanilla chunk dump, and a 724271-sample corpus.

## Surface rules: what the top of the terrain is made of

`surface.py` runs the preset's `surface_rule` tree over the default block in each
column, top-down. It only ever *replaces* the default block — with two
exceptions, badlands plateaus and frozen-ocean icebergs, which are column surgery
the pass runs before the rules see the column at all.

The traps are in the module docstring. The two that cost the most time here:

* **`Mth.getSeed` is a specific hash, not any position hash.** `random_source`
  carried an xxHash-shaped one because nothing called `split_pos`, so nothing
  ever exercised it. It is now pinned against eight vectors from a real
  Minecraft, and three separate subsystems depend on it.
* **`random_name` hashes `md5` big-endian**, the same recipe as a noise name —
  and *not* the little-endian reading Pumpkin's generated surface-rule constants
  carry. Big-endian reproduces 1280 bedrock cells exactly; the other reading
  misses 310 of them.

### What is checked against what

`tools/check_surface_rules.py` compares the pass cell for cell against seven
vanilla chunk dumps committed under `assets/vanilla_chunks/`. Each file carries
the same chunk twice — `before` is the noise stage and `after` is noise plus
surface — so a mismatch is attributable to this pass rather than to the terrain
under it.

| chunk | cells | differ | note |
|---|---|---|---|
| `overworld_0_0` | 98304 | **0** | plains and ocean |
| `overworld_frozen_ocean_-119_183` | 98304 | 7 | the 7 are `tuff` the noise dump does not have |
| `overworld_7_4` | 98304 | 1203 | deep sand under caves |
| `overworld_13579_-2_15` | 98304 | 4067 | badlands banding |
| `overworld_badlands_-595_544` | 98304 | 4700 | badlands banding |
| `nether_0_0` | 32768 | 803 | bedrock floor and roof |
| `end_0_0` | 32768 | **0** | |

Two of seven are exact and the overall disagreement is 1.9%. The residuals are
recorded as per-chunk ceilings in the tool rather than swept up, so a regression
shows as a number going *up* even in a chunk that was already imperfect.

**They are not explained yet.** The honest summary is that the shallow surface
rules are right and two families are not: the badlands `bandlands`/terracotta
path and the deep `stone_depth(add_surface_depth: true)` sand path, plus the
nether's bedrock bands. `minecraft:hole` is the most likely place to look — it
is implemented as `run_depth <= 0`, which is Pumpkin's reading and not something
this package has independent evidence for.

## Carvers and features: the two layers with no oracle

`carver.py` and `features.py` are ports of Minecraft's carver and
decorated-feature stages, and neither has a reference to check against: Pumpkin's
`assets/tests/` carries noise and surface dumps only, and this package will not
pretend a self-consistency check is a parity check.

**`tools/check_carvers.py`** checks determinism (same seed, same bytes, across
processes and carve orders), that the mask protects bedrock and the world top,
that the `replaceable` tag is never violated (20480 planted obsidian cells
survive), that lava stops at `lava_level`, that carved air is connected
(4.0 components/chunk, largest 85%), and it prints the numbers. It also prints
the two things that are *guesses*, with their sensitivity:

* **The RNG wrapper.** Whether vanilla wraps the algorithm's source in
  `WorldgenRandom` (the default here) or draws a bare `XoroshiroRandomSource`
  (Pumpkin's choice) is unverified, and it changes 163 carved cells to 2849.
* **`getRange() == 4`.** Taken from Pumpkin. If it were 8, tunnels would be 240
  blocks long and the carve would grow 755%.

**`tools/check_features.py`** is stronger, because placement *counts* can be
recomputed from the modifier definitions alone: **133 placed features are
recomputed independently and matched exactly, per chunk, over 36 chunks**
(`ore_iron_upper` 3240/3240, `bamboo` 1960/1960, `vines` 4572/4572, ...). It also
pins the `in_square` and `height_range` distributions, the `heightmap` offset,
and the out-of-footprint write accounting. Positions and block choices are *not*
verified - nothing offline can confirm a tree landed where vanilla's did.

**`tools/check_ore_veins.py`** is the one place where the *interpreted* reference
is itself under test: `features.py` runs `ore` through a numba kernel
(`_ore_kernel`, the same trade `carver.py` makes for its ellipsoids) and keeps
`_place_ore_python` beside it, and the tool compares the two block for block,
draw for draw, counter for counter - 1152 synthetic veins over every real ore
config, plus real chunks through the whole `decorate` pipeline. It also pins the
discard semantics as a truth table, because "discard chance on air exposure" is
easy to read as "never place when the chance is 1.0" and that reading makes
buried lapis, buried diamond and ancient debris generate nothing at all
(Pumpkin's `OreFeature::should_place` places them whenever no neighbour is air).
Three behavioural notes this port carries, each found while building the kernel:

* `should_not_discard || !is_exposed_to_air`, with the roll taken before the
  exposure scan and `discard = 1.0` consuming no roll.
* The vein occlusion squares are multiplies (`x * x`, like the Java and Rust);
  Python's `x ** 2` routes through libm `pow` and can sit one ULP off the
  product, which flips the drop decision when the sides nearly tie.
* `Surface.top` computes live, like vanilla's chunk heightmaps that `setBlock`
  keeps current: a tree's leaves count for `ocean_floor_wg`, and a later
  `surface_water_depth_filter` in the same feature sees them. The ore probe
  reads `Surface.columns`' frozen grid instead - the ore step replaces solid
  with solid and cannot move those rows, so the freeze is invisible there and
  keeps the probe's 240 boxes per chunk affordable.

Implemented: all 15 placement modifiers and 30 configured-feature types.
**Missing: 32 types, 52 of 2641 biome-step references (2.0%)**, enumerated by
`world.vanilla.features.missing_types()` and printed by the tool -
`kelp` (6), `fossil` (6), `vines` (4), `iceberg` (4), the mushroom and fungus
family, the dripstone family, and the End's types. They are never a silent
no-op: an unimplemented type is counted and reported.

Both stages read only the chunk they are given. Vanilla reads a 3x3 neighbourhood
for carvers and features alike, so a tunnel started next door does not continue
into this chunk, and a feature straddling a border is clipped. The clipped writes
are counted (`FeatureStats.writes_dropped`, ~76-313 per chunk in the fixtures),
not silently lost.

## Cost

Measured on one core, overworld, seed 0, warm, per chunk:

| stage | ms |
|---|---|
| density terrain fill (`world/vanilla/terrain.py`) | ~130–900 |
| terrain fill (`world/legacy/generator.py`, its own Perlin) | ~20 |
| biome volume: 6 density slots over 1536 quart cells | 200–450 |
| surface rules: one Python walk per stone cell | ~600 |
| carvers (compiled walk; interpreted is ~10x) | 35–55 |
| features (compiled ore walk; interpreted is ~8x) | ~25 |
| chunk encode (`world/chunk_codec.py`) | 1.1 |

The first row is the terrain the `vanilla` backend actually uses, and it is the
whole point of `world/vanilla/terrain.py` existing. It is one batched
`NoiseRouter.fill_chunk` over 98,304 samples: ~130 ms warm once the router's
`FlatCache` is populated, ~600–900 ms on the first few chunks of a session. The
spread is the cache, not the arithmetic — see "Performance" below.

So a fully vanilla chunk is ~1 s warm end to end, against ~1 s for the same
layers over the cheap Perlin heightmap: the decoration is what costs, and it
costs the same either way. `[world] generator = "legacy"` exists for that
reason, not because the terrain is the slow part. The biome stage has the
obvious win available: the six slots are sampled in six separate batched calls,
so they share no common subexpression, and `density.py`'s per-batch memo is not
reachable across calls.

## `Interpolated` cell sizes

The vanilla datapack carries `cell_size_xz`/`cell_size_y` per node, but the
resolved dump in `assets/density_function.json` drops them. The values used here
were read out of the datapack directly:

| source | cell |
|---|---|
| `overworld/chunk_surface_level.json` | 16, 1 |
| `overworld/final_density.json` | 4, 8 |
| `overworld/caves/noodle.json` (×4) | 4, 8 |
| `overworld/ore_vein/{mask,toggle}.json` | 4, 8 |
| nether | 4, 8 |
| end | 8, 4 |

So the overworld family is (4, 8) throughout, the surface-level function is the
lone (16, 1), and the End is (8, 4). `cell_size_for(preset, slot)` picks between
them.

## Validation

`tools/check_density_functions.py` runs four layers, strongest first. Layers 1–3
compare against numbers that came from somewhere else; layer 4 is a recorded
baseline of our own output and exists only to catch drift.

1. **Random sources** — Stafford mixing, Xoroshiro draws, the LCG, against
   vectors checked against Java.
2. **The overworld router at seed 0** — eleven slots against values verified
   against a real Minecraft. Reproducing all eleven at once means the RNG, the
   Perlin construction, all three octave-normalisation factors and every node
   type are correct simultaneously. Worst relative error: **2.5e-07**, which is
   f32 rounding.
3. **The vein tables** — 125 positions for `vein_toggle`. The other two are
   skipped; see below.
4. **The corpus** — all 11 functions in `density_function_tests.json` across 512
   positions, against `tools/baselines/density_functions.json`.

The harness is not vacuous: changing the `Interpolated` cell size from 4×8 to
8×4, or dropping one axis of the blend, both make layer 4 fail on
`overworld/caves/noodle` — the only corpus function that uses `Interpolated`.

`tools/check_density_vectorized.py` is the fifth layer, and it runs against the
*first* four: the batch path (`eval_array`, `fill_chunk`, `surface_heights`)
must be **bit-identical** to the scalar one, across the corpus and all fifteen
router slots in both the proto and chunk phases, at positions chosen to land on
and off interpolation-cell corners, on gradient breaks, and at negative
coordinates where Python and numpy would disagree about `%` if anything got it
wrong. "Close" is not good enough here: at f32, a rounding difference per
sample accumulates into a surface that moves.

## Known data gap: the dump predates 1.21.9

`assets/density_function.json` has no `find_top_surface` anywhere, which is what
25w31a (1.21.9) added. The ore-vein mask was reworked in the same era, so this
dump carries the older definitions:

| slot | this dump | 1.21.9+ |
|---|---|---|
| `veinGap` | `Noise(ore_gap)` | `sub(-0.3, Noise(ore_gap))` |
| `veinRidged` | `MAX(ABS(a), ABS(b)) + (-0.08)` | `range_choice(veinToggle, [-0.4,0.4), -1.0, 0.08 - MAX(ABS(a), ABS(b)))` |

The reference values are the newer form, so `veinGap` disagrees by exactly
`-mine - 0.3` — the signature of a missing `sub`. `vein_toggle` still matches
and is checked. `veinRidged` and `veinGap` are skipped with that note attached.

**Do not "fix" the interpreter to match them.** The dump is what the rest of the
project reads. To check those two, first replace the vein slots in the dump with
the definitions under
`Pumpkin/assets/datapack/data/minecraft/worldgen/density_function/overworld/ore_vein/`.

`preliminary_surface_level` is the other gap: vanilla's surface estimator is
`find_top_surface(final_density, preliminary_surface_level, -128, 8)`, and this
dump has no `find_top_surface`. The slot itself evaluates, but it is a
y-dependent candidate level, not a finished surface — evaluate it at y=0 and it
returns its low bias, not the answer. Use `router.surface_height`, which scans
`final_density` directly, until `find_top_surface` is implemented.

## Performance

Pure Python trees are large — `final_density` in the overworld is 3585 nodes,
of which only ~134 are distinct — and there are two ways to sample them: `eval`
for one position, `eval_array` for a batch. Both return the same bits;
`tools/check_density_vectorized.py` is what makes that a fact rather than a
hope. Measured on the overworld at `final_density`:

| workload | before | after |
|---|---|---|
| warm sample (y-scan, caches hit) | 9 ms | 0.13 ms |
| cold sample (fresh x, z) | 37 ms | 16 ms |
| `surface_height`, one column | 2.2 s | 35 ms |
| `fill_chunk`, 16×384×16 = 98k samples | minutes | ~130 ms |
| six slots per chunk (climate + depth + final) | minutes | ~270 ms |

What buys the time, in order of impact:

* **Numba kernels for the Perlin chain.** A single sample is one
  Python-to-JIT transition covering wrap, floor, eight gradient dots and the
  whole lerp tree; the array path runs the *same* kernel per point, so the two
  paths cannot drift. Without numba there is a pure-numpy fallback.
* **The batch path with a per-batch CSE memo.** Parse-time hash-consing turns
  the tree into a DAG, and the memo evaluates each unique node once per batch
  instead of once per visit — the 240 copies of a shift cluster cost one
  shift-field evaluation.
* **Point-set decompositions are cached per batch.** The memoisation wrappers
  (`Cache2D`, `FlatCache`, `CacheOnce`) drop duplicate columns/positions
  before evaluating their input; that split is a sort, so it runs once per
  distinct point set and is keyed on packed uint64 coordinates.
* **`Interpolated` corners are computed once per cell** — which is how vanilla
  defines the wrapper — instead of once per interior block.
* **Splines evaluate only the two points that bracket the location**, as
  vanilla's `CubicSpline` does; a nested value-spline no longer costs the whole
  curve per sample.

Which path to use where: scalar `eval` / `surface_height` / `evaluate_column`
for single points and single columns — below ~1000 points the batch path's
per-node numpy overhead is not amortised, and a column through
`evaluate_array` is *slower* than the scalar loop. Use `evaluate_array` /
`fill_chunk` / `surface_heights` for anything volume-shaped. One caveat if a
tool interleaves the two: right after a big batch, the first scalar call pays a
CPython gc pass over the freed temporaries (~200 ms once, then it is gone);
call `gc.collect()` between the paths to move that cost where it is convenient.

This package is still *the reference*: it is correct, and it is what any other
fast path is tested against. The node boundary is where f32 rounding happens,
and both evaluation paths keep it.
