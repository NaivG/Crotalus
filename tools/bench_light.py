"""What the light engine costs: one patch, one block change, one chunk.

    py -3.14t tools/bench_light.py [--repeats 30] [--changes 300] [--regen]

The numbers `LIGHT_PLAN.md` §7.4 asks for, measured on **real generated chunks**
(§5.4) so the terrain is not a synthetic best case:

* **(a) one chunk of sky + block light**, 16x16x384. The kernel works on a 3x3
  patch, so this is measured on a region of nine *identical* real chunks and
  divided by nine -- nine identical columns is exactly nine times one column's
  work, and it is the figure comparable to the plan's 0.70 ms warm per column.
  The raw patch time is printed next to it;
* **(b) one block change**, P50 and P99, both at the kernel (the plan's §5.3
  number) and through `DimensionLight` (what a tick actually pays: the queue,
  the region, the write-back);
* **(c) one real 3x3 patch end to end** -- nine *different* chunks, including
  unpacking 18 light arrays, the kernel and packing the centre chunk back.
  §5.5 lists this as untested; it is the number that decides whether
  `light-chunks-per-tick = 4` fits in `light-budget-ms = 4.0`;
* the **first call** in the process, which is the JIT (or the on-disk numba
  cache) rather than the engine: with a cold cache it is tens of seconds, which
  is why `world.light_engine.warm_up()` exists and why `cache=True` is not
  optional.

The world is `world.legacy` -- real generated terrain, since a synthetic flat
world would measure the loops and not the light. It is cached to a temporary
`.npz` after the first run (about 20-35 ms per chunk here; `LIGHT_PLAN.md` §5.4
quotes 1.6 s per chunk for the *preset-driven* pipeline, which this tool does not
pay because it asks for the dataclass terrain constants). Pass `--regen` to throw
the cache away. Exit code 0 when it completes: this is a benchmark, not a check,
so the targets are printed next to the measurements instead of being enforced.

Run: py -3.14t tools/bench_light.py
"""

from __future__ import annotations

import argparse
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from world.chunk import Chunk, ChunkPos  # noqa: E402
from world.generator import GeneratorSettings, create_generator  # noqa: E402
from world.light_engine import (  # noqa: E402
    NO_SECTION_TABLE,
    DimensionLight,
    LightQueues,
    relight_region,
    update_block_light,
    update_sky_column,
    warm_up,
)
from world.light_layout import CELL_COUNT, unpack_light  # noqa: E402
from world.light_region import REGION_CHUNKS, SECTION_AIR, LightRegion  # noqa: E402

#: The vertical window a light patch is measured over (the overworld's).
MIN_Y, HEIGHT = -64, 384
SECTIONS = HEIGHT // 16

#: The fixture is a 5x5 chunk grid, not 3x3: a chunk is only lit when all nine
#: of *its* neighbours are resident, so a 3x3 grid can light its middle chunk and
#: nothing else (`deferred_chunks` climbs forever). Five by five lights the
#: middle nine, which is what an incremental edit needs -- `DimensionLight`
#: sends an edit through a full patch while any chunk of its 3x3 is unlit.
GRID = 5
GRID_CHUNKS = GRID * GRID
GRID_RADIUS = GRID // 2

#: The fixture seed. Same seed as `tools/check_chunk_lifecycle.py`, so a chunk
#: from one tool is a chunk from the other.
SEED = 12345

#: 20 Hz: the budget light shares with chunk streaming and the entity tracker.
TICK_BUDGET_MS = 50.0

#: §4.8 defaults, for the "does 4 chunks fit in 4 ms" question.
LIGHT_BUDGET_MS = 4.0
CHUNKS_PER_TICK = 4


# ---------------------------------------------------------------------------
# fixture
# ---------------------------------------------------------------------------


def fixture_path(seed: int) -> Path:
    return Path(tempfile.gettempdir()) / f"crotalus_light_fixture_{seed}_{GRID}x{GRID}.npz"


def load_fixture(seed: int, regenerate: bool = False) -> np.ndarray:
    """A 5x5 grid of real chunks' block arrays, `(25, sections, 4096)` uint16.

    Generated, not tiled: a patch measurement is only real if the neighbours are
    different terrain, because that is what makes the boundary cells work.
    `grid_index(dx, dz) = (dz + 2) * 5 + (dx + 2)`. Cached because `world.legacy`
    is expensive per chunk and the numbers below should not depend on how many
    times the tool has been run.
    """
    path = fixture_path(seed)
    if path.exists() and not regenerate:
        with np.load(path) as data:
            blocks = data["blocks"]
        print(f"fixture: {path} ({blocks.shape}, cached)")
        return blocks

    print(f"fixture: generating {GRID}x{GRID} chunks with world.legacy (seed {seed}) ...")
    generator = create_generator(GeneratorSettings(seed=seed), backend="legacy")
    blocks = np.empty((GRID_CHUNKS, SECTIONS, CELL_COUNT), dtype=np.uint16)
    started = time.perf_counter()
    for dz in range(-GRID_RADIUS, GRID_RADIUS + 1):
        for dx in range(-GRID_RADIUS, GRID_RADIUS + 1):
            chunk = Chunk(ChunkPos(dx, dz), min_y=MIN_Y, height=HEIGHT)
            generator.generate_chunk(chunk)
            index = grid_index(dx, dz)
            for section_idx, section in enumerate(chunk.sections):
                blocks[index, section_idx] = section.blocks
    elapsed = time.perf_counter() - started
    np.savez_compressed(path, blocks=blocks)
    print(f"fixture: {elapsed:.2f} s for {GRID_CHUNKS} chunks "
          f"({elapsed / GRID_CHUNKS * 1e3:.0f} ms/chunk), cached to {path}")
    return blocks


def grid_index(dx: int, dz: int) -> int:
    """Row-major index into the fixture grid for a chunk offset from the centre."""
    return (dz + GRID_RADIUS) * GRID + (dx + GRID_RADIUS)


def build_chunks(blocks: np.ndarray) -> dict:
    """Independent `Chunk` objects for the whole grid, keyed by `(dx, dz)`."""
    chunks = {}
    for dz in range(-GRID_RADIUS, GRID_RADIUS + 1):
        for dx in range(-GRID_RADIUS, GRID_RADIUS + 1):
            index = grid_index(dx, dz)
            chunk = Chunk(ChunkPos(dx, dz), min_y=MIN_Y, height=HEIGHT)
            for section_idx, section in enumerate(chunk.sections):
                section.blocks[:] = blocks[index, section_idx]
                section.recompute_non_air_count()
            chunks[(dx, dz)] = chunk
    return chunks


def region_of(chunks: dict, cx: int = 0, cz: int = 0) -> LightRegion:
    region = LightRegion.build(chunks[(cx, cz)], lambda x, z: chunks.get((x, z)))
    if not region.ready:
        raise RuntimeError("fixture region is not ready")
    return region


# ---------------------------------------------------------------------------
# timing
# ---------------------------------------------------------------------------


def percentile(samples, fraction: float) -> float:
    """Nearest-rank percentile, in milliseconds."""
    if not samples:
        return float("nan")
    ordered = sorted(samples)
    rank = min(len(ordered) - 1, max(0, int(round(fraction * (len(ordered) - 1)))))
    return ordered[rank] * 1e3


def time_once(fn) -> float:
    started = time.perf_counter()
    fn()
    return time.perf_counter() - started


def time_many(fn, repeats: int):
    return [time_once(fn) for _ in range(repeats)]


def report(label: str, samples, note: str = "") -> None:
    best = min(samples) * 1e3
    print(f"  {label:<34} best {best:8.3f} ms   p50 {percentile(samples, 0.5):8.3f} ms"
          f"   p99 {percentile(samples, 0.99):8.3f} ms   {note}")


# ---------------------------------------------------------------------------
# (a) one chunk of sky and block light
# ---------------------------------------------------------------------------


def bench_full_chunk(chunks: dict, repeats: int) -> None:
    print("\n(a) one 16x16x384 column, sky + block light, full")
    # Nine identical real chunks: nine times one column's work, no synthetic
    # terrain anywhere, and the region layout the kernel requires.
    tiled = {}
    for dz in (-1, 0, 1):
        for dx in (-1, 0, 1):
            twin = Chunk(ChunkPos(dx, dz), min_y=MIN_Y, height=HEIGHT)
            for section_idx, section in enumerate(twin.sections):
                section.blocks[:] = chunks[(0, 0)].sections[section_idx].blocks
            tiled[(dx, dz)] = twin
    region = region_of(tiled)
    opacity, emitters, luminance = tables()
    sky, block, blocks = region.sky, region.block_light, region.blocks
    flat = np.concatenate([section.blocks for section in tiled[(0, 0)].sections])
    opacities = opacity[flat]
    print(f"  centre chunk: air {int((flat == 0).sum())},"
          f" opaque {int((opacities == 15).sum())},"
          f" dampening {int((opacities == 1).sum())},"
          f" emitters {int((luminance[flat] > 0).sum())}"
          f"  (LIGHT_PLAN §5.4: air 67499, opaque 30775, water 27, lava 3)")

    queues = LightQueues(SECTIONS).args()
    # `-1`: the unmasked kernel. (a) is a per-chunk figure for the plan's §5.1
    # targets, so it stays the scan-everything number and (c) carries the table.
    both = min(time_many(
        lambda: relight_region(blocks, opacity, emitters[0], emitters[1],
                               sky, block, True, NO_SECTION_TABLE, *queues),
        repeats,
    ))
    no_sky = min(time_many(
        lambda: relight_region(blocks, opacity, emitters[0], emitters[1],
                               sky, block, False, NO_SECTION_TABLE, *queues),
        repeats,
    ))
    relight_region(blocks, opacity, emitters[0], emitters[1], sky, block, True,
                   NO_SECTION_TABLE, *queues)
    print(f"  region of 9 identical real chunks: {both * 1e3:.3f} ms"
          f"  -> {both * 1e3 / 9:.3f} ms per chunk  (sky + block light)")
    print(f"  same region with do_sky=False:     {no_sky * 1e3:.3f} ms"
          f"  -> {no_sky * 1e3 / 9:.3f} ms per chunk  (block light only)")
    print(f"  LIGHT_PLAN §5.1 targets per chunk: 0.335 ms sky, 0.49 ms block light)")
    counts = light_stats(region)
    print(f"  centre chunk: sky>0 {counts['sky']}, sky in 1..14 {counts['sky_mid']},"
          f" block>0 {counts['block']}")


# ---------------------------------------------------------------------------
# (b) one block change
# ---------------------------------------------------------------------------


def bench_block_change(chunks: dict, changes: int, repeats: int) -> None:
    print(f"\n(b) one block change (P50/P99 over {changes} positions x place+break)")
    opacity, emitters, luminance = tables()
    stone = state_id("minecraft:stone")
    glowstone = state_id("minecraft:glowstone")
    # The engine allocates these once per dimension and reuses them, so
    # measuring a fresh `np.empty` per call would be measuring the allocator.
    queues = LightQueues(SECTIONS).args()

    rng = np.random.default_rng(7)
    positions = [
        (int(rng.integers(0, 16)), int(rng.integers(MIN_Y + 40, MIN_Y + 200)),
         int(rng.integers(0, 16)))
        for _ in range(changes)
    ]

    # -- bare kernel, region already open (the §5.3 measurement) -----------
    region = region_of(chunks)
    center = chunks[(0, 0)]
    bare = []
    for index, (x, y, z) in enumerate(positions):
        state = glowstone if index % 4 == 2 else stone
        for new_state in (state, 0):
            old_state = center.get_block(x, y, z)
            if old_state == new_state:
                continue
            center.set_block(x, y, z, new_state)
            section_idx = (y - MIN_Y) // 16
            cell = x + z * 16 + ((y - MIN_Y) % 16) * 256
            region.blocks[4, section_idx, cell] = new_state
            started = time.perf_counter()
            update_block_light(region.block_light, region.blocks, opacity,
                               emitters[0], emitters[1],
                               x, y - MIN_Y, z,
                               int(luminance[old_state]), int(luminance[new_state]),
                               *queues)
            update_sky_column(region.sky, region.blocks, opacity, x, z, *queues)
            bare.append(time.perf_counter() - started)
    report("kernel only (region open)", bare, f"{len(bare)} edits")

    # -- through the engine: queue + region + write-back -------------------
    chunks = build_chunks(load_fixture(SEED))
    light = DimensionLight(lambda x, z: chunks.get((x, z)), has_skylight=True,
                           budget_ms=LIGHT_BUDGET_MS,
                           chunks_per_tick=CHUNKS_PER_TICK)
    # The middle nine are the ones a 5x5 grid can light (each needs its own 3x3).
    for dz in (-1, 0, 1):
        for dx in (-1, 0, 1):
            light.queue_chunk(dx, dz)
    for _ in range(40):
        light.tick()
    print(f"  neighbourhood lit: {light.chunks_lit} chunks,"
          f" deferred {light.deferred_chunks}, fallbacks {light.fallback_patches}")
    engine = []
    for index, (x, y, z) in enumerate(positions):
        for new_state in ((glowstone if index % 4 == 2 else stone), 0):
            old_state = chunks[(0, 0)].get_block(x, y, z)
            if old_state == new_state:
                continue
            chunks[(0, 0)].set_block(x, y, z, new_state)
            light.block_changed(x, y, z, old_state, new_state)
            started = time.perf_counter()
            light.tick()
            engine.append(time.perf_counter() - started)
    report("DimensionLight change+tick", engine,
           f"{len(engine)} edits, includes the tick's write-back")
    print(f"  light stats: {light.stats()}")
    print(f"  one 50 ms tick fits ~{TICK_BUDGET_MS / percentile(bare, 0.5):.0f} bare changes,"
          f" ~{TICK_BUDGET_MS / percentile(engine, 0.5):.0f} engine changes")
    return chunks, light


# ---------------------------------------------------------------------------
# (c) a real 3x3 patch
# ---------------------------------------------------------------------------


def bench_patch(chunks: dict, repeats: int) -> None:
    print("\n(c) one real 3x3 patch: unpack + kernel + pack/write-back")
    opacity, emitters, _ = tables()
    queues = LightQueues(SECTIONS).args()

    def full_patch():
        # Exactly what `DimensionLight._light_chunk` does, section table included.
        region = LightRegion.build(chunks[(0, 0)], lambda x, z: chunks.get((x, z)),
                                   read_light=False)
        relight_region(region.blocks, opacity, emitters[0], emitters[1],
                       region.sky, region.block_light, True, region.air_sections,
                       *queues)
        region.write_back()

    samples = time_many(full_patch, repeats)
    report("patch total (9 different chunks)", samples, "LIGHT_PLAN §5.5 untested")

    build = time_many(
        lambda: LightRegion.build(chunks[(0, 0)], lambda x, z: chunks.get((x, z)),
                                  read_light=False),
        repeats,
    )
    report("  LightRegion.build (blocks only)", build, "9 chunks x 24 sections copied")

    unpack = time_many(
        lambda: LightRegion.build(chunks[(0, 0)], lambda x, z: chunks.get((x, z))),
        repeats,
    )
    report("  build with read_light=True", unpack, "the 432 unpack_light calls the above skips")

    region = region_of(chunks)
    write = time_many(region.write_back, repeats)
    report("  write_back (pack)", write, "centre chunk, 48 arrays")

    kernel = time_many(
        lambda: relight_region(region.blocks, opacity, emitters[0], emitters[1],
                               region.sky, region.block_light, True,
                               NO_SECTION_TABLE, *queues),
        repeats,
    )
    report("  kernel, no section table", kernel,
           "everything in (a) x 9 chunks, the pre-optimization kernel")

    masked = time_many(
        lambda: relight_region(region.blocks, opacity, emitters[0], emitters[1],
                               region.sky, region.block_light, True,
                               region.air_sections, *queues),
        repeats,
    )
    report("  kernel, with the section table", masked,
           f"{int((region.air_sections == SECTION_AIR).sum())} of"
           f" {region.air_sections.size} chunk-sections are all air")
    print(f"  => the table saves {percentile(kernel, 0.5) - percentile(masked, 0.5):.3f} ms"
          f" of {percentile(kernel, 0.5):.3f} ms p50 on this region")

    per_tick = max(1.0, LIGHT_BUDGET_MS) / max(percentile(samples, 0.5), 1e-9)
    print(f"  => light-budget-ms = {LIGHT_BUDGET_MS} fits {per_tick:.1f} patches"
          f" (light-chunks-per-tick = {CHUNKS_PER_TICK});"
          f" tick budget {TICK_BUDGET_MS} ms")
    counts = light_stats(region)
    print(f"  centre chunk: sky>0 {counts['sky']}, sky in 1..14 {counts['sky_mid']},"
          f" block>0 {counts['block']}")


def light_stats(region: LightRegion) -> dict:
    """Lit-cell counts of the centre chunk, the §5.2 sanity figures."""
    sky = region.sky[4].reshape(-1)
    block = region.block_light[4].reshape(-1)
    return {
        "sky": int(np.count_nonzero(sky)),
        "sky_mid": int(np.count_nonzero((sky > 0) & (sky < 15))),
        "block": int(np.count_nonzero(block)),
    }


# ---------------------------------------------------------------------------
# shared
# ---------------------------------------------------------------------------

_TABLES = None


def tables():
    """`(opacity, (emitter_ids, emitter_levels), luminance)` from the registry.

    Built once: `light_opacity_table` / `light_emitters` are the phase-1 tables
    and the kernels index the first two directly; the third is the Python-side
    state id -> luminance map a caller needs to say what a block change did.
    """
    global _TABLES
    if _TABLES is None:
        from registry.blocks import light_emitters, light_opacity_table

        opacity = light_opacity_table()
        emitters = light_emitters()
        luminance = np.zeros(65536, dtype=np.uint8)
        luminance[emitters[0]] = emitters[1]
        _TABLES = (opacity, emitters, luminance)
    return _TABLES


def state_id(name: str) -> int:
    from world.block_states import state_id as resolve

    return int(resolve(name))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repeats", type=int, default=30,
                        help="timed iterations for the full-chunk and patch sections")
    parser.add_argument("--changes", type=int, default=300,
                        help="block changes measured in section (b)")
    parser.add_argument("--regen", action="store_true", help="ignore the fixture cache")
    args = parser.parse_args()

    started = time.perf_counter()
    warm_up()
    first = time.perf_counter() - started
    second = time_once(warm_up)
    print(f"first warm_up(): {first:.2f} s (JIT compile, or the numba on-disk cache)")
    print(f"second warm_up(): {second * 1e3:.3f} ms   (an empty warm-up once compiled)")

    blocks = load_fixture(SEED, regenerate=args.regen)
    chunks = build_chunks(blocks)
    bench_full_chunk(chunks, args.repeats)
    chunks, _light = bench_block_change(chunks, args.changes, args.repeats)
    bench_patch(chunks, args.repeats)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
