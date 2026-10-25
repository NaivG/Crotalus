"""
Carver acceptance: does the ported carver stage still behave like vanilla's?

    python tools/check_carvers.py            # check
    python tools/check_carvers.py --emit-hash  # one carve, printed as a hash

What this guards, and what it cannot
------------------------------------
`Pumpkin/assets/tests/` has chunk dumps for the noise and surface stages and
none for the carvers, so there is no golden volume to compare against. What is
left is a set of properties that a wrong implementation violates:

* **The mask.** The read-only reference ships a unit test for
  `CarvingMask::visit` -- three bits in one column and one in another, yielding
  two runs. It is re-run here, together with the bounds and the bit layout.
* **Determinism and order independence.** Two carves of the same chunk must be
  bit-identical, and carving a neighbour first must not change the result. The
  second one is the real test: it fails for any implementation that shares an
  RNG stream between carver chunks or leaves `FlatCache`-like state behind. It
  is also run in a *fresh process*, which is the only way to catch a module
  level cache that leaks between runs.
* **The compiled walk vs the interpreted one.** `perlin.py`'s contract is that
  the numba path is bit-identical to the interpreted one, and this is the
  equivalent for the carvers: the RNG draw sequences, one ellipsoid, one
  tunnel, and whole chunks.
* **That it does something.** A large sample of chunks must produce a plausible
  carved fraction, measured against terrain from the real density function.
  Both "0% carved" and "everything carved" are failures, and the measured
  numbers are printed rather than only asserted.
* **Connectivity.** Caves are tunnels, so the carved cells in a chunk belong to
  a small number of connected components. The old carver this project started
  with produced thousands of isolated ellipsoids, which is why
  `check_cave_connectivity.py` exists; the distribution is printed here.
* **The `replaceable` tag.** No carved cell may replace a block outside the
  carver's tag. Asserted by planting blocks that are *not* in the tag and
  checking they survive, and by diffing before/after on real terrain.
* **Fluids.** Lava below `lava_level`, the nether's lava below sea level, and
  the overworld's documented gap (no aquifer, so no water).

Everything is measured and printed where a number is available, because this
repo prefers a measurement to an assertion.

Run: python tools/check_carvers.py
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
from loguru import logger  # noqa: E402

logger.remove()
logger.add(sys.stderr, level="ERROR")

from world.vanilla import carver as carver  # noqa: E402

SEED = 20261004
MIN_Y, HEIGHT = -64, 384
CHUNK = (16, 16)

failures: List[str] = []


def check(label: str, condition: bool, detail: str = "") -> bool:
    marker = "OK  " if condition else "FAIL"
    print(f"  {marker} {label}" + (f"  ({detail})" if detail else ""))
    if not condition:
        failures.append(label)
    return condition


def section(title: str) -> None:
    print(f"\n{title}")


# ==========================================================================
# Fixtures
# ==========================================================================

class World:
    """The real generator data a carve needs, built once."""

    def __init__(self, seed: int = SEED) -> None:
        from registry import init_registries

        self.seed = seed
        self.registry = init_registries(log_summary=False)
        self.tags = self.registry.tags
        self.carvers = carver.load_carvers()
        from world.vanilla.biome import MultiNoiseBiomeSource, biome_table

        self.biome_table = biome_table()
        self.biome_carvers = {
            biome_id: carver.carvers_for_biome(biome_id, self.biome_table)
            for biome_id in self.biome_table.ids.values()
        }
        self.sources = {}
        self.router = None
        from world.block_states import BLOCKS

        self.blocks = BLOCKS
        self.air = 0
        self.stone = BLOCKS.STONE
        self.bedrock = BLOCKS.BEDROCK
        self.lava = BLOCKS.LAVA
        self.water = BLOCKS.WATER
        self.obsidian = carver._block_id("minecraft:obsidian")

    def source(self, preset: str):
        if preset not in self.sources:
            from world.vanilla.biome import MultiNoiseBiomeSource

            self.sources[preset] = MultiNoiseBiomeSource.from_assets(preset, self.seed)
        return self.sources[preset]

    def router_for(self, preset: str):
        if self.router is None:
            from world.vanilla.router import NoiseRouter

            self.router = NoiseRouter.from_assets(preset, seed=self.seed)
        return self.router

    def terrain(self, chunk_x: int, chunk_z: int, preset: str = "overworld",
                min_y: int = MIN_Y, height: int = HEIGHT) -> np.ndarray:
        """A real chunk: the density function decides stone from air."""
        density = self.router_for(preset).fill_chunk(
            chunk_x, chunk_z, min_y, min_y + height, name="finalDensity"
        )
        blocks = np.where(density > 0, np.uint16(self.stone), np.uint16(self.air))
        blocks = blocks.astype(np.uint16)
        blocks[0, :, :] = self.bedrock
        return blocks

    def synthetic(self, min_y: int = MIN_Y, height: int = HEIGHT) -> np.ndarray:
        """A cheap solid chunk, for the checks that need volume but not terrain."""
        blocks = np.zeros((height, 16, 16), dtype=np.uint16)
        ys = np.arange(height) + min_y
        blocks[(ys < 60) & (ys >= min_y + 1)] = self.stone
        blocks[ys == min_y] = self.bedrock
        return blocks


def carve_chunk(world: World, chunk_x: int, chunk_z: int, blocks: np.ndarray,
                biomes: np.ndarray, *, preset: str = "overworld",
                exact_biomes: bool = False, **kwargs) -> None:
    carver.carve(
        chunk_x, chunk_z, blocks, biomes,
        preset=preset, seed=world.seed, min_y=MIN_Y, height=HEIGHT,
        biome_carvers=world.biome_carvers, carver_data=world.carvers,
        tags=world.tags,
        biome_at=world.source(preset).biome_id_at_quart if exact_biomes else None,
        **kwargs,
    )


def digest(blocks: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(blocks).tobytes()).hexdigest()


def blobs(volume: np.ndarray) -> Tuple[int, int]:
    """6-connected components of a boolean volume: count and largest size."""
    seen = np.zeros_like(volume)
    count = 0
    largest = 0
    for y, z, x in zip(*np.nonzero(volume)):
        if seen[y, z, x]:
            continue
        count += 1
        size = 0
        queue = deque([(y, z, x)])
        seen[y, z, x] = True
        while queue:
            cy, cz, cx = queue.popleft()
            size += 1
            for ny, nz, nx in ((cy + 1, cz, cx), (cy - 1, cz, cx),
                               (cy, cz + 1, cx), (cy, cz - 1, cx),
                               (cy, cz, cx + 1), (cy, cz, cx - 1)):
                if (0 <= ny < volume.shape[0] and 0 <= nz < volume.shape[1]
                        and 0 <= nx < volume.shape[2]
                        and volume[ny, nz, nx] and not seen[ny, nz, nx]):
                    seen[ny, nz, nx] = True
                    queue.append((ny, nz, nx))
        largest = max(largest, size)
    return count, largest


# ==========================================================================
# Layer 0: data
# ==========================================================================

def layer0_data(world: World) -> None:
    section("0. The carver data")
    check("four carvers load", len(world.carvers) == 4,
          ", ".join(sorted(world.carvers)))
    types = {config.type for config in world.carvers.values()}
    check("every carver type is implemented", types <= carver.SUPPORTED_TYPES,
          f"types {sorted(types)}, unsupported {sorted(types - carver.SUPPORTED_TYPES)}")

    with open(ROOT / "assets" / "biome.json", encoding="utf-8") as handle:
        raw = json.load(handle)
    named: Dict[str, set] = {}
    for name, entry in raw.items():
        value = entry.get("carvers") or ()
        names = (value,) if isinstance(value, str) else tuple(value)
        for item in names:
            named.setdefault(carver._canonical_id(item), set()).add(name)
    with open(ROOT / "assets" / "carver.json", encoding="utf-8") as handle:
        carver_payload = json.load(handle)
    unimplemented = []
    for name, biomes in sorted(named.items()):
        bare = name[len("minecraft:"):] if name.startswith("minecraft:") else name
        payload = carver_payload.get(bare)
        ctype = (payload or {}).get("type")
        state = "implemented" if ctype in carver.SUPPORTED_TYPES else "UNIMPLEMENTED"
        print(f"       {name:28s} {str(ctype):22s} {len(biomes):3d} biomes  {state}")
        if ctype not in carver.SUPPORTED_TYPES:
            unimplemented.append((name, ctype, len(biomes)))
    check("every biome carver resolves to an implemented type",
          not unimplemented,
          f"{len(named)} carver name(s) across {len(raw)} biomes, "
          f"{len(unimplemented)} unimplemented")

    # An unknown `type` and an unknown carver name are the two data problems
    # this stage can hit, and both must fail loudly rather than carve nothing.
    try:
        carver._build_config("mystery", {"type": "minecraft:tunnel", "config": {}})
        raised = None
    except carver.UnsupportedCarverError as error:
        raised = str(error)
    check("an unsupported carver type raises", raised is not None, raised or "no error")

    empty = world.synthetic()
    caught = None
    try:
        carver.carve(
            0, 0, empty, world.source("overworld").fill_volume(0, 0, MIN_Y, HEIGHT),
            preset="overworld", seed=world.seed, min_y=MIN_Y, height=HEIGHT,
            biome_carvers=world.biome_carvers, carver_data={}, tags=world.tags,
        )
    except KeyError as error:
        caught = str(error)
    check("a carver missing from carver_data raises", caught is not None, caught or "no error")

    lists = {names for names in world.biome_carvers.values()}
    check("biome carver lists are real ids, not characters",
          all(all(len(item) > 2 for item in names) for names in lists),
          f"distinct lists: {sorted(lists)}")
    nether = [world.biome_table.id_of(name) for name in (
        "minecraft:nether_wastes", "minecraft:basalt_deltas", "minecraft:crimson_forest",
        "minecraft:warped_forest", "minecraft:soul_sand_valley")]
    check("the nether's string-form carver list is repaired",
          all(world.biome_carvers[bid] == ("minecraft:nether_cave",) for bid in nether),
          str(world.biome_carvers[nether[0]]))


# ==========================================================================
# Layer 1: the RNG contract
# ==========================================================================

#: `Mth.getSeed` vectors, from a real Minecraft via Pumpkin's
#: `block_position_hash` test. Nothing in the carver calls `split_pos`, but the
#: hash is shared worldgen plumbing and must not regress under this stage.
HASH_VECTORS = [
    ((0, 0, 0), 0),
    ((1, 1, 1), 60311958971344),
    ((4, 4, 4), 120566413180880),
    ((25, 25, 25), 111753446486209),
    ((676, 676, 676), 75210837988243),
    ((458329, 458329, 458329), -43764888250),
    ((-387008604, -387008604, -387008604), 8437923733503),
    ((176771161, 176771161, 176771161), 18421337580760),
]


def layer1_rng(world: World) -> None:
    section("1. The random sources this stage draws from")
    from world.vanilla.random_source import LegacyRandom, WorldgenRandom, Xoroshiro, _hash_block_pos

    bad = []
    for (x, y, z), want in HASH_VECTORS:
        got = _hash_block_pos(x, y, z)
        signed = got - (1 << 64) if got >= (1 << 63) else got
        if signed != want:
            bad.append(((x, y, z), want, signed))
    check("Mth.getSeed vectors", not bad, f"{len(HASH_VECTORS)} positions"
          + (f"; wrong: {bad[:2]}" if bad else ""))

    # Names: the preset's `legacy_random_source` picks the generator class.
    flags = {preset: carver.preset_settings(preset)["legacy_random_source"]
             for preset in ("overworld", "nether")}
    check("preset random-source flags", flags == {"overworld": False, "nether": True},
          str(flags))

    # The compiled walk re-implements the draw methods; every draw it can make
    # is compared against the Python generator it was seeded from.
    kinds = (
        ("legacy (nether)", carver._RNG_LEGACY, lambda: LegacyRandom(0xC0FFEE)),
        ("worldgen (overworld)", carver._RNG_WORLDGEN,
         lambda: WorldgenRandom(Xoroshiro.from_seed(0xC0FFEE))),
        ("raw xoroshiro", carver._RNG_RAW, lambda: Xoroshiro.from_seed(0xC0FFEE)),
    )
    for label, kind, factory in kinds:
        python_side = factory()
        state = carver._rng_state_from(python_side)
        check(f"{label} state kind", int(state[0]) == kind, f"kind={int(state[0])}")
        python_side = factory()
        mismatches = 0
        first = None
        for i in range(2000):
            which = i % 4
            if which == 0:
                want = np.float32(python_side.next_f32())
                got = carver._rng_next_f32(state)
                same = np.float32(want) == np.float32(got)
            elif which == 1:
                want = python_side.next_bounded_i32(31)
                got = int(carver._rng_next_bounded_i32(state, 31))
                same = want == got
            elif which == 2:
                want = python_side.next_bounded_i32(64)
                got = int(carver._rng_next_bounded_i32(state, 64))
                same = want == got
            else:
                want = python_side.next_i64()
                got = int(carver._rng_next_i64(state))
                same = want == got
            if not same:
                mismatches += 1
                if first is None:
                    first = (i, want, got)
        check(f"{label} draws match the Python generator", mismatches == 0,
              f"2000 mixed draws, {mismatches} mismatches"
              + (f", first {first}" if first else ""))

    # The seed derivation itself: 289 carver positions, no collisions, stable.
    seeds = {carver.get_large_feature_seed(SEED + index, cx, cz)
             for index in range(3) for cx in range(-3, 4) for cz in range(-3, 4)}
    check("large feature seeds are distinct per position", len(seeds) == 3 * 49,
          f"{len(seeds)} distinct of {3 * 49}")


# ==========================================================================
# Layer 2: the mask
# ==========================================================================

def layer2_mask() -> None:
    section("2. The carving mask")
    mask = carver.CarvingMask(-64, 320)
    check("a fresh mask is empty", mask.is_empty() and list(mask.visit()) == [])

    mask.set(0, 10, 0)
    mask.set(0, 11, 0)
    mask.set(0, 12, 0)
    mask.set(1, 20, 2)
    visits = list(mask.visit())
    check("reference visit() semantics", visits == [(0, 0, 10, 12), (1, 2, 20, 20)],
          str(visits))

    bounds = carver.CarvingMask(-64, 320)
    bounds.set(5, -65, 5)
    bounds.set(5, 321, 5)
    bounds.set(16, 0, 5)
    bounds.set(-1, 0, 5)
    check("out-of-range writes are ignored", bounds.is_empty())

    single = carver.CarvingMask(-64, 320)
    single.set(3, -63, 7)
    check("lowest and highest cells round-trip",
          single.get(3, -63, 7) and not single.get(3, -64, 7)
          and list(single.visit()) == [(3, 7, -63, -63)],
          str(list(single.visit())))

    # The mask the stage actually builds: `min_y + 1` to `min_y + height - 8`.
    expected_min = MIN_Y + 1
    expected_max = MIN_Y + HEIGHT - 1 - carver.PROTECTED_BLOCKS_ON_TOP
    stage = carver.CarvingMask(MIN_Y + 1, MIN_Y + HEIGHT - 1 - carver.PROTECTED_BLOCKS_ON_TOP)
    check("stage mask bounds", (stage.min_y, stage.max_y) == (expected_min, expected_max),
          f"{stage.min_y}..{stage.max_y}")

    # Two different radii, and a relationship between them: a tunnel may not
    # outrun the area whose carver chunks are enumerated, or its far end is
    # never generated and a chunk border gets half a cave.
    check("the carver's reach and the loop radius are distinct",
          (carver.CARVER_RANGE, carver.RANGE) == (4, 8)
          and carver.MAX_DISTANCE == 112,
          f"CARVER_RANGE={carver.CARVER_RANGE}, RANGE={carver.RANGE}, "
          f"MAX_DISTANCE={carver.MAX_DISTANCE}")
    check("a tunnel cannot outrun the 17x17 loop",
          carver.MAX_DISTANCE <= carver.RANGE * 16,
          f"{carver.MAX_DISTANCE} <= {carver.RANGE * 16}")


# ==========================================================================
# Layer 3: the compiled kernels
# ==========================================================================

def layer3_kernels() -> None:
    section("3. Compiled kernels vs the interpreted reference")
    if not carver._HAS_NUMBA:
        check("numba is installed", False, "no compiled path to compare")
        return
    rng = np.random.default_rng(7)
    worst = 0
    for trial in range(60):
        mask_py = carver.CarvingMask(MIN_Y + 1, MIN_Y + HEIGHT - 8)
        mask_k = carver.CarvingMask(MIN_Y + 1, MIN_Y + HEIGHT - 8)
        chunk_x, chunk_z = int(rng.integers(-4, 4)), int(rng.integers(-4, 4))
        centre_x = (chunk_x << 4) + float(rng.uniform(-24, 24))
        centre_z = (chunk_z << 4) + float(rng.uniform(-24, 24))
        centre_y = float(rng.uniform(MIN_Y, 120))
        h_radius = float(rng.uniform(0.5, 12.0))
        v_radius = float(rng.uniform(0.5, 12.0))
        kind = int(rng.integers(0, 2))
        floor_level = float(rng.uniform(-1.0, 0.0))
        factors = np.zeros(HEIGHT, dtype=np.float32)
        if kind == carver.SKIP_CANYON:
            factors[:] = rng.uniform(0.2, 4.0, size=HEIGHT).astype(np.float32)
        carver._USE_NUMBA = False
        carver.carve_ellipsoid(mask_py, chunk_x, chunk_z, centre_x, centre_y, centre_z,
                               h_radius, v_radius, kind, floor_level, factors, MIN_Y)
        carver._USE_NUMBA = True
        carver.carve_ellipsoid(mask_k, chunk_x, chunk_z, centre_x, centre_y, centre_z,
                               h_radius, v_radius, kind, floor_level, factors, MIN_Y)
        worst = max(worst, int((mask_py.words != mask_k.words).sum()))
    check("ellipsoids agree bit for bit", worst == 0, f"60 random ellipsoids, worst {worst} words")

    # One tunnel per (generator, seed, thickness): branching included.
    diverged = []
    for legacy in (False, True):
        for seed in range(8):
            for thickness in (0.5, 2.0):
                args = (0, 0, seed, 0.0, 30.0, 0.0, 1.0, 1.0, np.float32(thickness),
                        np.float32(0.3), np.float32(0.1), 0, 112, 1.0, -0.7,
                        legacy, "worldgen", MIN_Y, HEIGHT)
                mask_py = carver.CarvingMask(MIN_Y + 1, MIN_Y + HEIGHT - 8)
                mask_k = carver.CarvingMask(MIN_Y + 1, MIN_Y + HEIGHT - 8)
                carver._USE_NUMBA = False
                carver._create_tunnel(mask_py, *args)
                carver._USE_NUMBA = True
                carver._create_tunnel(mask_k, *args)
                if not np.array_equal(mask_py.words, mask_k.words):
                    diverged.append((legacy, seed, thickness))
    check("tunnels agree bit for bit", not diverged,
          f"32 tunnels (branching included), {len(diverged)} diverged"
          + (f": {diverged[:3]}" if diverged else ""))


# ==========================================================================
# Layer 4: determinism and order independence
# ==========================================================================

def layer4_determinism(world: World) -> Tuple[np.ndarray, np.ndarray]:
    section("4. Determinism and order independence")
    source = world.source("overworld")
    biomes = source.fill_volume(0, 0, MIN_Y, HEIGHT)
    first = world.synthetic()
    second = world.synthetic()
    carve_chunk(world, 0, 0, first, biomes)
    carve_chunk(world, 0, 0, second, biomes)
    check("the same carve twice is bit-identical", digest(first) == digest(second),
          digest(first)[:16])

    neighbour = world.synthetic()
    carve_chunk(world, 1, 0, neighbour, source.fill_volume(1, 0, MIN_Y, HEIGHT))
    after = world.synthetic()
    carve_chunk(world, 0, 0, after, biomes)
    check("carving (1,0) first does not change (0,0)", digest(after) == digest(first),
          digest(after)[:16])

    flipped = world.synthetic()
    carve_chunk(world, -1, -1, flipped, source.fill_volume(-1, -1, MIN_Y, HEIGHT))
    carve_chunk(world, 0, -2, flipped, source.fill_volume(0, -2, MIN_Y, HEIGHT))
    order = world.synthetic()
    carve_chunk(world, 0, 0, order, biomes)
    check("a shuffled neighbourhood order does not change (0,0)",
          digest(order) == digest(first), digest(order)[:16])

    # `neighbour_blocks` must not be read: an object that raises on any use.
    class Trap:
        def __getattr__(self, name):
            raise AssertionError(f"carve() read neighbour_blocks.{name}")

        def __getitem__(self, key):
            raise AssertionError(f"carve() read neighbour_blocks[{key!r}]")

    trapped = world.synthetic()
    carve_chunk(world, 0, 0, trapped, biomes, neighbour_blocks=Trap())
    check("neighbour_blocks is never read (None == trap)", digest(trapped) == digest(first))

    # The 17x17 neighbourhood has to be doing something: with the carver-chunk
    # radius forced to zero only the tunnels this chunk starts itself survive,
    # so the difference is what neighbouring carver chunks contribute.
    radius = carver.RANGE
    probe = list(range(8))
    full_total = 0
    alone_total = 0
    try:
        carver.RANGE = 0
        for index in probe:
            alone = world.synthetic()
            carve_chunk(world, index, 0, alone, source.fill_volume(index, 0, MIN_Y, HEIGHT))
            alone_total += int((alone != world.synthetic()).sum())
    finally:
        carver.RANGE = radius
    for index in probe:
        full = world.synthetic()
        carve_chunk(world, index, 0, full, source.fill_volume(index, 0, MIN_Y, HEIGHT))
        full_total += int((full != world.synthetic()).sum())
    check("neighbouring carver chunks reach into this one",
          alone_total > 0 and full_total > 3 * alone_total,
          f"{len(probe)} chunks: {full_total} cells from the 17x17 area, "
          f"{alone_total} from the chunk's own carver chunk "
          f"({100 * (1 - alone_total / max(1, full_total)):.0f}% neighbour-owned)")

    # Data drives it: with no carver lists, nothing is carved at all.
    empty = world.synthetic()
    carver.carve(
        0, 0, empty, biomes, preset="overworld", seed=world.seed, min_y=MIN_Y, height=HEIGHT,
        biome_carvers={}, carver_data=world.carvers, tags=world.tags,
    )
    check("a biome with no carvers carves nothing",
          int((empty != world.synthetic()).sum()) == 0)

    # And the seed drives it.
    shifted = world.synthetic()
    carver.carve(
        0, 0, shifted, biomes, preset="overworld", seed=world.seed + 1, min_y=MIN_Y,
        height=HEIGHT, biome_carvers=world.biome_carvers, carver_data=world.carvers,
        tags=world.tags,
    )
    check("a different seed carves differently", digest(shifted) != digest(first))

    # `carver_data` may arrive as `load_carvers()` or as the game registry's
    # `CarverData` objects; both must carve the same world.
    worldgen = world.registry.worldgen
    if worldgen is not None:
        from_registry = {
            name: worldgen.carver(name) for name in world.carvers
        }
        registry_carve = world.synthetic()
        carver.carve(
            0, 0, registry_carve, biomes, preset="overworld", seed=world.seed, min_y=MIN_Y,
            height=HEIGHT, biome_carvers=world.biome_carvers, carver_data=from_registry,
            tags=world.tags,
        )
        check("registry CarverData carves the same as load_carvers()",
              digest(registry_carve) == digest(first))

    # Structural bound: nothing above the protected ceiling, nothing below it.
    ceiling = MIN_Y + HEIGHT - 1 - carver.PROTECTED_BLOCKS_ON_TOP
    rows = np.nonzero((first != world.synthetic()).any(axis=(1, 2)))[0] + MIN_Y
    check("the carve stays inside the mask bounds",
          rows.size > 0 and int(rows.min()) > MIN_Y and int(rows.max()) <= ceiling,
          f"carved y {int(rows.min())}..{int(rows.max())}, mask {MIN_Y + 1}..{ceiling}")
    return first, biomes


def layer4b_fresh_process(world: World, expected: str) -> None:
    """The same carve in a brand-new interpreter, twice.

    A module-level cache that leaks RNG state or biome state shows up here and
    nowhere else.
    """
    hashes = []
    for _ in range(2):
        result = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--emit-hash"],
            cwd=str(ROOT), capture_output=True, text=True, timeout=600,
        )
        line = [row for row in result.stdout.splitlines() if row.startswith("HASH ")]
        if result.returncode != 0 or not line:
            check("fresh-process carve", False,
                  f"exit {result.returncode}: {(result.stderr or '').strip()[-200:]}")
            return
        hashes.append(line[0].split()[1])
    check("fresh-process carve matches this process",
          hashes[0] == hashes[1] == expected,
          f"{hashes[0][:16]} / {hashes[1][:16]} / {expected[:16]}")


def emit_hash() -> int:
    """`--emit-hash`: one deterministic carve, one line, no checks.

    Must carve exactly what `layer4_determinism` carves in-process, or the
    comparison is meaningless -- hence the same terrain, the same biome array
    and the same (fallback) biome provider.
    """
    world = World()
    source = world.source("overworld")
    blocks = world.synthetic()
    carve_chunk(world, 0, 0, blocks, source.fill_volume(0, 0, MIN_Y, HEIGHT))
    print(f"HASH {digest(blocks)}")
    return 0


# ==========================================================================
# Layer 5: a real carve
# ==========================================================================

def layer5_real_carve(world: World, chunks: List[Tuple[int, int]]) -> Dict[Tuple[int, int], dict]:
    section("5. What the carver does to real terrain")
    source = world.source("overworld")
    router = world.router_for("overworld")
    replaceable = carver.resolve_replaceable(
        world.tags, [config.replaceable for config in world.carvers.values()]
    )
    stats = {}
    start = time.perf_counter()
    for chunk_x, chunk_z in chunks:
        blocks = world.terrain(chunk_x, chunk_z)
        before = blocks.copy()
        carve_chunk(world, chunk_x, chunk_z, blocks,
                    source.fill_volume(chunk_x, chunk_z, MIN_Y, HEIGHT), exact_biomes=True)
        changed = np.nonzero(before != blocks)
        solid_before = before[before != world.air]
        carved = int((before != blocks).sum())
        stats[(chunk_x, chunk_z)] = {
            "carved": carved,
            "solid": int(solid_before.size),
            "fraction": carved / max(1, solid_before.size),
            "changed": changed,
            "before": before,
            "after": blocks,
        }
    elapsed = (time.perf_counter() - start) * 1000 / len(chunks)
    fractions = [row["fraction"] for row in stats.values()]
    carved_counts = [row["carved"] for row in stats.values()]
    print(f"       {len(chunks)} real chunks, {elapsed:.0f} ms/chunk "
          f"(density + biomes + carve, exact biome lookups)")
    print(f"       solid cells -> carved: mean {100 * np.mean(fractions):.2f}%  "
          f"median {100 * np.median(fractions):.2f}%  "
          f"min {100 * min(fractions):.2f}%  max {100 * max(fractions):.2f}%")
    print(f"       carved cells per chunk: mean {np.mean(carved_counts):.0f}  "
          f"min {min(carved_counts)}  max {max(carved_counts)}")
    check("the carver does something", min(carved_counts) > 0, f"min {min(carved_counts)} cells")
    check("the carver does not eat the world", max(fractions) < 0.40,
          f"max {100 * max(fractions):.2f}% of solid cells")

    # The replaceable tag, cell by cell: every write must land on a block the
    # carver is allowed to replace.
    outside = 0
    example = None
    allowed = list(replaceable)
    for row in stats.values():
        before, after = row["before"], row["after"]
        originals = before[before != after]
        bad = np.isin(originals, allowed, invert=True)
        outside += int(bad.sum())
        if bad.any() and example is None:
            example = int(originals[bad][0])
    check("no cell outside the replaceable tag was carved", outside == 0,
          f"tag blocks={len(replaceable)}, violations={outside}"
          + (f", first state {example}" if example is not None else ""))

    # A planted block that is *not* in the tag must survive, inside a cave.
    blocks = world.terrain(*chunks[0])
    before = blocks.copy()
    plant_mask = np.zeros_like(blocks, dtype=bool)
    # Somewhere under the surface, in a band the carver reaches.
    plant_mask[(np.arange(HEIGHT) + MIN_Y >= -40) & (np.arange(HEIGHT) + MIN_Y < 40)] = True
    blocks[plant_mask] = world.obsidian
    carve_chunk(world, chunks[0][0], chunks[0][1], blocks,
                world.source("overworld").fill_volume(chunks[0][0], chunks[0][1], MIN_Y, HEIGHT),
                exact_biomes=True)
    survived = int((blocks[plant_mask] == world.obsidian).sum())
    planted = int(plant_mask.sum())
    check("tagged-out blocks survive a carve", survived == planted,
          f"{survived}/{planted} obsidian cells untouched")

    # Fluids. `lava_level` is `min_y + 8` for the overworld carvers, and every
    # carved cell at or below it must be lava -- vanilla tests it before the
    # aquifer, so it wins even when an aquifer is supplied.
    lava_level = MIN_Y + 8
    ys = (np.arange(HEIGHT) + MIN_Y)[:, None, None]
    wrong = 0
    lava_total = 0
    for row in stats.values():
        before, after = row["before"], row["after"]
        changed = before != after
        below = changed & (ys <= lava_level)
        lava_total += int((after == world.lava).sum())
        if below.any():
            wrong += int((after[below] != world.lava).sum())
    check("every carved cell at or below lava_level is lava", wrong == 0,
          f"lava_level={lava_level}, {lava_total} lava cells, {wrong} wrong")
    check("lava appears at the bottom", lava_total > 0, f"{lava_total} cells")
    water_total = sum(int((row["after"] == world.water).sum()) for row in stats.values())
    print(f"       water placed: {water_total} cells "
          f"(expected 0: no aquifer is implemented -- see the module docstring)")

    # A stub aquifer must be honoured, including its "do not carve" answer.
    origin = world.terrain(*chunks[0])
    biome_volume = world.source("overworld").fill_volume(
        chunks[0][0], chunks[0][1], MIN_Y, HEIGHT
    )
    stub_blocks = origin.copy()
    stub_calls = [0]

    def aquifer(x: int, y: int, z: int) -> Optional[int]:
        stub_calls[0] += 1
        return world.water

    carve_chunk(world, chunks[0][0], chunks[0][1], stub_blocks, biome_volume,
                exact_biomes=True, aquifer=aquifer)
    stub_water = int((stub_blocks == world.water).sum())
    check("a supplied aquifer is asked for every carved cell below the lava level",
          stub_calls[0] > 0 and stub_water > 0,
          f"{stub_calls[0]} aquifer queries, {stub_water} water cells")

    blocked = origin.copy()
    blocker_calls = [0]

    def refusing(x: int, y: int, z: int) -> Optional[int]:
        blocker_calls[0] += 1
        return None

    carve_chunk(world, chunks[0][0], chunks[0][1], blocked, biome_volume,
                exact_biomes=True, aquifer=refusing)
    carved_stub = int((stub_blocks != origin).sum())
    carved_refused = int((blocked != origin).sum())
    check("an aquifer returning None refuses the cell",
          blocker_calls[0] > 0 and carved_refused < carved_stub
          and int((blocked == world.water).sum()) == 0,
          f"{blocker_calls[0]} refusals, {carved_refused} cells carved vs "
          f"{carved_stub} with water")

    # The nether: aquifers off, so `default_fluid` (lava) fills below sea level.
    nether_lava = 0
    nether_carved = 0
    nether_wrong = 0
    nether_rows = []
    for chunk_x in range(16):
        nether_blocks = np.zeros((128, 16, 16), dtype=np.uint16)
        rows = np.arange(128)
        nether_blocks[(rows >= 1) & (rows < 120)] = world.blocks.NETHERRACK
        nether_blocks[0] = world.bedrock
        before_nether = nether_blocks.copy()
        carver.carve(
            chunk_x, 0, nether_blocks, world.source("nether").fill_volume(chunk_x, 0, 0, 128),
            preset="nether", seed=world.seed, min_y=0, height=128,
            biome_carvers=world.biome_carvers, carver_data=world.carvers, tags=world.tags,
        )
        changed = before_nether != nether_blocks
        below = changed & (rows[:, None, None] < 32)
        nether_carved += int(changed.sum())
        nether_lava += int((nether_blocks == world.lava).sum())
        if below.any():
            nether_rows.append(int(below.sum()))
            nether_wrong += int((nether_blocks[below] != world.lava).sum())
    check("the nether's carved cells below sea level are lava",
          sum(nether_rows) > 100 and nether_wrong == 0,
          f"16 chunks: {nether_carved} carved, {nether_lava} lava, "
          f"{sum(nether_rows)} below sea level, {nether_wrong} wrong")
    return stats


# ==========================================================================
# Layer 6: connectivity
# ==========================================================================

def layer6_connectivity(world: World, stats: Dict[Tuple[int, int], dict]) -> None:
    section("6. Are the carved cells connected?")
    components = []
    shares = []
    for (chunk_x, chunk_z), row in sorted(stats.items()):
        before, after = row["before"], row["after"]
        # Carved air only: cells that were solid and are now air. Pre-existing
        # air is the density function's business, not the carver's.
        carved_air = (before != world.air) & (after == world.air)
        if not carved_air.any():
            continue
        count, largest = blobs(carved_air)
        components.append(count)
        shares.append(largest / int(carved_air.sum()))
    if not components:
        check("there is carved air to measure", False)
        return
    print(f"       components per chunk: mean {np.mean(components):.1f}  "
          f"median {np.median(components):.0f}  max {max(components)}")
    print(f"       largest component share: mean {100 * np.mean(shares):.1f}%  "
          f"min {100 * min(shares):.1f}%  max {100 * max(shares):.1f}%")
    check("no scatter of tiny blobs", max(components) < 120,
          f"worst chunk has {max(components)} components")
    check("most carved air is in one piece of each chunk", np.median(shares) > 0.25,
          f"median share {100 * np.median(shares):.1f}%")


# ==========================================================================
# Layer 7: performance
# ==========================================================================

def layer7_performance(world: World) -> None:
    section("7. Cost")
    source = world.source("overworld")
    biomes = [source.fill_volume(cx, 0, MIN_Y, HEIGHT) for cx in range(6)]

    def timed(exact: bool, use_numba: bool) -> float:
        carver._USE_NUMBA = use_numba
        pool = [world.synthetic() for _ in range(6)]
        start = time.perf_counter()
        for index in range(6):
            carve_chunk(world, index, 0, pool[index], biomes[index], exact_biomes=exact)
        elapsed = (time.perf_counter() - start) * 1000 / 6
        carver._USE_NUMBA = True
        return elapsed

    timed(False, True)  # warm the JIT
    compiled = timed(False, True)
    interpreted = timed(False, False)
    exact = timed(True, True)
    print(f"       carve, compiled walk:      {compiled:7.1f} ms/chunk")
    print(f"       carve, interpreted walk:   {interpreted:7.1f} ms/chunk")
    print(f"       carve, + exact biome_at:   {exact:7.1f} ms/chunk "
          f"(the 289 neighbour lookups dominate)")
    check("the compiled walk is at least 4x the interpreted one",
          compiled * 4 < interpreted, f"{compiled:.1f} vs {interpreted:.1f} ms")
    check("a chunk carves in reasonable time", compiled < 150, f"{compiled:.1f} ms")


def layer8_open_questions(world: World) -> None:
    """Measure the two documented guesses, so they are visible rather than hidden.

    `random_source` selects how the carver generator is wrapped, and the two
    answers are not the same generator. This prints how much of the carve
    depends on the choice, which is the honest way to report an unverifiable
    decision.
    """
    section("8. The two open questions, measured")
    source = world.source("overworld")
    biomes = source.fill_volume(0, 0, MIN_Y, HEIGHT)
    worldgen = world.synthetic()
    carve_chunk(world, 0, 0, worldgen, biomes)
    raw = world.synthetic()
    carve_chunk(world, 0, 0, raw, biomes, random_source="raw")
    differing = int((worldgen != raw).sum())
    worldgen_cells = int((worldgen != world.synthetic()).sum())
    raw_cells = int((raw != world.synthetic()).sum())
    print(f"       WorldgenRandom wrapper: {worldgen_cells} cells carved")
    print(f"       bare Xoroshiro (Pumpkin): {raw_cells} cells carved")
    print(f"       cells that differ between the two: {differing} "
          f"({100 * differing / max(1, max(worldgen_cells, raw_cells)):.0f}% of the carve)")
    check("the random-source choice does not silently do nothing",
          differing > 0, f"{differing} cells differ")

    # `MAX_DISTANCE` rests on an inferred `getRange()` of 4; the other reading
    # (8, i.e. 240-block tunnels) is one no-vanilla-dump decision away.
    length = carver.MAX_DISTANCE
    long_length = (8 * 2 - 1) << 4
    try:
        carver.MAX_DISTANCE = long_length
        long_carve = world.synthetic()
        carve_chunk(world, 0, 0, long_carve, biomes)
    finally:
        carver.MAX_DISTANCE = length
    long_cells = int((long_carve != world.synthetic()).sum())
    print(f"       MAX_DISTANCE {length}: {worldgen_cells} cells; "
          f"if getRange() were 8 ({long_length}): {long_cells} cells "
          f"({100 * long_cells / max(1, worldgen_cells):.0f}%)")

    # The cave count is Pumpkin's `very_biased_to_bottom`; print what it
    # produces so a wrong count is at least visible as a density.
    count = world.carvers["cave"].count
    from world.vanilla.random_source import WorldgenRandom, Xoroshiro

    draws = [count.get(WorldgenRandom(Xoroshiro.from_seed(index))) for index in range(4000)]
    print(f"       cave systems per start chunk: mean {np.mean(draws):.2f}, "
          f"0 in {100 * sum(1 for d in draws if d == 0) / len(draws):.0f}%, "
          f"max {max(draws)}")
    check("the cave count is in range", 0 <= min(draws) and max(draws) <= 14,
          f"min {min(draws)}, max {max(draws)}")

    # How much does that unverifiable choice matter? `count` is not in this
    # repo's `carver.json` at all, and the readings of vanilla's
    # `very_biased_to_bottom` differ in cave density by a factor of a few.
    # Averaged over several chunks: one chunk is dominated by whichever
    # neighbouring carver chunks happened to start a tunnel that crosses it.
    variants = (
        ("pumpkin inner=1 (in use)", carver._VeryBiasedToBottomInt(0, 14, 1)),
        ("one bias step more", carver._VeryBiasedToBottomInt(0, 14, 2)),
        ("uniform 0..14", carver._UniformInt(0, 14)),
        ("constant 1", carver._ConstantInt(1)),
    )
    probe_chunks = [0, 1, 2, 3, 4, 5]
    counts: Dict[str, int] = {}
    for label, provider in variants:
        import dataclasses

        swap = dict(world.carvers)
        swap["cave"] = dataclasses.replace(world.carvers["cave"], count=provider)
        total = 0
        for chunk_x in probe_chunks:
            probe = world.synthetic()
            carver.carve(
                chunk_x, 0, probe, source.fill_volume(chunk_x, 0, MIN_Y, HEIGHT),
                preset="overworld", seed=world.seed, min_y=MIN_Y, height=HEIGHT,
                biome_carvers=world.biome_carvers, carver_data=swap, tags=world.tags,
            )
            total += int((probe != world.synthetic()).sum())
        counts[label] = total
    baseline = counts["pumpkin inner=1 (in use)"]
    for label, _ in variants:
        print(f"       count={label:26s} -> {counts[label]:6d} cells over "
              f"{len(probe_chunks)} chunks ({100 * counts[label] / max(1, baseline):.0f}% "
              f"of the {baseline} in use)")


# ==========================================================================

def main() -> int:
    if "--emit-hash" in sys.argv:
        return emit_hash()

    world = World()
    print(f"carver acceptance: seed {SEED}, overworld shape {MIN_Y}..{MIN_Y + HEIGHT - 1}")
    print(f"numba {getattr(carver, '_HAS_NUMBA', False)}, "
          f"SUPPORTED_TYPES={sorted(carver.SUPPORTED_TYPES)}")

    layer0_data(world)
    layer1_rng(world)
    layer2_mask()
    layer3_kernels()
    baseline, _ = layer4_determinism(world)
    layer4b_fresh_process(world, digest(baseline))
    stats = layer5_real_carve(world, [(0, 0), (1, 0), (2, 0), (0, 1), (1, 1), (-1, -1)])
    layer6_connectivity(world, stats)
    layer7_performance(world)
    layer8_open_questions(world)

    print()
    if failures:
        print(f"RESULT: FAIL ({len(failures)} check(s))")
        for label in failures:
            print(f"  - {label}")
        return 1
    print("RESULT: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
