"""Saving one chunk leaves the rest of its region file alone.

The regression this covers: `RegionFile.save` used to pop the staged chunks,
read every *other* chunk in the region into a local dict (`existing_data`),
reopen the file with `open(path, "wb")` -- truncating it -- and write back only
the chunks staged for that one save. The collected bytes were never used. Saving
one chunk therefore destroyed every other chunk in the region: the header still
pointed at the old offsets, which by then held a different chunk's bytes or
nothing at all. It was also O(region) per save, which is what made a
save-on-unload streaming saver (`world/chunk_saver.py`) impossible.

So every check here re-reads through a **fresh** `RegionManager`: a fresh header
read and no in-memory state. An in-memory header would happily agree with a file
it no longer matches, which is exactly how the old bug stayed invisible. The
coordinates are negative on purpose -- in region (-1, -1) the world coordinate
and the region-local slot never coincide, so a save that files a chunk under the
wrong index is visible too.

The one threading check shares a single `RegionManager`, because that is the
supported contract: one manager owns a region file's header, and `RegionFile`
serializes reads and writes of it under its own lock. A *second* manager on the
same directory has its own copy of the header and is only valid once the writer
is quiescent -- which is what checks 1 and 4 rely on.

Run: python tools/check_region_streaming_save.py

Exit code 0 when everything passes, 1 otherwise.
"""

from __future__ import annotations

import shutil
import struct
import sys
import tempfile
import threading
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
from loguru import logger  # noqa: E402

logger.remove()

from world import block_states  # noqa: E402
from world.anvil import (  # noqa: E402
    FIRST_DATA_SECTOR,
    LOCATION_TABLE_BYTES,
    RegionFile,
    RegionManager,
    SECTOR_BYTES,
)
from world.chunk import Chunk, ChunkPos  # noqa: E402

failures: List[str] = []

#: Distinct blocks to write with, filled in `main` once the registry is up.
#: Every chunk gets its own pair of them, so a mix-up between chunks is content,
#: not just a coordinate, and cannot pass unnoticed.
IDS: List[int] = []

BLOCK_NAMES = [
    "minecraft:stone",
    "minecraft:dirt",
    "minecraft:cobblestone",
    "minecraft:oak_planks",
    "minecraft:sand",
    "minecraft:gravel",
    "minecraft:glass",
    "minecraft:sandstone",
    "minecraft:deepslate",
    "minecraft:netherrack",
    "minecraft:end_stone",
    "minecraft:obsidian",
    "minecraft:coal_ore",
    "minecraft:iron_ore",
    "minecraft:gold_ore",
    "minecraft:bedrock",
]

#: Anything before this is not a timestamp a save in this run could have written.
MIN_EPOCH = 1_600_000_000

#: Chunks of region (-1, -1), which covers world chunks -32..-1 on both axes.
#: Ten of its 1024 slots, with local and world coordinates differing everywhere.
REGION_COORDS = [
    (-1, -1), (-1, -5), (-5, -1), (-32, -32), (-32, -1),
    (-1, -32), (-16, -16), (-2, -31), (-31, -2), (-7, -20),
]


def check(label: str, condition: bool, detail: str = "") -> None:
    marker = "OK  " if condition else "FAIL"
    print(f"  {marker} {label}" + (f"  ({detail})" if detail else ""))
    if not condition:
        failures.append(label)


def local_index(chunk_x: int, chunk_z: int) -> int:
    """Header slot for a chunk: the same `% 32` / `// 32` the writer uses."""
    return (chunk_x % 32) + (chunk_z % 32) * 32


# ==================== fixtures ====================


def build_chunk(chunk_x: int, chunk_z: int, variant: int, sections: int = 2) -> Chunk:
    """A small chunk whose content is unique to `(coordinate, variant)`.

    `variant` picks the fill and stripe ids and the row the stripe sits in, so
    no two variants of one coordinate -- and no two chunks of one region -- share
    a fingerprint. `set_block` keeps `_non_air_count` up to date, which the
    round trip then has to reproduce.
    """
    chunk = Chunk(position=ChunkPos(chunk_x, chunk_z))  # -64..319, 24 sections
    stripe = IDS[(variant + 5) % len(IDS)]
    for index in range(sections):
        chunk.fill_section(index, IDS[variant % len(IDS)])
    row = variant % 16
    for column in range(16):
        chunk.set_block(column, chunk.min_y + 16 + row, column, stripe)
    return chunk


def build_large_chunk(
    chunk_x: int, chunk_z: int, variant: int, sections: Optional[int] = None
) -> Chunk:
    """A chunk with non-empty, many-entry sections: an order of magnitude more
    sectors than `build_chunk`.

    Used two ways: with every section filled, so a rewrite has to move the chunk
    to a new run (and a rewrite back has to fit in the old one); and with a few
    sections, so a payload lands in a run freed by another chunk. Assigning
    `blocks` wholesale bypasses the non-air count, so it is recomputed by hand --
    that bookkeeping is what `ChunkSection` documents.
    """
    chunk = Chunk(position=ChunkPos(chunk_x, chunk_z))
    rng = np.random.default_rng(1000 + variant)
    targets = chunk.sections if sections is None else chunk.sections[:sections]
    for section in targets:
        section.blocks[:] = rng.choice(IDS, size=section.blocks.size).astype(np.uint16)
        section.recompute_non_air_count()
    return chunk


def fingerprint(chunk: Chunk) -> Tuple:
    """Everything about a chunk a save has to preserve, in a comparable form.

    The position is part of it: a chunk that reads back with another chunk's
    bytes is one failure, a chunk filed under the wrong coordinate is another.
    Per section the block array itself is compared, not a summary of it, so two
    different patterns cannot collide.
    """
    return (
        (chunk.position.x, chunk.position.z),
        tuple((section.non_air_count, section.blocks.tobytes()) for section in chunk.sections),
    )


def digest(fp: Tuple) -> str:
    """A short handle for a fingerprint, for the printouts."""
    position, sections = fp
    non_empty = sum(1 for count, _ in sections if count)
    return (
        f"({position[0]},{position[1]}) {non_empty}/{len(sections)} non-empty, "
        f"section 0 starts {sections[0][1][:8].hex()}"
    )


def read_back(
    region_dir: Path, coords: List[Tuple[int, int]]
) -> Dict[Tuple[int, int], Optional[Chunk]]:
    """Load coordinates through a brand-new manager: fresh header, no cache."""
    manager = RegionManager(region_dir)
    return {coord: manager.load_chunk(coord[0], coord[1]) for coord in coords}


def location_of(region_dir: Path, region: Tuple[int, int], coord: Tuple[int, int]) -> Tuple[int, int]:
    """`(offset, size)` for a chunk, read out of a fresh header."""
    path = region_dir / f"r.{region[0]}.{region[1]}.mca"
    region_file = RegionFile(path)
    region_file.load_header()
    location = region_file.header.locations[local_index(coord[0], coord[1])]
    return location.offset, location.size


# ==================== checks ====================


def check_cross_chunk_preservation(region_dir: Path) -> None:
    print("1. Saving one chunk does not disturb the others (the regression)")
    regions = {coord[0] // 32 for coord in REGION_COORDS} | {coord[1] // 32 for coord in REGION_COORDS}
    check("the fixture is one region", regions == {-1}, str(sorted(regions)))
    check(
        "world and local coordinates differ everywhere",
        all((coord[0] % 32, coord[1] % 32) != coord for coord in REGION_COORDS),
    )

    chunks = {coord: build_chunk(coord[0], coord[1], variant=index)
              for index, coord in enumerate(REGION_COORDS)}
    expected = {coord: fingerprint(chunk) for coord, chunk in chunks.items()}

    manager = RegionManager(region_dir)
    region = manager.get_region(-1, -1)

    saves = 0
    for coord in REGION_COORDS:
        region.save_chunk(chunks[coord])
        region.save()  # one chunk per flush: the save-on-unload shape
        saves += 1

        saved = REGION_COORDS[:saves]
        seen = read_back(region_dir, saved)
        damaged = [
            c for c in saved
            if seen[c] is None or fingerprint(seen[c]) != expected[c]
        ]
        check(
            f"after save {saves}, all {len(saved)} saved chunk(s) still read back",
            not damaged,
            f"damaged: {damaged}" if damaged else "",
        )

    print(f"  ({saves} single-chunk saves, each verified through a fresh manager)")


def check_rewrite_in_place(region_dir: Path) -> None:
    print("2. A chunk rewritten larger and smaller keeps its neighbours")
    target = (-16, -16)
    peer = (-1, -1)
    peers = [coord for coord in REGION_COORDS if coord != target]
    expected = {
        coord: fingerprint(build_chunk(coord[0], coord[1], variant=index))
        for index, coord in enumerate(REGION_COORDS)
    }
    path = region_dir / "r.-1.-1.mca"

    manager = RegionManager(region_dir)
    region = manager.get_region(-1, -1)
    for index, coord in enumerate(REGION_COORDS):
        region.save_chunk(build_chunk(coord[0], coord[1], variant=index))
    region.save()
    first = location_of(region_dir, (-1, -1), target)

    def peers_intact(step: str, current: Optional[Tuple]) -> None:
        seen = read_back(region_dir, REGION_COORDS)
        rewritten = seen[target]
        check(
            f"({step}) the rewritten chunk reads back as itself",
            rewritten is not None and fingerprint(rewritten) == current,
            "" if rewritten is None else digest(fingerprint(rewritten)),
        )
        damaged = [
            c for c in peers
            if seen[c] is None or fingerprint(seen[c]) != expected[c]
        ]
        check(f"({step}) all {len(peers)} neighbours are untouched", not damaged,
              f"damaged: {damaged}" if damaged else "")

    def all_intact(step: str, expected_now: Dict[Tuple[int, int], Tuple]) -> None:
        seen = read_back(region_dir, REGION_COORDS)
        damaged = [
            coord for coord in REGION_COORDS
            if seen[coord] is None or fingerprint(seen[coord]) != expected_now[coord]
        ]
        check(
            f"({step}) all {len(REGION_COORDS)} chunks read back as their own version",
            not damaged,
            f"damaged: {damaged}" if damaged else "",
        )

    big = build_large_chunk(target[0], target[1], variant=1)
    region.save_chunk(big)
    region.save()
    big_location = location_of(region_dir, (-1, -1), target)
    check(
        "(a) the larger payload really did need more sectors",
        big_location[1] > first[1],
        f"{first[1]} -> {big_location[1]} sector(s)",
    )
    peers_intact("a: much larger", fingerprint(big))
    expected[target] = fingerprint(big)

    small = build_chunk(target[0], target[1], variant=11, sections=1)
    region.save_chunk(small)
    region.save()
    small_location = location_of(region_dir, (-1, -1), target)
    check(
        "(b) the smaller payload fits in fewer sectors",
        small_location[1] < big_location[1],
        f"{big_location[1]} -> {small_location[1]} sector(s)",
    )
    peers_intact("b: much smaller", fingerprint(small))
    expected[target] = fingerprint(small)

    # (c) The shrink above left a run free. A chunk that needs a new run has to
    # take that one instead of appending: if the free list is not consulted, the
    # file grows here even though there is room in it.
    peer_first = location_of(region_dir, (-1, -1), peer)
    size_before = path.stat().st_size
    medium = build_large_chunk(peer[0], peer[1], variant=5, sections=4)
    region.save_chunk(medium)
    region.save()
    medium_location = location_of(region_dir, (-1, -1), peer)
    size_after = path.stat().st_size
    check(
        "(c) the rewritten chunk needed a bigger run than it had",
        medium_location[1] > peer_first[1],
        f"{peer_first[1]} -> {medium_location[1]} sector(s)",
    )
    check(
        "(c) that run came out of the freed space, so the file did not grow",
        size_after == size_before,
        f"{size_before} -> {size_after} bytes, offset {peer_first[0]} -> {medium_location[0]}",
    )
    expected[peer] = fingerprint(medium)
    all_intact("c: free-run reuse", expected)

    # compact() is the operation that reclaims what the shrink left behind.
    size_before_compact = path.stat().st_size
    compacted = region.compact()
    size_after_compact = path.stat().st_size
    check("compact() rewrote the file because the shrink left a hole", compacted)
    check(
        "compact() made the file smaller",
        size_after_compact < size_before_compact,
        f"{size_before_compact} -> {size_after_compact} bytes",
    )
    check("compact() on an already dense file is a no-op", region.compact() is False)
    all_intact("after compact()", expected)


def check_repeated_writes(region_dir: Path) -> None:
    print("3. Rewriting one chunk 100 times does not grow the file")
    coords = [(-1, -1), (-2, -2), (-3, -3), (-16, -16)]
    target = (-16, -16)
    expected = {
        coord: fingerprint(build_chunk(coord[0], coord[1], variant=index))
        for index, coord in enumerate(coords)
    }

    manager = RegionManager(region_dir)
    region = manager.get_region(-1, -1)
    for index, coord in enumerate(coords):
        region.save_chunk(build_chunk(coord[0], coord[1], variant=index))
    region.save()

    path = region_dir / "r.-1.-1.mca"
    size_first = path.stat().st_size
    slot_first = location_of(region_dir, (-1, -1), target)

    for _ in range(100):
        region.save_chunk(build_chunk(target[0], target[1], variant=3))
        region.save()

    size_last = path.stat().st_size
    slot_last = location_of(region_dir, (-1, -1), target)
    check(
        "100 rewrites grew the file by at most one sector",
        size_last - size_first <= SECTOR_BYTES,
        f"{size_first} -> {size_last} bytes",
    )
    check(
        "the rewrite stayed in its own sector run",
        slot_last == slot_first,
        f"offset/size {slot_first} -> {slot_last}",
    )

    seen = read_back(region_dir, coords)
    damaged = [
        c for c in coords
        if seen[c] is None or fingerprint(seen[c]) != expected[c]
    ]
    check(f"all {len(coords)} chunks read back unchanged after the 100 rewrites",
          not damaged, f"damaged: {damaged}" if damaged else "")


def check_manager_roundtrip(region_dir: Path) -> None:
    print("4. save_all() + close_all() round-trip through a new manager")
    coords = [
        (-1, 0), (0, -1), (-1, -1), (-4, 3), (3, -4), (-33, 17),
        (17, -33), (5, -1), (-1, 5), (0, 0), (31, 31), (-32, -32),
    ]
    chunks = {coord: build_chunk(coord[0], coord[1], variant=index)
              for index, coord in enumerate(coords)}
    expected = {coord: fingerprint(chunk) for coord, chunk in chunks.items()}

    manager = RegionManager(region_dir)
    for coord in coords:
        manager.save_chunk(chunks[coord])

    expected_regions = {(coord[0] // 32, coord[1] // 32) for coord in coords}
    saved = manager.save_all()
    check(
        "save_all flushed every region that staged a chunk",
        saved == len(expected_regions),
        f"{saved} of {len(expected_regions)}",
    )
    check(
        "every region file is on disk",
        all((region_dir / f"r.{x}.{z}.mca").exists() for x, z in expected_regions),
    )

    manager.close_all()  # flushes and compacts

    seen = read_back(region_dir, coords)
    damaged = [
        coord for coord in coords
        if seen[coord] is None or fingerprint(seen[coord]) != expected[coord]
    ]
    check(f"a brand-new manager reads all {len(coords)} chunks back identically",
          not damaged, f"damaged: {damaged}" if damaged else "")


def check_concurrent_read_write(region_dir: Path) -> None:
    print("5. A reader never sees a half-written chunk")
    coords = [(-1, -1), (-2, -2), (-3, -3), (-4, -4)]
    never_saved = [(-5, -5), (-6, -6), (-1, -2)]
    rounds = 12

    versions: Dict[Tuple[int, int], Tuple[Tuple, Tuple]] = {}
    for index, coord in enumerate(coords):
        versions[coord] = (
            fingerprint(build_chunk(coord[0], coord[1], variant=index)),
            fingerprint(build_large_chunk(coord[0], coord[1], variant=index)),
        )
    check("each chunk has two distinguishable versions",
          all(first != second for first, second in versions.values()))

    # Both threads share one manager, which is the design: it owns the region
    # header both of them use.
    manager = RegionManager(region_dir)
    for index, coord in enumerate(coords):
        manager.save_chunk(build_chunk(coord[0], coord[1], variant=index))
    manager.save_all()

    errors: List[str] = []
    raised: List[BaseException] = []
    seen_versions: Dict[Tuple[int, int], Set[Tuple]] = {coord: set() for coord in coords}
    guard = threading.Lock()
    reads = 0
    stop = threading.Event()

    def writer() -> None:
        try:
            for round_index in range(rounds):
                for index, coord in enumerate(coords):
                    builder = build_large_chunk if round_index % 2 else build_chunk
                    manager.save_chunk(builder(coord[0], coord[1], variant=index))
                manager.save_all()
        except BaseException as error:  # noqa: BLE001 - reported, not swallowed
            with guard:
                raised.append(error)
        finally:
            stop.set()

    def reader() -> None:
        nonlocal reads
        try:
            while not stop.is_set():
                for coord in coords:
                    chunk = manager.load_chunk(coord[0], coord[1])
                    if chunk is None:
                        with guard:
                            if len(errors) < 20:
                                errors.append(f"{coord} disappeared mid-save")
                        continue
                    fp = fingerprint(chunk)
                    if fp not in versions[coord]:
                        with guard:
                            if len(errors) < 20:
                                errors.append(
                                    f"{coord} read back as a chunk never written: {digest(fp)}"
                                )
                        continue
                    with guard:
                        seen_versions[coord].add(fp)
                        reads += 1
                for coord in never_saved:
                    if manager.load_chunk(coord[0], coord[1]) is not None:
                        with guard:
                            if len(errors) < 20:
                                errors.append(f"{coord} was never saved but loaded")
        except BaseException as error:  # noqa: BLE001 - reported, not swallowed
            with guard:
                raised.append(error)

    threads = [
        threading.Thread(target=writer, name="region-writer", daemon=True),
        threading.Thread(target=reader, name="region-reader", daemon=True),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=180)
    check("both threads ran to completion",
          not any(thread.is_alive() for thread in threads))

    check("neither thread raised", not raised, repr(raised[:1]))
    check("every load was a whole chunk that was really written", not errors,
          "; ".join(errors[:3]))
    overlapped = sum(1 for coord in coords if len(seen_versions[coord]) == 2)
    check(
        "the reader really did overlap the writer",
        reads > 0 and overlapped > 0,
        f"{reads} load(s), both versions seen for {overlapped}/{len(coords)} chunk(s)",
    )


def check_header_hygiene(region_dir: Path) -> None:
    print("6. The header names exactly the chunks that were saved")
    region = (3, -2)  # world chunks 96..127 x -64..-33
    saved = [(96, -64), (97, -63), (98, -62), (99, -61), (100, -60)]
    never = [(101, -59), (96, -33), (127, -64), (110, -40), (127, -33)]

    manager = RegionManager(region_dir)
    for index, coord in enumerate(saved):
        manager.save_chunk(build_chunk(coord[0], coord[1], variant=index))
    manager.save_all()

    path = region_dir / f"r.{region[0]}.{region[1]}.mca"
    raw = path.read_bytes()
    fresh = RegionFile(path)
    check("a fresh RegionFile loads the header", fresh.load_header())
    check(
        "the file has a header and at least one data sector",
        len(raw) > FIRST_DATA_SECTOR * SECTOR_BYTES,
        f"{len(raw)} bytes",
    )

    for coord in saved:
        index = local_index(coord[0], coord[1])
        value = struct.unpack_from(">I", raw, index * 4)[0]
        offset, size = value >> 8, value & 0xFF
        timestamp = struct.unpack_from(">I", raw, LOCATION_TABLE_BYTES + index * 4)[0]
        location = fresh.header.locations[index]
        check(
            f"{coord} (slot {index}) is named by the on-disk header",
            offset >= FIRST_DATA_SECTOR
            and size >= 1
            and (location.offset, location.size) == (offset, size)
            and timestamp >= MIN_EPOCH
            and fresh.chunk_exists(coord[0], coord[1])
            and manager.chunk_exists(coord[0], coord[1]),
            f"offset={offset} size={size} timestamp={timestamp}",
        )

    for coord in never:
        index = local_index(coord[0], coord[1])
        value = struct.unpack_from(">I", raw, index * 4)[0]
        location = fresh.header.locations[index]
        check(
            f"{coord} (slot {index}) was never saved and has no entry",
            value == 0
            and location.is_empty()
            and not fresh.chunk_exists(coord[0], coord[1])
            and not manager.chunk_exists(coord[0], coord[1]),
            f"entry={value}",
        )


def main() -> int:
    from registry import init_registries

    init_registries(log_summary=False)
    IDS.extend(block_states.default_state_id(name) for name in BLOCK_NAMES)
    check("the fixture blocks resolve to distinct, non-air state ids",
          len(set(IDS)) == len(IDS) and 0 not in IDS,
          f"{len(IDS)} ids, first={IDS[0]}")

    workdir = Path(tempfile.mkdtemp())
    try:
        check_cross_chunk_preservation(workdir / "1")
        print()
        check_rewrite_in_place(workdir / "2")
        print()
        check_repeated_writes(workdir / "3")
        print()
        check_manager_roundtrip(workdir / "4")
        print()
        check_concurrent_read_write(workdir / "5")
        print()
        check_header_hygiene(workdir / "6")
    finally:
        shutil.rmtree(workdir, ignore_errors=True)

    print()
    if failures:
        print(f"{len(failures)} check(s) FAILED:")
        for failure in failures:
            print(f"  - {failure}")
        print("RESULT:", f"FAIL ({len(failures)} check(s))")
        return 1
    print("RESULT:", "PASS")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception:
        import traceback

        traceback.print_exc()
        raise SystemExit(1)
