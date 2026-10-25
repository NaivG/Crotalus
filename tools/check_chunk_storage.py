"""Prove a chunk stores the registry's own state ids, and saves them by name.

Chunk storage used to number blocks with a private, boot-allocated table. That
made a saved chunk's meaning depend on what had been loaded when it was written,
and it silently turned a block into a different block when the allocation moved.
This tool checks the three properties that replaced it:

1. what a section holds is a global state id - including non-default states like
   `oak_log[axis=x]`;
2. a region file stores `{Name, Properties}`, so a save does not depend on the
   block data that wrote it;
3. anything the save names that this build does not have reads as **air with a
   warning** - never as a different block, and never as a crash;
4. `raw_sections()` hands out the section storage itself (the physics kernel's
   entry point), while `blocks_view()` hands out a copy - and the index
   arithmetic behind both is the same on a `-64..319` window as on `0..255`.

Run: python tools/check_chunk_storage.py

Exit code 0 when everything passes, 1 otherwise.
"""

from __future__ import annotations

import gzip
import shutil
import struct
import sys
import tempfile
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
from loguru import logger  # noqa: E402

from world import block_states, palette  # noqa: E402
from world.anvil import AnvilChunkData, RegionFile  # noqa: E402
from world.chunk import AIR, Chunk, ChunkPos, ChunkSection  # noqa: E402
from world.nbt import (  # noqa: E402
    NBTByte,
    NBTByteArray,
    NBTCompound,
    NBTInt,
    NBTIntArray,
    NBTList,
    NBTLongArray,
    NBTString,
    NBTType,
    read_nbt,
    write_nbt,
)

failures: List[str] = []


def check(label: str, condition: bool, detail: str = "") -> None:
    marker = "OK  " if condition else "FAIL"
    print(f"  {marker} {label}" + (f"  ({detail})" if detail else ""))
    if not condition:
        failures.append(label)


# ==================== fixtures ====================

def make_chunk() -> Chunk:
    """A chunk that uses a non-default state and more than one palette entry.

    Built on an explicit 0..255 window so the section indexes below read
    plainly; the negative-window round trip gets its own check in section 6.
    """
    chunk = Chunk(position=ChunkPos(0, 0), min_y=0, height=256)
    stone = block_states.default_state_id("minecraft:stone")
    log_x = block_states.state_id("minecraft:oak_log", {"axis": "x"})
    water = block_states.state_id("minecraft:water", {"level": "3"})
    chunk.fill_section(0, stone)
    for x in range(16):
        for z in range(16):
            chunk.set_block(x, 5, z, water)
            chunk.set_block(x, 9, z, log_x)
    return chunk


def section_palette(tag) -> tuple:
    """`(palette entries, data tag)` out of a serialized `block_states`."""
    entries = tag.get("palette")
    return getattr(entries, "tags", []), tag.get("data")


def write_region(path: Path, payload: bytes, index: int = 0) -> None:
    """A one-chunk region file, so the reader is exercised for real.

    `payload` is the gzip-compressed NBT as `RegionFile.save` would write it.
    """
    header = bytearray(8192)
    body = struct.pack(">I", len(payload) + 1) + struct.pack("B", 1) + payload
    offset = 2  # first sector after the 8 KiB header
    size = (len(body) + 4095) // 4096
    struct.pack_into(">I", header, index * 4, (offset << 8) | size)
    path.write_bytes(bytes(header) + body + b"\x00" * (size * 4096 - len(body)))


def read_region(path: Path, min_y: int = 0, height: int = 256) -> Chunk:
    region = RegionFile(path, min_y=min_y, height=height)
    assert region.load_header(), "region header did not load"
    return region.load_chunk(0, 0, 0, 0)


def main() -> int:
    registry = None
    try:
        from registry import init_registries

        registry = init_registries(log_summary=False)
    except Exception:  # pragma: no cover - the registry is required everywhere else
        pass

    print("1. Sections hold global state ids")
    check("AIR is 0", AIR == 0)
    section = ChunkSection(y=0)
    check("a new section is uint16", section.blocks.dtype == np.uint16, str(section.blocks.dtype))
    check("a new section is air", not section.blocks.any() and section.non_air_count == 0)

    log_x = block_states.state_id("minecraft:oak_log", {"axis": "x"})
    check("oak_log[axis=x] is state 136", log_x == 136, str(log_x))
    section.set_block(1, 2, 3, log_x)
    check("a non-default state survives a write", int(section.get_block(1, 2, 3)) == log_x)
    check("the non-air count followed", section.non_air_count == 1, str(section.non_air_count))
    check("air is the default everywhere else", int(section.blocks.sum()) == log_x)

    total = registry.blocks.total_states if registry is not None else 29671
    chunk = make_chunk()
    ids = np.concatenate([s.blocks.reshape(-1) for s in chunk.sections])
    check(
        "every stored id is inside the registry",
        int(ids.max()) < total,
        f"max={int(ids.max())} of {total}",
    )
    check(
        "no chunk id is outside the registry",
        registry is None or all(registry.blocks.state(int(v)) is not None for v in np.unique(ids)),
    )

    print("\n2. A region file stores names, not ids")
    root = read_nbt(AnvilChunkData(chunk).serialize(), compressed=True)
    level = root.get("Level")
    sections = level.get("Sections").tags
    non_empty = [s for s in chunk.sections if not s.is_empty]
    check("empty sections are skipped, the rest are written",
          len(sections) == len(non_empty) and len(sections) > 0,
          f"{len(sections)} written of {len(non_empty)} non-empty")
    entries, data = section_palette(sections[0].get("block_states"))
    names = [entry.get("Name").value for entry in entries]
    check("the palette names blocks", names and all(n.startswith("minecraft:") for n in names),
          str(names))
    check("the palette has no bare numbers", not any(str(n).isdigit() for n in names))
    check("stone is named", "minecraft:stone" in names, str(names))
    check("oak_log is named", "minecraft:oak_log" in names, str(names))
    log_entry = next(e for e in entries if e.get("Name").value == "minecraft:oak_log")
    props = log_entry.get("Properties")
    check(
        "its properties came along",
        props is not None and props.get("axis").value == "x",
        str({k: v.value for k, v in getattr(props, "tags", {}).items()}),
    )
    check("the data array is a LongArray", data is not None and data.get_type() is NBTType.LONG_ARRAY)
    check("the data array is not empty", len(data.value) == 256, str(len(data.value)))

    # The same packing the client is sent, so reading it back has to work.
    bits = max(4, palette.bits_for(len(entries)))
    unpacked = palette.unpack(data.value, bits, 4096)
    lookup = np.array([block_states.try_state_id(e.get("Name").value,
                                                {k: v.value for k, v in getattr(e.get("Properties"), "tags", {}).items()}
                                                if e.get("Properties") is not None else None)
                       for e in entries], dtype=np.uint16)
    check("the packed data reproduces the section", np.array_equal(lookup[unpacked], chunk.sections[0].blocks))

    print("\n3. A saved chunk survives a reload by name")
    workdir = Path(tempfile.mkdtemp())
    try:
        region_path = workdir / "r.0.0.mca"
        write_region(region_path, AnvilChunkData(chunk).serialize())
        reloaded = read_region(region_path)
        for index, original in enumerate(chunk.sections):
            check(
                f"section {index} round-trips",
                np.array_equal(reloaded.sections[index].blocks, original.blocks),
            )
        check("the reloaded chunk is marked generated", reloaded.is_generated)

        print("\n4. A block this build does not have reads as air, loudly")
        messages: List[str] = []
        sink = logger.add(lambda message: messages.append(str(message)), level="WARNING")
        try:
            # Rename one palette entry to a block that does not exist, leaving
            # the rest of the section intact. This is what a save from a build
            # with different block data looks like.
            root = read_nbt(AnvilChunkData(chunk).serialize(), compressed=True)
            level = root.get("Level")
            block_states_tag = level.get("Sections").tags[0].get("block_states")
            for entry in block_states_tag.get("palette").tags:
                if entry.get("Name").value == "minecraft:oak_log":
                    entry.tags["Name"] = NBTString("Name", "minecraft:not_a_block")
            unknown_path = workdir / "r.1.0.mca"
            write_region(unknown_path, write_nbt(root, compressed=True))
            recovered = read_region(unknown_path)
        finally:
            logger.remove(sink)

        grid = recovered.sections[0].blocks.reshape(16, 16, 16)  # [y, z, x]
        unknown_layer = grid[9]   # where the renamed block was
        kept_layer = grid[5]      # the water layer, which must be untouched
        check("the unknown block did not become another block",
              not unknown_layer.any(), str(np.unique(unknown_layer)))
        check("its cells are air", int(unknown_layer.sum()) == 0)
        check(
            "the remaining blocks are untouched",
            np.all(kept_layer == block_states.state_id("minecraft:water", {"level": "3"})),
            str(np.unique(kept_layer)),
        )
        check("the load reported the missing block", any("not_a_block" in m for m in messages),
              " | ".join(messages[:2]))

        print("\n5. The pre-rewrite format is not guessed at")
        messages.clear()
        sink = logger.add(lambda message: messages.append(str(message)), level="WARNING")
        try:
            legacy_root = NBTCompound()
            legacy_level = NBTCompound()
            legacy_level["xPos"] = NBTInt("xPos", 0)
            legacy_level["zPos"] = NBTInt("zPos", 0)
            legacy_level["Heightmap"] = NBTIntArray("Heightmap", [0] * 256)
            legacy_sections = NBTList()
            legacy_section = NBTCompound()
            legacy_section["Y"] = NBTByte("Y", 0)
            legacy_section["block_states"] = NBTIntArray("block_states", [1] * 4096)
            legacy_sections.tags.append(legacy_section)
            legacy_level["Sections"] = legacy_sections
            legacy_root["Level"] = legacy_level
            legacy_path = workdir / "r.2.0.mca"
            write_region(legacy_path, write_nbt(legacy_root, compressed=True))
            legacy = read_region(legacy_path)
        finally:
            logger.remove(sink)

        check("an old flat array does not become a palette", not legacy.sections[0].blocks.any())
        check(
            "the old format is called out",
            any("pre-rewrite" in m or "flat block array" in m for m in messages),
            " | ".join(messages[:2]),
        )

        print("\n6. A chunk below Y=0 round-trips at its own absolute Y")
        # The overworld window (min_y=-64, 24 sections) is the storage reality:
        # section Y tags go negative, and a reload has to land the blocks back
        # at the same world Y. This is the layout the `above_bottom` bedrock
        # band needs to survive a save.
        deep = Chunk(position=ChunkPos(0, 0))  # defaults: -64..319
        deep_stone = block_states.default_state_id("minecraft:stone")
        deep.set_block(3, -20, 4, deep_stone)
        deep.set_block(5, 300, 6, deep_stone)
        deep_path = workdir / "r.3.0.mca"
        write_region(deep_path, AnvilChunkData(deep).serialize())
        deep_back = read_region(deep_path, min_y=deep.min_y, height=deep.height)
        check("a block at Y=-20 returns", int(deep_back.get_block(3, -20, 4)) == deep_stone)
        check("a block at Y=300 returns", int(deep_back.get_block(5, 300, 6)) == deep_stone)
        deep_root = read_nbt(AnvilChunkData(deep).serialize(), compressed=True)
        deep_ys = sorted(int(s.get("Y").value)
                         for s in deep_root.get("Level").get("Sections").tags)
        check("section Y tags are absolute (negative below 0)",
              deep_ys == [-2, 18], str(deep_ys))
    finally:
        shutil.rmtree(workdir, ignore_errors=True)

    print("\n7. raw_sections() is the storage, blocks_view() is a copy")
    # The physics kernel runs every tick and cannot copy 128 KB per chunk per
    # tick, so it reads `raw_sections()`. "Really the storage" is the whole
    # contract, and the only way to check it is to write through the array and
    # ask `get_block` -- which reads the sections, not the array.
    live = Chunk(position=ChunkPos(0, 0))            # -64..319, 24 sections
    raw = live.raw_sections()
    check("one array per section", len(raw) == 24 == len(live.sections), str(len(raw)))
    check("they are the section's own arrays",
          all(array is section.blocks for array, section in zip(raw, live.sections)))
    check("and a second call hands out the same arrays",
          all(a is b for a, b in zip(raw, live.raw_sections())))
    check("a fresh call is a new list, not the internal one",
          live.raw_sections() is not live.sections)

    snap = live.blocks_view()
    marker = block_states.default_state_id("minecraft:gold_block")
    raw[0][0] = marker                                  # section 0, x=y=z=0
    check("a write through the array is visible at y=-64 immediately",
          int(live.get_block(0, -64, 0)) == marker, str(int(live.get_block(0, -64, 0))))
    check("the earlier blocks_view() copy did not change",
          int(snap[0, 0, 0]) == AIR, str(int(snap[0, 0, 0])))
    raw[23][4095] = marker                              # last section, x=y=z=15
    check("and at the top of the window (y=319)",
          int(live.get_block(15, 319, 15)) == marker, str(int(live.get_block(15, 319, 15))))

    # `x + z*16 + y*256` over `[y, z, x]`, absolute section Y, both windows.
    for window, probe in (("min_y=-64 height=384", Chunk(position=ChunkPos(3, -2))),
                          ("min_y=0 height=256",
                           Chunk(position=ChunkPos(0, 0), min_y=0, height=256))):
        arrays = probe.raw_sections()
        value = 0
        cells = 0
        mismatches = []
        for section_index in range(len(arrays)):
            for x, y, z in ((0, 0, 0), (7, 3, 11), (15, 15, 15), (1, 8, 14)):
                value += 1
                cells += 1
                arrays[section_index][x + z * 16 + y * 256] = value
                world_y = probe.min_y + section_index * 16 + y
                if int(probe.get_block(x, world_y, z)) != value:
                    mismatches.append((section_index, x, world_y, z))
        check(f"{window}: x + z*16 + y*256 indexes exactly what get_block reads",
              not mismatches,
              f"{cells} cells written" if not mismatches else f"first mismatch {mismatches[0]}")
    check("a cell outside the window still reads as air",
          int(live.get_block(0, 320, 0)) == AIR and int(live.get_block(0, -65, 0)) == AIR)

    # Writing through the raw array bypasses `set_block`, so `non_air_count`
    # (and therefore `is_empty`, which the chunk codec trusts) goes stale. That
    # is the documented cost of the zero copy, and `recompute_non_air_count` is
    # the documented recovery -- not a silent lie.
    check("a raw write leaves the non-air count stale, as documented",
          live.sections[0].is_empty, f"count={live.sections[0].non_air_count}")
    live.sections[0].recompute_non_air_count()
    check("recompute_non_air_count() repairs it",
          not live.sections[0].is_empty and live.sections[0].non_air_count == 1,
          f"count={live.sections[0].non_air_count}")

    print("\n8. Light survives a round trip, and a bad one is refused")
    # Light is the second thing this file has to carry now that a flood-fill
    # engine exists. Three properties, and the failure mode of each:
    #
    #   * the bytes come back identical -- otherwise a relit chunk reads old
    #     light on the next start and nothing ever corrects it;
    #   * an array of the wrong length is refused -- otherwise the light engine
    #     indexes 4096 nibbles into something shorter;
    #   * the copy is a copy -- `np.frombuffer` returns a view onto the NBT
    #     buffer, and the engine writes into these arrays in place.
    #
    # Its own temporary directory: section 3's `finally` already removed the
    # one above, and reusing it would fail on the first write.
    light_dir = Path(tempfile.mkdtemp())
    try:
        light_chunk = Chunk(position=ChunkPos(0, 0), min_y=0, height=256)
        light_chunk.fill_section(0, block_states.default_state_id("minecraft:stone"))
        rng = np.random.default_rng(20261006)
        for section in light_chunk.sections:
            section.sky_light[:] = rng.integers(0, 256, size=2048, dtype=np.uint8)
            section.block_light[:] = rng.integers(0, 256, size=2048, dtype=np.uint8)
        light_chunk.light_populated = True

        light_path = light_dir / "r.2.0.mca"
        write_region(light_path, AnvilChunkData(light_chunk).serialize())
        light_back = read_region(light_path)
        # The arrays are written for every section, so all 16 have to come back.
        check("the light arrays come back byte for byte",
              all(np.array_equal(a.sky_light, b.sky_light)
                  and np.array_equal(a.block_light, b.block_light)
                  for a, b in zip(light_back.sections, light_chunk.sections)))
        check("and the chunk is marked as having real light", light_back.light_populated)
        # The view bug is invisible until something writes: mutating the loaded
        # array must not reach the decoded tag, and two sections must not share
        # one buffer. Checking the base object is what catches it.
        check("the loaded arrays own their memory",
              all(s.sky_light.base is None and s.block_light.base is None
                  for s in light_back.sections))

        # A chunk whose light engine has not run yet must not come back marked
        # lit. Its arrays are `ChunkSection`'s defaults -- 15 for sky, which is
        # "nobody computed this" and not "daylight" (`world/chunk.py`) -- and
        # writing them out anyway turned the default into a claim: the loader
        # found an array, set `light_populated`, and the encoder shipped a fully
        # lit chunk. Measured live (real server, real generator pool, one
        # viewer, save flushed at the end): 48 of 169 chunks were still in the
        # light queue when they were written, all 48 of them reloading lit.
        unlit = Chunk(position=ChunkPos(0, 0), min_y=0, height=256)
        unlit.fill_section(0, block_states.default_state_id("minecraft:stone"))
        unlit_path = light_dir / "r.4.0.mca"
        unlit_root = read_nbt(AnvilChunkData(unlit).serialize(), compressed=True)
        unlit_sections = unlit_root.get("Level").get("Sections").tags
        write_region(unlit_path, AnvilChunkData(unlit).serialize())
        unlit_back = read_region(unlit_path)
        check("a chunk nobody lit is written without light arrays",
              not any("sky_light" in section_tag.tags
                      for section_tag in unlit_sections))
        check("so it reads back as not populated",
              not unlit_back.light_populated)
        check("and its blocks came back anyway",
              unlit_back.sections[0].blocks.tobytes()
              == unlit.sections[0].blocks.tobytes())

        # A short array: the reader keeps the default and says so.
        messages.clear()
        sink = logger.add(lambda message: messages.append(str(message)), level="WARNING")
        try:
            root = read_nbt(AnvilChunkData(light_chunk).serialize(), compressed=True)
            sections_tag = root.get("Level").get("Sections")
            sections_tag.tags[0].tags["sky_light"] = NBTByteArray("sky_light", b"\x00" * 16)
            short_path = light_dir / "r.3.0.mca"
            write_region(short_path, write_nbt(root, compressed=True))
            short_back = read_region(short_path)
        finally:
            logger.remove(sink)
        check("a short light array is refused, not truncated into place",
              not np.array_equal(short_back.sections[0].sky_light,
                                 light_chunk.sections[0].sky_light))
        check("the refusal is reported", any("light array" in m for m in messages),
              " | ".join(messages[:2]))
    finally:
        shutil.rmtree(light_dir, ignore_errors=True)

    print("\n9. A light-only change is worth saving; nothing else repeats")
    from world.chunk_saver import should_persist  # noqa: E402

    policy = Chunk(position=ChunkPos(0, 0), min_y=0, height=256)
    policy.is_generated = True
    policy.is_persisted = True
    check("a generated, persisted, unchanged chunk is skipped", not should_persist(policy))
    policy.is_light_dirty = True
    check("light alone makes it worth writing", should_persist(policy))
    policy.mark_saved()
    check("mark_saved() clears the light flag too",
          not policy.is_light_dirty and policy.is_persisted)
    check("so a second save is skipped again", not should_persist(policy))
    policy.is_dirty = True
    check("a block change still counts", should_persist(policy))

    print()
    if failures:
        print(f"{len(failures)} check(s) FAILED:")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print("All chunk storage checks passed.")
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
