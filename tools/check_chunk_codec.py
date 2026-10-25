"""
Round-trip check for world/chunk_codec.py.

Decodes the encoder's output with an independent parser and compares it back
against the source `Chunk`. The point is to catch byte-layout mistakes, which
a self-consistent encoder would happily reproduce.

Run: python tools/check_chunk_codec.py
"""

from __future__ import annotations

import struct
import sys
from pathlib import Path
from typing import Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from world.block_states import BLOCKS, state_id
from world.chunk import BlockPos, Chunk, ChunkPos
from world.chunk_codec import (
    BIOME_ENTRIES,
    BIOME_MAX_BITS,
    BLOCK_MAX_BITS,
    ChunkCodec,
    HEIGHTMAP_BITS,
    HEIGHTMAP_LONGS,
    LIGHT_BYTES,
    _encompassing_bits,
    _column_heights,
    _pack_bits,
    _classify_light,
)

#: `minecraft:torch`, resolved through the registry rather than hardcoded:
#: a state id is not stable across data updates, and this fixture is the one
#: place in the repo that needs a named light source.
TORCH = state_id("minecraft:torch")


class Reader:
    def __init__(self, data: bytes):
        self.d = data
        self.i = 0

    def varint(self) -> int:
        r = shift = 0
        while True:
            b = self.d[self.i]
            self.i += 1
            r |= (b & 0x7F) << shift
            if not b & 0x80:
                return r
            shift += 7

    def take(self, n: int) -> bytes:
        out = self.d[self.i : self.i + n]
        self.i += n
        return out

    def i16(self) -> int:
        return struct.unpack(">h", self.take(2))[0]

    def i32(self) -> int:
        return struct.unpack(">i", self.take(4))[0]

    def i64(self) -> int:
        return struct.unpack(">q", self.take(8))[0]

    def u8(self) -> int:
        return self.take(1)[0]


def unpack_bits(data: bytes, bits: int, count: int) -> list[int]:
    """Inverse of _pack_bits, written independently."""
    if bits == 0:
        return []
    per_long = 64 // bits
    n_longs = (count + per_long - 1) // per_long
    out = []
    for n in range(n_longs):
        long = struct.unpack(">q", data[n * 8 : n * 8 + 8])[0]
        for k in range(per_long):
            out.append((long >> (k * bits)) & ((1 << bits) - 1))
    return out[:count]


def parse_chunk(data: bytes, section_count: int, min_y: int = -64):
    r = Reader(data)
    cx, cz = r.i32(), r.i32()

    hm_count = r.varint()
    heightmaps = {}
    for _ in range(hm_count):
        idx = r.varint()
        length = r.varint()
        raw = [r.i64() for _ in range(length)]
        heightmaps[idx] = unpack_bits(struct.pack(f">{length}q", *raw), HEIGHTMAP_BITS, 256)

    blob_len = r.varint()
    blob = r.take(blob_len)

    br = Reader(blob)
    sections = []
    for _ in range(section_count):
        non_air = br.i16()
        bits = br.u8()
        if bits == 0:
            palette = [br.varint()]
        elif bits == BLOCK_MAX_BITS:
            palette = None
        else:
            palette = [br.varint() for _ in range(br.varint())]
        per_long = 64 // bits if bits else 0
        n_longs = (4096 + per_long - 1) // per_long if bits else 0
        indices = unpack_bits(br.take(n_longs * 8), bits, 4096)

        b_bits = br.u8()
        if b_bits == 0:
            b_palette = [br.varint()]
        elif b_bits == BIOME_MAX_BITS:
            b_palette = None
        else:
            b_palette = [br.varint() for _ in range(br.varint())]
        b_per = 64 // b_bits if b_bits else 0
        b_indices = unpack_bits(br.take(((BIOME_ENTRIES + b_per - 1) // b_per) * 8) if b_bits else b"",
                                b_bits, BIOME_ENTRIES)

        sections.append((non_air, palette, indices, b_palette, b_indices))

    assert br.i == len(blob), f"section blob not fully consumed: {br.i} != {len(blob)}"

    block_entities = r.varint()
    assert block_entities == 0, f"expected no block entities, got {block_entities}"

    masks = []
    for _ in range(4):
        n = r.varint()
        masks.append(struct.unpack(">q", r.take(8 * n))[0])

    sky_arrays = []
    for _ in range(r.varint()):
        sky_arrays.append(r.take(r.varint()))
    block_arrays = []
    for _ in range(r.varint()):
        block_arrays.append(r.take(r.varint()))

    assert r.i == len(data), f"payload not fully consumed: {r.i} != {len(data)}"

    # What the client ends up with, indexed by *light* section. The masks cover
    # `section_count + 2` light sections, not `section_count`, and the mapping is
    # the one thing a server gets wrong here:
    #
    #   bit 0                  -> the section below the dimension's min_y
    #   bit j, 1..section_count -> block section j - 1, i.e. y = min_y + (j-1)*16
    #   bit section_count + 1  -> the section above the dimension's ceiling
    #
    # (Protocol 774, "Light Data": "the least significant bit is for blocks 16
    # blocks to 1 block below the min world height".) Writing block section `s`
    # into bit `s` shifts every light array 16 blocks down on the client, which
    # is invisible in a flat fixture and wrong everywhere else.
    #
    # A section with no bit in either array mask is one the client has no data
    # for, and its own defaults are 15 for sky and 0 for block. The polarity of
    # the two empty masks is opposite, which is the other thing a reader gets
    # wrong: `empty sky` would mean 15 and `empty block` means 0. This encoder
    # writes no `empty sky` bits at all (a sky array always accompanies the bit
    # it needs, so "no bit" is the only way it says 15); the model below accepts
    # either.
    #
    # This is deliberately built from the *wire* masks rather than from
    # `chunk.sections`, so comparing it against the chunk tests the encoder's
    # classification rather than restating it.
    light_section_count = section_count + 2
    sky_light: list[list[int] | None] = [None] * light_section_count
    block_light: list[list[int] | None] = [None] * light_section_count
    sky_stream = list(sky_arrays)
    block_stream = list(block_arrays)
    for j in range(light_section_count):
        bit = 1 << j
        if masks[0] & bit:
            sky_light[j] = unpack_nibbles(sky_stream.pop(0))
        elif masks[2] & bit:
            # Dead for this codec (the mask is asserted empty below): written as
            # the client's *default* for a section it has no data for, which is
            # what "empty" means in the field name and the reading this check
            # holds the encoder to.
            sky_light[j] = [15] * 4096
        if masks[1] & bit:
            block_light[j] = unpack_nibbles(block_stream.pop(0))
        elif masks[3] & bit:
            block_light[j] = [0] * 4096

    return {
        "pos": (cx, cz),
        "min_y": min_y,
        "heightmaps": heightmaps,
        "sections": sections,
        "masks": masks,
        "sky_arrays": sky_arrays,
        "block_arrays": block_arrays,
        "sky_light": sky_light,
        "block_light": block_light,
    }


def client_light_for_section(parsed: dict, field: str, packet_index: int) -> list[int]:
    """The light the client holds for one *block* section, defaults filled in.

    `packet_index` is the packet's block section (0 = the chunk's first section),
    which is light section `packet_index + 1`.
    """
    values = parsed[field][packet_index + 1]
    if values is not None:
        return values
    return [15] * 4096 if field == "sky_light" else [0] * 4096


def unpack_nibbles(packed: bytes) -> list[int]:
    """Inverse of the codec's nibble packing, written independently.

    Low nibble first, which is the one detail worth stating twice: a swapped
    pair would round trip through the codec's own packer and only show up as
    light values one block off, which is the kind of bug that gets blamed on
    the client.
    """
    assert len(packed) == LIGHT_BYTES, f"light array is {len(packed)} B, want {LIGHT_BYTES}"
    out = [0] * (len(packed) * 2)
    for i, byte in enumerate(packed):
        out[2 * i] = byte & 0x0F
        out[2 * i + 1] = byte >> 4
    return out


def make_chunk(cx: int = 3, cz: int = -7) -> Chunk:
    """A chunk with a plausible terrain column so packing is non-trivial.

    The light is filled in the way the light engine would fill it for this
    terrain -- 15 above the surface, 14 directly under it, 0 further down --
    and `light_populated` is set, so encoding reads real arrays instead of the
    height fallback. A fixture that left the arrays at their defaults would
    test `_legacy_sky_light`, which is the path gameplay never takes.
    """
    chunk = Chunk(position=ChunkPos(cx, cz))
    for x in range(16):
        for z in range(16):
            h = 60 + ((x * 7 + z * 3 + cx) % 12)
            for y in range(0, h):
                block = BLOCKS.STONE
                if y == 0:
                    block = BLOCKS.BEDROCK
                elif y == h - 1:
                    block = BLOCKS.GRASS
                elif y > h - 4:
                    block = BLOCKS.DIRT
                chunk.set_block(x, y, z, block)
            if h < 64:
                for y in range(h, 64):
                    chunk.set_block(x, y, z, BLOCKS.WATER)  # water
    # One torch, so the block-light half of the light masks is exercised too.
    # Without it every section is block-light-empty and the encoder's block
    # path is never entered -- which is exactly how the old hardcoded
    # "no block light" survived this check. It goes in the air one block above
    # the surface, which is where a real one would stand.
    torch = (8, int(chunk.get_height_at(8, 8)) + 1, 8)
    chunk.set_block(torch[0], torch[1], torch[2], TORCH)
    _fill_hand_light(chunk, torch=torch)
    return chunk


def _fill_hand_light(chunk: Chunk, torch: Optional[tuple] = None) -> None:
    """Write plausible sky light (and one torch's block light) in place.

    Not a light engine: it is the vertical part of one plus one torch, which is
    all a fixture needs to exercise the encoder's three-way classification.
    Every column is lit at its top block and above, and one level darker for
    each block below, which gives the surface section a genuinely mixed array
    (15 next to 14), the sections above it a uniform 15, and the sections below
    a uniform 0 -- all three kinds the encoder has to tell apart, in one chunk.

    The torch is a real flood fill rather than a single lit cell, because the
    assertions compare the wire against a recomputed value and a single cell
    would only pin "the seed is in the right place", not the decay.
    """
    heights = _column_heights(chunk)
    for section in chunk.sections:
        section.sky_light[:] = 0
        section.block_light[:] = 0
    for x in range(16):
        for z in range(16):
            top = int(heights[z, x])
            for y in range(top + 1, chunk.min_y + chunk.height):
                _write_nibble(chunk.sections[(y - chunk.min_y) // 16].sky_light,
                              x + z * 16 + ((y - chunk.min_y) % 16) * 256, 15)
            _write_nibble(chunk.sections[(top - chunk.min_y) // 16].sky_light,
                          x + z * 16 + ((top - chunk.min_y) % 16) * 256, 14)
    if torch is not None:
        _flood_torch(chunk, torch)
    chunk.light_populated = True


def _write_nibble(packed: np.ndarray, cell: int, value: int) -> None:
    """Set one 4-bit cell in a 2048-byte packed light array, low nibble first."""
    byte = cell >> 1
    if cell & 1:
        packed[byte] = (int(packed[byte]) & 0x0F) | ((value & 0x0F) << 4)
    else:
        packed[byte] = (int(packed[byte]) & 0xF0) | (value & 0x0F)


def _read_nibble(packed: np.ndarray, cell: int) -> int:
    """Read one 4-bit cell out of a packed light array."""
    byte = int(packed[cell >> 1])
    return (byte >> 4) if cell & 1 else (byte & 0x0F)


#: A torch's light level, and its dampening into a neighbour: the vanilla
#: `level - max(opacity, 1)`, with the torch's own cell sitting on grass
#: (opacity 1) rather than in air.
TORCH_LEVEL = 14


def _flood_torch(chunk: Chunk, torch: tuple) -> None:
    """Six-directional BFS from one torch, into `ChunkSection.block_light`.

    Deliberately the simplest thing that can be right: a deque, no bitset, no
    early exit. It exists so the light section of this check has an independent
    value to compare the wire against, which is worth more than the
    microseconds a second clever implementation would save.
    """
    from collections import deque

    tx, ty, tz = torch

    def write(x: int, y: int, z: int, level: int) -> None:
        section = chunk.sections[(y - chunk.min_y) // 16]
        _write_nibble(section.block_light, x + z * 16 + ((y - chunk.min_y) % 16) * 256, level)

    def read(x: int, y: int, z: int) -> int:
        section = chunk.sections[(y - chunk.min_y) // 16]
        return _read_nibble(section.block_light, x + z * 16 + ((y - chunk.min_y) % 16) * 256)

    write(tx, ty, tz, TORCH_LEVEL)
    queue = deque([(tx, ty, tz, TORCH_LEVEL)])
    while queue:
        x, y, z, level = queue.popleft()
        if level <= 1:
            continue
        for dx, dy, dz in ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)):
            nx, ny, nz = x + dx, y + dy, z + dz
            if not (0 <= nx < 16 and 0 <= nz < 16):
                continue
            if not (chunk.min_y <= ny < chunk.min_y + chunk.height):
                continue
            if int(chunk.get_block(nx, ny, nz)) != 0:  # anything but air blocks it
                continue
            level_here = level - 1
            if level_here > read(nx, ny, nz):
                write(nx, ny, nz, level_here)
                queue.append((nx, ny, nz, level_here))


def check_light_classification() -> None:
    """`_classify_light` on the patterns a fixture cannot produce.

    This is the unit test for the one function whose mistakes are invisible:
    every array the `make_chunk` fixture writes happens to be uniform, so a
    classifier that combined a byte's two nibbles instead of decoding them
    would still pass every end-to-end assertion in this file. It did, for a
    while -- a byte of `0x0F` (cells 15 and 0) was read as "15 everywhere", so
    a half-dark section went out as a mask bit with no array and the client
    filled it with daylight.

    The rule being tested is byte-level: all cells equal `v` exactly when every
    byte equals `v | (v << 4)`, because a byte holds cell 2i in its low nibble
    and cell 2i+1 in its high one.
    """
    cases = [
        (0x00, 0, "all dark"),
        (0xFF, 15, "all lit"),
        (0x11, 1, "all dim"),
        (0x1E, -1, "cells 14 and 1 -- two different values"),
        (0x0F, -1, "cells 15 and 0 -- half dark"),
        (0xF0, -1, "cells 0 and 15 -- the same pair, swapped"),
        (0x12, -1, "cells 2 and 1"),
        (0x77, 7, "all seven"),
    ]
    for byte, want, label in cases:
        got = _classify_light(np.full(LIGHT_BYTES, byte, dtype=np.uint8))
        assert got == want, f"0x{byte:02X} ({label}) classified {got}, want {want}"

    # And a genuinely mixed array, which no single byte pattern expresses.
    mixed = np.zeros(LIGHT_BYTES, dtype=np.uint8)
    mixed[0] = 0xFF
    assert _classify_light(mixed) == -1, "an array with one bright byte is not uniform"
    print(f"  _classify_light OK ({len(cases)} byte patterns + one mixed array)")


def check_pack_bits() -> None:
    """The generic packer against a hand-computed case."""
    # 4 bits per entry: entry i lands at bit offset (i % 16) * 4 of long i // 16.
    vals = np.array([0, 15, 1, 2] + [0] * 4092, dtype=np.uint64)
    packed = _pack_bits(vals, 4)
    assert len(packed) == 256 * 8, len(packed)
    back = unpack_bits(packed, 4, 4096)
    assert back[:4] == [0, 15, 1, 2], back[:4]
    print("  _pack_bits 4-bit round trip OK")

    # 9 bits per entry (heightmaps): 7 per long, 37 longs.
    packed9 = _pack_bits(np.arange(256, dtype=np.uint64), HEIGHTMAP_BITS)
    assert len(packed9) == HEIGHTMAP_LONGS * 8, len(packed9)
    back9 = unpack_bits(packed9, HEIGHTMAP_BITS, 256)
    assert back9 == list(range(256)), back9[:10]
    print("  _pack_bits 9-bit round trip OK")

    assert _encompassing_bits(1) == 0
    assert _encompassing_bits(2) == 1
    assert _encompassing_bits(16) == 4
    assert _encompassing_bits(17) == 5
    print("  _encompassing_bits OK")


def make_biome_chunk(cx: int = -5, cz: int = 11) -> Chunk:
    """A chunk whose biome volume is mixed enough to leave the homogeneous path.

    Four biomes placed by a 3D pattern so that individual *sections* contain
    more than one, and one section is deliberately uniform - both branches of
    `_encode_biome_palette` have to be exercised, and a volume that happened to
    be one biome per section would pass while testing neither.
    """
    chunk = Chunk(position=ChunkPos(cx, cz))
    sections = len(chunk.sections)
    volume = np.zeros((sections, 4, 4, 4), dtype=np.int32)
    palette = [40, 21, 41, 3]  # plains, forest, river, beach
    for s in range(sections):
        if s % 7 == 3:
            volume[s] = palette[0]
            continue
        for y in range(4):
            for z in range(4):
                for x in range(4):
                    volume[s, y, z, x] = palette[(s + y + z * 2 + x) % len(palette)]
    chunk.biomes = volume
    return chunk


def main() -> int:
    check_pack_bits()
    check_light_classification()

    codec = ChunkCodec()
    chunk = make_chunk()

    data = codec.encode(chunk)
    parsed = parse_chunk(data, codec.section_count, codec.min_y)

    print(f"\n  chunk {parsed['pos']}  payload={len(data)} bytes")
    assert parsed["pos"] == (3, -7)

    # --- heightmaps -------------------------------------------------
    for idx, values in parsed["heightmaps"].items():
        assert len(values) == 256, f"heightmap {idx} has {len(values)} entries"
    hs = parsed["heightmaps"][4]
    # MOTION_BLOCKING raw = highest non-air y + 2, checked across every column
    # (a sampled check missed a per-column broadcasting bug here once).
    for x in range(16):
        for z in range(16):
            expected = chunk.get_height_at(x, z) + 2
            got = hs[z * 16 + x]
            assert got == expected, f"heightmap at (x={x},z={z}): {got} != {expected}"
    print("  heightmaps OK (3 maps, 256 entries, all columns match chunk)")

    # --- sections ---------------------------------------------------
    total_non_air = 0
    reconstructed = np.zeros((16, 16, 16), dtype=np.uint32)  # [y, z, x]
    for packet_index, (non_air, palette, indices, b_palette, b_indices) in enumerate(parsed["sections"]):
        total_non_air += non_air

        # Packet section i holds absolute Y codec.min_y + i*16; chunk section j
        # holds chunk.min_y + j*16. Same window here, so the two coincide, but
        # the formula keeps the check honest if that ever changes.
        server_index = packet_index + (codec.min_y - chunk.min_y) // 16
        if not (0 <= server_index < len(chunk.sections)):
            assert non_air == 0, f"section {packet_index} should be empty"
            continue

        section = chunk.sections[server_index]
        if non_air == 0:
            continue

        for i in range(4096):
            if palette is None:
                state = indices[i]
            elif len(palette) == 1:
                state = palette[0]
            else:
                state = palette[indices[i]]
            x = i % 16
            z = (i % 256) // 16
            y = i // 256
            reconstructed[y, z, x] = state
            # A section stores global state ids, so what the client is handed
            # has to be the number that was stored - identity, not a translation.
            assert state == int(section.blocks[i]), (
                f"section {server_index} block ({x},{y},{z}): "
                f"state {state} != stored {int(section.blocks[i])}"
            )

    print(f"  sections OK ({codec.section_count} written, {total_non_air} non-air blocks match)")

    # --- biomes -----------------------------------------------------
    # Before this existed the whole chunk shipped as one biome, so every
    # section took the homogeneous branch and nothing ever exercised the
    # palette path. `make_biome_chunk` below gives a section a real mix.
    biome_chunk = make_biome_chunk()
    biome_parsed = parse_chunk(codec.encode(biome_chunk), codec.section_count, codec.min_y)
    volume = biome_chunk.biome_volume()
    for packet_index, section in enumerate(biome_parsed["sections"]):
        _, _, _, b_palette, b_indices = section
        server_index = packet_index + (codec.min_y - biome_chunk.min_y) // 16
        if not (0 <= server_index < volume.shape[0] // 4):
            assert b_palette == [codec.default_biome] and not any(b_indices)
            continue
        want = volume[server_index * 4:(server_index + 1) * 4].reshape(-1)
        if b_palette is None:
            got = b_indices
        elif len(b_palette) == 1:
            got = [b_palette[0]] * BIOME_ENTRIES
        else:
            got = [b_palette[i] for i in b_indices]
        assert list(got) == [int(v) for v in want], (
            f"section {server_index}: biome cells differ\n  got  {list(got)[:16]}\n"
            f"  want {[int(v) for v in want][:16]}"
        )
    mixed = sum(1 for s in biome_parsed["sections"]
                if s[3] is None or len(s[3]) > 1)
    assert mixed >= 4, f"the biome fixture should exercise the palette path, got {mixed}"
    print(f"  biomes OK ({mixed} sections off the homogeneous path, all 64 cells match)")

    # Every non-air block in the source must appear in the decode.
    source_non_air = int(np.count_nonzero(chunk.sections[0].blocks != 0)) * 0
    for s in chunk.sections:
        source_non_air += int(np.count_nonzero(s.blocks))
    assert total_non_air == source_non_air, f"{total_non_air} != {source_non_air}"
    print(f"  non-air total matches source ({source_non_air})")

    # --- light ------------------------------------------------------
    # The four masks at 774. Two things about them are easy to get backwards,
    # and both are asserted below rather than described:
    #
    #   * the bit a section goes in is its *light* section, not its block
    #     section: block section s is bit s + 1, bit 0 is the section below the
    #     dimension's min_y;
    #   * the empty masks have opposite polarity: `empty block` means "no data",
    #     i.e. 0, and `empty sky` -- whose meaning sources disagree on, 15 or all
    #     zeros -- is not written by this codec at all.
    #
    #   masks[0] sky light   array    masks[2] empty sky   unused by this codec
    #   masks[1] block light array    masks[3] empty block 1 bit = 0
    sky_array_mask, block_array_mask, sky_empty_mask, block_empty_mask = parsed["masks"]

    for name, arr in (("sky", parsed["sky_arrays"]), ("block", parsed["block_arrays"])):
        for packed in arr:
            assert len(packed) == LIGHT_BYTES, f"{name} light array is {len(packed)} B"

    # Bits 0 and section_count + 1 are the two light sections outside the
    # dimension's own window. They are left unaddressed here; the client's
    # default is what they hold (15 for sky, 0 for block) and there are no
    # blocks in either of them.
    block_bits = ((1 << codec.section_count) - 1) << 1
    assert sky_array_mask & ~block_bits == 0, (
        f"a sky array addresses a section outside 1..{codec.section_count}: "
        f"{sky_array_mask:#x} -- the mask bit is the section + 1"
    )
    assert block_array_mask & ~block_bits == 0, (
        f"a block array addresses a section outside 1..{codec.section_count}: "
        f"{block_array_mask:#x}"
    )
    assert sky_empty_mask == 0, (
        "this codec never writes an `empty sky` bit: a section with no bit at "
        "all already reads as 15, a dark section ships an explicit array, and "
        "the empty bit's own meaning is described as both 15 and all zeros"
    )
    assert bin(sky_array_mask).count("1") == len(parsed["sky_arrays"]), (
        f"sky: {bin(sky_array_mask).count('1')} mask bits vs "
        f"{len(parsed['sky_arrays'])} arrays"
    )
    assert bin(block_array_mask).count("1") == len(parsed["block_arrays"]), (
        f"block: {bin(block_array_mask).count('1')} mask bits vs "
        f"{len(parsed['block_arrays'])} arrays"
    )
    # `empty block` and `block light` partition the block sections: a section is
    # either all-zero (bit in empty) or carries an array. `empty sky` is not
    # part of this codec's encoding, so it is only checked to be empty above.
    assert not (block_array_mask & block_empty_mask), "block masks overlap"

    # The wire has to reproduce the chunk's own light, byte for byte, *as the
    # client assembles it*. This is the assertion that catches the shift: with
    # the mask indexed by block section instead of light section, every section
    # compares against its neighbour and the check fails on the first one.
    for packet_index in range(codec.section_count):
        server_index = packet_index + (codec.min_y - chunk.min_y) // 16
        assert 0 <= server_index < len(chunk.sections)
        source = chunk.sections[server_index]
        got_sky = client_light_for_section(parsed, "sky_light", packet_index)
        got_block = client_light_for_section(parsed, "block_light", packet_index)
        want_sky = unpack_nibbles(bytes(source.sky_light))
        want_block = unpack_nibbles(bytes(source.block_light))
        assert got_sky == want_sky, (
            f"block section {packet_index} (light section {packet_index + 1}): "
            f"sky light differs on the wire "
            f"({sum(a != b for a, b in zip(got_sky, want_sky))} cells)"
        )
        assert got_block == want_block, (
            f"block section {packet_index} (light section {packet_index + 1}): "
            f"block light differs on the wire "
            f"({sum(a != b for a, b in zip(got_block, want_block))} cells)"
        )

    # The counterfactual, so this check cannot quietly go back to the old model:
    # indexing the masks by block section must *not* reproduce the chunk.
    shifted_mismatch = 0
    for packet_index in range(codec.section_count):
        server_index = packet_index + (codec.min_y - chunk.min_y) // 16
        if not (0 <= server_index < len(chunk.sections)):
            continue
        want_sky = unpack_nibbles(bytes(chunk.sections[server_index].sky_light))
        got = parsed["sky_light"][packet_index]
        if got is None:
            got = [15] * 4096
        shifted_mismatch += sum(a != b for a, b in zip(got, want_sky))
    assert shifted_mismatch > 0, (
        "the shift this check exists for is not detectable on this fixture: "
        "a section-indexed mask reproduces the chunk, so the fixture cannot "
        "tell the two layouts apart"
    )

    # All three classification kinds must actually occur, or the loop above is
    # passing because every section took the same branch.
    n_sky_arrays = bin(sky_array_mask).count("1")
    n_sky_dark = sum(1 for packed in parsed["sky_arrays"]
                     if all(value == 0 for value in unpack_nibbles(packed)))
    kinds = {
        "sky array": n_sky_arrays - n_sky_dark,
        "sky dark array": n_sky_dark,
        "sky daylight": codec.section_count - n_sky_arrays,
    }
    assert kinds["sky array"] >= 1, "no section needed a mixed sky light array"
    assert kinds["sky dark array"] >= 1, "no section was uniformly dark"
    assert kinds["sky daylight"] >= 1, "no section was uniform daylight"
    assert bin(block_array_mask).count("1") >= 1, "the torch did not light anything"

    # The torch itself: its own cell is lit, and one step away is one darker.
    # A mask-only check would pass with the seed cell written somewhere else.
    # The position is found by scanning the column rather than imported from
    # the fixture, so the fixture cannot move the torch and keep the check
    # green by moving the expectation with it.
    torch_y = None
    for y in range(chunk.min_y, chunk.min_y + chunk.height):
        if int(chunk.get_block(8, y, 8)) == TORCH:
            torch_y = y
            break
    assert torch_y is not None, "the fixture's torch is not in column (8, 8)"
    torch_section = (torch_y - chunk.min_y) // 16
    torch_cell = 8 + 8 * 16 + ((torch_y - chunk.min_y) % 16) * 256
    torch_light = client_light_for_section(parsed, "block_light", torch_section)
    assert torch_light[torch_cell] == TORCH_LEVEL, (
        f"torch cell reads {torch_light[torch_cell]}, want {TORCH_LEVEL}"
    )
    neighbour_cell = 9 + 8 * 16 + ((torch_y - chunk.min_y) % 16) * 256
    assert torch_light[neighbour_cell] == TORCH_LEVEL - 1, (
        f"next to the torch reads {torch_light[neighbour_cell]}, want {TORCH_LEVEL - 1}"
    )

    print(f"  light OK (bit = light section, {kinds['sky array']} mixed + "
          f"{kinds['sky dark array']} dark sky arrays, {kinds['sky daylight']} daylight "
          f"sections free, block light in {bin(block_array_mask).count('1')} section(s), "
          f"all cells match the chunk, {shifted_mismatch} cells would differ if the "
          f"mask were indexed by block section)")

    print("\nAll chunk codec checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
