"""
Chunk Codec - encodes a `Chunk` into the 1.21.11 `LevelChunkWithLight` payload.

This module is pure: `Chunk` in, `bytes` out. It holds no state about players,
view distance or send ordering -- that lives in `world/chunk_streamer.py`.

Wire format (protocol 774 / MC 1.21.11)
--------------------------------------
    i32 BE          chunk X
    i32 BE          chunk Z
    VarInt 3        heightmap count
      VarInt idx ; VarInt len ; len x i64 BE     (x3, see _encode_heightmaps)
    VarInt len      length of the sections blob
      per section:
        i16 BE      non-air block count
        u8          bits per entry
        VarInt      palette size, or the single value when bits == 0
        ...         palette entries / packed longs
        (then the identical structure for biomes, 64 entries)
    VarInt 0        block entity count
    light data      (see _encode_light)

Two details drive most of the design here:

1. **Sections have no Y on the wire.** The client infers position from array
   order, so a 24-section dimension needs 24 entries written. Empty sections
   use the homogeneous encoding (4 bytes) instead of being omitted, which is
   what makes this affordable: 24 x 4096 bytes of air becomes 24 x ~4.

2. **Light is read, not derived.** `ChunkSection.sky_light` / `.block_light`
   hold the packed nibbles the flood-fill engine (`world/light_engine.py`)
   writes, in exactly the format this module puts on the wire, so encoding is
   a classification -- uniform 15, uniform 0, or a real array -- and never a
   recomputation. `_legacy_sky_light` is the fallback for a chunk that was
   never lit (`Chunk.light_populated` is False), and that is **not** only a
   fixture path: `chunk_streamer` sends a chunk as soon as it is generated,
   while `DimensionLight` lights about one chunk per tick, so a chunk that just
   entered the player's view ships the height approximation and is corrected by
   the `CLightUpdate` that follows its patch. The 21:25 session's log has 81 of
   the 488 chunks the player received provably in that state.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
from loguru import logger

from network.buffer import ByteBuffer
from world import palette

# Palette encoding thresholds, from the 1.21.11 protocol constants.
# A palette that would need more than MAX_MAP_BITS per entry is sent "direct",
# with the raw state IDs packed straight into the data array.
BLOCK_MIN_MAP_BITS = 4
BLOCK_MAX_MAP_BITS = 8
BLOCK_MAX_BITS = 15

BIOME_MIN_MAP_BITS = 1
BIOME_MAX_MAP_BITS = 3
BIOME_MAX_BITS = 7

# One 4x4x4x4 cell per biome entry within a section.
BIOME_ENTRIES = 64

# Heightmap.Type ordinals as the client expects them.
HEIGHTMAP_WORLD_SURFACE = 1
HEIGHTMAP_MOTION_BLOCKING = 4
HEIGHTMAP_MOTION_BLOCKING_NO_LEAVES = 5

# Heightmap entries are 9 bits wide (the -64..320 range needs 9 bits), so seven
# fit in one int64 and 256 columns need 37.
HEIGHTMAP_BITS = 9
HEIGHTMAP_ENTRIES = 256
HEIGHTMAP_LONGS = (HEIGHTMAP_ENTRIES + (64 // HEIGHTMAP_BITS) - 1) // (64 // HEIGHTMAP_BITS)

# 4096 blocks at 4 bits each.
LIGHT_BYTES = 2048

# Index into `synced_registries.json`'s `worldgen/biome` (the order the client
# receives them in). 40 is `plains`, and that is the value the registry reports
# for `minecraft:plains` - `default_biome_id()` reads it from there, so the two
# cannot drift; this constant is the fallback for a caller with no registry.
#
# It is the fallback for a chunk that carries no biome volume too: a region file
# written before biomes existed, or a `Chunk` built by hand. `Chunk.biomes` is
# now filled by the generator from `world/vanilla/biome.py`, and each section is
# encoded from its own 64 cells.
BIOME_PLAINS = 40


def default_biome_id() -> int:
    """`minecraft:plains` as the client's biome index, from the registry."""
    try:
        from registry import get_game_registry

        worldgen = get_game_registry().worldgen
        if worldgen is not None:
            resolved = worldgen.biome_id("minecraft:plains")
            if resolved is not None:
                return resolved
    except Exception as error:  # pragma: no cover - only a broken install
        logger.warning(f"Biome id unavailable from the registry, using {BIOME_PLAINS}: {error}")
    return BIOME_PLAINS


# Vectorised lookup from the ids stored in a ChunkSection to what goes on the
# wire used to live here: sections held 1.8-era enum ids and this translated
# them. Sections now hold global block state ids - the number the palette wants
# - so encoding is `np.unique` and nothing else.

# Local Y within a section, broadcast against the (z, x) plane. The section
# array is indexed `x + z * 16 + y * 256`, which is exactly `reshape(16, 16, 16)`
# with axes (y, z, x).
_LOCAL_Y = np.arange(16, dtype=np.int32)[:, None, None]

# A section entirely above the terrain is uniform full daylight: every one of
# the 2048 bytes is 0xFF. Built once and shared, since most sections in a
# typical column fall in this range.
_FULL_SKY_LIGHT = b"\xff" * LIGHT_BYTES

# The same uniform states as the packed bytes `ChunkSection` stores, for the
# paths that have no section behind them (a packet section outside the chunk's
# window, or a section that was never lit). Built once and shared: most
# sections in a typical column are one of these two.
_FULL_SKY_VALUES = _FULL_SKY_LIGHT
_NO_LIGHT_VALUES = b"\x00" * LIGHT_BYTES


def _pack_bits(indices: np.ndarray, bits: int) -> bytes:
    """Pack fixed-width entries into a big-endian int64 array.

    At protocol 774 the data array carries no length prefix -- the reader
    derives the count from `bits_per_entry` -- so this returns exactly the
    bytes the client will read and nothing more. The layout itself is shared
    with region files; see `world/palette.py`.
    """
    return palette.pack(indices, bits)


def _encompassing_bits(count: int) -> int:
    """Bits needed to index `count` palette entries."""
    return palette.bits_for(count)


def _column_heights(chunk) -> np.ndarray:
    """Highest non-air block Y per column, as a (16, 16) [z, x] array.

    World Y - the same absolute coordinates `Chunk.heightmap` carries - so a
    block at the overworld floor reads as -64 rather than 0. Columns with
    nothing in them get `min_y - 1`, matching vanilla's "raw 0 means the
    column is empty" convention.

    Computed straight from the section arrays rather than via
    `Chunk.get_height_at`, which walks sections in a Python loop and would
    dominate encode time.
    """
    sections = chunk.sections
    min_y = chunk.min_y
    heights = np.full((16, 16), min_y - 1, dtype=np.int32)
    found = np.zeros((16, 16), dtype=bool)

    for index in range(len(sections) - 1, -1, -1):
        if found.all():
            break
        blocks = sections[index].blocks
        grid = (blocks != 0).reshape(16, 16, 16)  # [y, z, x]
        occupied = grid.any(axis=0)  # [z, x]
        fresh = occupied & ~found
        if not fresh.any():
            continue
        # Highest occupied local Y per column: argmax over the y-descending
        # view finds the first hit, so 15 - that index is the top block. This
        # has to run on the 3-D grid -- collapsing to [z, x] first would throw
        # away the Y axis the argmax needs.
        top = 15 - np.argmax(grid[::-1, :, :], axis=0)  # [z, x]
        heights[fresh] = min_y + index * 16 + top[fresh]
        found |= occupied

    return heights


def _encode_block_palette(blocks: np.ndarray) -> Tuple[int, List[int], bytes, int]:
    """Encode one section's blocks.

    Returns `(bits_per_entry, palette, packed_data, non_air_count)`.

    The array already holds global block state ids, which is exactly what a
    chunk palette carries, so there is no translation step and no lookup table:
    the ids were produced by the same registry the client's ids come from.

    `bits_per_entry == 0` means homogeneous: the palette holds the single value
    and `packed_data` is empty.
    """
    states = blocks.astype(np.uint16, copy=False)

    palette, inverse = np.unique(states, return_inverse=True)
    non_air = int(np.count_nonzero(states))

    if palette.size == 1:
        return 0, [int(palette[0])], b"", non_air

    raw_bits = _encompassing_bits(palette.size)
    if raw_bits > BLOCK_MAX_MAP_BITS:
        # Too many distinct states to index: send the state IDs themselves.
        return BLOCK_MAX_BITS, [], _pack_bits(states.astype(np.uint64), BLOCK_MAX_BITS), non_air

    bits = max(raw_bits, BLOCK_MIN_MAP_BITS)
    return bits, [int(v) for v in palette], _pack_bits(inverse, bits), non_air

def _encode_biome_palette(cells: Optional[np.ndarray], default_biome: int) -> Tuple[int, List[int], bytes]:
    """Encode the 64 biome cells of one section.

    Same paletted container as blocks, a different size and a different set of
    protocol limits: 4x4x4 cells instead of 4096, `MIN=1 / MAX=3` map bits
    instead of `4 / 8`, and biome ids are written as VarInts but are `u8` wide
    (<= 255), so the direct form caps at 7 bits.

    A section whose 64 cells are all one biome - which is most of them, including
    every section above the terrain and every section in a single-biome chunk -
    takes the homogeneous form: `bits=0` and one VarInt. That is 3 bytes instead
    of 64 entries plus a data array, and it is what the previous
    one-biome-per-chunk encoder always produced.

    `cells` is `[y, z, x]` over the 4x4x4 grid, i.e. the layout
    `Chunk.biomes[section]` already has.
    """
    if cells is None:
        return 0, [default_biome], b""

    flat = np.ascontiguousarray(cells, dtype=np.int32).reshape(-1)
    palette, inverse = np.unique(flat, return_inverse=True)
    count = int(palette.size)
    if count == 1:
        return 0, [int(palette[0])], b""

    raw_bits = max(1, (count - 1).bit_length())
    if raw_bits > BIOME_MAX_BITS:
        # No biome registry is that large; a section that somehow needs it is a
        # data problem, not a case to invent an encoding for. `worldgen`
        # validation reports the registry size, and 65 biomes never reaches this.
        return BIOME_MAX_BITS, [default_biome], b""

    bits = max(raw_bits, BIOME_MIN_MAP_BITS)
    if bits > BIOME_MAX_MAP_BITS:
        # Direct mode: the biome ids themselves, no palette.
        return BIOME_MAX_BITS, [], _pack_bits(flat.astype(np.uint64), BIOME_MAX_BITS)
    return bits, [int(v) for v in palette], _pack_bits(inverse.astype(np.uint64), bits)


def _legacy_sky_light(section_base_y: int, col_heights: np.ndarray) -> np.ndarray:
    """Height-derived sky light for one section, as 4096 nibble values.

    **The fallback, not the engine.** A column is lit at and above the cell
    directly above its highest block, which is what the flood fill produces for
    open ground (`raw = highest_block_y + 1`): full brightness above the
    surface, zero below, no intermediate values and no block light.
    `_collect_light` uses it only for a chunk whose light was never populated,
    so that a hand-built chunk still ships something sane instead of 24 sections
    of black.

    The `+ 1` is the whole difference between a surface that is lit and a
    surface that is black: the value the client reads for a block's *top face*
    is the light of the cell above it, so a rule that starts one cell higher
    renders every top face in the chunk at 0.

    Kept rather than deleted because the two cases it gets right -- open sky
    and solid ground -- are the two a fixture is made of.
    """
    raw = col_heights + 1  # [z, x]
    lit = (section_base_y + _LOCAL_Y) >= raw[None, :, :]
    return np.where(lit, np.uint8(15), np.uint8(0)).reshape(-1)


def _pack_light(values: np.ndarray) -> bytes:
    """Pack 4096 four-bit light values into 2048 bytes, low nibble first."""
    nibbles = np.asarray(values, dtype=np.uint8).reshape(-1)
    return (nibbles[0::2] | (nibbles[1::2] << 4)).astype(np.uint8).tobytes()


def unpack_light(packed, out: Optional[np.ndarray] = None) -> np.ndarray:
    """Unpack 2048 packed bytes into 4096 nibble values, low nibble first.

    The inverse of `_pack_light`, and public because more than one place needs
    to look at a light array cell by cell: `world/light_engine.py` before it
    floods, and the checks that compare a section against what went on the
    wire. The layout is the one `ChunkSection` stores and `world/palette.py`
    writes: cell `x + z*16 + y*256`, even cells in the low nibble.

    Reads bytes rather than relying on numpy, so a `bytes` object and a uint8
    array are both acceptable -- the two forms are the same thing and the
    callers do not agree on which they hold.
    """
    if not isinstance(packed, np.ndarray):
        packed = np.frombuffer(packed, dtype=np.uint8)
    if out is None:
        out = np.empty(4096, dtype=np.uint8)
    out[0::2] = packed & 0x0F
    out[1::2] = packed >> 4
    return out


#: The three answers `_classify_light` can give. Named rather than written as
#: bare 0/15/-1, because the two uniform answers have *opposite* meanings in
#: the two empty masks and a bare number hides that.
_LIGHT_NONE = 0
_LIGHT_FULL = 15
_LIGHT_MIXED = -1


def _classify_light(packed) -> int:
    """How one section's light array goes on the wire: 0, 15, or -1 for mixed.

    `packed` is the 2048-byte nibble form, as bytes or as the uint8 array
    `ChunkSection` holds -- both callers exist (a section's own array, and the
    height fallback, which builds bytes), and normalising here keeps them from
    having to agree on a type.

    "Uniform" is decided at the **byte** level, and that is not an
    optimisation: a byte holds two cells, so `bytes[i] == first` for all i is
    exactly "every cell has the same value" *only* when the two nibbles of
    `first` are equal. `first | (first << 4)` is the byte for `first` repeated
    in both nibbles, so testing against that covers both halves at once.

    Getting this wrong is silent and it is the bug this function was written
    to fix: `(byte & 0x0F) | (byte >> 4)` combines the two cells of a byte
    instead of decoding them, so `0x0F` (= cells 15 and 0) reads as 15, and a
    section that is half dark is declared uniform daylight -- its array is never
    sent, and the client fills the whole section with 15 it never received.
    """
    if not isinstance(packed, np.ndarray):
        packed = np.frombuffer(packed, dtype=np.uint8)
    first = int(packed[0])
    if bool(np.all(packed == first)):
        low, high = first & 0x0F, first >> 4
        if low == high:
            return low
    return _LIGHT_MIXED


def _collect_light(
    chunk,
    col_heights: np.ndarray,
    section_count: int,
    min_y: int,
    server_offset: int,
) -> bytes:
    """Build the light block (four masks + the arrays) from the chunk's light.

    **The bit a section goes in is not its section index.** The four masks cover
    `section_count + 2` light sections, not `section_count`: the client's window
    runs from one section *below* the dimension's `min_y` to one section *above*
    its ceiling (`LightEngine`: `minLightSection = minSectionY - 1`,
    `lightSectionCount = sectionsCount + 2`), so

        bit 0                -> y in [min_y - 16, min_y - 1]   (below the world)
        bit j, 1 <= j <= n   -> block section j - 1             (the chunk's own)
        bit n + 1            -> y in [min_y + height, ...]      (above the world)

    The protocol says the same thing in words: "the least significant bit is for
    blocks 16 blocks to 1 block below the min world height". Writing section `s`
    into bit `s` therefore lands every array **16 blocks too low** on the client,
    which is what this encoder used to do: the client then shades the surface
    with the light of the section above it (uniform daylight, no shadows at all)
    and every torch's pool appears a section below the torch. The two extra
    sections at the ends are left unmentioned -- the client's default for a
    section it has no data for is exactly right there (15 for sky, 0 for block)
    and there are no blocks in either of them.

    The polarity of the two empty masks is not symmetric:

    * `sky light` -- sections carrying a real sky light array. A section that is
      uniformly *dark* has nowhere else to say so: "no bit" means 15, so an
      unlit underground section ships 2048 bytes of zeroes. That is the price
      the format charges, and the reason a typical column sends two to five sky
      arrays -- the interesting ones -- and nothing for the daylight above the
      terrain. `empty sky` is never written: sources disagree on what that bit
      means (the field name and vanilla's use of it say "no data, i.e. 15"; the
      protocol wiki says "all zeros"), and both encodings this codec does use --
      no bit for daylight, an explicit array for anything darker -- read the
      same way under either description.
    * `block light` -- the same for block light, with the polarity of the empty
      mask flipped: a **1** bit in `empty block` means the client fills the
      section with 0, which is what makes "this section has no block light"
      free. A section where every cell is 15 has no such encoding and needs a
      real array -- rarer in practice than a dark one, and the reason the two
      encodings are not mirror images.
    """
    sky_arrays: List[bytes] = []
    block_arrays: List[bytes] = []
    sky_mask = 0
    block_mask = 0
    #: Always 0 -- see the docstring: daylight is "no bit", darkness is an
    #: explicit array. The wire still carries the field (`_write_light` writes
    #: four masks, not three).
    sky_empty_mask = 0
    block_empty_mask = 0

    # Read from the chunk's own sections, falling back to the height
    # approximation only when nobody has lit this chunk. See `_legacy_sky_light`.
    populated = bool(getattr(chunk, "light_populated", True))

    for packet_index in range(section_count):
        # +1: the packet's block section 0 is light section 1. See the docstring.
        bit = packet_index + 1
        server_index = packet_index + server_offset
        section = None
        if 0 <= server_index < len(chunk.sections):
            section = chunk.sections[server_index]

        if populated:
            sky_nibbles = _FULL_SKY_LIGHT if section is None else section.sky_light
            block_nibbles = _NO_LIGHT_BYTES if section is None else section.block_light
        else:
            # Nobody lit this chunk. The blocks are still there, so the height
            # approximation is the best available answer for sky; block light
            # has no such fallback (nothing in the block data says where the
            # torches would have lit), so it stays at whatever the section
            # holds -- the default arrays, i.e. dark.
            if section is None:
                sky_nibbles = _FULL_SKY_LIGHT
                block_nibbles = _NO_LIGHT_BYTES
            else:
                base_y = min_y + packet_index * 16
                sky_nibbles = _pack_light(_legacy_sky_light(base_y, col_heights))
                block_nibbles = section.block_light

        sky_class = _classify_light(sky_nibbles)
        if sky_class == _LIGHT_FULL:
            # No bit at all: a section the client has no data for reads as 15,
            # which is exactly what "the daylight above the terrain" is, and it
            # is the only sky encoding that is unambiguous -- sources disagree
            # on what an `empty sky` bit means (15, or all zeros), so this
            # encoder never writes one. The cost is that a section which *was*
            # darker and has become uniform daylight cannot be corrected by a
            # later `CLightUpdate`: "no bit" is also "leave what you have",
            # which is right until it is stale. Resending the chunk fixes it.
            pass
        else:
            # Everything that is not full daylight needs a real array, a dark
            # section included: "no bit" means 15, so an array of zeros is the
            # only way to state 0.
            sky_mask |= 1 << bit
            sky_arrays.append(bytes(sky_nibbles))

        block_class = _classify_light(block_nibbles)
        if block_class == _LIGHT_NONE:
            # 1 bit in `empty block` is "0 everywhere", and this is the common
            # case for every section without a light source in it.
            block_empty_mask |= 1 << bit
        else:
            block_mask |= 1 << bit
            block_arrays.append(bytes(block_nibbles))

    return _write_light(
        sky_mask=sky_mask,
        block_mask=block_mask,
        sky_empty_mask=sky_empty_mask,
        block_empty_mask=block_empty_mask,
        sky_arrays=sky_arrays,
        block_arrays=block_arrays,
    )


def _write_light(
    sky_mask: int,
    block_mask: int,
    sky_empty_mask: int,
    block_empty_mask: int,
    sky_arrays: List[bytes],
    block_arrays: List[bytes],
) -> bytes:
    """Serialise the four masks and their arrays, in the order 774 reads them.

    A BitSet is a VarInt long count followed by that many big-endian int64s,
    least significant long first -- `BitSet.toLongArray`. One long covers every
    dimension this server has (26 light sections in the overworld, 18 in the
    nether), but the count comes from the values rather than a hardcoded 1, and
    every mask is padded to the same width, so a dimension taller than 64
    sections would still encode correctly.
    """
    highest = max(sky_mask, block_mask, sky_empty_mask, block_empty_mask)
    longs = max(1, (highest.bit_length() + 63) // 64)

    def bitset(value: int) -> bytes:
        return b"".join(
            ((value >> (64 * index)) & 0xFFFFFFFFFFFFFFFF).to_bytes(8, "big")
            for index in range(longs)
        )

    buf = ByteBuffer()
    buf.write_varint(longs)
    buf.write_bytes(bitset(sky_mask))
    buf.write_varint(longs)
    buf.write_bytes(bitset(block_mask))
    buf.write_varint(longs)
    buf.write_bytes(bitset(sky_empty_mask))
    buf.write_varint(longs)
    buf.write_bytes(bitset(block_empty_mask))

    buf.write_varint(len(sky_arrays))
    for data in sky_arrays:
        buf.write_varint(len(data))
        buf.write_bytes(data)

    buf.write_varint(len(block_arrays))
    for data in block_arrays:
        buf.write_varint(len(data))
        buf.write_bytes(data)

    return bytes(buf)


class ChunkCodec:
    """Encodes chunks for one dimension.

    A codec is cheap to build but not free (it computes a few lookup tables),
    so keep one per dimension rather than one per packet.
    """

    def __init__(
        self,
        min_y: int = -64,
        height: int = 384,
        default_biome: int = BIOME_PLAINS,
    ):
        self.min_y = min_y
        self.height = height
        self.default_biome = default_biome

        self.section_count = height // 16

    def _write_section(self, buf: ByteBuffer, bits: int, palette: List[int], packed: bytes) -> None:
        """Write one paletted container (used for both blocks and biomes).

        `bits == 0` is the homogeneous case: a single VarInt value and no data
        array. That is what makes an all-air section cost 4 bytes instead of
        the 4096 a full data array would need.
        """
        buf.write_byte(bits)
        if bits == 0:
            buf.write_varint(palette[0])
        elif bits != BLOCK_MAX_BITS:
            # Direct mode writes no palette, just the packed state IDs.
            buf.write_varint(len(palette))
            for value in palette:
                buf.write_varint(value)
        if packed:
            buf.write_bytes(packed)

    def _encode_sections(self, chunk, col_heights: np.ndarray) -> Tuple[bytes, int]:
        """Build the sections blob. Returns `(blob, non_air_total)`.

        Light is *not* encoded here any more: it lives in `_collect_light`,
        which reads the chunk's own light arrays instead of deriving them from
        `col_heights`. The heights are still needed for the heightmaps, which
        is why they are still a parameter.
        """
        buf = ByteBuffer()
        non_air_total = 0

        air = [0]
        biomes = getattr(chunk, "biomes", None)

        # Packet section i holds absolute Y = self.min_y + i*16; chunk section j
        # holds chunk.min_y + j*16. Chunk windows normally match the dimension,
        # making the two indexes equal - but a hand-built or deserialized chunk
        # can differ, so the mapping is computed per chunk rather than assumed.
        chunk_min_y = getattr(chunk, "min_y", 0)
        server_offset = (self.min_y - chunk_min_y) // 16

        for packet_index in range(self.section_count):
            server_index = packet_index + server_offset
            section = None
            if 0 <= server_index < len(chunk.sections):
                section = chunk.sections[server_index]

            if section is None or not section.blocks.any():
                # Nothing here. Skip the palette machinery entirely -- running
                # np.unique over 4096 zeros costs more than every other part of
                # the encode put together, and 20-odd of a 24-section column
                # are air. Emptiness is read off the array rather than trusting
                # ChunkSection._non_air_count, which the Anvil reader can leave
                # stale.
                buf.write_short(0)
                self._write_section(buf, 0, air, b"")
            else:
                bits, palette, packed, non_air = _encode_block_palette(section.blocks)
                buf.write_short(non_air)
                non_air_total += non_air
                self._write_section(buf, bits, palette, packed)

            cells = None
            if biomes is not None and 0 <= server_index < biomes.shape[0]:
                cells = biomes[server_index]
            # `cells is None` is "this chunk has no biome data", not "biome 0".
            b_bits, b_palette, b_packed = _encode_biome_palette(cells, self.default_biome)
            self._write_section(buf, b_bits, b_palette, b_packed)

        return bytes(buf), non_air_total

    def _encode_heightmaps(self, col_heights: np.ndarray) -> bytes:
        """Write the three heightmaps the client expects.

        All three use the same column data for now: the server tracks one
        height, not one per predicate. Entries are 9-bit, indexed `z * 16 + x`,
        which is the order vanilla's `Heightmap` uses over its dense array.
        """
        values = (col_heights + 2).reshape(-1).astype(np.uint64)
        packed = _pack_bits(values, HEIGHTMAP_BITS)

        buf = ByteBuffer()
        buf.write_varint(3)
        for index in (
            HEIGHTMAP_WORLD_SURFACE,
            HEIGHTMAP_MOTION_BLOCKING,
            HEIGHTMAP_MOTION_BLOCKING_NO_LEAVES,
        ):
            buf.write_varint(index)
            buf.write_varint(HEIGHTMAP_LONGS)
            buf.write_bytes(packed)
        return bytes(buf)

    def _encode_light(self, chunk, col_heights: np.ndarray) -> bytes:
        """The light block for one chunk: four masks, then the arrays.

        A thin wrapper over `_collect_light` that supplies the two mappings
        (chunk section index -> packet section index) and the fallback heights.
        `encode` and `encode_light` both come through here, which is what keeps
        the tail of a full chunk packet and the body of a `CLightUpdate`
        byte-identical -- `tools/check_chunk_packets.py` pins that.
        """
        chunk_min_y = getattr(chunk, "min_y", 0)
        return _collect_light(
            chunk,
            col_heights=col_heights,
            section_count=self.section_count,
            min_y=self.min_y,
            server_offset=(self.min_y - chunk_min_y) // 16,
        )

    def encode(self, chunk) -> bytes:
        """Encode a `Chunk` as a full `LevelChunkWithLight` payload."""
        col_heights = _column_heights(chunk)

        buf = ByteBuffer()
        buf.write_int(chunk.position.x)
        buf.write_int(chunk.position.z)
        buf.write_bytes(self._encode_heightmaps(col_heights))

        blob, _non_air = self._encode_sections(chunk, col_heights)
        buf.write_varint(len(blob))
        buf.write_bytes(blob)

        buf.write_varint(0)  # block entities

        buf.write_bytes(self._encode_light(chunk, col_heights))
        return bytes(buf)

    def encode_light(self, chunk) -> bytes:
        """Encode just the light block for an already-sent chunk.

        This is the body of a `CLightUpdate` packet. `tools/check_chunk_packets.py`
        pins `encode(chunk).endswith(encode_light(chunk))`; both go through
        `_encode_light` so the two layouts cannot drift.
        """
        return self._encode_light(chunk, _column_heights(chunk))
