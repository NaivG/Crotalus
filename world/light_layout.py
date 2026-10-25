"""
Light storage layout - the packed-nibble section format, and nothing else.

One section's light is **2048 bytes holding 4096 four-bit values**, one nibble per
cell, and this module is the only place that knows it. Three places in the
repository already read or write exactly these bytes, and they are the reason to
give the arithmetic one owner instead of a copy per consumer:

* `world/chunk.py` -- `ChunkSection.sky_light` / `.block_light` are those two
  2048-byte arrays, and a cell sits at the same `x + z*16 + y_local*256` index
  `ChunkSection.blocks` uses;
* `world/chunk_codec.py` -- `_pack_light` turns 4096 nibbles into the 2048 bytes
  of the `LevelChunkWithLight` light block, low nibble first;
* `world/anvil.py` -- the same buffer, written to and read from a region file as an
  NBT byte array.

So "which byte holds cell 2047" has one answer and it is here; everything else
imports `SECTION_BYTES` / `CELL_COUNT` rather than writing 2048 down again.

Values are 0..15 and nothing else is meaningful: 15 is full light, 0 is none.
Nibble order is **low half first**: `packed[i]` carries cell `2*i` in its low
nibble and cell `2*i + 1` in its high nibble.

The module imports numpy and nothing else, on purpose: the numba kernel, a
standalone tool and the storage layer all have to be able to import it without a
world, a registry or a logger. It holds no state.

`world/light_engine.py` is the algorithm and `world/light_region.py` is the 3x3
accessor; this file is only the format.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

#: Bits per light value. Also the reason a section is half its cell count in bytes.
LIGHT_BITS = 4

#: The brightest level. A section that is full of it packs as `0xFF` per byte --
#: `0x0F` would be one cell at 15 and the next at 0 -- which is what
#: `ChunkSection.sky_light` is initialised to.
MAX_LIGHT_LEVEL = (1 << LIGHT_BITS) - 1

#: Edge length of a section, in blocks.
SECTION_SIZE = 16

#: Cells in one horizontal layer of a section (the `y_local * 256` stride).
CELLS_PER_LAYER = SECTION_SIZE * SECTION_SIZE

#: Cells in a section: 16 x 16 x 16, stored flat (see `cell_index`).
CELL_COUNT = CELLS_PER_LAYER * SECTION_SIZE

#: Packed values per byte.
NIBBLES_PER_BYTE = 8 // LIGHT_BITS

#: Bytes one section's light occupies, on disk and on the wire. The one number
#: this module exists to own.
SECTION_BYTES = CELL_COUNT // NIBBLES_PER_BYTE

#: Low-nibble mask, spelled once.
_NIBBLE_MASK = MAX_LIGHT_LEVEL


# ==================== coordinates ====================


def cell_index(x: int, y_local: int, z: int) -> int:
    """Flat cell index for section-local coordinates.

    The same layout `ChunkSection` indexes its block array with: `x` varies
    fastest, then `z`, then `y_local`, so `0 <= index < 4096` for in-range
    coordinates and one cell lies in exactly one place.
    """
    return x + z * SECTION_SIZE + y_local * CELLS_PER_LAYER


def cell_coords(cell: int) -> Tuple[int, int, int]:
    """`(x, y_local, z)` of a flat cell index - the inverse of `cell_index`."""
    x = cell % SECTION_SIZE
    z = (cell % CELLS_PER_LAYER) // SECTION_SIZE
    return x, cell // CELLS_PER_LAYER, z


def section_index(y: int, min_y: int) -> int:
    """Index into `Chunk.sections` holding world Y `y`.

    `//` floors, so a `y` below the window gives a negative index rather than a
    wrapped one: the caller is expected to reject it. Overworld `min_y` is -64,
    which is what makes this more than a shift.
    """
    return (y - min_y) // SECTION_SIZE


def section_base_y(section: int, min_y: int) -> int:
    """World Y of the lowest layer of a section (the inverse of `section_index`)."""
    return min_y + section * SECTION_SIZE


def local_y(y: int, min_y: int) -> int:
    """`y` within its section, 0..15 - the `y_local` of `cell_index`."""
    return (y - min_y) % SECTION_SIZE


def split_y(y: int, min_y: int) -> Tuple[int, int]:
    """`(section_index, y_local)` for a world Y, the pair the kernels carry."""
    return section_index(y, min_y), local_y(y, min_y)


def section_count(height: int) -> int:
    """Sections a dimension of this height is made of (`384 -> 24`).

    Rejects a height that is not a whole number of sections, which is the same
    rule `Chunk.__post_init__` enforces.
    """
    if height <= 0 or height % SECTION_SIZE:
        raise ValueError(
            f"height must be a positive multiple of {SECTION_SIZE}, got {height}"
        )
    return height // SECTION_SIZE


# ==================== packing ====================


def _as_section_bytes(packed, label: str) -> np.ndarray:
    """A flat uint8 view of `packed`, checked to hold exactly one section."""
    if isinstance(packed, (bytes, bytearray, memoryview)):
        # `bytes` is a buffer, not a list of integers: `np.asarray` would make a
        # 0-d array out of it, which is the classic way to get a silent size of 1.
        array = np.frombuffer(packed, dtype=np.uint8)
    else:
        array = np.asarray(packed, dtype=np.uint8)
    array = array.reshape(-1)
    if array.size != SECTION_BYTES:
        raise ValueError(
            f"{label} must hold {SECTION_BYTES} bytes, got {array.size}"
        )
    return array


def _write_target(out, count: int, label: str) -> np.ndarray:
    """The array unpack/pack fills: a fresh one, or `out` written in place.

    The result has to be the caller's own array, so the shape is checked exactly
    rather than flattened: a reshape of a non-contiguous input returns a copy, and
    a copy is a write that silently goes nowhere.
    """
    if out is None:
        return np.empty(count, dtype=np.uint8)
    array = out if isinstance(out, np.ndarray) else np.frombuffer(out, dtype=np.uint8)
    if array.dtype != np.uint8 or array.shape != (count,):
        raise ValueError(
            f"{label} buffer must be a {count}-element uint8 array, "
            f"got {array.shape} of {array.dtype}"
        )
    if not array.flags.writeable:
        raise ValueError(f"{label} buffer is read-only")
    return array


def unpack_light(packed: np.ndarray, out: Optional[np.ndarray] = None) -> np.ndarray:
    """Expand 2048 packed bytes into the 4096 nibble values they hold.

    `cells[2*i]` is the low nibble of `packed[i]` and `cells[2*i + 1]` the high
    one -- the order `world/chunk_codec._pack_light` writes and `world/anvil`
    stores. The result is indexed by `cell_index`, so
    `cells[cell_index(x, y_local, z)]` is the light at that cell.

    `out`, when given, is filled in place and returned: a `(9, 24, 4096)` region
    buffer can take one section per call without a temporary per section.

    **This is the reference implementation, not the fast one.** The strided
    stores below measure **~10 us** per section, and `LightRegion.build` makes
    432 of them -- ~6.5 ms, which is more than the whole light kernel costs.
    `world/light_region.py` therefore unpacks with an njit loop
    (`light_region._unpack_section`, ~0.5 us) and this stays where it is on
    purpose: it is the thing the fast path is checked against, in
    `tools/check_light_layout.py`, and it is what the codec and the tools read
    one section at a time. Do not "optimise" it and expect the engine to notice.
    """
    source = _as_section_bytes(packed, "packed light")
    cells = _write_target(out, CELL_COUNT, "cells")
    cells[0::2] = source & _NIBBLE_MASK
    cells[1::2] = source >> LIGHT_BITS
    return cells


def pack_light(cells: np.ndarray, out=None) -> np.ndarray:
    """Pack 4096 nibble values into the 2048 bytes a `ChunkSection` stores.

    The exact inverse of `unpack_light`, and byte for byte the same function as
    `world/chunk_codec._pack_light` -- which it deliberately does not import: the
    codec is a *consumer* of the format, so the dependency has to point this way
    (`tools/check_light_layout.py` pins the two outputs together, so "the same
    format" is asserted rather than assumed). `out` may be a
    `ChunkSection.sky_light` and `out=` writes straight into it.

    Values are expected in 0..15. Above that the shift drops the high bits, which
    is the same behaviour as the codec's packer; the kernel is what keeps light
    clipped.
    """
    values = np.asarray(cells, dtype=np.uint8).reshape(-1)
    if values.size != CELL_COUNT:
        raise ValueError(f"cells must hold {CELL_COUNT} values, got {values.size}")
    packed = _write_target(out, SECTION_BYTES, "packed light")
    np.bitwise_or(values[0::2], values[1::2] << LIGHT_BITS, out=packed)
    return packed


__all__ = [
    "LIGHT_BITS",
    "MAX_LIGHT_LEVEL",
    "SECTION_SIZE",
    "CELLS_PER_LAYER",
    "CELL_COUNT",
    "NIBBLES_PER_BYTE",
    "SECTION_BYTES",
    "cell_index",
    "cell_coords",
    "section_index",
    "section_base_y",
    "local_y",
    "split_y",
    "section_count",
    "unpack_light",
    "pack_light",
]
