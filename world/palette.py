"""
Packed palette arrays - the bit layout a chunk section shares between an
anvil region file and a chunk packet.

A palette stores indices into a small list of values, `bits` per entry, packed
into big-endian int64s. Minecraft uses this one layout in both places, with a
rule that is easy to get wrong: **an entry never straddles a long boundary**.
`64 // bits` entries fit in each long and the leftover bits are padding, so a
5-bit palette wastes 4 bits per long instead of letting entry 12 continue into
the next long. The padding is why the packed length is `ceil(count * bits / 64)`
only by accident and must be computed from the entries per long.

`world/chunk_codec.py` writes this for the wire and `world/anvil.py` for region
files; both go through here so the two cannot drift apart, and so a reader has
exactly one inverse to get right.
"""

from __future__ import annotations

from typing import List, Sequence, Union

import numpy as np


def bits_for(count: int) -> int:
    """Bits needed to index `count` palette entries; 0 for a single entry."""
    if count <= 1:
        return 0
    return (count - 1).bit_length()


def entries_per_long(bits: int) -> int:
    """How many entries fit in one int64 at this width (0 when there is no data)."""
    if bits <= 0:
        return 0
    return 64 // bits


def pack(indices: Sequence[int], bits: int) -> bytes:
    """Pack `indices` into big-endian int64s, `bits` wide, first entry lowest.

    Args:
        indices: Palette indices, in the order the consumer will read them back.
        bits: Entry width. 0 means a homogeneous palette: no data array at all.

    Returns:
        The packed bytes, `ceil(len(indices) / entries_per_long) * 8` long.
    """
    if bits <= 0:
        return b""

    flat = np.asarray(indices).reshape(-1)
    per_long = entries_per_long(bits)
    n_longs = (flat.size + per_long - 1) // per_long

    padded = np.zeros(n_longs * per_long, dtype=np.uint64)
    padded[: flat.size] = flat

    shifts = np.arange(per_long, dtype=np.uint64) * np.uint64(bits)
    packed = np.bitwise_or.reduce(padded.reshape(n_longs, per_long) << shifts, axis=1)
    return packed.astype(">u8").tobytes()


def pack_longs(indices: Sequence[int], bits: int) -> List[int]:
    """`pack` as the signed int64s an NBT `LongArray` holds."""
    data = pack(indices, bits)
    if not data:
        return []
    # `>i8` to `int`: the values are bit patterns, and NBT stores them signed.
    return [int(value) for value in np.frombuffer(data, dtype=">i8")]


def unpack(data: Union[bytes, bytearray, Sequence[int], np.ndarray], bits: int, count: int) -> np.ndarray:
    """Inverse of `pack`. Returns `count` values as `uint16`.

    Args:
        data: Packed bytes, or an NBT `LongArray`'s list of signed int64s.
        bits: Entry width, as written.
        count: How many entries to return (4096 for a section).

    Raises:
        ValueError: If `bits` is 0, which means the palette has a single entry
            and no data array - the caller has to use that entry directly.
    """
    if bits <= 0:
        raise ValueError("a homogeneous palette has no data array to unpack")

    if isinstance(data, (bytes, bytearray)):
        words = np.frombuffer(bytes(data), dtype=">u8")
    else:
        # A list of signed int64s from NBT. `view` keeps the bit pattern; astype
        # would be a value conversion and negative longs would be a problem.
        words = np.asarray(data, dtype=np.int64).view(np.uint64).reshape(-1)

    per_long = entries_per_long(bits)
    shifts = np.arange(per_long, dtype=np.uint64) * np.uint64(bits)
    mask = np.uint64((1 << bits) - 1)
    values = (words[:, None] >> shifts) & mask
    return values.reshape(-1)[:count].astype(np.uint16)


__all__ = ["bits_for", "entries_per_long", "pack", "pack_longs", "unpack"]
