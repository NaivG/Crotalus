"""The 3x3 patch accessor: nine chunks' blocks and light as one flat array.

A light patch is computed over a **3x3 block of chunks** and written back to the
middle one only (`LIGHT_PLAN.md` §4.5). That is the whole point of this module:
it assembles the nine chunks into the arrays the kernel wants and puts the
centre chunk's light back, and it does **no algorithm at all** -- the same split
`world/light_layout.py` (format only) and `world/light_engine.py` (algorithm
only) use.

Layout
------

`region[chunk_idx, section_idx, cell]`, with

* `chunk_idx = (dz + 1) * 3 + (dx + 1)`, `dx`/`dz` being the chunk offset from
  the centre chunk -- so the centre is index 4 and the eight neighbours ring it;
* `section_idx = (world_y - min_y) // 16`, the same index `Chunk.sections` and
  `light_layout.section_index` use;
* `cell = x + z*16 + y_local*256`, exactly `ChunkSection.blocks`'s own index
  (`light_layout.cell_index`).

That last point is why this is a copy and not a transposition: element `i` of a
section's block array already means the same cell as element `i` of its light
arrays, so the nine chunks go in with one `array[...] = section.blocks` per
section and nothing is permuted. The `(9, 24, 4096)` block buffer is ~1.1 MB and
the two light buffers ~0.9 MB each (`LIGHT_PLAN.md` §4.1: "about 884 KB"), which
is the price of a kernel that takes plain numpy arrays instead of a list of
arrays -- a `numba.typed.List` of 216 section arrays costs ~222 ms *per call*
(`physics/kernel.py:56`), which would dwarf the patch itself.

Sources are read through `Chunk.raw_sections()`, which is zero-copy and
read-only, and called on the owning thread (`world/chunk.py:412`). Copying into
the region is what makes the kernel's single-array signature possible; the
*chunk's* arrays are never written here except by `write_back`, which writes the
centre chunk through the `ChunkSection` objects that own them.

Section uniformity (`region.air_sections`)
-----------------------------------------

The **third** thing `build` produces, after the blocks and the two light layers,
is a `(9, sections)` uint8 table of `SECTION_SKIP` flags: 1 where a section holds
nothing but air, 0 otherwise. It is the engine's light-data management in the
Starlight sense -- the metadata the kernels were missing.

It is derived from `ChunkSection.non_air_count` (`section.is_empty`,
`world/chunk.py:152-154`), which is maintained by `set_block` / `fill` and
recomputed by `recompute_non_air_count` after a wholesale assignment. That is
the same flag `world/chunk_codec.py` already trusts to decide whether a section
is uniform air on the wire, so nothing new is being believed here: a section
whose `is_empty` lies is already a broken chunk, not a broken light patch
(`world/chunk.py:156-168`).

Why the metadata and not the arrays. `not blocks[index, section_idx].any()` is
the same predicate computed from the buffer the kernel is handed, and it was the
first choice, because then the table provably matches the blocks and no metadata
has to be trusted at all. It measured **+0.74 ms** on a region build (0.60 ms ->
1.34 ms for the 216 copies, `tools/bench_light.py` fixture) against **+0.02 ms**
for 216 `is_empty` reads -- and a region build is paid by every incremental edit
that moves to another chunk, not just by a patch. The two agree exactly on the
fixture (138 of 216 sections flagged, 0 disagreements), and
`tools/check_light_batch.py` pins that agreement on real generated terrain so a
divergence is a test failure rather than a mystery darkness.

Only `relight_region` consumes the table, and only through the region its own
`_light_chunk` built from the current blocks, so an edit cannot leave it stale.
Two different mechanisms keep that true, and it is worth being explicit about
which is which:

* a chunk re-lit after an edit gets a **new region** (`DimensionLight._light_chunk`
  calls `build` every time), so the table is re-derived from the new blocks;
* the region an incremental edit keeps open across ticks does have a table that
  goes stale the moment `DimensionLight` writes a block into `region.blocks` --
  and nothing reads it until the next `build`, because the incremental kernels
  (`update_block_light`, `update_sky_column`) do not take one. The stale window
  is real; it just has no consumer.

Readiness
---------

If any of the nine chunks is missing -- or has a different vertical window --
the region is built with `ready = False` and empty buffers, and the caller must
skip the patch and leave the chunk in its queue. `LIGHT_PLAN.md` §4.5 gives the
reason, and it is the same rule the spawner and the physics view use for an
unloaded chunk: **"unloaded" is a refusal, not air**. Substituting air for a
missing neighbour would seed sky light that nothing else ever corrects, and it
would be written back to the centre chunk as if it were real.
"""

from __future__ import annotations

from typing import Callable, List, Optional

import numpy as np

from world.chunk import Chunk
from world.light_layout import CELL_COUNT, pack_light, unpack_light

try:  # pragma: no cover - numba is a declared dependency
    from numba import njit

    _HAS_NUMBA = True
except ImportError:  # pragma: no cover
    _HAS_NUMBA = False

    def njit(*args, **kwargs):
        def decorator(func):
            return func

        return decorator if not args else decorator(args[0])

#: Edge length of the patch, in chunks.
REGION_EDGE = 3

#: `chunk_idx` of the chunk a patch lights and writes back -- the middle of the 3x3.
CENTER_CHUNK_INDEX = (REGION_EDGE // 2) * REGION_EDGE + (REGION_EDGE // 2)

#: Chunks one patch covers. Also the number of `raw_sections()` calls per patch.
REGION_CHUNKS = REGION_EDGE * REGION_EDGE

#: `air_sections` flags. `SECTION_MIXED` means "the kernels must look at this
#: section"; `SECTION_AIR` means it holds nothing but air, so there is nothing in
#: it for the column-top scan, the emitter scan or the sky fill to find.
SECTION_MIXED = 0
SECTION_AIR = 1

#: `SECTION_AIR`, plus "and the sky fill leaves it at 15" -- which is the second
#: and stronger half, and the reason the flag is worth having. `relight_region`
#: raises a `SECTION_AIR` section to this once pass 1 has told it that every
#: column top in the section's 16x16 footprint is *below* it: then the fill's
#: above-the-top branch would write 15 into every one of its 4096 cells, and the
#: loop can skip a whole section with one `fill` instead.
#:
#: Only the kernel writes this value: `LightRegion.air_sections` comes out of
#: `build` as `SECTION_AIR` / `SECTION_MIXED` and nothing else, so an edit that
#: raises a top is re-decided by the next patch rather than trusted.
SKY_SEEDED = 2


def chunk_index(dx: int, dz: int) -> int:
    """`chunk_idx` of the chunk `dx`/`dz` chunks away from the centre one.

    The +1s are what put the centre at 4 and keep the index non-negative; the
    order (z major, x minor) is the order the three nested loops in
    `LightRegion.build` visit the chunks in, so index `i` is the `i`-th chunk
    appended there.
    """
    return (dz + 1) * REGION_EDGE + (dx + 1)


# ---------------------------------------------------------------------------
# Unpacking
# ---------------------------------------------------------------------------
#
# One nibble expansion is 2048 bytes in and 4096 bytes out, and a 3x3 build does
# 432 of them (9 chunks x 24 sections x 2 layers). `light_layout.unpack_light`
# does it with two strided numpy stores plus the per-call shape validation, which
# measures **~10 us** each -- **~6.5 ms** for a build, more than the entire
# light kernel. The loop below is the same arithmetic with no strided access
# and no validation: **~0.5 us**, so the 432 calls cost ~0.6 ms and the strided
# form's other half (the temporaries it allocated) is gone as well.
#
# It is *not* the format's definition. `light_layout.unpack_light` stays the
# reference and `tools/check_light_layout.py` pins the two together byte for
# byte, which is what makes this a speedup rather than a second format.


@njit(cache=True, inline="always")
def _unpack_section(packed, out):
    """`light_layout.unpack_light` one section, into a region buffer slice.

    `packed` is the section's 2048 stored bytes, `out` the 4096-cell view it
    expands into. Both are 1-D uint8 and both are already contiguous views into
    the region arrays, so this is a straight sequential write with no temporary.
    """
    for i in range(packed.shape[0]):
        byte = packed[i]
        out[2 * i] = byte & 0x0F
        out[2 * i + 1] = byte >> 4


if _HAS_NUMBA:
    _unpack_into = _unpack_section
else:  # pragma: no cover - the pure-Python fallback is far slower than numpy
    def _unpack_into(packed, out):
        """Without numba, the reference implementation is still the fast one."""
        unpack_light(packed, out=out)


class LightRegion:
    """Nine chunks as `(9, sections, 4096)` block and light arrays, plus write-back.

    Build one per patch and throw it away; nothing here is cached or reused.
    The class is deliberately dumb: it copies blocks in, unpacks light in, and
    packs the centre chunk's light back out. It never decides what a light value
    should be.
    """

    __slots__ = (
        "center",
        "chunks",
        "ready",
        "all_lit",
        "min_y",
        "sections",
        "blocks",
        "air_sections",
        "sky",
        "block_light",
        "_scratch",
    )

    def __init__(self, center: Chunk) -> None:
        self.center = center
        #: The nine chunks, index `chunk_index(dx, dz)` order; `None` entries when
        #: `ready` is False.
        self.chunks: List[Optional[Chunk]] = [None] * REGION_CHUNKS
        self.ready: bool = False
        #: True when all nine chunks carry engine-written light
        #: (`Chunk.light_populated`). Only then may a kernel *read* the region's
        #: light: an unlit chunk's arrays are defaults (`sky_light` is
        #: `np.full(2048, 15)`, which is not "dark"), and reading them as if
        #: they were light is how a stale neighbour lights the centre chunk
        #: from nowhere. `LightRegion.build` unpacks an unlit chunk as zeroes
        #: and leaves this False.
        self.all_lit: bool = False
        self.min_y: int = center.min_y
        self.sections: int = len(center.sections)
        #: `(9, sections, 4096)` uint16 block state ids, and the two light layers
        #: as `(9, sections, 4096)` uint8 nibble values. `None` until `build`.
        self.blocks: Optional[np.ndarray] = None
        #: `(9, sections)` uint8 of `SECTION_AIR` / `SECTION_MIXED`: which
        #: sections the kernels may skip. See the module docstring for why this
        #: comes from `ChunkSection.is_empty` rather than from `blocks`.
        self.air_sections: Optional[np.ndarray] = None
        self.sky: Optional[np.ndarray] = None
        self.block_light: Optional[np.ndarray] = None
        # One section's packed bytes, reused by `write_back` so a write-back of 48
        # arrays allocates nothing.
        self._scratch: Optional[np.ndarray] = None

    # ---------------- construction ----------------

    @classmethod
    def build(
        cls,
        center: Chunk,
        get_chunk: Callable[[int, int], Optional[Chunk]],
        read_light: bool = True,
    ) -> "LightRegion":
        """Assemble the 3x3 patch around `center`.

        `get_chunk(chunk_x, chunk_z)` is the caller's "is this chunk resident"
        question -- on the server that is `Dimension.get_chunk`, on the actor
        thread, which is also the thread that owns the arrays being read here.

        A missing neighbour, or one whose `min_y`/`height` do not match the
        centre chunk's, leaves `ready` False and the buffers `None`. It does not
        raise: "not ready yet" is the normal state of a chunk at the edge of the
        loaded area, and the caller's answer is to keep the chunk queued.

        `read_light=False` skips unpacking the 18 light arrays and hands back
        zeroed buffers instead. It is only legal for a kernel that treats both
        layers as **outputs** -- `light_engine.relight_region` zeroes them
        itself -- and it is worth a flag because the unpack dominates a build:
        the 432 section expansions measure **~0.6 ms** through `_unpack_section`
        (and **~6.5 ms** through `light_layout.unpack_light`, which is why the
        fast path exists), against 1.0 ms for the 216 block copies and 0.03 ms
        for the allocations (`tools/bench_light.py`). A full patch reads no
        neighbour light at all (see the module docstring of
        `world/light_engine.py`), so that cost would buy nothing.

        `air_sections` is filled on both paths: it describes `blocks`, which is
        copied either way.
        """
        region = cls(center)
        centre_pos = center.position
        chunks: List[Optional[Chunk]] = []
        for dz in (-1, 0, 1):
            for dx in (-1, 0, 1):
                chunks.append(get_chunk(centre_pos.x + dx, centre_pos.z + dz))
        region.chunks = chunks

        sections = len(center.sections)
        for chunk in chunks:
            if chunk is None or len(chunk.sections) != sections or chunk.min_y != center.min_y:
                return region

        region.sections = sections
        blocks = np.empty((REGION_CHUNKS, sections, CELL_COUNT), dtype=np.uint16)
        sky = np.zeros((REGION_CHUNKS, sections, CELL_COUNT), dtype=np.uint8)
        block_light = np.zeros((REGION_CHUNKS, sections, CELL_COUNT), dtype=np.uint8)
        air_sections = np.full((REGION_CHUNKS, sections), SECTION_MIXED, dtype=np.uint8)
        region.all_lit = all(
            bool(getattr(chunk, "light_populated", False)) for chunk in chunks
        )
        for index, chunk in enumerate(chunks):
            assert chunk is not None  # checked above
            raw = chunk.raw_sections()
            for section_idx, section in enumerate(chunk.sections):
                blocks[index, section_idx] = raw[section_idx]
                if section.is_empty:
                    air_sections[index, section_idx] = SECTION_AIR
            if not read_light:
                continue
            if not getattr(chunk, "light_populated", False):
                # Never lit: the section arrays are the constructor's defaults,
                # not light. Zeroes are the safe reading -- they can only make
                # the centre chunk darker, and `DimensionLight` sends an edit
                # through a full patch instead of the incremental kernels while
                # any of the nine is in this state.
                continue
            for section_idx, section in enumerate(chunk.sections):
                # `_unpack_into` writes straight into the region slice: one
                # 4096-element expansion per section, no temporary per section
                # (§4.1).
                _unpack_into(section.sky_light, sky[index, section_idx])
                _unpack_into(section.block_light, block_light[index, section_idx])

        region.blocks = blocks
        region.air_sections = air_sections
        region.sky = sky
        region.block_light = block_light
        region._scratch = np.empty(CELL_COUNT // 2, dtype=np.uint8)
        region.ready = True
        return region

    # ---------------- write-back ----------------

    def write_back(self) -> int:
        """Pack the centre chunk's light into its `ChunkSection` arrays.

        Only the centre chunk, and only the two light layers: the eight
        neighbours' unpacked light was an *input* to the patch and is dropped
        with the region (`LIGHT_PLAN.md` §4.1).

        The target arrays are written **in place**, through the same
        `np.ndarray` objects `ChunkSection` holds, because `world/anvil.py` and
        `world/chunk_codec.py` both alias those buffers. Writes are skipped when
        the packed bytes are unchanged, so a patch that changes nothing costs no
        array writes and reports 0.

        `Chunk.light_populated` is set here -- these bytes are now the engine's
        answer rather than a default -- while `Chunk.is_light_dirty` is left to
        the caller, which owns the "someone has to be told" policy.

        Returns the number of section-arrays whose bytes actually changed (out
        of `2 * sections`), which is what the caller marks the chunk dirty on.
        """
        if not self.ready:
            return 0
        center = self.center
        sky = self.sky
        block_light = self.block_light
        assert sky is not None and block_light is not None
        changed = 0
        for section_idx, section in enumerate(center.sections):
            changed += self._write_one(sky[CENTER_CHUNK_INDEX, section_idx], section.sky_light)
            changed += self._write_one(
                block_light[CENTER_CHUNK_INDEX, section_idx], section.block_light
            )
        # The bytes are the engine's answer now, not a default: the chunk codec
        # reads them instead of falling back to its heightmap approximation
        # (`Chunk.light_populated`), and `world/anvil.py` persists them.
        center.light_populated = True
        return changed

    def _write_one(self, values: np.ndarray, target: np.ndarray) -> int:
        """Pack one layer of one section into `target`; 1 when it changed."""
        scratch = self._scratch
        assert scratch is not None
        pack_light(values, out=scratch)
        if np.array_equal(scratch, target):
            return 0
        target[:] = scratch
        return 1

    # ---------------- inspection helpers ----------------

    def center_section(self, section_idx: int):
        """The centre chunk's `ChunkSection` for an index, for callers that need it."""
        return self.center.sections[section_idx]

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        pos = self.center.position
        return (
            f"LightRegion(chunk=({pos.x}, {pos.z}), sections={self.sections}, "
            f"ready={self.ready})"
        )


__all__ = [
    "CENTER_CHUNK_INDEX",
    "LightRegion",
    "REGION_CHUNKS",
    "REGION_EDGE",
    "SECTION_AIR",
    "SECTION_MIXED",
    "SKY_SEEDED",
    "chunk_index",
]
