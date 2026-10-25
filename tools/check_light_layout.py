"""
Data-layer check for the light engine: the two light tables in `registry/blocks.py`
and the packed-nibble section format in `world/light_layout.py`.

Every assertion here is about *data and format*, not about lighting: that the
`opacity` column is still the three-valued `getLightBlock()` the engine's decay
rules are written against, that the emitter set is still the 1,626 states the
kernel screens a section with, and that a section's light is still the 2048 bytes
`world/chunk_codec.py` and `world/anvil.py` read and write. `check` collects every
failure instead of stopping at the first, because on a data regression the whole
distribution is the interesting part.

The nibble order gets its own section on purpose: `pack` and `unpack` written
mirrored still round trip, so a round trip alone proves nothing about which half
of a byte a cell lands in. The direction is pinned against hand-computed bytes and
an independent per-byte Python reference.

Run: py -3.14t tools/check_light_layout.py
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from loguru import logger

# The block registry loads lazily on first use below; its startup log is not this
# tool's output, and the double-load warning it would print is the documented lazy
# path, not a problem.
logger.remove()

from registry import get_game_registry
from registry.blocks import light_emitters, light_opacity_table
from tools.check_chunk_codec import make_chunk
from world import light_layout
from world.chunk import ChunkSection
from world.chunk_codec import LIGHT_BYTES, _pack_light
from world import light_region

#: Measured on `assets/blocks.json` for 1.21.11 (protocol 774) and recorded in
#: `LIGHT_PLAN.md`. These are regression pins, not derivations.
EXPECTED_STATES = 29671
EXPECTED_OPACITY = {0: 16240, 1: 10872, 15: 2559}
EXPECTED_EMITTERS = 1626
EXPECTED_LEVEL_15 = 611

FAILURES: list = []


def check(label: str, ok: bool, detail: str = "") -> bool:
    print(f"  {'OK  ' if ok else 'FAIL'} {label}" + (f"  ({detail})" if detail else ""))
    if not ok:
        FAILURES.append(label)
    return ok


def note(text: str) -> None:
    """Something worth recording that is not a failure of this layer."""
    print(f"  NOTE {text}")


def rejects(call, *args) -> bool:
    """Whether `call(*args)` raises `ValueError` (the guard on a bad length)."""
    try:
        call(*args)
    except ValueError:
        return True
    return False


# ==================== 1. the state tables ====================


def check_state_tables() -> None:
    print("\n1. state tables (registry/blocks.py)")
    registry = get_game_registry().blocks
    total = registry.total_states

    check("the registry is frozen before a light table is built", registry.frozen,
          "the tables cache for the life of the registry; GameRegistry.freeze() "
          "is what makes that safe")
    check(f"the registry holds {EXPECTED_STATES} states", total == EXPECTED_STATES,
          str(total))

    table = light_opacity_table()
    check("light_opacity_table() is a uint8[65536]",
          table.dtype == np.uint8 and table.shape == (1 << 16,),
          f"{table.dtype} x {table.shape}")
    check("it is built once: the second call returns the same array",
          light_opacity_table() is table)
    check("and an explicit registry returns that registry's table",
          light_opacity_table(registry) is table)

    # The regression that matters: three values, and vanilla's getLightBlock() is
    # exactly this column. A fourth value appearing means the decay rules changed
    # shape underneath the engine.
    counts = {int(value): int(count) for value, count in
              enumerate(np.bincount(table[:total], minlength=16)) if count}
    check("opacity is exactly {0: 16240, 1: 10872, 15: 2559}",
          counts == EXPECTED_OPACITY, repr(counts))

    # Every state id, covered exactly once, and equal to the registry's own value.
    covered = 0
    mismatched = []
    for block in registry.blocks.values():
        covered += block.state_count
        first = block.first_state_id
        column = np.asarray(block.opacity, dtype=np.uint8)
        if not np.array_equal(table[first:first + block.state_count], column):
            mismatched.append(block.name)
    check("the table covers every state id exactly once", covered == total,
          f"{covered} covered vs {total} states")
    check("light_opacity_table()[id] == BlockState(id).opacity for every state",
          not mismatched, f"{len(mismatched)} block(s) differ: {mismatched[:3]}")
    check("state ids past the registry read as 0 (no block, no opacity)",
          not table[total:].any(), f"{int(np.count_nonzero(table[total:]))} nonzero")

    # Why the flags column must not be used as a second opinion: the occlusion bits
    # are not set on anything, the opaque states included.
    flagged = sum(
        1
        for block in registry.blocks.values()
        for flags, opacity in zip(block.state_flags, block.opacity)
        if opacity == 15 and flags & 0x0C00
    )
    check("no opacity==15 state sets state_flags bit 10/11 (IS_SOLID_RENDER/CAN_OCCLUDE)",
          flagged == 0, f"{flagged} of {counts.get(15, 0)}")

    ids, levels = light_emitters()
    check(f"light_emitters() is (int32[{EXPECTED_EMITTERS}], uint8[{EXPECTED_EMITTERS}])",
          ids.dtype == np.int32 and levels.dtype == np.uint8
          and ids.shape == (EXPECTED_EMITTERS,) and levels.shape == (EXPECTED_EMITTERS,),
          f"{ids.dtype}{ids.shape} / {levels.dtype}{levels.shape}")
    check("it is built once: the second call returns the same arrays",
          light_emitters()[0] is ids and light_emitters()[1] is levels)
    check(f"{EXPECTED_LEVEL_15} of them are level 15",
          int(np.count_nonzero(levels == 15)) == EXPECTED_LEVEL_15,
          str(int(np.count_nonzero(levels == 15))))
    check("every level is 1..15 (a level-0 state is not an emitter)",
          int(levels.min()) > 0 and int(levels.max()) <= light_layout.MAX_LIGHT_LEVEL,
          f"{int(levels.min())}..{int(levels.max())}")
    check("state ids are unique and inside the registry",
          np.unique(ids).size == ids.size
          and int(ids.min()) >= 0 and int(ids.max()) < total,
          f"ids {int(ids.min())}..{int(ids.max())}")
    check("the ids are ascending, so a searchsorted screen is valid",
          bool((np.diff(ids) > 0).all()))

    # The same 1,626 states the registry enumerates, each at the level it reports
    # -- a set comparison, because the two orderings are deliberately different
    # (the arrays are ascending by state id, `emitting_states()` is brightest
    # first).
    reference = registry.emitting_states()
    emitter_level = {int(i): int(v) for i, v in zip(ids, levels)}
    check("light_emitters() covers emitting_states() exactly (same states, same levels)",
          len(reference) == ids.size
          and all(emitter_level.get(state.id) == state.luminance
                  for state in reference),
          f"{len(reference)} states, brightest {reference[0]}")


# ==================== 2. semantics ====================


def check_semantics() -> None:
    print("\n2. semantics (the blocks the engine's rules are named after)")
    registry = get_game_registry().blocks
    table = light_opacity_table()
    ids, levels = light_emitters()
    emitter_level = {int(i): int(v) for i, v in zip(ids, levels)}

    water = registry.get_block("minecraft:water")
    check("water (minecraft:water, all 16 levels) is opacity 1",
          water is not None and water.state_count == 16 and set(water.opacity) == {1}
          and int(table[water.default_state_id]) == 1,
          f"default {water.default_state_id} "
          f"(minecraft:water[level=0]), opacity {sorted(set(water.opacity))}"
          if water is not None else "missing")

    stone = registry.get_block("minecraft:stone")
    check("stone is opacity 15", stone is not None and int(table[stone.default_state_id]) == 15,
          f"id {stone.default_state_id}" if stone is not None else "missing")

    glass = registry.get_block("minecraft:glass")
    check("glass is opacity 0", glass is not None and int(table[glass.default_state_id]) == 0,
          f"id {glass.default_state_id}" if glass is not None else "missing")

    torch = registry.get_block("minecraft:torch")
    torch_id = torch.default_state_id if torch is not None else -1
    check("torch is luminance 14, and is in light_emitters() at 14",
          torch is not None and int(torch.luminance[0]) == 14
          and emitter_level.get(torch_id) == 14,
          f"id {torch_id}, emitters[torch]={emitter_level.get(torch_id)}")


# ==================== 3. the layout ====================


def check_layout() -> None:
    print("\n3. section layout (world/light_layout.py)")
    check("one section is 4096 cells of 4 bits -> 2048 bytes",
          light_layout.CELL_COUNT == 4096 and light_layout.SECTION_BYTES == 2048
          and light_layout.SECTION_BYTES * light_layout.NIBBLES_PER_BYTE
          == light_layout.CELL_COUNT,
          f"{light_layout.CELL_COUNT} cells -> {light_layout.SECTION_BYTES} bytes")
    check("light_layout agrees with chunk_codec.LIGHT_BYTES",
          light_layout.SECTION_BYTES == LIGHT_BYTES, f"codec LIGHT_BYTES={LIGHT_BYTES}")

    def walk():
        for y in range(16):
            for z in range(16):
                for x in range(16):
                    yield x, y, z

    indexes = [light_layout.cell_index(x, y, z) for x, y, z in walk()]
    check("cell_index covers 0..4095 exactly once",
          sorted(indexes) == list(range(4096)),
          f"{len(set(indexes))} distinct")
    check("cell_index is x + z*16 + y_local*256",
          light_layout.cell_index(1, 0, 0) == 1
          and light_layout.cell_index(0, 0, 1) == 16
          and light_layout.cell_index(0, 1, 0) == 256
          and light_layout.cell_index(15, 15, 15) == 4095)
    check("cell_coords inverts cell_index",
          all(light_layout.cell_coords(light_layout.cell_index(x, y, z)) == (x, y, z)
              for x, y, z in walk()))

    check("world Y -> section: -64 -> 0, 0 -> 4, 319 -> 23 (overworld window)",
          light_layout.section_index(-64, -64) == 0
          and light_layout.section_index(0, -64) == 4
          and light_layout.section_index(319, -64) == 23,
          f"{light_layout.section_index(-64, -64)}/"
          f"{light_layout.section_index(0, -64)}/"
          f"{light_layout.section_index(319, -64)}")
    check("section_base_y + local_y invert it for every cell of 24 sections",
          all(light_layout.split_y(light_layout.section_base_y(section, -64) + y, -64)
              == (section, y)
              for section in range(24) for y in range(16)))
    check("section_count: 384 -> 24, 256 -> 16",
          light_layout.section_count(384) == 24
          and light_layout.section_count(256) == 16)
    check("a height that is not whole sections is refused",
          rejects(light_layout.section_count, 383)
          and rejects(light_layout.section_count, 0)
          and rejects(light_layout.section_count, -16))

    chunk = make_chunk()
    check("a real Chunk's section list matches the mapping",
          all(light_layout.section_index(section.y * 16, chunk.min_y) == index
              for index, section in enumerate(chunk.sections)),
          f"{len(chunk.sections)} sections from min_y={chunk.min_y}")


# ==================== 4. round trips ====================


def mixed_section() -> ChunkSection:
    """A section with light that is neither uniform nor a single value.

    Real light is a surface: 15 above a per-column height, decreasing below it,
    plus a couple of torch-sized spots below. A uniform vector would exercise the
    nibble order but not the classification the codec does on top of it.
    """
    section = ChunkSection(y=0)
    sky = np.zeros(light_layout.CELL_COUNT, dtype=np.uint8)
    block = np.zeros(light_layout.CELL_COUNT, dtype=np.uint8)
    for z in range(16):
        for x in range(16):
            top = 12 - ((x + z) % 5)
            for y in range(16):
                level = 15 if y >= top else max(0, 15 - (top - y))
                sky[light_layout.cell_index(x, y, z)] = level
    block[light_layout.cell_index(3, 4, 5)] = 14
    block[light_layout.cell_index(9, 2, 11)] = 14
    section.sky_light[:] = light_layout.pack_light(sky)
    section.block_light[:] = light_layout.pack_light(block)
    return section


def check_round_trips() -> None:
    print("\n4. packed <-> unpacked round trips")
    rng = np.random.default_rng(2048)
    random_bytes = rng.integers(0, 256, light_layout.SECTION_BYTES, dtype=np.uint8)

    for label, packed in (
        ("random 2048 bytes", random_bytes),
        ("all zeros", np.zeros(light_layout.SECTION_BYTES, dtype=np.uint8)),
        ("all 0xFF", np.full(light_layout.SECTION_BYTES, 0xFF, dtype=np.uint8)),
        ("a real sky_light default", ChunkSection(y=0).sky_light),
        ("a real block_light default", ChunkSection(y=0).block_light),
    ):
        cells = light_layout.unpack_light(packed)
        back = light_layout.pack_light(cells)
        check(f"unpack -> pack is byte-identical ({label})",
              back.tobytes() == packed.tobytes(),
              f"{int(np.count_nonzero(back != packed))} byte(s) differ")

    section = mixed_section()
    for label in ("sky_light", "block_light"):
        array = getattr(section, label)
        check(f"a real section's {label} round trips byte-for-byte",
              light_layout.pack_light(light_layout.unpack_light(array)).tobytes()
              == array.tobytes())

    full_bytes = np.full(light_layout.SECTION_BYTES, 0xFF, dtype=np.uint8)
    check("4096 fifteens pack to 2048 x 0xFF (all-full is 0xFF, not 0x0F bytes)",
          light_layout.pack_light(np.full(4096, 15, dtype=np.uint8)).tobytes()
          == full_bytes.tobytes())
    check("2048 x 0xFF unpacks to 4096 fifteens",
          bool((light_layout.unpack_light(full_bytes) == 15).all()))
    # The array a fresh section starts with has to be the packed all-15 form.
    check("a fresh ChunkSection's sky_light is the packed all-15 array (0xFF per byte)",
          ChunkSection(y=0).sky_light.tobytes() == full_bytes.tobytes(),
          f"byte 0x{int(ChunkSection(y=0).sky_light[0]):02X}")

    # ...and `0x0F` is the byte that looks like "15" without being it: one cell at
    # 15, the next at 0. That was `ChunkSection.sky_light`'s default until this
    # round -- a half-dark section -- and the classifier of the day, which OR'd a
    # byte's two nibbles together instead of decoding them, read it as full
    # daylight. Both halves of that are pinned here so neither can come back.
    half_bytes = np.full(light_layout.SECTION_BYTES, 0x0F, dtype=np.uint8)
    half = light_layout.unpack_light(half_bytes)
    check("2048 x 0x0F is a half-lit field, not all 15",
          int(np.count_nonzero(half == 15)) == 2048
          and int(np.count_nonzero(half == 0)) == 2048,
          f"{int(np.count_nonzero(half == 15))}/4096 cells at 15")
    note("ChunkSection's default sky_light used to be 2048 x 0x0F, i.e. 2048/4096 "
         "lit cells (byte 0x0F); full is 0xFF, and the default is that now")

    # The writers on the other side, both directions.
    cells = light_layout.unpack_light(section.sky_light)
    check("pack_light == chunk_codec._pack_light on the same input",
          light_layout.pack_light(cells).tobytes() == _pack_light(cells))
    check("unpack_light inverts chunk_codec._pack_light",
          light_layout.unpack_light(_pack_light(cells)).tolist() == [int(v) for v in cells])

    # `out=` as the kernel will use it: into a region buffer, and into the section.
    packed = np.zeros(light_layout.SECTION_BYTES, dtype=np.uint8)
    returned = light_layout.pack_light(cells, out=packed)
    check("pack_light(out=) writes into the caller's array and returns it",
          returned is packed and packed.tobytes() == section.sky_light.tobytes())
    region = np.zeros((9, 24, light_layout.CELL_COUNT), dtype=np.uint8)
    light_layout.unpack_light(section.sky_light, out=region[4, 7])
    check("unpack_light(out=) fills one slice of a (9, 24, 4096) region buffer",
          np.array_equal(region[4, 7], cells) and not region[0, 0].any())

    check("a short or long vector is refused (not silently truncated)",
          rejects(light_layout.unpack_light, np.zeros(2047, dtype=np.uint8))
          and rejects(light_layout.unpack_light, np.zeros(2049, dtype=np.uint8))
          and rejects(light_layout.pack_light, np.zeros(4095, dtype=np.uint8))
          and rejects(light_layout.pack_light, np.zeros(4097, dtype=np.uint8)))

    # `world/light_region.py` unpacks with an njit loop instead of the strided
    # numpy stores above, because `LightRegion.build` makes 432 of these and the
    # numpy form measures ~6.5 ms against the light kernel's whole cost. Two
    # unpackers means two chances to be wrong about nibble order, so the fast
    # one is pinned against the reference on inputs chosen to separate the two
    # nibbles: an asymmetric byte, the 0x0F/0xF0 field that a mis-ordered
    # decoder turns into the wrong half, and a real section.
    fast = np.zeros(light_layout.CELL_COUNT, dtype=np.uint8)
    for label, packed in (
        ("0x0F (the half-lit field)", half_bytes),
        ("0xF0", np.full(light_layout.SECTION_BYTES, 0xF0, dtype=np.uint8)),
        ("0x5A alternating nibbles",
         np.full(light_layout.SECTION_BYTES, 0x5A, dtype=np.uint8)),
        ("a real section", section.sky_light),
        ("random bytes", np.frombuffer(random_bytes, dtype=np.uint8)),
    ):
        fast[:] = 0
        light_region._unpack_into(np.ascontiguousarray(packed, dtype=np.uint8), fast)
        check(f"the njit unpack == light_layout.unpack_light ({label})",
              np.array_equal(fast, light_layout.unpack_light(packed)),
              f"{int(np.count_nonzero(fast != light_layout.unpack_light(packed)))}"
              " cell(s) differ")
    note("light_region._unpack_into is the engine's path; light_layout.unpack_light "
         "stays the reference and the codec's, and the njit loop is checked against "
         "it here -- if they ever disagree, the format is in question, not the loop")


# ==================== 5. nibble order ====================


def check_nibble_order() -> None:
    print("\n5. nibble order (a mirrored pack/unpack would still round trip)")
    zero = np.zeros(light_layout.SECTION_BYTES, dtype=np.uint8).tobytes()

    def single(cell: int, value: int) -> bytes:
        cells = np.zeros(light_layout.CELL_COUNT, dtype=np.uint8)
        cells[cell] = value
        return light_layout.pack_light(cells).tobytes()

    check("cell 0 = 1 sets the low nibble of byte 0 (0x01)",
          single(0, 1) == b"\x01" + zero[1:], single(0, 1)[:1].hex())
    check("cell 1 = 1 sets the high nibble of byte 0 (0x10)",
          single(1, 1) == b"\x10" + zero[1:], single(1, 1)[:1].hex())
    check("cell 0 = 15 fills the low nibble only (0x0F)",
          single(0, 15) == b"\x0f" + zero[1:], single(0, 15)[:1].hex())
    check("cell 2 = 1 lands in byte 1, not byte 0",
          single(2, 1) == zero[:1] + b"\x01" + zero[2:], single(2, 1)[:2].hex())
    check("cell 3 = 1 lands in the high nibble of byte 1",
          single(3, 1) == zero[:1] + b"\x10" + zero[2:], single(3, 1)[:2].hex())
    check("cell 4095 = 7 lands in the high nibble of the last byte",
          single(4095, 7) == zero[:-1] + b"\x70", single(4095, 7)[-1:].hex())

    low = light_layout.unpack_light(np.frombuffer(b"\x01" + zero[1:], dtype=np.uint8))
    high = light_layout.unpack_light(np.frombuffer(b"\x10" + zero[1:], dtype=np.uint8))
    check("unpacking 0x01 gives cell 0 (and only cell 0)",
          int(low[0]) == 1 and int(low.sum()) == 1)
    check("unpacking 0x10 gives cell 1 (and only cell 1)",
          int(high[1]) == 1 and int(high.sum()) == 1)

    # An independent, numpy-free implementation of the same format: if the numpy
    # shifts above ever drift, these two disagree.
    rng = np.random.default_rng(4096)
    cells = rng.integers(0, 16, light_layout.CELL_COUNT, dtype=np.uint8)
    reference = bytes(
        (int(cells[2 * i]) & 0x0F) | ((int(cells[2 * i + 1]) & 0x0F) << 4)
        for i in range(light_layout.SECTION_BYTES)
    )
    check("matches a per-byte Python reference on 4096 random nibbles",
          light_layout.pack_light(cells).tobytes() == reference)


# ==================== 6. what light_layout may depend on ====================


def check_module_imports() -> None:
    print("\n6. light_layout.py stands alone")
    path = ROOT / "world" / "light_layout.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    check("imports nothing but numpy/typing (no world, no registry, no logger)",
          imported <= {"__future__", "numpy", "typing"}, f"imports {sorted(imported)}")


def main() -> int:
    print("Light layout checks (state tables, packed nibbles, section coordinates)")
    check_state_tables()
    check_semantics()
    check_layout()
    check_round_trips()
    check_nibble_order()
    check_module_imports()

    print()
    if FAILURES:
        print(f"RESULT: FAIL ({len(FAILURES)})")
        for name in FAILURES:
            print(f"  - {name}")
        return 1
    print("RESULT: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
