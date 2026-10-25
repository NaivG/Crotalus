"""
Chunk Management Module

Handles chunk data structures, sections, and block operations.

Block storage
-------------
`ChunkSection.blocks` holds one **global block state id** per cell, as `uint16`
- the same number the client is sent in a chunk palette, and the number
`assets/blocks.json` assigns through `registry.entries.BlockState.id`.

This module is the storage layer and deliberately knows nothing else about
blocks: naming one is `world/block_states.py`'s job. The only id it needs by
name is air, which is 0 by definition - an empty section is a palette holding
that one value, and the generator fills skipped sections with it.

Earlier versions stored a 1.8-era enum id here and translated on the way out.
That made the meaning of a saved chunk depend on boot order - an id space
allocated per boot cannot be persisted safely - which is why the numbering was
replaced outright rather than kept in step.
"""

import numpy as np
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Callable, Optional, Dict, List, Tuple, Any
import threading
import time

from loguru import logger

#: Air. Fixed at 0 by the registry and by the protocol's empty-section encoding.
AIR = 0

#: Rate limiter for the "everything in memory is pinned" warning: it is a
#: per-tick condition, and logging it per tick would bury the log.
_prune_warn_at = 0.0


@dataclass
class ChunkPos:
    """Chunk position coordinates."""
    x: int
    z: int
    
    def __hash__(self) -> int:
        return hash((self.x, self.z))
    
    def __eq__(self, other: object) -> bool:
        if isinstance(other, ChunkPos):
            return self.x == other.x and self.z == other.z
        return False
    
    def to_tuple(self) -> Tuple[int, int]:
        return (self.x, self.z)
    
    @property
    def key(self) -> str:
        """Get string key for dictionary storage."""
        return f"{self.x},{self.z}"
    
    def distance_to(self, other: "ChunkPos") -> float:
        """Calculate Euclidean distance to another chunk."""
        dx = (self.x - other.x) * 16
        dz = (self.z - other.z) * 16
        return np.sqrt(dx * dx + dz * dz)


@dataclass
class BlockPos:
    """Block position within a chunk."""
    x: int  # 0-15
    y: int  # min_y .. min_y + height - 1 (the owning Chunk's window)
    z: int  # 0-15
    
    def __hash__(self) -> int:
        return hash((self.x, self.y, self.z))
    
    def to_global(self, chunk_x: int, chunk_z: int) -> Tuple[int, int, int]:
        """Convert to global world coordinates."""
        return (
            chunk_x * 16 + self.x,
            self.y,
            chunk_z * 16 + self.z
        )


@dataclass
class ChunkSection:
    """
    A 16x16x16 section of a chunk.

    Uses a single array of 4096 global block state ids (uint16) for storage.
    Block at (x, y, z) is at index: x + (z * 16) + (y * 256)
    """
    y: int  # Absolute section Y: min_y//16 .. min_y//16 + section_count - 1
    blocks: np.ndarray = field(default_factory=lambda: np.zeros(4096, dtype=np.uint16))
    block_light: np.ndarray = field(default_factory=lambda: np.zeros(2048, dtype=np.uint8))
    #: Sky light is packed two cells to a byte, so "15 everywhere" is every byte
    #: `0xFF` -- not `0x0F`, which is one cell at 15 and the next at 0 (a field
    #: of light with a black cell between every pair). The default means
    #: "nobody has lit this section yet"; a chunk whose light is real says so
    #: with `Chunk.light_populated`, which is what the encoder reads, so this
    #: value is never sent as if it were an engine result.
    sky_light: np.ndarray = field(default_factory=lambda: np.full(2048, 0xFF, dtype=np.uint8))
    
    # Non-air block count for optimization
    _non_air_count: int = field(default=0, repr=False)
    
    def get_block(self, x: int, y: int, z: int) -> int:
        """
        Get block ID at local coordinates.
        
        Args:
            x: Block X (0-15)
            y: Block Y within section (0-15)
            z: Block Z (0-15)
        
        Returns:
            Block ID
        """
        if not (0 <= x < 16 and 0 <= z < 16 and 0 <= y < 16):
            return AIR
        
        index = x + (z * 16) + (y * 256)
        return self.blocks[index]
    
    def set_block(self, x: int, y: int, z: int, block_id: int) -> None:
        """
        Set block ID at local coordinates.
        
        Args:
            x: Block X (0-15)
            y: Block Y within section (0-15)
            z: Block Z (0-15)
            block_id: Global block state id to set
        """
        if not (0 <= x < 16 and 0 <= z < 16 and 0 <= y < 16):
            return
        
        index = x + (z * 16) + (y * 256)
        old_block = self.blocks[index]
        self.blocks[index] = block_id
        
        # Update non-air count
        if old_block == AIR and block_id != AIR:
            self._non_air_count += 1
        elif old_block != AIR and block_id == AIR:
            self._non_air_count -= 1
    
    @property
    def is_empty(self) -> bool:
        """Check if section contains no non-air blocks."""
        return self._non_air_count == 0

    @property
    def non_air_count(self) -> int:
        """Number of non-air blocks, as the chunk protocol wants to report it."""
        return self._non_air_count

    def recompute_non_air_count(self) -> None:
        """Recalculate the non-air count from `blocks`.

        `set_block` and `fill` keep the count up to date, but code that assigns
        `blocks` wholesale -- the Anvil reader does this -- bypasses it. Call
        this after any such assignment or `is_empty` will lie.
        """
        self._non_air_count = int(np.count_nonzero(self.blocks))

    def fill(self, block_id: int) -> None:
        """Fill entire section with a single block type."""
        self.blocks.fill(block_id)
        self._non_air_count = 4096 if block_id != AIR else 0
    
    def get_section_data(self) -> bytes:
        """Serialize section data for network/storage."""
        return self.blocks.tobytes()


@dataclass
class Chunk:
    """
    A 16 x height x 16 chunk spanning world Y `min_y .. min_y + height - 1`.

    The window is the *dimension's* vertical extent, not a 1.17-era 0..255
    column: the overworld is `min_y=-64, height=384` (24 sections), the nether
    and the end are `0, 256` (16 sections). Section `j` holds world Y
    `min_y + j*16 .. +15`, so a rule anchored to the world floor
    (`above_bottom: 0`) lands inside the array instead of falling outside it.
    """
    position: ChunkPos
    min_y: int = -64
    height: int = 384
    sections: List[ChunkSection] = field(default_factory=list)
    last_update: float = 0.0
    is_generated: bool = False
    is_dirty: bool = False

    #: Whether this chunk's `sky_light` / `block_light` arrays were produced by
    #: the flood-fill light engine (`world/light_engine.py`).
    #:
    #: Three states matter, and only the first is a special case:
    #:
    #: * `False` -- nobody lit this chunk. The arrays hold `ChunkSection`'s
    #:   defaults (sky 15 everywhere, block light 0), which describe a world
    #:   with no terrain in it. The encoder falls back to the old height-based
    #:   approximation rather than shipping those defaults as real light.
    #:   This is what a hand-built chunk, a test fixture and a chunk whose
    #:   region file predates light persistence all look like.
    #: * `True` -- the engine wrote them; the encoder reads them as they are.
    #: * A `False` set by the Anvil reader means the opposite of the above: the
    #:   file *did* carry light and it was unusable (wrong length), so the
    #:   array is the default and the light is known to be wrong rather than
    #:   merely absent. The relight pass treats both the same way; the
    #:   distinction is there so a future migration command can tell "never
    #:   lit" from "lit and lost".
    #:
    #: Light never sets `is_dirty`. `world/README.md` keeps "a chunk is written
    #: because its blocks changed" as the save rule; a light-only change sets
    #: `is_light_dirty` instead, which the saver also honours.
    light_populated: bool = False

    #: Whether the light changed since it was last written or sent. Separate
    #: from `is_dirty` on purpose: every freshly lit chunk in a player's view
    #: would otherwise look modified and be rewritten to disk.
    is_light_dirty: bool = False

    #: Whether this chunk's contents are known to be in a region file. Set by
    #: the Anvil reader (the bytes came from disk) and by `mark_saved`; stays
    #: False for a chunk the generator built. Read together with `is_dirty` to
    #: decide whether an unloaded chunk needs writing at all: a generated chunk
    #: has never been saved (and regenerating it costs seconds per chunk on the
    #: vanilla backend), while a chunk read from disk and untouched since costs
    #: nothing to drop -- which is what keeps a player walking around from
    #: rewriting their whole view to disk.
    is_persisted: bool = False
    
    # Entity data (to be implemented)
    entities: List[Any] = field(default_factory=list)
    block_entities: Dict[Tuple[int, int, int], Any] = field(default_factory=dict)
    
    # Heightmap for rendering
    heightmap: np.ndarray = field(default_factory=lambda: np.zeros((16, 16), dtype=np.int32))

    # Biomes, one per 4x4x4 cell: shape (sections, 4, 4, 4), indexed
    # `[section, y_in_section // 4, z // 4, x // 4]`. That is exactly the 64-cell
    # paletted container the protocol puts after each section's block palette, so
    # encoding is a reshape and no second coordinate system is involved.
    #
    # The ids are the numeric `worldgen/biome` registry ids - the same numbers
    # `world/vanilla/biome.py` produces and the client indexes its own registry
    # with. The previous `get_biome_at` guessed from height, which is why every
    # chunk used to ship as one biome.
    #
    # `None` means "nobody generated biomes for this chunk", which is not the
    # same as "biome 0": a hand-built `Chunk` (a test, a flat world, a chunk read
    # from a region file written before biomes existed) must keep shipping as
    # `codec.default_biome`, not silently become badlands. `set_biome_volume`
    # allocates it; `biome_volume()` still hands back a zero array so downstream
    # maths has the right shape.
    biomes: Optional[np.ndarray] = field(default=None, repr=False)

    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    
    def __post_init__(self):
        """Initialize chunk sections over the chunk's vertical window."""
        if self.height <= 0 or self.height % 16:
            raise ValueError(f"chunk height must be a positive multiple of 16, got {self.height}")
        if not self.sections:
            base = self.min_y // 16
            self.sections = [ChunkSection(y=base + j) for j in range(self.height // 16)]

    def get_block(self, x: int, y: int, z: int) -> int:
        """
        Get block ID at world-relative coordinates.

        Args:
            x: Block X within chunk (0-15)
            y: World Y (min_y .. min_y + height - 1)
            z: Block Z within chunk (0-15)

        Returns:
            Block ID
        """
        if not (0 <= x < 16 and 0 <= z < 16 and self.min_y <= y < self.min_y + self.height):
            return AIR

        offset = y - self.min_y
        section_index = offset // 16
        section_y = offset % 16

        return self.sections[section_index].get_block(x, section_y, z)

    def set_block(self, x: int, y: int, z: int, block_id: int) -> None:
        """
        Set block ID at world-relative coordinates.

        Args:
            x: Block X within chunk (0-15)
            y: World Y (min_y .. min_y + height - 1)
            z: Block Z within chunk (0-15)
            block_id: Block ID to set
        """
        with self._lock:
            if not (0 <= x < 16 and 0 <= z < 16 and self.min_y <= y < self.min_y + self.height):
                return

            offset = y - self.min_y
            section_index = offset // 16
            section_y = offset % 16

            self.sections[section_index].set_block(x, section_y, z, block_id)
            self.is_dirty = True
            self.last_update = 0.0  # Will be set by tick thread
    
    def get_height_at(self, x: int, z: int) -> int:
        """
        Get the highest non-air block Y at the given local coordinates.

        Args:
            x: Block X within chunk (0-15)
            z: Block Z within chunk (0-15)

        Returns:
            World Y of the highest block, or `min_y - 1` if the column is
            empty (the same "nothing above the floor" convention
            `chunk_codec._column_heights` uses).
        """
        if not (0 <= x < 16 and 0 <= z < 16):
            return self.min_y - 1

        for section_index in range(len(self.sections) - 1, -1, -1):
            section = self.sections[section_index]
            if section.is_empty:
                continue

            for y in range(15, -1, -1):
                if section.get_block(x, y, z) != AIR:
                    return self.min_y + section_index * 16 + y

        return self.min_y - 1

    def update_heightmap(self) -> None:
        """Update the heightmap for rendering.

        Per column, the height is the topmost non-air block anywhere in the
        chunk. The obvious spelling of that -- 256 columns times a scan
        through the sections -- is what this replaced, at roughly 65k
        `get_block` calls per chunk. Each section is one pass instead: which
        columns it touches at all, and the topmost y among them.

        Sections are walked bottom-up so a later (higher) section simply
        overwrites the columns it occupies, which is what "topmost" means.
        Columns no section reaches keep `min_y - 1`, matching `get_height_at`.
        Values are world Y, so a block in section 0 reads as `min_y + y`.
        """
        # [z, x], transposed into the heightmap's [x, z] at the end.
        best = np.full((16, 16), self.min_y - 1, dtype=np.int32)

        for section_index, section in enumerate(self.sections):
            if section.is_empty:
                continue
            # Flat index is x + z*16 + y*256, i.e. [y, z, x] in C order.
            occupied = section.blocks.reshape(16, 16, 16) != np.uint16(AIR)
            touched = occupied.any(axis=0)            # [z, x]
            if not touched.any():
                continue
            local_y = np.where(
                occupied,
                np.arange(16, dtype=np.int32)[:, None, None],
                np.int32(-1),
            ).max(axis=0)                            # [z, x]
            best[touched] = self.min_y + section_index * 16 + local_y[touched]

        self.heightmap[:, :] = best.T
    
    def fill_section(self, section_y: int, block_id: int) -> None:
        """
        Fill an entire section with a block type.

        Args:
            section_y: Section index (0 .. height//16 - 1, bottom first)
            block_id: Block ID to fill with
        """
        if 0 <= section_y < len(self.sections):
            self.sections[section_y].fill(block_id)
            self.is_dirty = True

    def get_biome_at(self, x: int, y: int, z: int) -> int:
        """Numeric biome id at a block position, from the 4x4x4 cell holding it.

        A missing volume reads as 0, which is `minecraft:badlands` - the same
        "first entry" a zero-initialised array means. `chunk_codec` uses
        `default_biome_id()` for that case, so a chunk generated before biomes
        existed is still sent as plains rather than as badlands.
        """
        if (self.biomes is None or not (0 <= x < 16 and 0 <= z < 16)
                or not (self.min_y <= y < self.min_y + self.height)):
            return 0
        section = (y - self.min_y) // 16
        if section >= self.biomes.shape[0]:
            return 0
        return int(self.biomes[section, ((y - self.min_y) % 16) // 4, z // 4, x // 4])

    def set_biome_volume(self, volume: np.ndarray, *, first_section: int = 0) -> None:
        """Install a `(sections, 4, 4, 4)` or `(quart_y, 4, 4)` biome volume.

        The `(quart_y, 4, 4)` form is what `world.vanilla.biome` produces, with
        quart rows running upward from the window's bottom; it is split into
        sections here so nothing downstream has to know about the window.
        """
        volume = np.asarray(volume, dtype=np.int32)
        if volume.ndim == 3:
            if volume.shape[0] % 4:
                raise ValueError(f"{volume.shape[0]} quart rows do not fill whole sections")
            # [quart_y, z, x] -> [section, y_in_section//4, z, x]
            volume = volume.reshape(volume.shape[0] // 4, 4, volume.shape[1], volume.shape[2])
        if volume.ndim != 4 or volume.shape[1:] != (4, 4, 4):
            raise ValueError(f"biome volume must be (n, 4, 4, 4), got {volume.shape}")
        end = first_section + volume.shape[0]
        if self.biomes is None or self.biomes.shape[0] < end:
            grown = np.zeros((max(end, len(self.sections)), 4, 4, 4), dtype=np.int32)
            if self.biomes is not None:
                grown[: self.biomes.shape[0]] = self.biomes
            self.biomes = grown
        self.biomes[first_section:end] = volume

    def matches_biomes(self, other: np.ndarray) -> bool:
        """Whether the stored volume already is `other`. Used to skip rework."""
        return self.biomes is not None and np.array_equal(self.biomes, other)

    def biome_volume(self) -> np.ndarray:
        """The whole chunk as `(height // 4, 4, 4)` `[quart_y, z, x]`.

        A reshape, not a copy: the stored axes are
        `(section, y_in_section // 4, z, x)` and quart rows run upward, so
        flattening the first two gives `quart_y` in order. That is the layout
        `world.vanilla.biome` produces and `world.vanilla.surface` reads.

        A chunk with no biome data hands back zeros - the right shape, and the
        wrong ids. `codec` does not use this; it checks `biomes` directly.
        """
        if self.biomes is None:
            return np.zeros((4 * len(self.sections), 4, 4), dtype=np.int32)
        return self.biomes.reshape(-1, 4, 4)

    def raw_sections(self) -> List[np.ndarray]:
        """The section arrays themselves, to read (or write) in place.

        This is *not* `blocks_view()`: no copy is made, so a write through one
        of these arrays is visible to `get_block` immediately and no commit is
        needed. That is what the physics kernel wants -- it runs every tick over
        the same chunks and cannot afford 128 KB of copying per chunk per tick.

        Two rules come with the zero copy:

        * **Read-only for physics.** The kernel takes these as read-only inputs;
          mutating them behind `Chunk`'s back leaves `non_air_count` (and
          therefore `is_empty`, which the chunk codec trusts) lying. Writers
          belong in `set_block`, `fill_section` or `commit_blocks_view`.
        * **Call it on the owning thread.** A `Chunk` is not thread-safe (`_lock`
        covers `set_block`/`commit_blocks_view` only); the arrays handed out
        here have no protection at all.

        Element `i` of a section's array is `x + z*16 + y*256`, i.e. the section
        reshaped to `[y, z, x]`, and `sections[j]` covers world Y
        `min_y + j*16` through `min_y + j*16 + 15`.
        """
        return [section.blocks for section in self.sections]

    def blocks_view(self) -> np.ndarray:
        """The whole chunk as one **copy** of a `(height, 16, 16)` `uint16` array, `[y, z, x]`.

        Sections are separate 4096-cell arrays indexed `x + z*16 + y*256`, which
        is `reshape(16, 16, 16)` over `(y, z, x)` - so stacking them *is* the
        volume, in the axis order the worldgen passes want. A copy, because the
        alternative is every pass knowing about section boundaries; 128 KB per
        chunk against ~20 ms of generation is not the cost that matters.

        **This is a copy, not a view, despite the name.** Write it back with
        `commit_blocks_view`; until then `get_block` still sees the old states,
        and mutating the returned array changes nothing. Hot paths that only
        read -- the physics kernel, which runs every tick -- want
        `raw_sections()` instead, which hands out the storage itself.
        """
        height = 16 * len(self.sections)
        out = np.empty((height, 16, 16), dtype=np.uint16)
        for index, section in enumerate(self.sections):
            out[index * 16:(index + 1) * 16] = section.blocks.reshape(16, 16, 16)
        return out

    def commit_blocks_view(self, view: Optional[np.ndarray] = None) -> None:
        """Write a `blocks_view()` array back into the sections.

        Recomputes the non-air count, because assigning `section.blocks`
        wholesale bypasses `set_block`, which is what keeps it current.
        """
        if view is None:
            return
        height = 16 * len(self.sections)
        if view.shape != (height, 16, 16):
            raise ValueError(f"expected {(height, 16, 16)}, got {view.shape}")
        with self._lock:
            for index, section in enumerate(self.sections):
                section.blocks[:] = view[index * 16:(index + 1) * 16].reshape(-1)
                section.recompute_non_air_count()
            self.is_dirty = True
    
    def mark_saved(self) -> None:
        """Mark chunk as saved (no longer dirty).

        Clears the light flag too: `_serialize_section` writes both light
        arrays, so a save carries the light along with the blocks and there is
        nothing left for a second write to add.
        """
        self.is_dirty = False
        self.is_light_dirty = False
        self.is_persisted = True
    
    def get_data_size(self) -> int:
        """Get approximate size of chunk data in bytes."""
        size = 0
        for section in self.sections:
            size += section.blocks.nbytes
            size += section.block_light.nbytes
            size += section.sky_light.nbytes
        size += self.heightmap.nbytes
        return size


class ChunkManager:
    """
    Manages loaded chunks for a dimension.

    Handles chunk loading, unloading, and caching. The manager stamps every
    chunk it creates with the dimension's vertical window (`min_y`/`height`),
    so a bare `Chunk()` never shows up in a live dimension with the wrong
    number of sections.

    Three rules that are easy to get wrong and expensive to get wrong:

    * **Nothing is dropped without being offered to the save handler first.**
      Both `unload_chunk` and the LRU valve call `eviction_handler(chunk)`
      before the entry disappears. The handler returns False to mean "could not
      take it", and the chunk then stays: an unsaved dirty chunk is worse than
      a chunk that is late to leave. `ChunkLifecycle` hands over chunks this
      way, and `core/server.py` points the handler at the streaming saver.
    * **`max_chunks` is a safety valve, not the unloading policy.** The loading
      policy is ticket-driven (`world/chunk_lifecycle.py`): chunks a ticket
      covers are *pinned* and are never evicted, because evicting one would
      only make the next tick load it again. The cap catches the chunks that
      nothing asked to keep -- explicit loads from tools or commands -- and
      when every chunk is pinned the manager logs it rather than thrashing.
    * **`chunks` is an `OrderedDict` in LRU order.** The previous list-based
      access order made every `get_chunk` an O(n) list removal, and the
      streamer calls it once per queued coordinate per tick: at radius 10 that
      is hundreds of O(n) scans per tick, for bookkeeping nobody reads.
    """

    def __init__(self, max_chunks: int = 4096, min_y: int = -64, height: int = 384):
        """
        Initialize chunk manager.

        Args:
            max_chunks: Safety cap on resident chunks (0 or None = unlimited).
                Only unpinned chunks can be evicted to satisfy it.
            min_y: World Y of the bottom block of every chunk it creates
            height: Window height of every chunk it creates (multiple of 16)
        """
        self.max_chunks = max_chunks or 0
        self.min_y = min_y
        self.height = height
        #: LRU-ordered: the front is the least recently used chunk.
        self.chunks: "OrderedDict[ChunkPos, Chunk]" = OrderedDict()
        self._lock = threading.RLock()

        #: Called with a chunk that is about to leave memory. True = the
        #: handler took ownership (so the manager may drop it), False = keep it.
        self._eviction_handler: Optional[Callable[[Chunk], bool]] = None
        #: Chunks this predicate answers True for are never evicted.
        self._pin_predicate: Optional[Callable[[ChunkPos], bool]] = None

    # ---------------- policy hooks (set once, by the owner) ----------------

    def set_eviction_handler(self, handler: Optional[Callable[[Chunk], bool]]) -> None:
        """Install the "a chunk is leaving memory" callback (see class docs)."""
        self._eviction_handler = handler

    def set_pin_predicate(self, predicate: Optional[Callable[[ChunkPos], bool]]) -> None:
        """Install the "this chunk must not be evicted" predicate."""
        self._pin_predicate = predicate

    # ---------------- reads ----------------

    def get_chunk(self, pos: ChunkPos) -> Optional[Chunk]:
        """
        Get a chunk by position, marking it as recently used.

        Args:
            pos: Chunk position

        Returns:
            Chunk if loaded, None otherwise
        """
        with self._lock:
            chunk = self.chunks.get(pos)
            if chunk is not None:
                self.chunks.move_to_end(pos)
            return chunk

    def peek(self, pos: ChunkPos) -> Optional[Chunk]:
        """Get a chunk without touching the LRU order.

        For hot paths that scan many coordinates per tick (the streamer's queue
        and the lifecycle's load/unload passes): they are not "uses" in the LRU
        sense, and paying for the reordering would make them O(n).
        """
        with self._lock:
            return self.chunks.get(pos)

    # ---------------- writes ----------------

    def load_chunk(self, pos: ChunkPos) -> Optional[Chunk]:
        """
        Create (or return) a blank chunk at this position over the dimension's
        vertical window. Does NOT generate terrain -- `World.load_chunk` and the
        generator claim path own that.

        Args:
            pos: Chunk position

        Returns:
            Loaded chunk
        """
        with self._lock:
            # Check if already loaded
            if pos in self.chunks:
                self.chunks.move_to_end(pos)
                return self.chunks[pos]

            self._make_room(pos)

            # Create new chunk over this dimension's window
            chunk = Chunk(position=pos, min_y=self.min_y, height=self.height)
            self.chunks[pos] = chunk

            return chunk

    def unload_chunk(self, pos: ChunkPos) -> bool:
        """
        Unload a chunk from memory, offering it to the eviction handler first.

        Args:
            pos: Chunk position

        Returns:
            True if chunk was unloaded, False if not found or the handler
            refused to take it (in which case it stays loaded)
        """
        with self._lock:
            chunk = self.chunks.get(pos)
        if chunk is None:
            return False
        if self._eviction_handler is not None and not self._eviction_handler(chunk):
            return False
        return self.remove(pos) is not None

    def remove(self, pos: ChunkPos) -> Optional[Chunk]:
        """Drop a chunk without asking the eviction handler.

        For callers that have already dealt with the data (the lifecycle hands
        the chunk to the saver itself) or that know there is nothing to save
        (a blank claim whose generation failed).
        """
        with self._lock:
            chunk = self.chunks.pop(pos, None)
        return chunk

    def set_chunk(self, chunk: Chunk) -> None:
        """
        Set a chunk directly.

        Args:
            chunk: Chunk to set
        """
        with self._lock:
            pos = chunk.position

            existing = self.chunks.get(pos)
            if existing is not None and existing is not chunk and existing.is_dirty:
                # Replacing a dirty chunk would drop edits nothing has written
                # yet. It should not happen (a replacement is a generated copy
                # racing a disk read, and neither is dirty), so it is logged
                # rather than silently swallowed.
                logger.warning(
                    f"Replacing dirty chunk {pos} with a generated copy; "
                    f"its unsaved changes are lost"
                )

            self._make_room(pos)
            self.chunks[pos] = chunk
            self.chunks.move_to_end(pos)

    def _make_room(self, incoming: ChunkPos) -> None:
        """Evict unpinned LRU chunks until one more entry fits."""
        if not self.max_chunks or len(self.chunks) < self.max_chunks:
            return
        if incoming in self.chunks:
            return
        if not self._evict_lru():
            global _prune_warn_at
            now = time.monotonic()
            if now - _prune_warn_at >= 60.0:
                _prune_warn_at = now
                logger.warning(
                    f"Chunk manager is at its cap ({self.max_chunks}) with every "
                    f"chunk pinned by a ticket; memory is bounded by the view/"
                    f"simulation distances, not by max_chunks"
                )

    def _evict_lru(self) -> bool:
        """Evict the least recently used *unpinned* chunk.

        Returns False when every resident chunk is pinned or refused by the
        handler -- the caller then leaves the cap exceeded rather than dropping
        something that must stay.
        """
        with self._lock:
            for pos in list(self.chunks.keys()):
                if self._pin_predicate is not None and self._pin_predicate(pos):
                    continue
                chunk = self.chunks[pos]
                if self._eviction_handler is not None and not self._eviction_handler(chunk):
                    continue
                del self.chunks[pos]
                return True
        return False
    
    def get_loaded_chunks(self) -> List[ChunkPos]:
        """Get list of all loaded chunk positions."""
        with self._lock:
            return list(self.chunks.keys())
    
    def get_chunk_count(self) -> int:
        """Get number of loaded chunks."""
        with self._lock:
            return len(self.chunks)
    
    def get_chunks_in_range(
        self,
        center: ChunkPos,
        radius: int
    ) -> List[Chunk]:
        """
        Get all chunks within a radius of a center chunk.
        
        Args:
            center: Center chunk position
            radius: Radius in chunks
        
        Returns:
            List of chunks within range
        """
        with self._lock:
            chunks = []
            for x in range(center.x - radius, center.x + radius + 1):
                for z in range(center.z - radius, center.z + radius + 1):
                    pos = ChunkPos(x, z)
                    chunk = self.chunks.get(pos)
                    if chunk:
                        chunks.append(chunk)
            return chunks
    
    def clear(self) -> None:
        """Clear all chunks, without offering them to the save handler.

        The caller is responsible for the data: `World.unload` saves first, and
        an explicit clear is a "throw it away" operation.
        """
        with self._lock:
            self.chunks.clear()
    
    def get_memory_usage(self) -> int:
        """Get approximate memory usage in bytes."""
        with self._lock:
            return sum(chunk.get_data_size() for chunk in self.chunks.values())
