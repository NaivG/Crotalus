"""
Anvil Chunk Format Module

Implements Minecraft's Anvil chunk format for region storage.
- Region files: r.{region_x}.{region_z}.mca (32x32 chunks per region)
- Chunk payloads: gzip-compressed NBT (compression type 1; see `COMPRESSION_*`)
- Header: Location table (4096 bytes) + Timestamp table (4096 bytes)

Writing a region file
---------------------
A region file is a container, not a document: a save touches one chunk, and the
other 1023 entries in it have to come back byte for byte. This writer therefore
never rewrites the file. `save_chunk` serializes one chunk and *stages* it;
`save` writes the staged payloads into their own sector runs and patches two
4-byte header fields per chunk - the location entry at `chunk_index * 4` and the
timestamp at `4096 + chunk_index * 4`. Both live in the first 8 KiB and chunk
data starts at sector 2 (byte 8192), so a header write can never land on a
payload.

Why the file must be written in place
-------------------------------------
The writer this replaced popped the staged chunks, read every *other* chunk into
a local dict (`existing_data`), reopened the file with `open(path, "wb")` -
truncating it - and wrote back only the chunks staged for that one save. The
collected bytes were never used. Saving one chunk destroyed every other chunk in
the region: the header still pointed at the old offsets, which now held a
different chunk's bytes or nothing at all. It was also O(region) per save.

Save-on-unload streaming (`world/chunk_saver.py`) cannot live with either
property. A player crossing one chunk border unloads ~25 chunks; each is
serialized on a saver thread and written when its batch flushes. Under the old
writer every one of those flushes rewrote the whole region file (quadratic in
the chunks it holds), and every chunk not in the same batch was collateral
damage. In-place sector writes make a save cost the chunk, not the region, which
is what allows a chunk to be written the moment it leaves memory.

Sector allocation
-----------------
A payload occupies `ceil(len(payload) / 4096)` whole sectors. If the chunk's
current run is at least that long it is overwritten where it stands; otherwise a
run is allocated - first fit over the freed runs, else appended at the end of
the file - and the old run is handed back (freed *before* allocating, so a run
that grew can absorb the hole its own old copy leaves). `_free_runs` is an
offset-sorted list of `(offset, size)` pairs that splits on partial reuse and
merges neighbours, and it is rebuilt from the header when the file is opened, so
holes survive a restart. The file never shrinks while saving; `compact()` is the
only operation that rewrites it densely.

A chunk that shrinks keeps its offset, shortens its own location entry, and
returns the tail to the free list. That is the one way a save can leave a hole
behind, and the reason `compact()` exists at all.

Section block data
------------------
A section's `block_states` is a paletted container: a `palette` list of
`{Name, Properties}` compounds and a packed `data` long array, which is the same
shape vanilla writes. Blocks are stored **by name**, not by state id, because a
region file outlives the `assets/blocks.json` it was written against.

Anything else - including the flat `IntArray` of ids an earlier build wrote -
reads as air with a warning. That asymmetry is deliberate: a missing block is
recoverable, a misread one is not, and this server is a demo with no worlds
worth guessing at.
"""

import gzip
import os
import struct
import threading
import time
import zlib
from pathlib import Path
from typing import Optional, Dict, Any, List, Sequence, Set, Tuple
from dataclasses import dataclass, field
from enum import IntEnum
from io import BytesIO
import numpy as np
from loguru import logger

from world.nbt import NBTCompound, NBTInt, NBTIntArray, NBTList, NBTString, NBTByte, NBTByteArray, NBTEnd, NBTLongArray, NBTType, write_nbt, read_nbt
from world import block_states, palette
from world.chunk import Chunk, ChunkPos, ChunkSection


# Anvil format constants
REGION_SIZE = 32  # 32x32 chunks per region
SECTOR_BYTES = 4096  # 4KB sectors
CHUNKS_PER_REGION = REGION_SIZE * REGION_SIZE  # 1024

# Header layout: sector 0 is the location table, sector 1 the timestamp table,
# and chunk data starts at sector 2. Everything the header touches is inside the
# first 8192 bytes, which is why a save can patch four bytes of it and leave the
# rest of the file alone.
LOCATION_TABLE_BYTES = CHUNKS_PER_REGION * 4   # 4096
TIMESTAMP_TABLE_BYTES = CHUNKS_PER_REGION * 4  # 4096
HEADER_BYTES = LOCATION_TABLE_BYTES + TIMESTAMP_TABLE_BYTES  # 8192
FIRST_DATA_SECTOR = HEADER_BYTES // SECTOR_BYTES  # 2

#: A location entry packs its length into one byte, so no chunk can be longer
#: than this many sectors. Vanilla has the same ceiling (a 1 MiB chunk).
MAX_CHUNK_SECTORS = 0xFF

# Compression types
COMPRESSION_GZIP = 1
COMPRESSION_DEFLATE = 2
COMPRESSION_NONE = 3

# World data version
WORLD_VERSION = 4671  # 1.21.x


# ==================== Section block data ====================

#: Names already reported as unresolvable, so a world full of them logs once per
#: name instead of once per section.
_REPORTED_MISSING: Set[str] = set()


def _decode_palette(tag, section: ChunkSection) -> int:
    """Fill a section from a `{palette, data}` container; returns unknown cells.

    A palette entry this build cannot resolve - a block that was renamed or
    removed from `assets/blocks.json` - becomes air and is counted. The caller
    reports the count; `_REPORTED_MISSING` keeps the log to one line per name.
    """
    entries = tag.get("palette")
    resolved_ids: List[int] = []
    missing = 0

    for entry in getattr(entries, "tags", None) or []:
        name_tag = entry.get("Name")
        name = name_tag.value if name_tag is not None else "minecraft:air"
        properties: Dict[str, str] = {}
        properties_tag = entry.get("Properties")
        if properties_tag is not None:
            for key, value in getattr(properties_tag, "tags", {}).items():
                properties[key] = value.value

        resolved = block_states.try_state_id(name, properties or None)
        if resolved is None:
            missing += 1
            if name not in _REPORTED_MISSING:
                _REPORTED_MISSING.add(name)
                logger.warning(
                    f"saved chunk names block {name} "
                    f"{properties or ''}, which this build does not have; "
                    f"cells using it read as air"
                )
            resolved = block_states.AIR
        resolved_ids.append(resolved)

    if not resolved_ids:
        section.blocks.fill(block_states.AIR)
        return 0

    lookup = np.asarray(resolved_ids, dtype=np.uint16)
    data = tag.get("data")
    packed = getattr(data, "value", None) if data is not None else None
    if lookup.size == 1 or not packed:
        # A one-entry palette carries no data array; vanilla omits it.
        section.blocks.fill(int(lookup[0]))
        return missing

    # On disk the width is never below 4 bits, regardless of palette size.
    bits = max(4, palette.bits_for(int(lookup.size)))
    indices = palette.unpack(packed, bits, section.blocks.size)
    section.blocks = lookup[indices]
    return missing


def _decode_block_states(tag, section: ChunkSection) -> int:
    """Fill a section from its `block_states` tag; returns cells that had no block."""
    tag_type = tag.get_type() if hasattr(tag, "get_type") else None
    if tag_type is NBTType.COMPOUND:
        return _decode_palette(tag, section)

    if tag_type is NBTType.INT_ARRAY:
        # What builds before the block-storage rewrite wrote: a flat array of ids
        # from a private numbering that no longer exists, so there is no name to
        # recover and nothing is guessed at. Deliberately not migrated.
        logger.warning(
            "section uses the pre-rewrite flat block array; this build does not "
            "migrate it, so the section reads as air"
        )
    else:
        logger.warning(f"unknown block_states tag type {tag_type}; reading it as air")

    section.blocks.fill(block_states.AIR)
    return 0


#: A light array on disk is 4096 cells at four bits: 2048 bytes. `world/chunk.py`
#: and `world/chunk_codec.py` both assume it; this is where a file that says
#: otherwise gets caught.
LIGHT_BYTES = 2048

#: The `ChunkSection` defaults, as the packed bytes they are: sky light 15 in
#: every cell (`0xFF` per byte), and block light 0 everywhere.
_DEFAULT_SKY_LIGHT = b"\xff" * LIGHT_BYTES
_DEFAULT_BLOCK_LIGHT = b"\x00" * LIGHT_BYTES


def _light_is_worth_writing(section: ChunkSection) -> bool:
    """Whether an all-air section carries light that is not the default.

    The cost of getting this wrong in the permissive direction is 4096 bytes an
    air section; in the restrictive direction it is light that silently
    disappears on reload. A byte-level comparison against the defaults is the
    cheapest test that cannot confuse the two, and it is exact: a section that
    equals `np.full(2048, 15)` has no light information in it.
    """
    return (
        section.sky_light.tobytes() != _DEFAULT_SKY_LIGHT
        or section.block_light.tobytes() != _DEFAULT_BLOCK_LIGHT
    )


def _read_light_array(tag, target: np.ndarray) -> bool:
    """Copy a stored light array into `target`. True when one was found.

    Returns False rather than raising on anything unusable -- a missing tag, a
    wrong length, a type this build does not know -- because the caller's next
    move is the same in every case: keep the `ChunkSection` default and let
    `light_populated` tell the encoder not to trust it.

    The copy is the point. `np.frombuffer` hands back a view onto the decoded
    NBT buffer, and the light engine mutates these arrays in place; a view
    would write through into the tag and make two chunks share one buffer.
    """
    if tag is None or not hasattr(tag, "value"):
        return False
    data = tag.value
    if isinstance(data, list):
        data = bytes(byte & 0xFF for byte in data)
    if not isinstance(data, (bytes, bytearray, memoryview)):
        return False
    if len(data) != LIGHT_BYTES:
        logger.warning(
            f"light array is {len(data)} B, expected {LIGHT_BYTES}; "
            f"keeping the default and marking the chunk's light unusable"
        )
        return False
    target[:] = np.frombuffer(bytes(data), dtype=np.uint8)
    return True


class ChunkStatus(IntEnum):
    """Minecraft chunk statuses."""
    EMPTY = 0
    STRUCTURE_STARTS = 1
    STRUCTURE_REFERENCES = 2
    BIOMES = 3
    NOISE = 4
    SURFACE = 5
    CARVERS = 6
    LIQUID_CARVERS = 7
    FEATURES = 8
    LIGHT = 9
    SPAWN = 10
    HEIGHTMAPS = 11
    FULL = 12


@dataclass
class ChunkLocation:
    """Location entry in region file header."""
    offset: int  # Offset in sectors
    size: int    # Size in sectors
    
    def is_empty(self) -> bool:
        return self.offset == 0 and self.size == 0


@dataclass
class RegionFileHeader:
    """Header for a region file."""
    locations: List[ChunkLocation]  # 1024 entries
    timestamps: List[int]  # 1024 entries


class AnvilChunkData:
    """Container for serialized chunk data."""
    
    def __init__(self, chunk: Chunk):
        self.chunk = chunk
        self.data: Optional[bytes] = None
    
    def serialize(self) -> bytes:
        """Serialize chunk to NBT format."""
        root = NBTCompound()
        
        # Level data
        level = NBTCompound()
        pos = self.chunk.position
        
        level["xPos"] = NBTInt("xPos", pos.x)
        level["zPos"] = NBTInt("zPos", pos.z)
        level["LastUpdate"] = NBTInt("LastUpdate", int(self.chunk.last_update * 1000))
        level["Status"] = NBTString("Status", "full")
        
        # Heightmap (MOTION_BLOCKING) - convert to Python native ints
        heightmap = []
        for i in range(16):
            for j in range(16):
                heightmap.append(int(self.chunk.heightmap[i, j]))
        level["Heightmap"] = NBTIntArray("Heightmap", heightmap)
        
        # Sections
        #
        # A section is written when it has blocks *or* when it has light that
        # its light-array contents say is not the default. Skipping it for
        # being empty was right when light was a constant nobody read: every
        # all-air section carried the same `ChunkSection` defaults, so nothing
        # was lost. Now an all-air section can hold the light a torch threw
        # into a cave, or the tail of a shaft from the surface, and dropping it
        # would put the chunk back to "black underground" on the next load with
        # nothing to correct it (there is no relight-on-load pass; see
        # LIGHT_PLAN.md §4.7).
        #
        # The light arrays go in only for a chunk whose light engine has
        # actually run. A chunk a worker just built has `light_populated` False
        # and `ChunkSection`'s defaults in both layers -- 15 everywhere for sky,
        # which is not "daylight", it is "nobody computed this" (`world/chunk.py`
        # says so at the field). Writing those bytes anyway made the next load
        # believe them: `_read_light_array` finds an array, `light_seen` becomes
        # True, the chunk comes back marked lit with an all-15 sky, and the
        # encoder ships "everything is daylight" to the client with a straight
        # face. Measured on a live probe (real server, real generator pool, one
        # viewer, save flushed at the end): 48 of 169 chunks were still in the
        # light queue when they were written, and every one of them reloaded as
        # a fully lit chunk. Leaving the tags out is the same trade
        # `_light_is_worth_writing` already makes for an empty section: a
        # default carries no information, so it is not written down.
        light_populated = bool(getattr(self.chunk, "light_populated", False))
        sections = NBTList()
        for section in self.chunk.sections:
            if section.is_empty and not _light_is_worth_writing(section):
                continue
            section_tag = self._serialize_section(section, light=light_populated)
            sections.tags.append(section_tag)
        level["Sections"] = sections
        
        # Block entities (tile entities)
        block_entities = NBTList()
        level["block_entities"] = block_entities
        
        # Entities
        entities = NBTList()
        level["Entities"] = entities
        
        root["Level"] = level
        
        return write_nbt(root, compressed=True)
    
    def _serialize_section(self, section: ChunkSection, light: bool = True) -> NBTCompound:
        """Serialize a chunk section to NBT.

        `light=False` leaves the two light arrays out, which is what a chunk
        whose light engine has not run yet needs: see `serialize`. The default
        is True so a direct caller that asked for one section still gets the
        whole section.

        `block_states` is the paletted container vanilla writes: a `palette` of
        `{Name, Properties}` compounds plus a packed `data` long array. Storing
        *names* rather than state ids is the point. A region file outlives the
        `assets/blocks.json` that produced it, and the same number means a
        different block under a different build - while a name either resolves
        or is visibly absent.

        Older builds wrote `block_states` as a flat `IntArray` of ids from a
        private numbering. `_decode_block_states` still reads those; nothing
        writes them any more.
        """
        section_tag = NBTCompound()
        section_tag["Y"] = NBTByte("Y", int(section.y))

        values, inverse = np.unique(section.blocks, return_inverse=True)

        block_states_tag = NBTCompound("block_states")
        entries = NBTList("palette")
        for state_id in values:
            entry = NBTCompound()
            codec = block_states.codec(int(state_id))
            entry["Name"] = NBTString("Name", codec["Name"])
            properties = codec.get("Properties")
            if properties:
                props_tag = NBTCompound("Properties")
                for key, value in properties.items():
                    props_tag[key] = NBTString(key, value)
                entry["Properties"] = props_tag
            entries.tags.append(entry)
        block_states_tag["palette"] = entries

        if values.size > 1:
            # Vanilla never writes fewer than 4 bits per entry on disk, even for
            # a two-entry palette.
            bits = max(4, palette.bits_for(int(values.size)))
            block_states_tag["data"] = NBTLongArray("data", palette.pack_longs(inverse, bits))
        section_tag["block_states"] = block_states_tag

        # Light data
        if light:
            section_tag["block_light"] = NBTByteArray("block_light", section.block_light.tobytes())
            section_tag["sky_light"] = NBTByteArray("sky_light", section.sky_light.tobytes())

        return section_tag


class RegionFile:
    """
    An Anvil region file containing 32x32 chunks.

    Layout:
    - Sector 0: location table, 1024 entries of 4 bytes each
      (`offset_in_sectors << 8 | length_in_sectors`)
    - Sector 1: timestamp table, 1024 entries of 4 bytes each
    - Sector 2 onward: chunk payloads, variable length, padded to sectors

    Only the chunks handed to `save_chunk` are ever written; see the module
    docstring for why that has to be true and `_write_chunk_locked` for how a
    run is chosen.

    Threading: one `RegionFile` is one file, and every read and write of it goes
    through `self._lock`. A load therefore never sees a half-written payload,
    and the header (`self.header`) is only ever mutated inside the lock. The
    lock is re-entrant so the public methods may keep calling the `*_locked`
    helpers, which are written as if they hold it.
    """
    
    def __init__(self, file_path: Path, *, min_y: int = -64, height: int = 384):
        """
        Initialize region file handler.

        Args:
            file_path: Path to the .mca file
            min_y: World Y of the bottom block of the chunks it holds (the
                owning dimension's floor; section Y tags are absolute and are
                rebased onto this window when read)
            height: Window height of the chunks it holds (multiple of 16)
        """
        self.file_path = file_path
        self.min_y = min_y
        self.height = height
        self.header = RegionFileHeader(
            locations=[ChunkLocation(0, 0) for _ in range(CHUNKS_PER_REGION)],
            timestamps=[0] * CHUNKS_PER_REGION
        )
        self.dirty_chunks: Set[int] = set()
        self._chunk_data_cache: Dict[int, bytes] = {}  # chunk_index -> serialized data
        #: Freed sector runs as `(offset, size)`, sorted by offset, disjoint and
        #: with neighbours already merged. Rebuilt from the header on load, so a
        #: hole left by an earlier process is reusable again. See
        #: `_allocate_sectors_locked` / `_free_sectors_locked`.
        self._free_runs: List[Tuple[int, int]] = []
        #: Sectors the file occupies, header included. Chunks appended past the
        #: end come from here; this is the only thing that grows the file.
        self._file_sectors: int = FIRST_DATA_SECTOR
        self._lock = threading.RLock()
        self._loaded = False
    
    def _get_chunk_index(self, chunk_x: int, chunk_z: int) -> int:
        """Get index in location table for a chunk."""
        return (chunk_x % REGION_SIZE) + (chunk_z % REGION_SIZE) * REGION_SIZE
    
    def _get_chunk_coords(self, index: int) -> Tuple[int, int]:
        """Get chunk coordinates from index."""
        z = index // REGION_SIZE
        x = index % REGION_SIZE
        return x, z
    
    def load_header(self) -> bool:
        """
        Load header from file.

        Also derives the free-run list and the file's sector count from the
        header and the file length. A run freed by an earlier process (the tail
        of a chunk that shrank) is recorded nowhere else, so it would otherwise
        be unusable until `compact()`.

        Returns:
            True if file exists and header loaded
        """
        with self._lock:
            if not self.file_path.exists():
                return False

            try:
                with open(self.file_path, "rb") as f:
                    raw = f.read(HEADER_BYTES)
            except OSError as error:
                logger.exception(f"Failed to load region header: {error}")
                return False

            if len(raw) < HEADER_BYTES:
                # Half a header is not something to guess at: every location in
                # it would be read out of a partial table. Refuse the file
                # rather than invent 1024 offsets from whatever is there.
                logger.warning(
                    f"{self.file_path.name} is {len(raw)} bytes, shorter than the "
                    f"{HEADER_BYTES}-byte header; reading it as empty"
                )
                return False

            for i in range(CHUNKS_PER_REGION):
                value = struct.unpack_from(">I", raw, i * 4)[0]
                self.header.locations[i] = ChunkLocation((value >> 8) & 0xFFFFFF, value & 0xFF)
                self.header.timestamps[i] = struct.unpack_from(
                    ">I", raw, LOCATION_TABLE_BYTES + i * 4
                )[0]

            self._rebuild_free_runs_locked()
            self._loaded = True
            return True

    def _rebuild_free_runs_locked(self) -> None:
        """Derive the free runs and the file length from the header.

        Everything between sector 2 and the end of the file that the header does
        not claim is free. Entries pointing outside the file are ignored, so a
        truncated file cannot make the allocator hand out sectors that another
        chunk still occupies.
        """
        try:
            size_bytes = self.file_path.stat().st_size
        except OSError:
            size_bytes = 0
        self._file_sectors = max(
            FIRST_DATA_SECTOR, (size_bytes + SECTOR_BYTES - 1) // SECTOR_BYTES
        )

        used = sorted(
            (location.offset, location.size)
            for location in self.header.locations
            if location.offset >= FIRST_DATA_SECTOR
            and location.size > 0
            and location.offset + location.size <= self._file_sectors
        )

        runs: List[Tuple[int, int]] = []
        cursor = FIRST_DATA_SECTOR
        for offset, size in used:
            if offset > cursor:
                runs.append((cursor, offset - cursor))
            cursor = max(cursor, offset + size)
        if cursor < self._file_sectors:
            runs.append((cursor, self._file_sectors - cursor))
        self._free_runs = runs

    def save(self) -> None:
        """Write the staged chunks into the file, in place.

        Only the staged chunks are touched: no other chunk's bytes move and the
        file is never truncated. Each write is one payload plus its own two
        header fields, so the cost of a save is the size of the chunk, not the
        size of the region.
        """
        with self._lock:
            self._flush_locked()

    def _flush_locked(self) -> int:
        """Write every staged chunk. Returns how many were written."""
        staged: Dict[int, bytes] = {}
        for chunk_index in list(self.dirty_chunks):
            payload = self._chunk_data_cache.pop(chunk_index, None)
            if payload is not None:
                staged[chunk_index] = payload
        self.dirty_chunks.clear()

        if not staged:
            return 0

        if not self._loaded and self.file_path.exists():
            # A writer that never read the header would put its chunk on top of
            # whatever already lives at sector 2. Cheap to rule out.
            self.load_header()

        self._ensure_file_locked()

        # Keys are dropped as they land, so whatever is left in `pending` after
        # a failure is exactly what did not reach the file.
        pending = dict(staged)
        try:
            with open(self.file_path, "r+b") as f:
                for chunk_index in sorted(staged):
                    self._write_chunk_locked(f, chunk_index, staged[chunk_index])
                    del pending[chunk_index]
        except Exception:
            # Re-stage the unwritten chunks so a later save() can still write
            # them. An interrupted chunk is re-staged too, which is the useful
            # recovery: its payload may be torn, while its header entry is
            # written last and therefore still names the previous version.
            for chunk_index, payload in pending.items():
                self._chunk_data_cache[chunk_index] = payload
                self.dirty_chunks.add(chunk_index)
            raise

        logger.debug(f"Saved {len(staged)} chunk(s) to {self.file_path.name}")
        return len(staged)

    def _ensure_file_locked(self) -> None:
        """Create the file with a zeroed 8 KiB header if it is missing."""
        if not self.file_path.exists():
            self.file_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.file_path, "wb") as f:
                f.write(b"\x00" * HEADER_BYTES)
            self._file_sectors = FIRST_DATA_SECTOR
            return

        try:
            size_bytes = self.file_path.stat().st_size
        except OSError as error:
            logger.warning(f"cannot stat {self.file_path}: {error}")
            return

        if size_bytes < HEADER_BYTES:
            # A crash between creating the file and writing its header. Nothing
            # but a header can live before byte 8192, so the bytes there are
            # zeroed rather than extended past: junk left in the location table
            # would read back as locations for slots that were never written.
            # The in-memory header is empty here too - `load_header` refused
            # this file - so the two cannot disagree.
            with open(self.file_path, "r+b") as f:
                f.write(b"\x00" * HEADER_BYTES)
            size_bytes = HEADER_BYTES

        self._file_sectors = max(
            self._file_sectors, (size_bytes + SECTOR_BYTES - 1) // SECTOR_BYTES
        )

    def _write_chunk_locked(self, f, chunk_index: int, payload: bytes) -> None:
        """Write one payload into its own run and patch its two header fields.

        `payload` is the on-disk frame: a 4-byte big-endian length that counts
        the compression byte, the compression type, then the compressed NBT.
        The header entry is written *after* the payload, so a failure (or a
        crash) in between leaves the previous version of the chunk reachable at
        the location the file still points to.
        """
        sectors = (len(payload) + SECTOR_BYTES - 1) // SECTOR_BYTES
        if sectors > MAX_CHUNK_SECTORS:
            # The length field of a location entry is one byte. Writing this
            # would wrap the size and corrupt the entry for good, so the chunk
            # is dropped (loudly) and the previous version stays in place.
            logger.error(
                f"chunk {chunk_index} of {self.file_path.name} needs {sectors} "
                f"sectors, more than the {MAX_CHUNK_SECTORS} the format allows; "
                f"it was not written"
            )
            return

        location = self.header.locations[chunk_index]
        if location.offset >= FIRST_DATA_SECTOR and location.size >= sectors:
            # Fits where it stands. Overwriting in place is the whole point:
            # nothing else in the file moves.
            offset = location.offset
            if location.size > sectors:
                # The chunk shrank. Hand the tail back so a later chunk can use
                # it; the location entry below shrinks to match.
                self._free_sectors_locked(offset + sectors, location.size - sectors)
        else:
            if location.offset >= FIRST_DATA_SECTOR and location.size > 0:
                # Free before allocating: a chunk that grew can then absorb the
                # hole its own old copy leaves, when the two are adjacent.
                self._free_sectors_locked(location.offset, location.size)
            offset = self._allocate_sectors_locked(sectors)

        f.seek(offset * SECTOR_BYTES)
        f.write(payload)
        padding = sectors * SECTOR_BYTES - len(payload)
        if padding:
            f.write(b"\x00" * padding)

        timestamp = int(time.time())
        self.header.locations[chunk_index] = ChunkLocation(offset, sectors)
        self.header.timestamps[chunk_index] = timestamp

        # Surgical header writes: 4 bytes in the location table, 4 in the
        # timestamp table, both inside the first 8 KiB. The data area starts at
        # sector 2 (byte 8192) and no run is ever allocated below it, so these
        # cannot land on a payload.
        f.seek(chunk_index * 4)
        f.write(struct.pack(">I", (offset << 8) | sectors))
        f.seek(LOCATION_TABLE_BYTES + chunk_index * 4)
        f.write(struct.pack(">I", timestamp))

    def _allocate_sectors_locked(self, sectors: int) -> int:
        """Reserve `sectors` contiguous sectors and return the offset.

        First fit over `_free_runs`, which is sorted by offset - so this is also
        the lowest-offset fit - and otherwise at the end of the file.
        """
        for i, (offset, size) in enumerate(self._free_runs):
            if size >= sectors:
                if size == sectors:
                    del self._free_runs[i]
                else:
                    # Split: the tail stays free, still sorted by offset.
                    self._free_runs[i] = (offset + sectors, size - sectors)
                return offset

        offset = self._file_sectors
        self._file_sectors += sectors
        return offset

    def _free_sectors_locked(self, offset: int, size: int) -> None:
        """Return a run to the free list, merging it with its neighbours.

        The caller frees a run the header no longer claims (one it is about to
        overwrite, or the tail of a chunk that shrank), so runs never overlap
        and at most one merge is needed on each side.
        """
        if size <= 0 or offset < FIRST_DATA_SECTOR:
            return

        runs = self._free_runs
        i = 0
        while i < len(runs) and runs[i][0] < offset:
            i += 1

        if i > 0 and runs[i - 1][0] + runs[i - 1][1] == offset:
            offset, size = runs[i - 1][0], runs[i - 1][1] + size
            del runs[i - 1]
            i -= 1
        if i < len(runs) and offset + size == runs[i][0]:
            size += runs[i][1]
            del runs[i]

        runs.insert(i, (offset, size))

    def compact(self) -> bool:
        """Rewrite the file densely and clear the free list.

        Chunk payloads are copied in chunk-index order, each into the smallest
        run it needs, and the whole 8192-byte header is rewritten - the one
        place that is worth doing, because every location may have moved. The
        new file is built beside the old one and moved into place, so a failure
        leaves the region as it was.

        Returns True when the file was rewritten. A region with no holes is left
        alone, which is what makes it cheap to call from `close_all()`.

        This is the only operation that makes a region file smaller: a chunk
        that shrank leaves its tail in the free list until it runs. Nothing
        needs it for correctness, so a caller may never call it.
        """
        with self._lock:
            self._flush_locked()

            live = [
                (index, location)
                for index, location in enumerate(self.header.locations)
                if location.offset >= FIRST_DATA_SECTOR and location.size > 0
            ]
            used = sum(location.size for _, location in live)

            try:
                on_disk = max(
                    FIRST_DATA_SECTOR,
                    (self.file_path.stat().st_size + SECTOR_BYTES - 1) // SECTOR_BYTES,
                )
            except OSError:
                on_disk = FIRST_DATA_SECTOR

            if not self._free_runs and used == on_disk - FIRST_DATA_SECTOR:
                return False  # already dense; nothing to gain

            temp_path = self.file_path.with_name(self.file_path.name + ".compact")
            locations = list(self.header.locations)
            cursor = FIRST_DATA_SECTOR
            try:
                with open(self.file_path, "rb") as src, open(temp_path, "wb") as out:
                    out.write(b"\x00" * HEADER_BYTES)
                    for index, location in live:
                        payload = self._read_payload(src, location)
                        if payload is None:
                            # Unreadable entry: copy its sectors verbatim rather
                            # than drop bytes this method does not understand.
                            logger.warning(
                                f"{self.file_path.name}: entry {index} does not fit "
                                f"its location; copying its sectors as they are"
                            )
                            sectors = location.size
                            src.seek(location.offset * SECTOR_BYTES)
                            payload = src.read(sectors * SECTOR_BYTES)
                        else:
                            sectors = (len(payload) + SECTOR_BYTES - 1) // SECTOR_BYTES

                        out.seek(cursor * SECTOR_BYTES)
                        out.write(payload)
                        out.write(b"\x00" * (sectors * SECTOR_BYTES - len(payload)))
                        locations[index] = ChunkLocation(cursor, sectors)
                        cursor += sectors

                    out.seek(0)
                    for location in locations:
                        out.write(struct.pack(">I", (location.offset << 8) | location.size))
                    for timestamp in self.header.timestamps:
                        out.write(struct.pack(">I", timestamp))
            except Exception:
                temp_path.unlink(missing_ok=True)
                raise

            os.replace(temp_path, self.file_path)
            self.header.locations = locations
            self._file_sectors = cursor
            self._free_runs = []
            self._loaded = True
            logger.debug(f"Compacted {self.file_path.name} to {cursor} sectors")
            return True

    def _read_payload(self, f, location: ChunkLocation) -> Optional[bytes]:
        """The payload frame at `location`, or None when the entry does not fit.

        Only the payload is read, not the whole run: a run can be longer than
        what lives in it (the file never shrinks, and a reused hole can be
        bigger than the chunk that lands in it).
        """
        f.seek(location.offset * SECTOR_BYTES)
        head = f.read(4)
        if len(head) < 4:
            return None
        length = struct.unpack(">I", head)[0]
        total = 4 + length
        if length < 1 or total > location.size * SECTOR_BYTES:
            return None
        f.seek(location.offset * SECTOR_BYTES)
        return f.read(total)
    
    def save_chunk(self, chunk: Chunk) -> None:
        """
        Serialize a chunk and stage it for the next `save()`.

        Staging rather than writing here is what lets a batch of chunks share
        one flush: `ChunkSaver` stages a batch through `RegionManager.save_chunk`
        and calls `save_all()` once, and the old writer needed exactly that
        shape too (it just did the writing wrong). The header is deliberately
        left untouched until the payload is on disk - it describes what is *in
        the file*, and a reader on another thread may still be using it - so a
        staged chunk is still read back from disk at its previous version until
        the flush lands (`ChunkSaver.flush_key` is what waits for that).

        Two threads staging the *same* coordinate can land in either order, and
        the later one wins. That was already true of a batch (one slot holds one
        chunk, and two entries for a coordinate are two points in time), and
        `ChunkSaver` keeps submissions in order for the common single worker.

        Args:
            chunk: Chunk to save
        """
        # Calculate position within region
        chunk_index = self._get_chunk_index(chunk.position.x, chunk.position.z)

        # Serialize outside the lock: it is the expensive part (palette + gzip),
        # it needs nothing but the chunk, and holding the region lock across it
        # would stall a concurrent load of the same region.
        serialized = AnvilChunkData(chunk).serialize()

        # Length (4 bytes, and it counts the compression byte) + compression
        # type + data. Compression type 1 = gzip; 2 is zlib/deflate and 3 is
        # uncompressed.
        data_with_header = (
            struct.pack(">I", len(serialized) + 1)
            + struct.pack("B", COMPRESSION_GZIP)
            + serialized
        )

        with self._lock:
            self._chunk_data_cache[chunk_index] = data_with_header
            self.dirty_chunks.add(chunk_index)
    
    def load_chunk(self, local_x: int, local_z: int, world_x: int, world_z: int) -> Optional[Chunk]:
        """
        Load a chunk from the region file.

        Two coordinate systems are in play and they are not
        interchangeable. `local_*` addresses the slot in this region's 32x32
        header and is what `_get_chunk_index` needs; `world_*` is where the
        chunk actually lives and is what the Chunk must be positioned at.

        They only coincide for the origin region. For chunk (-1, 0) the local
        slot is (31, 0), and building the Chunk from the local pair filed it
        in the chunk manager under (31, 0) -- unreachable, so every retry to
        generate it re-read the same mis-keyed chunk from disk and returned
        before reaching the generator. With the streamer's strict send order
        that stalled the whole view behind one coordinate.

        Args:
            local_x: Chunk X within this region (0-31)
            local_z: Chunk Z within this region (0-31)
            world_x: Chunk X in world coordinates
            world_z: Chunk Z in world coordinates

        Returns:
            Loaded chunk, positioned at the world coordinates, or None if not found
        """
        chunk_index = self._get_chunk_index(local_x, local_z)

        # Hold the lock only for the read, so the bytes come back as one whole
        # payload: a saver thread writing this region either finished its chunk
        # or has not started it. Decoding needs no lock - `payload` is a private
        # copy by then - and a load of a region being written does not wait for
        # gzip and NBT parsing.
        with self._lock:
            location = self.header.locations[chunk_index]
            if location.is_empty():
                return None

            try:
                with open(self.file_path, "rb") as f:
                    payload = self._read_payload(f, location)
            except OSError as error:
                logger.exception(f"Failed to load chunk ({world_x}, {world_z}): {error}")
                return None

        if payload is None:
            logger.warning(
                f"region entry for chunk ({world_x}, {world_z}) in "
                f"{self.file_path.name} does not fit its run; reading nothing"
            )
            return None

        try:
            # Compression type, then the compressed NBT (the length prefix
            # counts the type byte, so what follows it is `length - 1` long).
            compression_type = payload[4]
            chunk_data = payload[5:]

            if compression_type == COMPRESSION_GZIP:
                chunk_data = gzip.decompress(chunk_data)
            elif compression_type == COMPRESSION_DEFLATE:
                chunk_data = zlib.decompress(chunk_data)

            # Parse NBT
            root = read_nbt(chunk_data, compressed=False)
            return self._deserialize_chunk(root, world_x, world_z)

        except Exception as e:
            logger.exception(f"Failed to load chunk ({world_x}, {world_z}): {e}")
            return None
    
    def _deserialize_chunk(self, root: NBTCompound, chunk_x: int, chunk_z: int) -> Chunk:
        """Deserialize NBT to Chunk."""
        pos = ChunkPos(chunk_x, chunk_z)
        chunk = Chunk(position=pos, min_y=self.min_y, height=self.height)

        level = root.get("Level")
        if not level:
            return chunk

        # Parse sections. The `Y` tag is the *absolute* section Y (vanilla
        # writes -4..19 for the overworld), so it maps into the chunk's window
        # by rebasing on min_y//16. A section outside the window is skipped -
        # a save from a taller dimension reads as the air it would occupy.
        sections = level.get("Sections")
        #: Whether any section carried a light array. Drives `light_populated`:
        #: a region file written before light persistence, or by another build,
        #: has no light, and saying "populated" would make the encoder ship the
        #: `ChunkSection` defaults -- full daylight everywhere -- as if the
        #: engine had produced them.
        light_seen = False
        if sections and hasattr(sections, 'tags'):
            base = self.min_y // 16
            for section_tag in sections.tags:
                y_tag = section_tag.get("Y")
                if not y_tag:
                    continue
                section_y = int(y_tag.value)
                y = section_y - base
                if 0 <= y < len(chunk.sections):
                    section = chunk.sections[y]

                    # Parse block states - either the paletted container this
                    # build writes (and vanilla writes) or the flat IntArray of
                    # ids that builds before the storage rewrite wrote.
                    block_states_tag = section_tag.get("block_states")
                    if block_states_tag is not None:
                        missing = _decode_block_states(block_states_tag, section)
                        if missing:
                            logger.warning(
                                f"chunk ({chunk_x}, {chunk_z}) section {section_y}: "
                                f"{missing} cell(s) held a block id with no name "
                                f"in this build; they read as air"
                            )
                        section.recompute_non_air_count()
                    
                    # Parse lights.
                    #
                    # Three details, all of them learned the hard way:
                    #
                    # 1. The array must be exactly 2048 bytes. A region file
                    #    from another build, or a truncated one, would
                    #    otherwise hand the light engine an array of the wrong
                    #    size, and every kernel indexes 4096 nibbles into it.
                    # 2. `.copy()` is not optional. `np.frombuffer` returns a
                    #    view onto the NBT buffer, and the light engine writes
                    #    into these arrays in place -- a view would corrupt the
                    #    decoded tag and alias two chunks' light together.
                    # 3. A missing or unusable array leaves the `ChunkSection`
                    #    default in place and clears `light_populated`, so the
                    #    encoder falls back instead of shipping a default
                    #    ("everything is daylight") as if it were real.
                    block_light = section_tag.get("block_light")
                    sky_light = section_tag.get("sky_light")
                    found_block = _read_light_array(block_light, section.block_light)
                    found_sky = _read_light_array(sky_light, section.sky_light)
                    if found_block or found_sky:
                        # One of the two is enough to say "this file carries
                        # light"; the missing half stays at its default, which
                        # is the right value in every case but a nether chunk
                        # written before sky light existed.
                        light_seen = True
        
        # Parse heightmap
        heightmap_tag = level.get("Heightmap")
        if heightmap_tag and hasattr(heightmap_tag, 'value'):
            heights = np.array(heightmap_tag.value, dtype=np.int32)
            if len(heights) == 256:
                chunk.heightmap = heights.reshape(16, 16)
            elif len(heights) == 36 * 36:
                # Take only the 16x16 portion
                chunk.heightmap = heights[:256].reshape(16, 16)
        
        chunk.is_generated = True
        chunk.light_populated = light_seen
        chunk.last_update = 0
        return chunk
    
    def chunk_exists(self, chunk_x: int, chunk_z: int) -> bool:
        """Check if chunk exists in region file.

        Only a real run counts: an entry with a zero offset cannot address a
        payload (sector 0 is the location table), and staging a chunk no longer
        touches the header, so the answer describes the file on disk.
        """
        chunk_index = self._get_chunk_index(chunk_x, chunk_z)
        with self._lock:
            location = self.header.locations[chunk_index]
            return location.offset >= FIRST_DATA_SECTOR and location.size > 0


class RegionManager:
    """
    Manages region files for a dimension.
    
    Handles loading, saving, and caching of region files.
    """
    
    def __init__(self, region_dir: Path, *, min_y: int = -64, height: int = 384):
        """
        Initialize region manager.

        Args:
            region_dir: Directory containing region files
            min_y: The owning dimension's floor; stamped onto every chunk read
            height: The owning dimension's height; ditto
        """
        self.region_dir = region_dir
        self.min_y = min_y
        self.height = height
        self.region_dir.mkdir(parents=True, exist_ok=True)
        self._regions: Dict[Tuple[int, int], RegionFile] = {}
        # Re-entrant so a public method may keep calling another one that takes
        # it (`close_all` -> `save_all`) without a second lock object.
        self._lock = threading.RLock()

        # Cache settings
        self._max_cached_regions = 32
    
    def _get_region_path(self, region_x: int, region_z: int) -> Path:
        """Get path for a region file."""
        return self.region_dir / f"r.{region_x}.{region_z}.mca"
    
    def _get_region_coords(self, chunk_x: int, chunk_z: int) -> Tuple[int, int]:
        """Get region coordinates for a chunk."""
        return chunk_x // REGION_SIZE, chunk_z // REGION_SIZE
    
    def get_region(self, region_x: int, region_z: int) -> RegionFile:
        """
        Get or create a region file.
        
        Args:
            region_x: Region X coordinate
            region_z: Region Z coordinate
        
        Returns:
            RegionFile instance
        """
        with self._lock:
            key = (region_x, region_z)
            
            if key not in self._regions:
                path = self._get_region_path(region_x, region_z)
                region = RegionFile(path, min_y=self.min_y, height=self.height)
                region.load_header()
                self._regions[key] = region
                
                # Evict old regions if cache is full
                if len(self._regions) > self._max_cached_regions:
                    self._evict_oldest()
            
            return self._regions[key]
    
    def _evict_oldest(self) -> None:
        """Evict oldest region from cache."""
        # Simple eviction - save and remove oldest
        for key, region in list(self._regions.items()):
            if not region.dirty_chunks:
                del self._regions[key]
                break
    
    def save_chunk(self, chunk: Chunk) -> None:
        """
        Stage a chunk for its region file.

        The payload is serialized and held until `save_all()` flushes it (which
        is `RegionFile.save` behind the scenes). A batch of chunks therefore
        costs one flush per region instead of one per chunk, which is the shape
        `ChunkSaver` batches for. Staging is complete in itself: nothing is lost
        if the flush happens later, and the chunk can be read back unchanged
        from disk in the meantime.

        Args:
            chunk: Chunk to save
        """
        region_x, region_z = self._get_region_coords(
            chunk.position.x, chunk.position.z
        )
        region = self.get_region(region_x, region_z)
        region.save_chunk(chunk)
    
    def load_chunk(self, chunk_x: int, chunk_z: int) -> Optional[Chunk]:
        """
        Load a chunk from its region file.
        
        Args:
            chunk_x: Chunk X coordinate
            chunk_z: Chunk Z coordinate
        
        Returns:
            Loaded chunk or None if not found
        """
        region_x, region_z = self._get_region_coords(chunk_x, chunk_z)
        region = self.get_region(region_x, region_z)

        # Convert to local coordinates for the region header lookup, but keep
        # the world coordinates for the Chunk itself -- see RegionFile.load_chunk.
        local_x = chunk_x % REGION_SIZE
        local_z = chunk_z % REGION_SIZE

        return region.load_chunk(local_x, local_z, chunk_x, chunk_z)
    
    def chunk_exists(self, chunk_x: int, chunk_z: int) -> bool:
        """Check if chunk exists in any region file."""
        region_x, region_z = self._get_region_coords(chunk_x, chunk_z)
        path = self._get_region_path(region_x, region_z)
        
        if not path.exists():
            return False
        
        region = self.get_region(region_x, region_z)
        local_x = chunk_x % REGION_SIZE
        local_z = chunk_z % REGION_SIZE
        return region.chunk_exists(local_x, local_z)
    
    def save_all(self) -> int:
        """
        Flush every region that has staged chunks.

        With the incremental writer this writes only the chunks staged since the
        last flush - it no longer rewrites a region file per save - so calling
        it once per saver batch is cheap.

        Returns:
            Number of regions saved
        """
        saved = 0
        with self._lock:
            for region in self._regions.values():
                if region.dirty_chunks:
                    region.save()
                    saved += 1
        return saved
    
    def close_all(self) -> None:
        """Flush every region file, compact it, and drop the cache.

        Compacting here is safe: `close_all` is the shutdown path
        (`World.unload`), nothing writes to these regions afterwards, and
        `compact` returns immediately for a region with no holes. It is also the
        only moment a region file can shrink, which is why it is worth doing at
        all. The manager lock is held so a concurrent `get_region` cannot build a
        second `RegionFile` for a file that is being replaced under it.
        """
        self.save_all()
        with self._lock:
            for region in self._regions.values():
                try:
                    region.compact()
                except Exception as error:  # pragma: no cover - shutdown must finish
                    logger.exception(
                        f"Failed to compact {region.file_path.name}: {error}"
                    )
            self._regions.clear()


def create_world_directories(world_dir: Path, world_name: str) -> Dict[str, Path]:
    """
    Create world directory structure.
    
    Args:
        world_dir: Base world directory
        world_name: Name of the world
    
    Returns:
        Dictionary mapping dimension names to their directories
    """
    world_path = world_dir / world_name
    
    dirs = {
        "overworld": world_path / "region",
        "nether": world_path / "region" / "_DIM-1",
        "end": world_path / "region" / "DIM1",
    }
    
    # Create directories
    for name, path in dirs.items():
        path.mkdir(parents=True, exist_ok=True)
    
    # Create playerdata, stats, etc.
    (world_path / "playerdata").mkdir(parents=True, exist_ok=True)
    (world_path / "stats").mkdir(parents=True, exist_ok=True)
    (world_path / "advancements").mkdir(parents=True, exist_ok=True)
    (world_path / "poi").mkdir(parents=True, exist_ok=True)
    
    return dirs


def world_exists(world_dir: Path, world_name: str) -> bool:
    """
    Check if a world exists.
    
    Args:
        world_dir: Base directory
        world_name: World name
    
    Returns:
        True if world exists
    """
    world_path = world_dir / world_name
    level_dat = world_path / "level.dat"
    return level_dat.exists()


def list_worlds(world_dir: Path) -> List[str]:
    """
    List all worlds in a directory.
    
    Args:
        world_dir: Base directory
    
    Returns:
        List of world names
    """
    if not world_dir.exists():
        return []
    
    worlds = []
    for path in world_dir.iterdir():
        if path.is_dir() and (path / "level.dat").exists():
            worlds.append(path.name)
    
    return sorted(worlds)
