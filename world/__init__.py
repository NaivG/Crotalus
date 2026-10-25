"""
World Module

Manages worlds, dimensions, terrain generation, loading, and saving.
Uses NBT format for level.dat and Anvil format for region storage.
"""

from world.world import World, WorldData, WorldManager, world_manager
from world.dimension import Dimension, DimensionType, DimensionSettings
from world.chunk import Chunk, ChunkSection, ChunkPos, AIR
from world.chunk_ticket import (
    MAX_LEVEL,
    MAX_LOADED_LEVEL,
    DIMENSION_IDLE_TIMEOUT,
    LoadType,
    Ticket,
    TicketFlags,
    TicketManager,
    TicketType,
)
from world.chunk_lifecycle import ChunkLifecycle
from world.chunk_saver import ChunkSaver, should_persist
from world.block_states import (
    BLOCKS,
    codec as block_state_codec,
    default_state_id,
    name_of,
    state_id,
)
from world import palette
from world.generator import (
    BaseGenerator,
    GeneratorType,
    GeneratorSettings,
    create_generator,
    create_default_generator,
    register_generator,
)
from world.chunk_gen_thread import ChunkGeneratorThread, chunk_generator
from world.nbt import save_level_dat, load_level_dat, NBTCompound, write_nbt, read_nbt
from world.anvil import (
    RegionFile,
    RegionManager,
    AnvilChunkData,
    create_world_directories,
    world_exists,
    list_worlds,
)

__all__ = [
    # World
    "World",
    "WorldData",
    "WorldManager",
    "world_manager",
    # Dimension
    "Dimension",
    "DimensionType",
    "DimensionSettings",
    # Chunk
    "Chunk",
    "ChunkSection",
    "ChunkPos",
    "AIR",
    # Load levels / tickets
    "MAX_LEVEL",
    "MAX_LOADED_LEVEL",
    "DIMENSION_IDLE_TIMEOUT",
    "LoadType",
    "Ticket",
    "TicketFlags",
    "TicketManager",
    "TicketType",
    # Automatic load/unload and streaming save
    "ChunkLifecycle",
    "ChunkSaver",
    "should_persist",
    # Block states (global state ids, from assets/blocks.json)
    "BLOCKS",
    "state_id",
    "default_state_id",
    "name_of",
    "block_state_codec",
    "palette",
    # Generator
    "BaseGenerator",
    "GeneratorType",
    "GeneratorSettings",
    "create_generator",
    "create_default_generator",
    "register_generator",
    # Noise
    "PerlinNoise",
    "SimplexNoise",
    "create_noise_generator",
    "get_chunk_heightmap",
    # Chunk Generation Thread
    "ChunkGeneratorThread",
    "chunk_generator",
    # NBT
    "save_level_dat",
    "load_level_dat",
    "NBTCompound",
    "write_nbt",
    "read_nbt",
    # Anvil
    "RegionFile",
    "RegionManager",
    "AnvilChunkData",
    "create_world_directories",
    "world_exists",
    "list_worlds",
]

#: Names re-exported from `world.legacy.noise` on first attribute access.
#:
#: They are here because they were part of this package's public surface before
#: the noise module moved to `world/legacy/`, and removing them would break
#: `from world import PerlinNoise` for no gain. They are lazy so that
#: `import world` does not drag the numba kernels in for a caller that never
#: touches noise - which is most of them, now that `[world] generator =
#: "vanilla"` means the legacy Perlin field is not on the boot path at all.
_LAZY_NOISE = {
    "PerlinNoise": "PerlinNoise",
    "SimplexNoise": "SimplexNoise",
    "create_noise_generator": "create_noise_generator",
    "get_chunk_heightmap": "get_chunk_heightmap",
}


def __getattr__(name: str):
    source = _LAZY_NOISE.get(name)
    if source is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from world.legacy import noise

    value = getattr(noise, source)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(_LAZY_NOISE))
