"""
World Serializer Module

Handles saving and loading of world data including chunks, tiles, and entities.
Supports Anvil format and simplified JSON format for easier debugging.
"""

import gzip
import json
import struct
import zlib
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass
from loguru import logger

from world.chunk import Chunk, ChunkPos, ChunkSection
from world.world import World, WorldData


# NBT-like compression types
COMPRESSION_NONE = 0
COMPRESSION_ZLIB = 1
COMPRESSION_GZIP = 2


@dataclass
class RegionFileHeader:
    """Header for a .mca region file."""
    locations: List[int]  # 4096 bytes, 1024 entries (4 bytes each)
    timestamps: List[int]  # 4096 bytes, 1024 entries


class RegionSerializer:
    """
    Handles serialization of chunks to Anvil format (.mca files).
    
    This is a simplified implementation that stores chunks in a format
    compatible with Minecraft's Anvil format.
    """
    
    def __init__(self, region_dir: Path):
        """
        Initialize the region serializer.
        
        Args:
            region_dir: Directory to store region files
        """
        self.region_dir = region_dir
        self.region_dir.mkdir(parents=True, exist_ok=True)
    
    def get_region_file(self, chunk_x: int, chunk_z: int) -> Path:
        """Get the region file path for a chunk."""
        region_x = chunk_x // 32
        region_z = chunk_z // 32
        return self.region_dir / f"r.{region_x}.{region_z}.mca"
    
    def save_chunk(self, chunk: Chunk) -> bool:
        """
        Save a chunk to its region file.
        
        Args:
            chunk: Chunk to save
        
        Returns:
            True if successful
        """
        try:
            pos = chunk.position
            region_file = self.get_region_file(pos.x, pos.z)
            
            # Serialize chunk data
            chunk_data = self._serialize_chunk(chunk)
            
            # Compress data
            compressed_data = gzip.compress(chunk_data)
            
            # Calculate offset (location in file)
            offset = self._get_chunk_offset(region_file, pos.x, pos.z)
            
            # Write to region file
            self._write_chunk_to_region(region_file, offset, compressed_data)
            
            return True
            
        except Exception as e:
            logger.exception(f"Failed to save chunk {pos}: {e}")
            return False
    
    def load_chunk(self, chunk_x: int, chunk_z: int) -> Optional[Dict[str, Any]]:
        """
        Load a chunk from its region file.
        
        Args:
            chunk_x: Chunk X coordinate
            chunk_z: Chunk Z coordinate
        
        Returns:
            Chunk data dict, or None if not found
        """
        try:
            region_file = self.get_region_file(chunk_x, chunk_z)
            
            if not region_file.exists():
                return None
            
            # Read chunk from region
            chunk_data = self._read_chunk_from_region(region_file, chunk_x, chunk_z)
            
            if chunk_data:
                return self._deserialize_chunk(chunk_data)
            
            return None
            
        except Exception as e:
            logger.exception(f"Failed to load chunk ({chunk_x}, {chunk_z}): {e}")
            return None
    
    def _serialize_chunk(self, chunk: Chunk) -> bytes:
        """Serialize chunk to bytes."""
        sections = []
        
        for section in chunk.sections:
            if section.is_empty:
                sections.append(None)
                continue
            
            section_data = {
                "y": section.y,
                "blocks": section.blocks.tolist(),
                "blockLight": section.block_light.tolist(),
                "skyLight": section.sky_light.tolist(),
            }
            sections.append(section_data)
        
        heightmap = chunk.heightmap.tolist()
        
        chunk_dict = {
            "sections": sections,
            "heightmap": heightmap,
            "blockEntities": [],  # Would contain tile entities
            "entities": [],  # Would contain entities
        }
        
        return json.dumps(chunk_dict).encode('utf-8')
    
    def _deserialize_chunk(self, data: bytes) -> Dict[str, Any]:
        """Deserialize chunk from bytes."""
        return json.loads(data.decode('utf-8'))
    
    def _get_chunk_offset(self, region_file: Path, chunk_x: int, chunk_z: int) -> int:
        """Get the offset of a chunk in the region file."""
        if not region_file.exists():
            return 0
        
        # Read header
        with open(region_file, 'rb') as f:
            # Location table is at beginning
            index = ((chunk_x % 32) + (chunk_z % 32) * 32) * 4
            f.seek(index)
            location = struct.unpack('>I', f.read(4))[0]
            
            if location == 0:
                return 0
            
            # Offset is in 4KB sectors
            offset = (location >> 8) * 4096
            return offset
    
    def _write_chunk_to_region(
        self,
        region_file: Path,
        offset: int,
        data: bytes
    ) -> None:
        """Write chunk data to region file."""
        # Create file if doesn't exist
        if not region_file.exists():
            region_file.touch()
        
        # For simplicity, append to end (real implementation would update in place)
        with open(region_file, 'r+b') as f:
            f.seek(0, 2)  # Seek to end
            sector_offset = f.tell()
            
            # Write header if file is small
            if sector_offset < 8192:  # 2 sectors for header
                f.seek(0)
                # Write zero header
                f.write(b'\x00' * 8192)
            
            # Write chunk data
            f.seek(0, 2)
            
            # Chunk header: length (4 bytes) + compression type (1 byte) + data
            chunk_length = len(data) + 1  # Include compression byte
            header = struct.pack('>I', chunk_length) + struct.pack('B', COMPRESSION_GZIP)
            
            f.write(header)
            f.write(data)
            
            # Pad to 4KB boundary
            padding = (4096 - (f.tell() % 4096)) % 4096
            if padding:
                f.write(b'\x00' * padding)
    
    def _read_chunk_from_region(
        self,
        region_file: Path,
        chunk_x: int,
        chunk_z: int
    ) -> Optional[bytes]:
        """Read chunk data from region file."""
        # Simplified implementation
        # In real use, would parse the location table and read from offset
        
        try:
            # For debugging, try reading as JSON
            chunk_file = region_file.with_suffix('.json')
            if chunk_file.exists():
                with open(chunk_file, 'r') as f:
                    return f.read().encode('utf-8')
            
            return None
            
        except Exception as e:
            logger.debug(f"Could not read chunk from region: {e}")
            return None
    
    def chunk_exists(self, chunk_x: int, chunk_z: int) -> bool:
        """Check if a chunk exists in the region file."""
        region_file = self.get_region_file(chunk_x, chunk_z)
        
        if not region_file.exists():
            return False
        
        # Check if chunk file exists (for JSON fallback)
        chunk_file = region_file.with_suffix('.json')
        return chunk_file.exists()
    
    def delete_chunk(self, chunk_x: int, chunk_z: int) -> bool:
        """Delete a chunk."""
        try:
            region_file = self.get_region_file(chunk_x, chunk_z)
            
            # Try JSON file first
            chunk_file = region_file.with_suffix('.json')
            if chunk_file.exists():
                chunk_file.unlink()
                return True
            
            return False
            
        except Exception as e:
            logger.exception(f"Failed to delete chunk: {e}")
            return False


class LevelDatSerializer:
    """Handles serialization of level.dat (world metadata)."""
    
    @staticmethod
    def save(world_data: WorldData, file_path: Path) -> bool:
        """
        Save world data to level.dat.
        
        Args:
            world_data: World data to save
            file_path: Path to level.dat file
        
        Returns:
            True if successful
        """
        try:
            data = {
                "Data": {
                    "LevelName": world_data.name,
                    "generatorName": "default",
                    "generatorVersion": 2,
                    "generatorOptions": "",
                    "seed": world_data.seed,
                    # Only present once a spawn has been chosen; an absent
                    # key is what tells the loader to derive one.
                    **({"SpawnX": world_data.spawn_x,
                       "SpawnY": world_data.spawn_y,
                       "SpawnZ": world_data.spawn_z}
                      if world_data.spawn_x is not None else {}),
                    "WorldGenSettings": {},
                    "GameType": 0,  # 0=Survival, 1=Creative, 2=Adventure, 3=Spectator
                    "Difficulty": 2,  # Normal
                    "Hardcore": 1 if world_data.hardcore else 0,
                    "Time": world_data.world_time,
                    "DayTime": world_data.day_time,
                    "RainTime": world_data.rain_time,
                    "Raining": 1 if world_data.is_raining else 0,
                    "ThunderTime": world_data.thunder_time,
                    "Thundering": 1 if world_data.is_thundering else 0,
                    "GameRules": world_data.game_rules,
                    "BorderCenterX": world_data.border_center_x,
                    "BorderCenterZ": world_data.border_center_z,
                    "BorderSize": world_data.border_size,
                    "BorderSizeLerpTarget": world_data.border_size_lerp_target,
                    "BorderSizeLerpTime": world_data.border_size_lerp_time,
                    "BorderSafeZone": world_data.border_safe_zone,
                    "BorderWarningBlocks": world_data.border_warning_blocks,
                    "BorderWarningTime": world_data.border_warning_time,
                    "LastPlayed": int(world_data.last_loaded_time * 1000),
                }
            }
            
            # Save as JSON for debugging
            json_path = file_path.with_suffix('.json')
            with open(json_path, 'w') as f:
                json.dump(data, f, indent=2)
            
            logger.debug(f"Saved level.dat to {json_path}")
            return True
            
        except Exception as e:
            logger.exception(f"Failed to save level.dat: {e}")
            return False
    
    @staticmethod
    def load(file_path: Path) -> Optional[WorldData]:
        """
        Load world data from level.dat.
        
        Args:
            file_path: Path to level.dat file
        
        Returns:
            WorldData object, or None if loading failed
        """
        try:
            # Try JSON format first
            json_path = file_path.with_suffix('.json')
            
            if not json_path.exists():
                json_path = file_path
            
            if not json_path.exists():
                logger.error(f"No level.dat found at {file_path}")
                return None
            
            with open(json_path, 'r') as f:
                data = json.load(f)
            
            data = data.get("Data", {})
            
            world_data = WorldData()
            world_data.name = data.get("LevelName", "Unknown")
            world_data.seed = data.get("seed", 0)
            world_data.spawn_x = data.get("SpawnX")
            world_data.spawn_y = data.get("SpawnY")
            world_data.spawn_z = data.get("SpawnZ")
            world_data.world_time = data.get("Time", 0)
            world_data.day_time = data.get("DayTime", 6000)
            world_data.game_rules = data.get("GameRules", world_data.game_rules)
            world_data.border_center_x = data.get("BorderCenterX", 0.0)
            world_data.border_center_z = data.get("BorderCenterZ", 0.0)
            world_data.border_size = data.get("BorderSize", 59999968.0)
            world_data.border_size_lerp_target = data.get("BorderSizeLerpTarget", 59999968.0)
            world_data.border_size_lerp_time = data.get("BorderSizeLerpTime", 0)
            world_data.border_safe_zone = data.get("BorderSafeZone", 5.0)
            world_data.border_warning_blocks = data.get("BorderWarningBlocks", 5.0)
            world_data.border_warning_time = data.get("BorderWarningTime", 15)
            world_data.rain_time = data.get("RainTime", 0)
            world_data.is_raining = data.get("Raining", 0) == 1
            world_data.thunder_time = data.get("ThunderTime", 0)
            world_data.is_thundering = data.get("Thundering", 0) == 1
            world_data.hardcore = data.get("Hardcore", 0) == 1
            
            # Map difficulty
            difficulty_map = {0: "peaceful", 1: "easy", 2: "normal", 3: "hard"}
            difficulty = data.get("Difficulty", 2)
            world_data.difficulty = difficulty_map.get(difficulty, "normal")
            
            # Map gamemode
            gamemode_map = {0: "survival", 1: "creative", 2: "adventure", 3: "spectator"}
            gamemode = data.get("GameType", 0)
            world_data.gamemode = gamemode_map.get(gamemode, "survival")
            
            return world_data
            
        except Exception as e:
            logger.exception(f"Failed to load level.dat: {e}")
            return None


class PlayerSerializer:
    """Handles serialization of player data."""
    
    @staticmethod
    def save_player(
        player_id: str,
        data: Dict[str, Any],
        players_dir: Path
    ) -> bool:
        """
        Save player data.
        
        Args:
            player_id: Player UUID
            data: Player data dict
            players_dir: Directory to save player files
        
        Returns:
            True if successful
        """
        try:
            players_dir.mkdir(parents=True, exist_ok=True)
            
            # Save as JSON
            player_file = players_dir / f"{player_id}.json"
            
            with open(player_file, 'w') as f:
                json.dump(data, f, indent=2)
            
            return True
            
        except Exception as e:
            logger.exception(f"Failed to save player {player_id}: {e}")
            return False
    
    @staticmethod
    def load_player(player_id: str, players_dir: Path) -> Optional[Dict[str, Any]]:
        """
        Load player data.
        
        Args:
            player_id: Player UUID
            players_dir: Directory containing player files
        
        Returns:
            Player data dict, or None if not found
        """
        try:
            player_file = players_dir / f"{player_id}.json"
            
            if not player_file.exists():
                return None
            
            with open(player_file, 'r') as f:
                return json.load(f)
            
        except Exception as e:
            logger.exception(f"Failed to load player {player_id}: {e}")
            return None
    
    @staticmethod
    def list_players(players_dir: Path) -> List[str]:
        """
        List all saved players.
        
        Args:
            players_dir: Directory containing player files
        
        Returns:
            List of player IDs
        """
        try:
            if not players_dir.exists():
                return []
            
            return [
                f.stem for f in players_dir.glob("*.json")
            ]
            
        except Exception as e:
            logger.exception(f"Failed to list players: {e}")
            return []


# World directory operations
def create_world_directory(name: str, base_dir: Path) -> Path:
    """
    Create a new world directory structure.
    
    Args:
        name: World name
        base_dir: Base directory for worlds
    
    Returns:
        Path to the created world directory
    """
    world_dir = base_dir / name
    
    # Create directory structure
    (world_dir / "region" / "overworld").mkdir(parents=True, exist_ok=True)
    (world_dir / "region" / "the_nether").mkdir(parents=True, exist_ok=True)
    (world_dir / "region" / "the_end").mkdir(parents=True, exist_ok=True)
    (world_dir / "playerdata").mkdir(parents=True, exist_ok=True)
    (world_dir / "stats").mkdir(parents=True, exist_ok=True)
    (world_dir / "advancements").mkdir(parents=True, exist_ok=True)
    (world_dir / "data").mkdir(parents=True, exist_ok=True)
    
    # Create level.dat
    world_data = WorldData(name=name)
    LevelDatSerializer.save(world_data, world_dir / "level.dat")
    
    logger.info(f"Created world directory: {world_dir}")
    return world_dir


def list_worlds(base_dir: Path) -> List[str]:
    """
    List all worlds in the base directory.
    
    Args:
        base_dir: Base directory for worlds
    
    Returns:
        List of world names
    """
    try:
        if not base_dir.exists():
            return []
        
        worlds = []
        for path in base_dir.iterdir():
            if path.is_dir() and (path / "level.dat").exists() or (path / "level.dat.json").exists():
                worlds.append(path.name)
        
        return sorted(worlds)
        
    except Exception as e:
        logger.exception(f"Failed to list worlds: {e}")
        return []


def delete_world(name: str, base_dir: Path) -> bool:
    """
    Delete a world.
    
    Args:
        name: World name
        base_dir: Base directory for worlds
    
    Returns:
        True if successful
    """
    try:
        import shutil
        
        world_dir = base_dir / name
        
        if not world_dir.exists():
            return False
        
        # Delete world directory
        shutil.rmtree(world_dir)
        
        logger.info(f"Deleted world: {name}")
        return True
        
    except Exception as e:
        logger.exception(f"Failed to delete world {name}: {e}")
        return False
