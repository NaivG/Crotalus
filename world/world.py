"""
World Module

Main world management class that handles dimensions, players, and world operations.
Uses NBT format for level.dat and Anvil format for region storage.
"""

import json
import os
import time
from pathlib import Path
from typing import Optional, List, Dict, Any, Callable, Tuple
from dataclasses import dataclass, field
import threading
from loguru import logger

from world.chunk_saver import ChunkSaver, should_persist
from world.chunk_lifecycle import ChunkLifecycle, TicketMsg
from world.chunk_ticket import FORCED
from world.dimension import Dimension, DimensionType, DimensionSettings
from world.generator import BaseGenerator, GeneratorSettings, GeneratorType, create_default_generator, create_generator
from world.block_states import BLOCKS
from world.chunk import Chunk, ChunkPos
from world.chunk_gen_thread import ChunkGeneratorThread, chunk_generator
from world.nbt import save_level_dat, load_level_dat
from world.anvil import RegionManager, create_world_directories, world_exists, list_worlds


@dataclass
class WorldData:
    """World metadata and settings."""
    name: str = "world"
    seed: int = 0
    gamemode: str = "survival"
    difficulty: str = "easy"
    hardcore: bool = False
    
    # Spawn settings. None means the world has never had a spawn picked --
    # the terrain decides where players start, and the first one to join
    # settles it. A concrete int here is a spawn that has been chosen and
    # saved, and it wins from then on.
    spawn_x: Optional[int] = None
    spawn_y: Optional[int] = None
    spawn_z: Optional[int] = None
    
    # World border
    border_center_x: float = 0.0
    border_center_z: float = 0.0
    border_size: float = 59999968.0
    border_size_delta: float = 0.0
    border_size_lerp_target: float = 59999968.0
    border_size_lerp_time: int = 0
    border_safe_zone: float = 5.0
    border_warning_blocks: float = 5.0
    border_warning_time: int = 15
    
    # Time settings
    world_time: int = 0
    day_time: int = 6000
    game_time: int = 0
    
    # Rain/weather
    rain_time: int = 0
    is_raining: bool = False
    rain_intensity: float = 0.0
    thunder_time: int = 0
    is_thundering: bool = False
    thunder_intensity: float = 0.0
    
    # Game rules
    game_rules: Dict[str, str] = field(default_factory=lambda: {
        "doMobSpawning": "true",
        "doDaylightCycle": "true",
        "doWeatherCycle": "true",
        "doFireTick": "true",
        "doMobLoot": "true",
        "doTileDrops": "true",
        "doEntityDrops": "true",
        "keepInventory": "false",
        "randomTickSpeed": "3",
        "spawnRadius": "10",
        "maxEntityCramming": "24",
    })
    
    # Last saved
    last_loaded_time: int = 0
    last_save_time: int = 0
    
    # Dimension settings
    level_type: str = "minecraft:normal"
    
    def to_nbt_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for NBT serialization."""
        return {
            "LevelName": self.name,
            "RandomSeed": self.seed,
            "SpawnX": self.spawn_x,
            "SpawnY": self.spawn_y,
            "SpawnZ": self.spawn_z,
            "WorldGenSettings": {},
            "generatorName": "minecraft:noise",
            "generatorOptions": "",
            "Difficulty": {"peaceful": 0, "easy": 1, "normal": 2, "hard": 3}.get(self.difficulty.lower(), 2),
            "Hardcore": self.hardcore,
            "GameType": {"survival": 0, "creative": 1, "adventure": 2, "spectator": 3}.get(self.gamemode.lower(), 0),
            "Time": self.world_time,
            "DayTime": self.day_time,
            "GameRules": self.game_rules,
            "BorderCenterX": self.border_center_x,
            "BorderCenterZ": self.border_center_z,
            "BorderSize": self.border_size,
            "BorderSizeLerpTarget": self.border_size_lerp_target,
            "BorderSizeLerpTime": self.border_size_lerp_time,
            "BorderSafeZone": self.border_safe_zone,
            "BorderWarningBlocks": self.border_warning_blocks,
            "BorderWarningTime": self.border_warning_time,
            "RainTime": self.rain_time,
            "raining": self.is_raining,
            "ThunderTime": self.thunder_time,
            "thundering": self.is_thundering,
            "LevelType": self.level_type,
        }
    
    @classmethod
    def from_nbt_dict(cls, data: Dict[str, Any]) -> "WorldData":
        """Create from NBT dictionary."""
        world_data = cls()
        
        # Basic info
        world_data.name = data.get("LevelName", "Unknown")
        # Support both 'seed' and 'RandomSeed' field names
        world_data.seed = data.get("seed", data.get("RandomSeed", 0))
        world_data.spawn_x = data.get("SpawnX")
        world_data.spawn_y = data.get("SpawnY")
        world_data.spawn_z = data.get("SpawnZ")
        world_data.world_time = data.get("Time", 0)
        world_data.day_time = data.get("DayTime", 6000)
        
        # Difficulty - support both 'difficulty' and 'Difficulty'
        diff_val = data.get("difficulty", data.get("Difficulty", 2))
        diff_map = {0: "peaceful", 1: "easy", 2: "normal", 3: "hard"}
        world_data.difficulty = diff_map.get(diff_val, "normal")
        
        # Gamemode
        gm_map = {0: "survival", 1: "creative", 2: "adventure", 3: "spectator"}
        world_data.gamemode = gm_map.get(data.get("GameType", 0), "survival")
        
        # Hardcore - handle both bool and int types
        hardcore_val = data.get("hardcore", data.get("Hardcore", False))
        world_data.hardcore = bool(hardcore_val) if isinstance(hardcore_val, int) else hardcore_val
        
        # Game rules
        game_rules = data.get("GameRules", {})
        if isinstance(game_rules, dict):
            world_data.game_rules = game_rules
        elif isinstance(game_rules, list):
            # Parse list format
            for rule in game_rules:
                if isinstance(rule, dict):
                    name = rule.get("name", "")
                    value = rule.get("value", "")
                    if name:
                        world_data.game_rules[name] = value
        
        # Border
        world_data.border_center_x = data.get("BorderCenterX", 0.0)
        world_data.border_center_z = data.get("BorderCenterZ", 0.0)
        world_data.border_size = data.get("BorderSize", 59999968.0)
        world_data.border_size_lerp_target = data.get("BorderSizeLerpTarget", 59999968.0)
        world_data.border_size_lerp_time = data.get("BorderSizeLerpTime", 0)
        world_data.border_safe_zone = data.get("BorderSafeZone", 5.0)
        world_data.border_warning_blocks = data.get("BorderWarningBlocks", 5.0)
        world_data.border_warning_time = data.get("BorderWarningTime", 0)
        
        # Weather - support both lowercase and capitalized field names
        world_data.rain_time = data.get("RainTime", 0)
        # Handle both 'is_raining', 'raining', and 'Raining' variants
        raining_val = data.get("is_raining", data.get("raining", data.get("Raining", 0)))
        world_data.is_raining = bool(raining_val) if isinstance(raining_val, int) else raining_val
        
        world_data.thunder_time = data.get("ThunderTime", 0)
        # Handle both 'is_thundering', 'thundering', and 'Thundering' variants
        thundering_val = data.get("is_thundering", data.get("thundering", data.get("Thundering", 0)))
        world_data.is_thundering = bool(thundering_val) if isinstance(thundering_val, int) else thundering_val
        
        return world_data


#: Budget for `World.light_chunks_blocking`'s inline drain. A 3x3 patch
#: measures ~5 ms (tools/bench_light.py), so this is ~10 patches per tick --
#: the radius-4 spawn disc's interior lands in a fraction of a second instead
#: of the many ticks the per-tick `[world]` budget would spend on it. This is
#: boot latency on the main thread, where nothing else is competing, which is
#: the same trade `core/server.py::_warm_up_kernels` already makes.
SPAWN_LIGHT_BUDGET_MS = 50.0


def _generated_3x3(dimension: Dimension, chunk_x: int, chunk_z: int) -> bool:
    """Whether all nine chunks of `(chunk_x, chunk_z)`'s region have terrain.

    `LightRegion.build` reads a chunk's blocks whether or not that chunk has
    ever been lit, so the only thing a patch needs is nine *generated* chunks:
    a blank claim reads as all air and would have the patch compute a field
    that a terrain-less neighbour contradicts. This is the same readiness test
    `light_engine._light_chunk` makes, asked before queueing instead of after
    failing.
    """
    for dz in (-1, 0, 1):
        for dx in (-1, 0, 1):
            chunk = dimension.get_chunk(chunk_x + dx, chunk_z + dz)
            if chunk is None or not chunk.is_generated:
                return False
    return True


class World:
    """
    Main world class that manages all dimensions and world operations.
    
    This is the central manager for the world system, handling:
    - Multiple dimensions (overworld, nether, end)
    - World saving and loading (NBT + Anvil format)
    - Player management
    - Game tick operations
    - Background chunk generation
    """
    
    def __init__(
        self,
        name: str = "world",
        seed: int = 0,
        generator: Optional[BaseGenerator] = None,
        world_dir: Optional[Path] = None,
        create_if_missing: bool = True
    ):
        """
        Initialize a world.
        
        Args:
            name: World name
            seed: World seed for generation
            generator: Custom world generator
            world_dir: Directory to store world data
            create_if_missing: Create world directory if it doesn't exist
        """
        self.name = name
        self.seed = seed
        # If world_dir is provided, use it directly as the world directory
        # Otherwise, use data/{name} as the world directory
        self.world_dir = world_dir if world_dir else Path("data") / name
        
        # Check if world exists
        # The world directory is self.world_dir, so check if level.dat exists there
        level_dat = self.world_dir / "level.dat"
        if not level_dat.exists():
            if create_if_missing:
                logger.info(f"World '{name}' not found, creating new world...")
                self._create_world()
                # Initialize data from the newly created world
                self.data = WorldData(name=self.name, seed=self.seed)
            else:
                raise FileNotFoundError(f"World '{name}' does not exist")
        else:
            # Backup level.dat before loading
            backup_path = self.world_dir / "level.dat_old"
            try:
                import shutil
                shutil.copy2(level_dat, backup_path)
                logger.debug(f"Backed up level.dat to level.dat_old")
            except Exception as e:
                logger.warning(f"Failed to backup level.dat: {e}")
            
            # Load existing world (sets self.data)
            self._load_world()
        
        # Ensure data object is initialized
        if not hasattr(self, 'data') or self.data is None:
            self.data = WorldData(name=self.name, seed=self.seed)
        
        # Region managers for each dimension
        self.region_managers: Dict[DimensionType, RegionManager] = {}
        self._init_region_managers()

        # Streaming chunk save. Chunks that leave memory are handed to this
        # pipeline *before* they are dropped (see world/chunk_saver.py): the
        # actor thread must not pay a region write inside its 50 ms tick, and
        # `max_chunks` eviction must never silently discard an edited chunk the
        # way it used to.
        from core.config import config as app_config

        self.chunk_saver = ChunkSaver(
            self._region_manager_for,
            workers=app_config.get("world.chunk-save-workers", 1),
            sync_writes=bool(app_config.get("server.sync-chunk-writes", True)),
        )

        # Dimensions
        self.dimensions: Dict[DimensionType, Dimension] = {}

        # Automatic load/unload drivers, one per dimension. Created here rather
        # than in the actor so that `World` -- and anything holding only a
        # `World` -- can add tickets, force-load chunks and read the level map;
        # the actor is what ticks them while the server runs, and a tool can
        # tick them by hand.
        self.chunk_lifecycles: Dict[DimensionType, ChunkLifecycle] = {}
        
        # Generator
        if generator:
            self.generator = generator
        else:
            self.generator = create_default_generator("normal", self.seed)
        
        # Initialize default dimensions
        self._init_dimensions()
        self._wire_chunk_storage()

        # Force-loaded chunks (and live portal tickets) come back from the
        # previous session. They are restored onto the lifecycles, so the first
        # tick loads exactly those coordinates again.
        restored = self._load_tickets()
        if restored:
            logger.info(f"Restored {restored} persisted chunk ticket(s)")
        
        # Player spawn tracking
        self.players: Dict[str, str] = {}  # player_id -> dimension_name
        
        #: Set by `core/server` while the dimension actors are running:
        #: `(dimension) -> inbox or None`. It is how the console and the world
        #: API reach the single owner of a dimension's ticket state.
        self.chunk_actor_inbox: Optional[Callable[[DimensionType], Optional[object]]] = None

        # Tick management (ticking runs on the dimension actors,
        # see world/dimension_actor.py)
        self._tick_rate = 20  # TPS
        self._tick_count = 0
        
        # Tick callbacks
        self._tick_callbacks: List[Callable[[], None]] = []
        
        # Lock
        self._lock = threading.RLock()
        
        # Wire up background generation. The worker pool itself starts lazily
        # on the first request: a pool of interpreter processes is a real
        # cost (spawn + registry import each), and callers that only want a
        # World for synchronous work -- tools, tests -- should not pay it.
        # The seed is not part of the pool either: it travels on every
        # request, so the pool stays world-agnostic.
        chunk_generator.configure(
            workers=app_config.chunk_gen_workers,
            backend=app_config.generator_backend,
            mode=app_config.chunk_gen_mode,
        )
        chunk_generator.set_completion_sink(self.insert_generated_chunk)
        chunk_generator.set_failure_sink(self.discard_failed_chunk)
        
        logger.info(f"World '{name}' initialized with seed {self.seed}")
    
    def _create_world(self) -> None:
        """Create a new world directory structure."""
        # Create the world directory
        self.world_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories (region, playerdata, etc.)
        (self.world_dir / "region" / "overworld").mkdir(parents=True, exist_ok=True)
        (self.world_dir / "region" / "_DIM-1").mkdir(parents=True, exist_ok=True)
        (self.world_dir / "region" / "DIM1").mkdir(parents=True, exist_ok=True)
        (self.world_dir / "playerdata").mkdir(parents=True, exist_ok=True)
        (self.world_dir / "stats").mkdir(parents=True, exist_ok=True)
        (self.world_dir / "advancements").mkdir(parents=True, exist_ok=True)
        (self.world_dir / "poi").mkdir(parents=True, exist_ok=True)
        
        # Create level.dat with NBT format
        level_dat_path = self.world_dir / "level.dat"
        save_level_dat(
            str(level_dat_path),
            level_name=self.name,
            seed=self.seed,
            # No spawn: it is picked from the terrain the first time it is
            # asked for, and saved then. Hardcoding one here would fix every
            # new world to the same Y regardless of what the generator makes.
            spawn_x=None,
            spawn_y=None,
            spawn_z=None,
            world_time=0,
            day_time=6000,
            game_rules={
                "doMobSpawning": "true",
                "doDaylightCycle": "true",
                "doWeatherCycle": "true",
                "keepInventory": "false",
                "randomTickSpeed": "3",
            },
            difficulty=2,  # Normal
            hardcore=False,
            game_type=0,  # Survival
        )
        logger.info(f"Created new world: {self.name}")
    
    def _load_world(self) -> None:
        """Load world data from disk."""
        level_dat_path = self.world_dir / "level.dat"
        
        if level_dat_path.exists():
            try:
                data = load_level_dat(str(level_dat_path))
                self.name = data.get("LevelName", self.name)
                self.seed = data.get("seed", self.seed)
                
                # Update WorldData object with loaded data
                self.data = WorldData.from_nbt_dict(data)
                
                logger.info(f"Loaded world data for '{self.name}' (seed: {self.seed})")
            except Exception as e:
                logger.warning(f"Failed to load level.dat: {e}")
                logger.info("Using default world settings")
        else:
            logger.warning(f"level.dat not found in {self.world_dir}")
    
    def _init_region_managers(self) -> None:
        """Initialize region managers for each dimension.

        Each manager stamps its dimension's vertical window onto the chunks it
        reads, so a reloaded chunk has the same section layout a generated one
        does (24 sections in the overworld, 16 in the nether and the end).
        """
        overworld_dir = self.world_dir / "region"
        nether_dir = self.world_dir / "region" / "_DIM-1"
        end_dir = self.world_dir / "region" / "DIM1"

        self.region_managers = {}
        for dim_type, path in (
            (DimensionType.OVERWORLD, overworld_dir),
            (DimensionType.NETHER, nether_dir),
            (DimensionType.END, end_dir),
        ):
            settings = DimensionSettings.for_dimension(dim_type)
            self.region_managers[dim_type] = RegionManager(
                path, min_y=settings.min_y, height=settings.height)
    
    def _init_dimensions(self) -> None:
        """Initialize all dimensions.

        Each generator is built with the `chunk_gen_settings.json` preset its
        dimension type names (`registry.worldgen`), so the nether gets
        netherrack and lava rather than the overworld's stone and water. The
        height band stays hand-set: it is this Perlin generator's own shape, not
        a value the data carries.
        """
        # Overworld
        self.dimensions[DimensionType.OVERWORLD] = Dimension(
            DimensionType.OVERWORLD,
            self.seed,
            self.generator
        )
        
        # Nether. The same settings object feeds whichever backend
        # `[world] generator` selected. `base_height`/`height_variance` are
        # legacy-only and shape the Perlin terrain the legacy backend extrudes;
        # the vanilla backend reads its shape from the nether's density
        # function and ignores both. `sea_level` is not really set here - it
        # names the preset's value, and `_apply_registry` overwrites it with
        # whatever `chunk_gen_settings.json` says, which is also 32.
        nether_gen = create_generator(GeneratorSettings(
            generator_type=GeneratorType.NORMAL,
            seed=self.seed + 1,
            preset="nether",
            base_height=32,
            height_variance=32
        ))
        self.dimensions[DimensionType.NETHER] = Dimension(
            DimensionType.NETHER,
            self.seed,
            nether_gen
        )
        
        # End
        self.dimensions[DimensionType.END] = Dimension(
            DimensionType.END,
            self.seed
        )
    
    def _region_manager_for(self, dimension: DimensionType) -> Optional[RegionManager]:
        """Region manager for a dimension, or None (the saver's lookup hook)."""
        return self.region_managers.get(dimension)

    def _wire_chunk_storage(self) -> None:
        """Connect dimensions to the save pipeline and the auto load/unload driver.

        Three connections per dimension, and they have to be made here because
        `World` is the only object that owns both the region managers and the
        dimensions:

        * the `ChunkManager` eviction handler -- every chunk that leaves memory
          is offered to the streaming saver first, so the LRU cap can no longer
          discard an edited chunk the way it used to;
        * the dimension's synchronous save handler, for `save_all_chunks()`;
        * a `ChunkLifecycle`, which turns the dimension's tickets into loaded
          and unloaded chunks. Its save hook is the same saver.
        """
        for dim_type, dimension in self.dimensions.items():
            manager = dimension.chunk_manager
            manager.set_eviction_handler(self._eviction_handler(dim_type))
            dimension.save_handler = (
                lambda dt=dim_type: self._save_dimension(dt)
            )
            self.chunk_lifecycles[dim_type] = ChunkLifecycle(
                dimension,
                save_chunk=self._save_hook(dim_type),
                # Disk first: a chunk a player edited in an earlier session must
                # come back as it was saved, not regenerated from the seed.
                load_chunk=lambda x, z, dt=dim_type: self.load_chunk(
                    x, z, dimension=dt
                ),
            )

    def _eviction_handler(self, dim_type: DimensionType) -> Callable[[Chunk], bool]:
        """Hand a chunk that is leaving memory to the streaming saver.

        Returns False only when the saver has no region manager for the
        dimension -- the manager then keeps the chunk instead of losing it.
        """

        def handler(chunk: Chunk) -> bool:
            if not should_persist(chunk):
                # Nothing to write: a blank claim, or a chunk read from disk and
                # untouched since. Dropping it is free.
                return True
            if self.region_managers.get(dim_type) is None:
                logger.error(
                    f"Refusing to unload chunk {chunk.position} in "
                    f"{dim_type.value}: no region manager to save it to"
                )
                return False
            return self.chunk_saver.submit(dim_type, chunk, force=True)

        return handler

    def _save_hook(self, dim_type: DimensionType) -> Callable[[Chunk], bool]:
        """The lifecycle's save callback: same saver, same ownership transfer."""

        def hook(chunk: Chunk) -> bool:
            if self.region_managers.get(dim_type) is None:
                return False
            return self.chunk_saver.submit(dim_type, chunk, force=True)

        return hook

    def _save_dimension(self, dim_type: DimensionType) -> int:
        """Save one dimension's dirty chunks synchronously.

        Snapshot the dict: iterating it live while a chunk is inserted (a
        finished background generation) raises "dictionary changed size during
        iteration" and aborts the whole save.
        """
        dimension = self.dimensions.get(dim_type)
        region_manager = self.region_managers.get(dim_type)
        if dimension is None or region_manager is None:
            return 0

        saved = 0
        for chunk in list(dimension.chunk_manager.chunks.values()):
            if chunk.is_dirty:
                region_manager.save_chunk(chunk)
                chunk.mark_saved()
                saved += 1
        if saved:
            region_manager.save_all()
        return saved

    # ---------------- load levels / automatic loading ----------------

    def lifecycle(self, dimension: DimensionType = DimensionType.OVERWORLD) -> Optional[ChunkLifecycle]:
        """The auto load/unload driver for a dimension.

        Reading it is fine from any thread; *calling* into it is not once an
        actor owns it. `core/server` installs `chunk_actor_inbox` for exactly
        that reason -- see `forceload` and `unload_chunk_now`.
        """
        return self.chunk_lifecycles.get(dimension)

    def _post_to_actor(self, dimension: DimensionType, msg) -> bool:
        """Hand a message to the dimension's actor, if one is running.

        Returns False when nothing owns the dimension (tools, tests, or a
        stopped server), in which case the caller may touch the lifecycle
        directly: nobody else is.
        """
        hook = self.chunk_actor_inbox
        if hook is None:
            return False
        try:
            inbox = hook(dimension)
        except Exception:  # pragma: no cover - defensive
            logger.exception(f"Chunk actor lookup failed for {dimension}")
            return False
        if inbox is None:
            return False
        inbox.put(msg)
        return True

    def forceload(
        self,
        x: int,
        z: int,
        dimension: DimensionType = DimensionType.OVERWORLD,
        loaded: bool = True,
    ) -> bool:
        """Keep a chunk loaded across the auto-unload pass (and across restarts).

        Force-loaded chunks are persisted: they are written to
        `forced_chunks.json` next to `level.dat` and restored on the next boot,
        which is what the ticket type's PERSIST flag promises.

        With a running actor the request is posted to the owning thread rather
        than applied here, because the ticket table is single-owner state.
        """
        lifecycle = self.chunk_lifecycles.get(dimension)
        if lifecycle is None:
            return False
        action = "force" if loaded else "unforce"
        if self._post_to_actor(dimension, TicketMsg(action=action, x=x, z=z)):
            return True
        return lifecycle.force_load(x, z, loaded)

    def unload_chunk_now(
        self,
        x: int,
        z: int,
        dimension: DimensionType = DimensionType.OVERWORLD,
        *,
        save: bool = True,
        force: bool = False,
    ) -> bool:
        """Take a chunk out of memory immediately, on the owning thread.

        Refuses while a ticket still covers the chunk (it would be reloaded on
        the next tick); `force=True` is for operators and tests.
        """
        lifecycle = self.chunk_lifecycles.get(dimension)
        if lifecycle is None:
            return False
        if self._post_to_actor(
            dimension,
            TicketMsg(action="unload", x=x, z=z, save=save, force=force),
        ):
            return True
        return lifecycle.unload_chunk(x, z, save=save, force=force)

    def get_load_level(self, x: int, z: int, dimension: DimensionType = DimensionType.OVERWORLD) -> Optional[int]:
        """Level of a chunk in a dimension (see `Dimension.get_load_level`).

        A read of the dimension's ticket table: safe from any thread that is not
        racing the actor. The console reads the same information through
        `get_chunk_stats`, which asks the owning thread instead.
        """
        d = self.dimensions.get(dimension)
        return None if d is None else d.get_load_level(x, z)

    def get_load_type(self, x: int, z: int, dimension: DimensionType = DimensionType.OVERWORLD):
        d = self.dimensions.get(dimension)
        return None if d is None else d.get_load_type(x, z)

    # ---------------- forced-chunk persistence ----------------

    def _tickets_path(self) -> Path:
        return self.world_dir / "forced_chunks.json"

    def _save_tickets(self) -> None:
        """Persist every PERSIST-flagged ticket, per dimension."""
        if not self.chunk_lifecycles:
            return
        payload: Dict[str, Any] = {"version": 1, "dimensions": {}}
        for dim_type, lifecycle in self.chunk_lifecycles.items():
            records = lifecycle.persistable_tickets()
            if records:
                payload["dimensions"][dim_type.value] = records
        try:
            self._tickets_path().write_text(
                json.dumps(payload, indent=1), encoding="utf-8"
            )
        except Exception as error:
            logger.warning(f"Failed to write forced chunk list: {error}")

    def _load_tickets(self) -> int:
        """Restore persisted tickets (force-loaded chunks, live portals)."""
        path = self._tickets_path()
        if not path.exists():
            return 0
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception as error:
            logger.warning(f"Failed to read forced chunk list: {error}")
            return 0

        restored = 0
        for name, records in (payload.get("dimensions") or {}).items():
            try:
                dim_type = DimensionType(name)
            except ValueError:
                logger.warning(f"forced_chunks.json names unknown dimension {name!r}")
                continue
            lifecycle = self.chunk_lifecycles.get(dim_type)
            if lifecycle is None:
                continue
            restored += lifecycle.restore_tickets(records)
            forced = [
                ticket for ticket in lifecycle.tickets if ticket.type is FORCED
            ]
            logger.info(
                f"Restored {len(forced)} force-loaded chunk(s) in {dim_type.value}"
            )
        return restored

    @property
    def overworld(self) -> Dimension:
        """Get the overworld dimension."""
        return self.dimensions[DimensionType.OVERWORLD]
    
    @property
    def nether(self) -> Dimension:
        """Get the nether dimension."""
        return self.dimensions[DimensionType.NETHER]
    
    @property
    def end(self) -> Dimension:
        """Get the end dimension."""
        return self.dimensions[DimensionType.END]
    
    def get_dimension(self, dim_type: DimensionType) -> Optional[Dimension]:
        """Get a dimension by type."""
        return self.dimensions.get(dim_type)
    
    def get_spawn_point(self) -> tuple[int, int, int]:
        """Where players start, resolving it against the terrain if unset.

        A world that has never picked a spawn returns whatever
        `find_surface_spawn` makes of the origin chunk. The caller is
        expected to have loaded that chunk first -- asking before the terrain
        exists would hand back a height of zero and drop the player into
        bedrock.
        """
        if (self.data.spawn_x is not None
                and self.data.spawn_y is not None
                and self.data.spawn_z is not None):
            return (self.data.spawn_x, self.data.spawn_y, self.data.spawn_z)

        x, z = (self.data.spawn_x or 0), (self.data.spawn_z or 0)
        y = self.find_surface_y(x, z)
        self.set_spawn_point(x, y, z)
        logger.info(f"No saved spawn; settled on surface at ({x}, {y}, {z})")
        return (x, y, z)

    def find_surface_y(self, x: int, z: int) -> int:
        """The first Y at (x, z) that a player can stand on.

        Scans down from the top of the generated column for solid ground and
        returns the block above it, skipping water so the answer is dry land
        rather than the top of an ocean. Falls back to the generator's sea
        level if the column somehow has nothing solid in it.
        """
        dim = self.overworld
        chunk = dim.get_chunk(x // 16, z // 16)
        if chunk is None or not chunk.is_generated:
            chunk = self.load_chunk(x // 16, z // 16,
                                    dimension=DimensionType.OVERWORLD)
        if chunk is None:
            return dim.settings.sea_level + 1

        local_x, local_z = x % 16, z % 16
        top = chunk.get_height_at(local_x, local_z)
        for y in range(top, chunk.min_y - 1, -1):
            block = chunk.get_block(local_x, y, local_z)
            if block != BLOCKS.AIR and block != BLOCKS.WATER:
                return y + 1
        return dim.settings.sea_level + 1

    def set_spawn_point(self, x: int, y: int, z: int) -> None:
        """Set world spawn point."""
        self.data.spawn_x = x
        self.data.spawn_y = y
        self.data.spawn_z = z

        # Update overworld spawn
        self.overworld.spawn_point = (x, y, z)
    
    def claim_chunk(
        self,
        x: int,
        z: int,
        dimension: DimensionType = DimensionType.OVERWORLD,
    ) -> bool:
        """Create the "being generated" placeholder for a chunk, or load it
        from disk. Returns True when a generation request is still needed.

        MUST run on the dimension's actor thread: it mutates the chunk
        manager, and the actor is the single owner of that state. The blank
        chunk it leaves behind is the streamer's dedupe mark -- present but
        `is_generated=False` means "claimed, being filled right now", so
        other viewers do not queue duplicate generations for the same coord.

        The one thing this must not do is assume the chunk already exists.
        Everything the streamer asks for is a chunk nobody has loaded, and
        `ChunkManager.get_chunk` is a pure lookup -- it answers None for all
        of them. Guarding generation on that answer made every request a
        no-op that still counted as success, and because the streamer sends
        strictly nearest-first and stops at the first chunk that is not
        ready, one missing chunk there blocks the entire view: the client
        waits on its loading screen until the connection times out.
        """
        dim = self.dimensions.get(dimension)
        if not dim:
            return False

        pos = ChunkPos(x, z)
        if dim.chunk_manager.get_chunk(pos) is not None:
            return False

        # Disk first: a chunk saved by a previous session must not be
        # regenerated (the seed alone would rebuild it, but player edits
        # live only in the region file).
        region_manager = self.region_managers.get(dimension)
        if region_manager:
            chunk = self._read_from_disk(region_manager, x, z, dimension)
            if chunk:
                dim.chunk_manager.set_chunk(chunk)
                return False

        # Blank claim: is_generated stays False until the worker's finished
        # chunk is inserted.
        dim.chunk_manager.load_chunk(pos)
        return True

    def insert_generated_chunk(self, dimension: str, chunk: Chunk) -> None:
        """Completion sink for the chunk generator (direct-insert variant).

        This is the fallback path for when no dimension actor is running
        (tools, tests, shutdown drain after the actors stopped). While the
        tick pipeline is up, `core.server` routes completions through the
        actor inboxes instead -- same final insert, but on the owning thread.
        """
        try:
            dim_type = DimensionType(dimension)
        except ValueError:
            logger.warning(f"Generated chunk for unknown dimension {dimension!r}")
            return

        dim = self.dimensions.get(dim_type)
        if not dim:
            return

        existing = dim.chunk_manager.get_chunk(chunk.position)
        if existing is not None and existing.is_generated:
            # Another path (region load, spawn preload) beat the worker to it.
            return

        dim.chunk_manager.set_chunk(chunk)

    def discard_failed_chunk(self, dimension: str, chunk_x: int, chunk_z: int) -> None:
        """Failure sink for the chunk generator: drop the blank claim.

        A claim that stays ungenerated would stall the streamer's strict
        ordering forever (the nearest not-ready chunk holds the whole view).
        Removing it lets the next pump re-request the coord from scratch.
        Only ever discards a chunk that is neither generated nor dirty --
        real chunk data is never touched.
        """
        try:
            dim_type = DimensionType(dimension)
        except ValueError:
            return

        dim = self.dimensions.get(dim_type)
        if not dim:
            return

        pos = ChunkPos(chunk_x, chunk_z)
        chunk = dim.chunk_manager.get_chunk(pos)
        if chunk is not None and not chunk.is_generated and not chunk.is_dirty:
            dim.chunk_manager.unload_chunk(pos)
    
    def _read_from_disk(
        self,
        region_manager: RegionManager,
        x: int,
        z: int,
        dimension: DimensionType,
    ) -> Optional[Chunk]:
        """Read a chunk from its region file, after its own pending save landed.

        Two things happen here that a bare `region_manager.load_chunk` misses:

        * **The save queue is drained for this coordinate first.** A chunk that
          was unloaded moments ago may still be sitting in the streaming saver;
          reading the region file before that write lands would return the
          *previous* copy, and the newer in-memory copy would then write that
          staleness back when it is unloaded in turn. The wait is one chunk's
          serialization and only happens when the coordinate really is in
          flight.
        * **`is_persisted` is stamped**, which is what tells the unload path
          that this chunk's bytes are already on disk and that dropping it
          costs nothing (see `chunk_saver.should_persist`).
        """
        self.chunk_saver.flush_key(dimension, x, z)
        chunk = region_manager.load_chunk(x, z)
        if chunk is not None:
            chunk.is_persisted = True
            chunk.is_dirty = False
        return chunk

    def load_chunk(
        self,
        x: int,
        z: int,
        dimension: DimensionType = DimensionType.OVERWORLD,
        generate: bool = True,
        priority: int = 0
    ) -> Optional[Chunk]:
        """
        Load or generate a chunk.
        
        Args:
            x: Chunk X coordinate
            z: Chunk Z coordinate
            dimension: Dimension to load from
            generate: Whether to generate if not exists
            priority: Generation priority (higher = more urgent)
        
        Returns:
            The chunk, or None if dimension doesn't exist
        """
        dim = self.dimensions.get(dimension)
        if not dim:
            return None
        
        pos = ChunkPos(x, z)
        chunk = dim.chunk_manager.get_chunk(pos)
        
        if chunk:
            return chunk
        
        # Try to load from disk first
        region_manager = self.region_managers.get(dimension)
        if region_manager:
            chunk = self._read_from_disk(region_manager, x, z, dimension)
            if chunk:
                dim.chunk_manager.set_chunk(chunk)
                return chunk
        
        # Generate new chunk if requested
        if generate:
            chunk = dim.chunk_manager.load_chunk(pos)
            # The dimension's own generator, not the world's: the world-level
            # one is seeded for the overworld, and the nether and the end each
            # got theirs (see _init_dimensions).
            dim.generator.generate_chunk(chunk)
            return chunk
        
        return None
    
    def request_chunk(
        self,
        x: int,
        z: int,
        dimension: DimensionType = DimensionType.OVERWORLD,
        priority: int = 0,
        callback: Optional[Callable[[], None]] = None
    ) -> bool:
        """
        Request a chunk to be generated in background.

        Args:
            x: Chunk X coordinate
            z: Chunk Z coordinate
            dimension: Dimension
            priority: Generation priority
            callback: Callback when generation completes

        Returns:
            True if request was queued
        """
        dim = self.dimensions.get(dimension)
        if not dim:
            return False

        # Only a *generated* chunk short-circuits the request. A blank claim
        # (is_generated False) does not: it means a generation is already
        # claimed somewhere in the pipeline, but queueing a second request
        # for the same coord is harmless (the losing insert is discarded),
        # while refusing to queue would leave a claim with no worker behind
        # it -- a hole the streamer waits on forever.
        pos = ChunkPos(x, z)
        chunk = dim.chunk_manager.get_chunk(pos)
        if chunk is not None and chunk.is_generated:
            if callback:
                callback()
            return True

        # Request background generation. The seed rides along: the worker
        # processes are world-agnostic.
        return chunk_generator.request_chunk(
            x, z, dimension.value, priority, callback, seed=self.seed
        )

    def generate_chunks_blocking(
        self,
        coords: List[Tuple[int, int]],
        dimension: DimensionType = DimensionType.OVERWORLD,
        priority: int = 100,
        progress: Optional[Callable[[int], None]] = None,
        timeout: Optional[float] = None,
    ) -> int:
        """Generate a batch of chunks through the worker pool and wait.

        The server's spawn preload uses this: at ~1.2 s/chunk the old
        synchronous loop pinned one core for over a minute on a radius-4
        spawn area, while the pool does the same work in a fraction of the
        time and the wait overlaps the workers' own spin-up.

        Disk hits settle immediately; only genuinely missing chunks are
        dispatched. Every request settles exactly once -- inserted or failed
        -- so the wait cannot hang on a worker error.

        Args:
            coords: Chunk coordinates to generate.
            dimension: Which dimension to generate in.
            priority: Queue priority (preload outranks view streaming).
            progress: Called with the running total each time a chunk settles.
            timeout: Wait cap; defaults to a generous per-chunk budget.

        Returns:
            How many chunks settled (disk hits, inserted, or failed).
        """
        unique = list(dict.fromkeys(coords))
        if not unique:
            return 0

        if timeout is None:
            timeout = 60.0 + 5.0 * len(unique)

        cond = threading.Condition()
        state = {"remaining": len(unique), "done": 0}

        def settled() -> None:
            with cond:
                state["remaining"] -= 1
                state["done"] += 1
                cond.notify_all()
            if progress:
                try:
                    progress(state["done"])
                except Exception:
                    logger.exception("Error in generation progress callback")

        for x, z in unique:
            # Region load or blank claim. A claim that is already present
            # (never happens at boot, but stay defensive) still gets a
            # request queued: a claim without a worker behind it is a hole.
            try:
                self.claim_chunk(x, z, dimension)
            except Exception:
                logger.exception(f"Failed to claim chunk ({x}, {z}) for preload")

            dim = self.dimensions.get(dimension)
            chunk = dim.chunk_manager.get_chunk(ChunkPos(x, z)) if dim else None
            if chunk is not None and chunk.is_generated:
                settled()  # disk hit, or already generated this run
                continue

            queued = chunk_generator.request_chunk(
                x, z, dimension.value, priority, callback=settled, seed=self.seed
            )
            if not queued:
                # Pool down or queue full: generate in-process rather than
                # leave a blank claim and a hole behind.
                logger.warning(
                    f"Generation queue refused ({x}, {z}); generating synchronously"
                )
                self.discard_failed_chunk(dimension.value, x, z)
                self.load_chunk(x, z, dimension=dimension, generate=True)
                settled()

        deadline = time.time() + timeout
        with cond:
            while state["remaining"] > 0:
                left = deadline - time.time()
                if left <= 0:
                    logger.warning(
                        f"Chunk batch timed out with {state['remaining']} "
                        f"chunk(s) unsettled"
                    )
                    break
                cond.wait(min(0.5, left))

        return state["done"]

    def light_chunks_blocking(
        self,
        coords: List[Tuple[int, int]],
        dimension: DimensionType = DimensionType.OVERWORLD,
        timeout: float = 30.0,
    ) -> Dict[str, Any]:
        """Give every listed chunk its first light pass, and wait for it.

        The spawn preload's other half, and the answer to a bug the reload path
        does not have: generation writes no light. A worker ships blocks and the
        receiver's sections carry `ChunkSection`'s **default** arrays
        (`world/chunk_gen_worker.py`), so a chunk enters memory with
        `light_populated` False and has to be lit before anyone may read it.
        Normally the two places that learn a coordinate became readable --
        `ChunkLifecycle.on_chunk_ready` (a worker finished) and its
        `_load_missing` pass (a chunk came back from disk) -- both queue that
        pass. Neither exists while the preload runs: `_init_world` is still
        inside `Server.start`, and the lifecycles are built later in
        `_start_ticking`. So every spawn chunk used to arrive unlit and stay
        that way -- the engine's queue stayed empty, each chunk packet went out
        with the encoder's heightmap approximation (15 above the surface, 0
        below, no block light), and the world the client drew was that
        approximation until an operator ran `/light relight`. Chunks loaded
        later were fine, which is exactly the asymmetry this closes.

        The drain runs here, inline, rather than being left to the actor. Two
        reasons, in order of weight: nothing else owns the dimension yet (the
        batch has settled, so no worker result is in flight -- the same
        reasoning `console/commands.py::_relight_inline` relies on), and the
        listener has not opened yet, so the first player to connect is sent lit
        terrain instead of a correctable first frame.

        **The outer ring of a disc cannot be lit here.** A patch is one chunk's
        3x3 (`light_engine._light_chunk`), so the rim of any finite area is
        missing neighbours until something loads outside it. Those coordinates
        are *not* queued: the engine re-queues a chunk whose 3x3 is incomplete
        and would retry it on every tick of the drain, so a naive "queue
        everything, tick until empty" turns a 0.4 s pass into a full-timeout
        spin. They come back in `deferred` for the caller to hand to the real
        engine, which retries them until their outer neighbours stream in.

        Chunks that are not resident, not generated, or already
        `light_populated` are skipped: a chunk read back from a region file
        carries the light the engine wrote last session, and re-patching it
        would spend a 3x3 for nothing.

        Args:
            coords: Chunk coordinates to light. Deduplicated.
            dimension: Which dimension they are in.
            timeout: Backstop on the drain. Every queued chunk is patchable by
                construction, so this only fires if a kernel is pathologically
                slow.

        Returns:
            `queued`, `skipped`, `lit`, `pending`, `dropped` and
            `engine_errors` counters, plus `deferred`: the coordinates left for
            a later engine, as `(x, z)` pairs.
        """
        answer: Dict[str, Any] = {
            "queued": 0, "skipped": 0, "lit": 0, "pending": 0,
            "dropped": 0, "engine_errors": 0, "deferred": [],
        }

        dim = self.dimensions.get(dimension)
        if dim is None:
            return answer

        targets: List[Tuple[int, int]] = []
        for x, z in dict.fromkeys(coords):
            chunk = dim.chunk_manager.peek(ChunkPos(x, z))
            if chunk is None or not chunk.is_generated or chunk.light_populated:
                answer["skipped"] += 1
                continue
            if not _generated_3x3(dim, x, z):
                answer["deferred"].append((x, z))
                continue
            targets.append((x, z))

        if not targets:
            return answer

        # An engine already attached to the dimension owns the light field and
        # the tick that drains it; a second engine over the same chunks would
        # compute the same numbers into the same arrays, and queueing onto the
        # live one is what the console path does.
        engine = getattr(dim, "light", None)
        if engine is not None:
            for x, z in targets:
                engine.queue_chunk(x, z)
            answer["queued"] = len(targets)
            answer["pending"] = engine.pending_chunks
            answer["lit"] = engine.chunks_lit
            return answer

        from world.light_engine import DimensionLight

        engine = DimensionLight(
            dim,
            budget_ms=SPAWN_LIGHT_BUDGET_MS,
            chunks_per_tick=max(1, len(targets)),
        )
        for x, z in targets:
            engine.queue_chunk(x, z)
        answer["queued"] = len(targets)

        deadline = time.monotonic() + timeout
        while engine.pending_chunks and time.monotonic() < deadline:
            engine.tick()

        answer["lit"] = engine.chunks_lit
        answer["pending"] = engine.pending_chunks
        answer["dropped"] = engine.dropped_unloaded
        answer["engine_errors"] = engine.errors
        return answer

    def unload_chunk(
        self,
        x: int,
        z: int,
        dimension: DimensionType = DimensionType.OVERWORLD,
        save: bool = True
    ) -> bool:
        """Unload a chunk, handing it to the streaming saver first.

        The save is asynchronous: it is queued (or written inline only if the
        queue is full) and the chunk leaves memory immediately. A later read of
        the same coordinate waits for that write via `_read_from_disk`, so the
        asynchrony is invisible from the outside.

        A ticket that still covers the coordinate will bring the chunk back on
        the lifecycle's next tick; use `forceload`/`add_ticket` to keep one
        resident, or `remove_ticket` before unloading.
        """
        dim = self.dimensions.get(dimension)
        if not dim:
            return False
        return dim.unload_chunk(x, z, save=save)
    
    def get_chunks_in_range(
        self,
        center_x: int,
        center_z: int,
        radius: int,
        dimension: DimensionType = DimensionType.OVERWORLD
    ) -> List[Chunk]:
        """Get all chunks within a radius."""
        dim = self.dimensions.get(dimension)
        if dim:
            return dim.get_chunks_in_range(center_x, center_z, radius)
        return []
    
    def tick(self) -> None:
        """Tick the world (called every game tick)."""
        with self._lock:
            # Update all dimensions
            for dimension in self.dimensions.values():
                dimension.tick()
            
            # Update world time
            self.data.world_time += 1
            self.data.day_time = (self.data.day_time + 1) % 24000
            self.data.game_time += 1
            
            # Execute tick callbacks
            for callback in self._tick_callbacks:
                try:
                    callback()
                except Exception as e:
                    logger.exception(f"Error in world tick callback: {e}")
            
            self._tick_count += 1
    
    def register_tick_callback(self, callback: Callable[[], None]) -> None:
        """Register a callback to be called each tick."""
        self._tick_callbacks.append(callback)
    
    @property
    def current_tick(self) -> int:
        """Get current tick count."""
        return self._tick_count
    
    # ============ Save/Load Operations ============
    
    def save(self) -> bool:
        """
        Save the world to disk.

        Order matters: the streaming saver is drained *first*, so a chunk the
        actors unloaded a moment ago is on disk before this pass walks the chunk
        managers, and so the region files have exactly one writer at a time.
        Force-loaded chunks are written last: they are state, not terrain.
        
        Returns:
            True if successful
        """
        try:
            # Anything in flight from the unload path lands before we walk the
            # managers; otherwise the two writers would interleave on the same
            # region file.
            self.chunk_saver.flush()

            # Save level.dat
            self._save_level_dat()
            
            # Save all chunks via region managers
            self._save_chunks()

            # Persist force-loaded chunks (and live portal tickets)
            self._save_tickets()
            
            # Save player data
            self._save_player_data()
            
            self.data.last_save_time = int(time.time())
            
            logger.info(f"World '{self.name}' saved successfully")
            return True
            
        except Exception as e:
            logger.exception(f"Failed to save world '{self.name}': {e}")
            return False
    
    def _save_level_dat(self) -> None:
        """Save level.dat using NBT format."""
        level_file = self.world_dir / "level.dat"
        
        # Get data from to_nbt_dict and use as keyword arguments
        data = self.data.to_nbt_dict()
        save_level_dat(
            str(level_file),
            level_name=data.pop("LevelName"),
            seed=data.pop("RandomSeed"),
            spawn_x=data.pop("SpawnX"),
            spawn_y=data.pop("SpawnY"),
            spawn_z=data.pop("SpawnZ"),
            world_time=data.pop("Time"),
            day_time=data.pop("DayTime"),
            game_rules=data.pop("GameRules", {}),
            difficulty=data.pop("Difficulty", 2),
            hardcore=data.pop("Hardcore", False),
            game_type=data.pop("GameType", 0),
            **data  # Pass remaining kwargs
        )
        
        logger.debug(f"Saved level.dat for '{self.name}'")
    
    def _save_chunks(self) -> None:
        """Save all dirty chunks via region managers."""
        saved_total = 0

        for dim_type in self.dimensions:
            saved = self._save_dimension(dim_type)
            if saved > 0:
                logger.debug(f"Saved {saved} chunks for {dim_type.value}")
                saved_total += saved
        
        if saved_total > 0:
            # Flush all region files
            for region_manager in self.region_managers.values():
                region_manager.save_all()
    
    def _save_player_data(self) -> None:
        """Save player data."""
        players_dir = self.world_dir / "playerdata"
        players_dir.mkdir(parents=True, exist_ok=True)
        
        # TODO: Implement player data saving
        pass
    
    @classmethod
    def load(cls, world_dir: Path) -> Optional["World"]:
        """
        Load a world from disk.
        
        Args:
            world_dir: Directory containing the world
        
        Returns:
            Loaded world, or None if loading failed
        """
        try:
            if not world_exists(world_dir.parent, world_dir.name):
                logger.error(f"World not found: {world_dir}")
                return None
            
            world = cls(
                name=world_dir.name,
                world_dir=world_dir,
                create_if_missing=False
            )
            
            logger.info(f"World '{world.name}' loaded from {world_dir}")
            return world
            
        except Exception as e:
            logger.exception(f"Failed to load world from {world_dir}: {e}")
            return None
    
    @property
    def difficulty(self) -> str:
        """Get difficulty name."""
        return self.data.difficulty
    
    @difficulty.setter
    def difficulty(self, value: str) -> None:
        """Set difficulty."""
        if value.lower() in ("peaceful", "easy", "normal", "hard"):
            self.data.difficulty = value.lower()
    
    @property
    def hardcore(self) -> bool:
        """Check if world is hardcore."""
        return self.data.hardcore
    
    @hardcore.setter
    def hardcore(self, value: bool) -> None:
        """Set hardcore mode."""
        self.data.hardcore = value
    
    @property
    def gamemode(self) -> str:
        """Get gamemode name."""
        return self.data.gamemode
    
    @gamemode.setter
    def gamemode(self, value: str) -> None:
        """Set gamemode."""
        self.data.gamemode = value.lower()
    
    def get_total_chunk_count(self) -> int:
        """Get total loaded chunks across all dimensions."""
        return sum(dim.chunk_count for dim in self.dimensions.values())
    
    def get_world_info(self) -> Dict[str, Any]:
        """Get world information for status display."""
        gen_stats = chunk_generator.get_stats()
        save_stats = self.chunk_saver.stats()
        
        return {
            "name": self.name,
            "seed": self.seed,
            "gamemode": self.gamemode,
            "difficulty": self.difficulty,
            "hardcore": self.hardcore,
            "world_time": self.data.world_time,
            "day_time": self.data.day_time,
            "total_chunks": self.get_total_chunk_count(),
            "players": len(self.players),
            "tick_rate": self._tick_rate,
            "generator_queue": gen_stats.get("queue_size", 0),
            "generator_in_flight": gen_stats.get("in_flight", 0),
            "generator_workers": gen_stats.get("workers", 0),
            "chunks_generated": gen_stats.get("generated", 0),
            "chunks_pending_save": save_stats.get("pending", 0) + save_stats.get("in_flight", 0),
            "chunks_saved": save_stats.get("saved", 0),
        }

    def get_chunk_stats(self, timeout: float = 2.0) -> Dict[str, Any]:
        """Per-dimension load/unload picture: tickets, levels, memory, saves.

        `World.get_world_info` answers "how big is the world"; this answers
        "what is in memory right now and why", which is the question the ticket
        system exists to make answerable.

        The numbers come from the owning actor when one is running (via
        `StatsMsg`), because they are read out of sets the tick mutates: a
        direct read from the console thread is a real race on a free-threaded
        build, not a theoretical one.
        """
        from world.chunk_lifecycle import StatsMsg

        out: Dict[str, Any] = {
            "saver": self.chunk_saver.stats(),
            "dimensions": {},
        }
        for dim_type, lifecycle in self.chunk_lifecycles.items():
            snapshot = None
            if self.chunk_actor_inbox is not None:
                msg = StatsMsg()
                if self._post_to_actor(dim_type, msg) and msg.done.wait(timeout):
                    snapshot = msg.box.get("stats")
            if snapshot is None:
                if self.chunk_actor_inbox is None:
                    # No actor: this thread is the only one touching it.
                    snapshot = lifecycle.stats()
                else:
                    snapshot = {"dimension": dim_type.value, "error": "actor busy"}
            out["dimensions"][dim_type.value] = snapshot
        return out
    
    def unload(self, save: bool = True) -> None:
        """
        Unload the world.

        Args:
            save: Whether to save before unloading
        """
        # Generation stops FIRST, before any save: a chunk landing in the
        # chunk manager after the save pass has iterated would stay dirty
        # forever, and `_save_chunks` walks the manager's dict directly, so
        # concurrent inserts can break the iteration outright. stop() drains
        # in-flight generations (bounded) and discards the rest --
        # deterministic generation rebuilds them next boot.
        chunk_generator.stop()

        # Force-loaded chunks are state the world owns, not terrain: write them
        # even when the caller asked for a save-less shutdown, or the next boot
        # silently forgets what was force-loaded.
        self._save_tickets()

        if save:
            self.save()

        # The saver is drained by save(); close it so its threads end here and
        # not at interpreter exit.
        self.chunk_saver.close()

        # Close region managers
        for region_manager in self.region_managers.values():
            region_manager.close_all()

        # Unload all dimensions
        for dimension in self.dimensions.values():
            dimension.unload_all_chunks()

        self.dimensions.clear()

        logger.info(f"World '{self.name}' unloaded")


# World manager for handling multiple worlds
class WorldManager:
    """Manages multiple worlds and their lifecycle."""
    
    def __init__(self, base_dir: Path = Path("data")):
        """
        Initialize world manager.
        
        Args:
            base_dir: Base directory for worlds
        """
        self.base_dir = base_dir
        self.worlds: Dict[str, World] = {}
        self.current_world: Optional[World] = None
    
    def get_world(self, name: str) -> Optional[World]:
        """Get a loaded world by name."""
        return self.worlds.get(name)
    
    def load_world(self, name: str) -> Optional[World]:
        """
        Load a world.
        
        Args:
            name: World name
        
        Returns:
            Loaded world or None
        """
        if name in self.worlds:
            return self.worlds[name]
        
        world = World.load(self.base_dir / name)
        if world:
            self.worlds[name] = world
            if self.current_world is None:
                self.current_world = world
        
        return world
    
    def create_world(self, name: str, seed: int = 0) -> World:
        """
        Create a new world.
        
        Args:
            name: World name
            seed: World seed
        
        Returns:
            Created world
        """
        world = World(
            name=name,
            seed=seed,
            world_dir=self.base_dir / name,
            create_if_missing=True
        )
        self.worlds[name] = world
        
        if self.current_world is None:
            self.current_world = world
        
        return world
    
    def unload_world(self, name: str, save: bool = True) -> bool:
        """
        Unload a world.
        
        Args:
            name: World name
            save: Whether to save before unloading
        
        Returns:
            True if successful
        """
        world = self.worlds.pop(name, None)
        if world:
            world.unload(save=save)
            if self.current_world == world:
                self.current_world = next(iter(self.worlds.values()), None)
            return True
        return False
    
    def list_worlds(self) -> List[str]:
        """List all available worlds."""
        return list_worlds(self.base_dir)
    
    def shutdown(self) -> None:
        """Shutdown all worlds."""
        for world in list(self.worlds.values()):
            world.unload(save=True)
        self.worlds.clear()
        self.current_world = None


# Global world manager
world_manager = WorldManager()
