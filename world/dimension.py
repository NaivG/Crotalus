"""
Dimension Module

Manages dimensions (overworld, nether, end) with their specific properties.
"""

from enum import Enum
from typing import Optional, List, Dict, Any, Callable, Tuple
from dataclasses import dataclass, field
import threading
from loguru import logger

from world.chunk import Chunk, ChunkPos, ChunkManager
from world.chunk_ticket import (
    DIMENSION_IDLE_TIMEOUT,
    LoadType,
    TicketManager,
    TicketType,
)
from world.generator import BaseGenerator, GeneratorSettings, GeneratorType, create_generator


class DimensionType(Enum):
    """Minecraft dimension types."""
    OVERWORLD = "minecraft:overworld"
    NETHER = "minecraft:the_nether"
    END = "minecraft:the_end"
    
    # Custom dimensions
    AETHER = "aether"
    TWILIGHT_FOREST = "twilight_forest"


@dataclass
class DimensionSettings:
    """Settings specific to a dimension.

    The values come from `assets/dimension.json` (via `registry.worldgen`) when
    the dimension is a vanilla one - see `for_dimension`. The defaults below are
    the fallback for a dimension the registry does not describe (a custom
    dimension type) and for a caller that builds the dataclass by hand; before
    the registry was wired in they were the only source, which is how the nether
    ended up 128 blocks tall when its dimension type says 256.

    Three heights, and they are not interchangeable:

    * `height` - the dimension's full height, what the client is told and what
      the chunk codec walks.
    * `logical_height` - the build limit inside it (the nether is 256 tall but
      you cannot build above 128).
    * `worldgen.settings(preset).shape.height` - the height the *terrain*
      occupies, which can be smaller than both (the nether's shape is 128).
    """

    dimension_type: DimensionType = DimensionType.OVERWORLD

    # Coordinate scales (nether is 8x overworld)
    coordinate_scale: float = 1.0

    # Height settings
    min_y: int = -64
    max_y: int = 320
    height: int = 384
    logical_height: int = 384

    #: Sea level of the dimension's generator preset (`chunk_gen_settings.json`).
    #: 63 in the overworld, 32 in the nether, 0 in the end. The dataclass default
    #: matches the overworld, which is also the default dimension type.
    sea_level: int = 63

    #: The `chunk_gen_settings.json` preset this dimension generates with.
    preset: str = "overworld"

    #: `"registry"` when the values above came from `assets/dimension.json`,
    #: `"builtin"` when they are the fallback table.
    source: str = "builtin"

    # Ambient settings
    ambient_light: float = 0.0
    fixed_time: Optional[int] = None
    
    # Environmental
    has_skylight: bool = True
    has_ceiling: bool = False
    ultra_warm: bool = False
    natural: bool = True
    
    # Effects
    infiniburn: str = "#minecraft:infiniburn_overworld"
    respawn_anchor_works: bool = False
    has_raids: bool = True
    monster_spawn_block_light_limit: int = 0
    monster_spawn_light_level: int = 7
    
    # Generator overrides
    generator_type: Optional[GeneratorType] = None
    
    @classmethod
    def for_dimension(cls, dim_type: DimensionType) -> "DimensionSettings":
        """Get the settings for a dimension type.

        Prefers `assets/dimension.json`: the four vanilla dimension types are
        read out of the registry, and anything the registry does not describe
        (a custom dimension, or a tool that imported `world` without booting)
        falls back to the table below. The fallback is also what keeps the
        vanilla values honest - `tools/check_worldgen_data.py` asserts the two
        agree on everything they both define.
        """
        data = registry_dimension(dim_type)
        if data is None:
            return cls.builtin(dim_type)
        return cls._from_registry(dim_type, data)

    @classmethod
    def builtin(cls, dim_type: DimensionType) -> "DimensionSettings":
        """The hand-written table for a dimension the registry cannot answer.

        Public because it is not only a fallback: `tools/check_worldgen_data.py`
        asserts it agrees with `assets/dimension.json` on everything both of them
        define, which is what stops the two from drifting while custom
        dimensions still depend on it.
        """
        if dim_type == DimensionType.OVERWORLD:
            return cls(
                dimension_type=dim_type,
                coordinate_scale=1.0,
                min_y=-64,
                max_y=320,
                height=384,
                logical_height=384,
                sea_level=63,
                preset="overworld",
                has_skylight=True,
                has_raids=True,
                infiniburn="#minecraft:infiniburn_overworld"
            )
        elif dim_type == DimensionType.NETHER:
            return cls(
                dimension_type=dim_type,
                coordinate_scale=8.0,  # 1 block in nether = 8 in overworld
                min_y=0,
                max_y=256,
                height=256,
                logical_height=128,
                sea_level=32,
                preset="nether",
                has_skylight=False,
                has_ceiling=True,
                ultra_warm=True,
                natural=True,
                has_raids=False,
                respawn_anchor_works=True,
                infiniburn="#minecraft:infiniburn_nether"
            )
        elif dim_type == DimensionType.END:
            return cls(
                dimension_type=dim_type,
                coordinate_scale=1.0,
                min_y=0,
                max_y=256,
                height=256,
                logical_height=256,
                sea_level=0,
                preset="end",
                has_skylight=False,
                has_ceiling=False,
                ultra_warm=False,
                natural=False,
                has_raids=True,
                fixed_time=6000,  # End always has twilight
                infiniburn="#minecraft:infiniburn_end"
            )
        else:
            return cls(dimension_type=dim_type)

    @classmethod
    def _from_registry(cls, dim_type: DimensionType, data) -> "DimensionSettings":
        """Build from a `registry.worldgen.DimensionTypeData`.

        `ultra_warm`, `natural`, `has_raids` and `respawn_anchor_works` are not
        fields of the 1.21.11 dimension type any more: the game moves them into
        `attributes` (`minecraft:gameplay/...`), and where the attribute is
        absent the built-in table supplies the vanilla answer rather than a
        guess.
        """
        builtin = cls.builtin(dim_type)
        attributes = data.attributes or {}

        def attribute(name: str, default: bool) -> bool:
            value = attributes.get(f"minecraft:gameplay/{name}")
            return default if value is None else bool(value)

        fixed_time = None
        if data.raw.get("fixed_time") is not None:
            fixed_time = int(data.raw["fixed_time"])
        elif data.has_fixed_time:
            fixed_time = builtin.fixed_time

        sea_level = builtin.sea_level
        settings = registry_generator_settings(data.preset)
        if settings is not None:
            sea_level = settings.sea_level

        return cls(
            dimension_type=dim_type,
            coordinate_scale=data.coordinate_scale,
            min_y=data.min_y,
            max_y=data.max_y,
            height=data.height,
            logical_height=data.logical_height,
            sea_level=sea_level,
            preset=data.preset,
            source="registry",
            ambient_light=data.ambient_light,
            fixed_time=fixed_time,
            has_skylight=data.has_skylight,
            has_ceiling=data.has_ceiling,
            ultra_warm=attribute("water_evaporates", builtin.ultra_warm),
            natural=builtin.natural,
            infiniburn=data.infiniburn or builtin.infiniburn,
            respawn_anchor_works=attribute("respawn_anchor_works", builtin.respawn_anchor_works),
            has_raids=attribute("can_start_raid", builtin.has_raids),
            monster_spawn_block_light_limit=data.monster_spawn_block_light_limit,
            monster_spawn_light_level=builtin.monster_spawn_light_level,
        )


def _worldgen():
    """The registry's worldgen views, or None when the registry cannot load.

    Imported lazily: `registry` must not need `world`, and `import world` should
    not be what parses 54MB of JSON. A failure here is not fatal - the built-in
    table still describes the vanilla dimensions - but it is logged, because a
    server generating from the wrong dimension height is worse than a slow boot.
    """
    try:
        from registry import get_game_registry

        return get_game_registry().worldgen
    except Exception as error:  # pragma: no cover - only a broken install
        logger.warning(f"Worldgen data unavailable, using built-in dimension settings: {error}")
        return None


def registry_dimension(dim_type: DimensionType):
    """`registry.worldgen`'s view of a dimension type, or None."""
    worldgen = _worldgen()
    return None if worldgen is None else worldgen.dimension(dim_type.value)


def registry_generator_settings(preset: str):
    """`registry.worldgen`'s generator preset, or None."""
    worldgen = _worldgen()
    return None if worldgen is None else worldgen.settings(preset)


class Dimension:
    """
    Represents a dimension (world layer) in Minecraft.
    
    Each dimension has its own chunks, generation rules, and gameplay.
    """
    
    def __init__(
        self,
        dimension_type: DimensionType,
        world_seed: int,
        generator: Optional[BaseGenerator] = None
    ):
        """
        Initialize a dimension.

        Args:
            dimension_type: Type of dimension
            world_seed: World seed for generation
            generator: World generator (creates the configured backend if not
                provided)
        """
        self.dimension_type = dimension_type
        self.settings = DimensionSettings.for_dimension(dimension_type)
        self.seed = world_seed
        
        # Chunk management. Every chunk spans this dimension's vertical window
        # (overworld: Y -64..319), so worldgen rules anchored to the world
        # floor land inside the chunk instead of outside it.
        #
        # `max_chunks` is a safety valve for chunks nothing asked to keep (an
        # explicit `World.load_chunk` from a tool or a command), not the
        # unloading policy: chunks a ticket covers are pinned and are never
        # evicted, because evicting one only means loading it again next tick.
        # A view-distance-10 player legitimately holds ~625 chunks (a 25x25
        # square: the 21x21 of level <= 32 plus the two border rings), so a cap
        # in the low thousands would fight the policy instead of guarding it.
        self.chunk_manager = ChunkManager(
            max_chunks=4096,
            min_y=self.settings.min_y,
            height=self.settings.height,
        )

        # Load-level tickets. This is the authority on what belongs in memory
        # and what may tick; `ChunkLifecycle` is the driver that turns it into
        # loaded chunks, and the level maths itself lives in chunk_ticket.py.
        self.tickets = TicketManager(dimension_type.value)

        #: Ticks since a ticket that keeps this dimension active last existed
        #: (Chunk.mdx "Idle timeout"). At DIMENSION_IDLE_TIMEOUT the dimension
        #: stops ticking entities and block entities.
        self.idle_ticks = 0

        #: Synchronous "write this dimension's dirty chunks now" hook, set by
        #: `World` (which owns the region managers). The streaming path used
        #: when chunks are unloaded is a different hook, on `ChunkManager`.
        self.save_handler: Optional[Callable[[], int]] = None
        
        # Generator
        if generator:
            self.generator = generator
        else:
            gen_type = self.settings.generator_type or GeneratorType.NORMAL
            gen_settings = GeneratorSettings(
                generator_type=gen_type,
                seed=world_seed,
                preset=self.settings.preset,
            )
            self.generator = create_generator(gen_settings)
        
        # Player tracking
        self.players: List[str] = []  # Player UUIDs

        # Non-player entities. Owned by this dimension's actor thread - the
        # single-owner rule (world/dimension_actor.py's docstring) is what
        # makes them lock-free. Players are deliberately NOT here: a Player
        # belongs to the event loop that owns its socket, and the actor only
        # ever sees position snapshots through PlayerViewMsg. Empty until
        # mobs/items exist; the spawn/remove API is what the future summon
        # path and chunk entity-loading will call.
        self.entities: Dict[int, "Entity"] = {}

        #: Entities skipped by the last `tick()` because their chunk is not
        #: entity-ticking. Vanilla's `EntityTickList` is the same idea: an
        #: entity in a lazy or border chunk is in the world, in the tracker and
        #: in every table, and simply does not run.
        self.paused_entities = 0

        # Tick tracking
        self.world_time: int = 0
        self.day_time: int = 6000
        
        # Spawn point (relative to dimension)
        self._spawn_point = (0, 64, 0)
        
        # Lock for thread safety
        self._lock = threading.Lock()
        
        logger.info(f"Dimension {dimension_type.value} initialized")
    
    @property
    def name(self) -> str:
        """Get dimension name."""
        return self.dimension_type.value
    
    @property
    def spawn_point(self) -> tuple[int, int, int]:
        """Get dimension spawn point."""
        return self._spawn_point
    
    @spawn_point.setter
    def spawn_point(self, pos: tuple[int, int, int]) -> None:
        """Set dimension spawn point."""
        self._spawn_point = pos
    
    def get_chunk(self, x: int, z: int) -> Optional[Chunk]:
        """
        Get a loaded chunk.
        
        Args:
            x: Chunk X coordinate
            z: Chunk Z coordinate
        
        Returns:
            Chunk if loaded, None otherwise
        """
        return self.chunk_manager.get_chunk(ChunkPos(x, z))
    
    def load_chunk(self, x: int, z: int, generate: bool = True) -> Chunk:
        """
        Load or generate a chunk.

        This is the *synchronous* interface: with `generate` it builds the
        terrain on the calling thread, which is what tools and the spawn
        preload want, and what a tick must not do (the vanilla backend costs
        about a second per chunk). The automatic, tick-driven path is
        `ChunkLifecycle`, and the ticket interface that decides what should be
        resident at all is `Dimension.tickets`.

        Args:
            x: Chunk X coordinate
            z: Chunk Z coordinate
            generate: Whether to generate if not exists

        Returns:
            The loaded/generated chunk
        """
        pos = ChunkPos(x, z)
        chunk = self.chunk_manager.get_chunk(pos)
        
        if chunk is None and generate:
            chunk = self.chunk_manager.load_chunk(pos)
            self.generator.generate_chunk(chunk)
        
        return chunk
    
    def unload_chunk(self, x: int, z: int, save: bool = True) -> bool:
        """
        Unload a chunk.

        With `save` the chunk is offered to the manager's eviction handler
        first (the streaming saver, in a live server) and only dropped if the
        handler takes it; with `save=False` it is dropped outright, which is
        for chunks known to hold nothing -- a blank claim whose generation
        failed, or a build/discard path.

        A ticket-driven loader will bring the chunk back on its next tick if a
        ticket still covers it (`load_type` answers whether one does).

        Args:
            x: Chunk X coordinate
            z: Chunk Z coordinate
            save: Whether to hand the chunk to the save pipeline first

        Returns:
            True if the chunk left memory
        """
        pos = ChunkPos(x, z)
        if save:
            return self.chunk_manager.unload_chunk(pos)
        return self.chunk_manager.remove(pos) is not None

    # ---------------- load levels ----------------

    def get_load_level(self, x: int, z: int) -> int:
        """Level of a chunk (31 entity-ticking ... 33 border, 44 = unreachable)."""
        return self.tickets.level(ChunkPos(x, z))

    def get_load_type(self, x: int, z: int) -> LoadType:
        """Which load type the chunk's level puts it in."""
        return self.tickets.load_type(ChunkPos(x, z))

    def is_chunk_loaded(self, x: int, z: int) -> bool:
        """Whether the chunk is resident (or claimed and on its way in)."""
        return self.chunk_manager.peek(ChunkPos(x, z)) is not None

    def ticks_entities(self, x: int, z: int) -> bool:
        """Whether entities tick here: inside a player's simulation square."""
        return self.tickets.ticks_entities(ChunkPos(x, z))

    def ticks_entity(self, entity: "Entity") -> bool:
        """Whether this entity's own tick runs this game tick.

        `Chunk.mdx`'s entity-ticking rule, asked about one entity: level <= 31
        for the chunk the entity currently stands in. Levels 32 ("block
        ticking") and 33 ("border") keep the entity accessible -- it stays in
        `entities`, keeps its position, is sent to clients, can be teleported --
        but nothing about it is processed. This is vanilla's `EntityTickList`,
        which only holds entities whose chunk is ticking.

        The entity's *current* chunk decides, evaluated when the tick starts:
        like vanilla, an entity that crosses into a lazy chunk during its own
        move finishes that move and is suspended from the next tick on.
        """
        chunk_x, chunk_z = entity.position.chunk
        return self.ticks_entities(chunk_x, chunk_z)

    def ticks_blocks(self, x: int, z: int) -> bool:
        """Whether blocks/block entities/random ticks run here: one ring wider."""
        return self.tickets.ticks_blocks(ChunkPos(x, z))

    def add_ticket(
        self,
        ttype: TicketType,
        x: int,
        z: int,
        *,
        level: Optional[int] = None,
        radius: Optional[int] = None,
        key=None,
    ) -> Any:
        """Keep a chunk loaded at a given level (the explicit load interface).

        Anything stronger than a one-tick read should go through a ticket rather
        than a bare `load_chunk`: a ticket is what the auto-unload pass and the
        eviction valve consult, so a chunk held only by a direct call is a chunk
        they are allowed to take away.
        """
        return self.tickets.add(
            ttype, ChunkPos(x, z), level=level, radius=radius, key=key
        )

    def remove_ticket(self, ttype: TicketType, x: int, z: int, key=None) -> bool:
        """Drop a ticket. The chunk unloads after the grace period."""
        return self.tickets.remove(ttype, ChunkPos(x, z), key)

    @property
    def is_active(self) -> bool:
        """Whether the dimension still processes entity/block ticking.

        False once `DIMENSION_IDLE_TIMEOUT` ticks have passed without a ticket
        that keeps the dimension active (a simulating player, force-loaded
        chunks, a portal, an ender pearl). The chunks stay loaded; only the
        ticking stops, which is what vanilla does with an idle dimension.
        """
        return self.idle_ticks < DIMENSION_IDLE_TIMEOUT

    def get_block(self, x: int, y: int, z: int) -> int:
        """
        Get block at world coordinates.
        
        Args:
            x: World X coordinate
            y: World Y coordinate
            z: World Z coordinate
        
        Returns:
            Block ID
        """
        chunk_x = x // 16
        chunk_z = z // 16
        local_x = x % 16
        local_z = z % 16
        
        chunk = self.get_chunk(chunk_x, chunk_z)
        if chunk:
            return chunk.get_block(local_x, y, local_z)
        return 0  # Air
    
    def set_block(self, x: int, y: int, z: int, block_id: int) -> bool:
        """
        Set block at world coordinates.
        
        Args:
            x: World X coordinate
            y: World Y coordinate
            z: World Z coordinate
            block_id: Block ID to set
        
        Returns:
            True if block was set
        """
        chunk_x = x // 16
        chunk_z = z // 16
        local_x = x % 16
        local_z = z % 16
        
        chunk = self.get_chunk(chunk_x, chunk_z)
        if chunk:
            chunk.set_block(local_x, y, local_z, block_id)
            return True
        return False
    
    def get_chunks_in_range(
        self,
        center_x: int,
        center_z: int,
        radius: int
    ) -> List[Chunk]:
        """
        Get all chunks within a radius.
        
        Args:
            center_x: Center chunk X
            center_z: Center chunk Z
            radius: Radius in chunks
        
        Returns:
            List of chunks
        """
        chunks = []
        for x in range(center_x - radius, center_x + radius + 1):
            for z in range(center_z - radius, center_z + radius + 1):
                chunk = self.get_chunk(x, z)
                if chunk:
                    chunks.append(chunk)
        return chunks
    
    def tick(self) -> None:
        """Tick the dimension (called every game tick).

        Order: tickets, then the idle clock, then entities. Expiring tickets
        first means a ticket that ran out this tick already stopped keeping its
        chunks loaded before anything reads the level map. This is the *only*
        place tickets age -- `ChunkLifecycle.tick` deliberately does not purge,
        or every timeout would count down at twice the rate.

        Entity ticking has two gates, and they answer different questions:

        * `is_active` -- a dimension nobody is simulating stops processing
          entities after 300 idle ticks, while its chunks stay loaded. The
          counter is reset by any ticket that keeps the dimension active (a
          simulating player, forced chunks, a portal, a thrown ender pearl).
        * `ticks_entity` -- per entity, the load level of the chunk it stands
          in (`Chunk.mdx`: only level <= 31 processes entities at all). A mob in
          a lazy (32) or border (33) chunk is skipped, not removed: it keeps its
          position, its velocity and its place in every table, and resumes the
          moment its chunk ticks again. Physics asks the same question of the
          same chunks (`world/dimension_actor.py`), which is what makes "the AI
          ran but gravity did not" impossible. Players are not in `entities` and
          are not gated: vanilla ticks a player wherever they stand, and the
          movement validator is the only thing judging them.
        """
        self.tickets.tick(self.world_time)
        if self.tickets.activates_dimension():
            self.idle_ticks = 0
        else:
            self.idle_ticks += 1

        self.advance_clock(1)
        # Reset before the early returns, not inside the loop: the gauge
        # describes *this* tick, so a dimension that went idle or emptied out
        # must not keep reporting the last entity it paused.
        self.paused_entities = 0
        if not self.is_active:
            return
        if self.entities:
            # A per-tick memo, local to this call and thrown away with it, so
            # there is nothing to invalidate. It exists because the ticket query
            # costs microseconds (it walks the ticket table) and a herd shares a
            # chunk: without it, four cows in one chunk pay for the same answer
            # four times. Rebuilt from scratch next tick, so a ticket that moved
            # in between is picked up -- the physics gate dedups the same way and
            # for the same reason (`PhysicsSolver._ticking_mask`).
            ticking: Dict[Tuple[int, int], bool] = {}
            for entity in list(self.entities.values()):
                if entity.removed:
                    continue
                chunk = entity.position.chunk
                here = ticking.get(chunk)
                if here is None:
                    here = self.ticks_entity(entity)
                    ticking[chunk] = here
                if not here:
                    self.paused_entities += 1
                    continue
                entity.tick()

    def spawn_entity(self, entity: "Entity") -> bool:
        """Admit a non-player entity to this dimension.

        Assigns a runtime entity id if the caller did not, stamps the
        dimension name, and registers it. Refuses duplicates (by id or by
        uuid): the tracker's bookkeeping assumes one live object per id.

        MUST be called on this dimension's actor thread - the event loop's
        path into here is a message, never a direct call.
        """
        from entity import allocate_entity_id

        if entity.entity_id == 0:
            entity.entity_id = allocate_entity_id()
        if entity.entity_id in self.entities:
            logger.warning(
                f"Refusing duplicate entity id {entity.entity_id} in "
                f"{self.name} (existing: {self.entities[entity.entity_id]})"
            )
            return False
        entity.dimension = self.name
        entity.removed = False
        self.entities[entity.entity_id] = entity
        return True

    def remove_entity(self, entity_id: int) -> Optional["Entity"]:
        """Take a non-player entity out of the world, marking it removed.

        The tracker's removal broadcast is the caller's job - this method
        only owns the table and the flag.
        """
        entity = self.entities.pop(entity_id, None)
        if entity is not None:
            entity.removed = True
        return entity

    def get_entity(self, entity_id: int) -> Optional["Entity"]:
        """A live non-player entity by id, or None."""
        return self.entities.get(entity_id)

    def advance_clock(self, ticks: int) -> None:
        """Advance world time by n ticks.

        Also used to fast-forward past ticks discarded while the dimension
        was busy, so its clock keeps up with the wall clock.
        """
        with self._lock:
            self.world_time += ticks
            if self.settings.fixed_time is None:
                self.day_time = (self.day_time + ticks) % 24000
            else:
                self.day_time = self.settings.fixed_time
    
    def add_player(self, player_id: str) -> None:
        """Add a player to this dimension."""
        if player_id not in self.players:
            self.players.append(player_id)
    
    def remove_player(self, player_id: str) -> None:
        """Remove a player from this dimension."""
        if player_id in self.players:
            self.players.remove(player_id)
    
    @property
    def player_count(self) -> int:
        """Get number of players in this dimension."""
        return len(self.players)
    
    @property
    def chunk_count(self) -> int:
        """Get number of loaded chunks."""
        return self.chunk_manager.get_chunk_count()
    
    def get_dimension_data(self) -> Dict[str, Any]:
        """Get dimension data for network transmission."""
        return {
            "name": self.name,
            "dimension_type": self.dimension_type.value,
            "coordinate_scale": self.settings.coordinate_scale,
            "min_y": self.settings.min_y,
            "max_y": self.settings.max_y,
            "height": self.settings.height,
            "logical_height": self.settings.logical_height,
            "sea_level": self.settings.sea_level,
            "generator_preset": self.settings.preset,
            "has_skylight": self.settings.has_skylight,
            "has_ceiling": self.settings.has_ceiling,
            "ultra_warm": self.settings.ultra_warm,
            "natural": self.settings.natural,
            "fixed_time": self.settings.fixed_time,
            "ambient_light": self.settings.ambient_light,
        }
    
    def unload_all_chunks(self) -> int:
        """
        Unload all chunks *without* saving -- the caller has already done that.

        Returns:
            Number of chunks unloaded
        """
        count = self.chunk_manager.get_chunk_count()
        self.chunk_manager.clear()
        logger.info(f"Unloaded {count} chunks from dimension {self.name}")
        return count
    
    def save_all_chunks(self) -> int:
        """
        Save all dirty chunks of this dimension, synchronously.

        The durable path (world save, shutdown). Chunks that leave memory while
        the server runs go through the streaming saver instead, which owns them
        on a different thread -- see `world/chunk_saver.py`.
        """
        if self.save_handler is None:
            logger.debug(f"{self.name}: no save handler installed, nothing written")
            return 0
        return self.save_handler()
