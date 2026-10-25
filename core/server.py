"""
Crotalus Server

Main server class that manages network, threads, and game state.
"""

import asyncio
import os
import queue
import signal
import sys
import time
import datetime
import threading

from tqdm import tqdm
from pathlib import Path
from typing import Optional
from loguru import logger

from core.config import config
from core.logs import compress_old_logs
from core.threads import thread_manager
from network.buffer import ByteBuffer
from network.connection import Connection
from network.packet import GLOBAL_REGISTRY
from network.packets import (
    SHandshake, CStatusRequest, SPingResponse, SStatusResponse, CPingRequest
)
from network.protocol import ProtocolState, PROTOCOL_VERSION
from world.chunk import ChunkPos

def is_port_in_use(port: int, host: str = "127.0.0.1") -> bool:
    """检测端口是否被占用"""
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.connect((host, port))
            s.close()
            return True  # 连接成功，端口被占用
        except PermissionError:
            return True  # 没有权限访问端口，视为占用
        except (ConnectionRefusedError, OSError):
            return False  # 连接失败，端口可用

class CrotalusServer:
    """Main Crotalus Minecraft Server."""

    def __init__(self):
        self._running = False
        self._server: Optional[asyncio.Server] = None
        self._connections: dict[str, Connection] = {}
        self._lock = asyncio.Lock()
        # The network loop where every connection, writer and handler task
        # lives. The console thread runs commands on its own throwaway loop,
        # so stop() must hop back here before touching any of them.
        self._loop: Optional[asyncio.AbstractEventLoop] = None

        # Server info for status
        self._server_version = "Crotalus 1.21.11"
        self._protocol_version = PROTOCOL_VERSION

        # World system
        self._world = None
        self._world_manager = None
        # Loaded before the world; see _init_registry.
        self._registry = None
        # Resolved against the generated terrain during _init_world.
        self._spawn_point: tuple[int, int, int] = (0, 64, 0)
        # Spawn chunks the boot light pass could not patch (the disc's rim);
        # handed to the overworld actor's engine in _start_ticking.
        self._spawn_light_backlog: list = []

        # Console system
        self._console = None
        self._console_thread: Optional[threading.Thread] = None

        # Tick pipeline: Ticker paces, one DimensionActor per dimension executes
        self._ticker = None
        self._actors: list = []
        self._actors_by_dim: dict = {}
    
    async def start(self) -> None:
        """Start the server."""
        if self._running:
            logger.warning("Server is already running")
            return
        
        logger.info("=" * 50)
        logger.info(f"Starting {self._server_version}")
        logger.info(f"Protocol: {self._protocol_version}")
        logger.info(f"Port: {config.server_port}")
        logger.info(f"Max Players: {config.max_players}")
        logger.info("=" * 50)
        
        logger.info(f"Starting crotalus on {config.server_ip or '0.0.0.0'}:{config.server_port}")

        self._loop = asyncio.get_running_loop()

        if is_port_in_use(config.server_port, "127.0.0.1"):
            logger.warning("**** FAILED TO BIND TO PORT! ****")
            logger.warning(f"Port {config.server_port} is already in use. Please free the port or use a different port.")
            logger.info("Server will now exit.")
            sys.exit(1)
        
        if not config.online_mode:
            logger.warning("**** SERVER IS RUNNING IN OFFLINE/INSECURE MODE! ****")
            logger.warning("The server will make no attempt to authenticate usernames. Beware.")
            logger.warning("While this makes the game possible to play without internet access, it also opens up the ability for hackers to connect with any username they choose.")
            logger.warning("To change this, set online-mode to true in config.toml file.")
            logger.warning("*************** USE AT YOUR OWN RISK! ***************")
        
        start_time = time.time()

        # Load the complete game registry before anything touches the world.
        #
        # This is not a warm-up. Every block state id that goes into a chunk
        # palette is resolved from these tables, and the login path sends the
        # synced registries out of them, so a world generated or loaded before
        # this point would encode state ids from a half-built registry. It also
        # has to happen before the world *load* path, not just generation, for
        # the same reason: reading a saved chunk back needs the same table.
        #
        # Plugins run inside this call and the registry is frozen at the end of
        # it, which is the last moment an id may be allocated.
        self._init_registry()

        # Initialize world system
        await self._init_world()
        
        # Start threads
        thread_manager.start_all()

        # Start the tick pipeline: Ticker paces, dimension actors execute
        self._start_ticking()

        # 统一命令线程：控制台与客户端命令都投递到这里排队执行。必须在
        # 监听端口之前启动 —— 客户端的第一条命令也要有调度方在等它。
        from console.dispatcher import command_dispatcher
        command_dispatcher.start()

        # Start network server
        self._server = await asyncio.start_server(
            self._handle_client,
            config.server_ip or None,
            config.server_port,
            reuse_address=True,
        )
        
        self._running = True
        
        init_time = time.time() - start_time
        
        logger.info(f"Done({init_time:.3f}s)! For help, type \"help\" or \"?\"")

        self._start_console()

        # Player-facing commands, dispatched from the Chat Command packet.
        from player.commands import register_player_commands
        register_player_commands()

        # Keep server running
        async with self._server:
            await self._server.serve_forever()
    
    def _init_registry(self) -> None:
        """Load blocks, items, recipes, tags and the synced registries.

        Failure here is fatal on purpose. A server that cannot build its
        registry can still accept a connection, and then sends chunks whose
        palette entries mean nothing - a desync that looks like a client bug.
        Better to refuse to start.
        """
        from registry import init_registries
        from registry.loader import validate_registry

        plugins_dir = Path(config.plugins_dir)
        datapacks_dir = Path(config.datapacks_dir)

        logger.info(f"Loading game registry...")
        
        try:
            self._registry = init_registries(
                plugins_dir=plugins_dir,
                datapacks_dir=datapacks_dir,
                strict=False,
            )
        except Exception as e:
            logger.exception(f"Failed to load the game registry: {e}")
            raise

        # Re-run the checks so their result is visible in the server log even
        # when the loader was already initialized by an earlier import.
        problems = validate_registry(self._registry)
        if problems:
            for problem in problems[:10]:
                logger.warning(f"Registry problem: {problem}")
            if len(problems) > 10:
                logger.warning(f"... and {len(problems) - 10} more registry problems")

        summary = self._registry.summary()
        logger.info(
            f"Registry ready: {summary['blocks']} blocks / "
            f"{summary['block_states']} states, {summary['items']} items, "
            f"{summary['recipes']} recipes, {summary['tags']} tags, "
            f"{summary['synced_entries']} synced entries"
            + (f", {summary['plugins']} plugin(s)" if summary["plugins"] else "")
        )

        self._warm_up_kernels()

    def _warm_up_kernels(self) -> None:
        """Compile the numba kernels before any tick depends on them.

        `cache=True` makes the *second* process cheap, not the first, and the
        first call after a source change is seconds of LLVM for kernels this
        size (`LIGHT_PLAN.md` §5.1: ~4.8 s for the light kernels, ~0.8 s once
        cached). Inside a tick that is a multi-second stall on a server whose
        Ticker discards late ticks; here it is startup latency with a log line
        next to it, which is the trade `physics.kernel.warm_up` already makes.

        Runs after the registry is frozen because the light tables are built
        from it, and deliberately does not fail the boot: a warm-up that throws
        means the kernels will compile on first use instead, which is slower
        but not broken.
        """
        import time

        started = time.perf_counter()
        try:
            from physics.kernel import warm_up as warm_physics  # type: ignore

            warm_physics()
        except Exception as error:  # pragma: no cover - optional dependency
            logger.warning(f"Physics kernel warm-up skipped: {error}")
        try:
            from world.light_engine import warm_up as warm_light

            warm_light()
        except Exception as error:  # pragma: no cover - defensive
            logger.warning(f"Light kernel warm-up skipped: {error}")
        logger.info(
            f"Kernels warmed in {time.perf_counter() - started:.2f}s "
            f"(the first tick pays none of it)"
        )

    async def _init_world(self) -> None:
        """Initialize the world system."""
        try:
            from world import WorldManager, world_exists
            from world.dimension import DimensionType

            # Get world name from config
            world_name = config.level_name or "world"
            world_seed = self._parse_seed(config.level_seed)

            logger.info(f"Loading world: {world_name}")

            # Create world manager
            self._world_manager = WorldManager(base_dir=Path("data"))

            # Check if world exists
            if world_exists(Path("data"), world_name):
                # Load existing world
                self._world = self._world_manager.load_world(world_name)
                if self._world:
                    logger.info(f"World '{world_name}' loaded successfully")
                else:
                    logger.warning(f"Failed to load world '{world_name}', creating new one")
                    self._world = self._world_manager.create_world(world_name, seed=world_seed)
            else:
                # Create new world
                self._world = self._world_manager.create_world(world_name, seed=world_seed)
                logger.info(f"Created new world: {world_name}")
            
            # Generate spawn chunks
            #
            # Order matters: the spawn *column* is resolved after the area is
            # generated, because picking a surface height out of terrain that
            # does not exist yet reads a height of zero. On a world with no
            # saved spawn the X/Z default to the origin, so the disc below is
            # always centred on the right place.
            #
            # Through the worker pool, not a synchronous loop: at ~1.2 s per
            # vanilla chunk the old loop pinned one core for over a minute on
            # a radius-4 area, while the pool cuts that to a fraction (and
            # the wait overlaps the workers' own spin-up).
            logger.info("Preparing spawn area...")
            spawn_x = self._world.data.spawn_x or 0
            spawn_z = self._world.data.spawn_z or 0
            spawn_chunk_x, spawn_chunk_z = spawn_x // 16, spawn_z // 16
            spawn_radius = config.spawn_radius
            start_time = time.time()
            spawn_coords = [
                (spawn_chunk_x + dx, spawn_chunk_z + dz)
                for dx in range(-spawn_radius, spawn_radius + 1)
                for dz in range(-spawn_radius, spawn_radius + 1)
            ]
            with tqdm(total=len(spawn_coords), desc="Preparing spawn area", unit="chunk") as pbar:
                self._world.generate_chunks_blocking(
                    spawn_coords,
                    progress=lambda done: pbar.update(1),
                )
            logger.info(f"Time elapsed: {time.time() - start_time:.2f}s")

            # Light the disc before the actor loop exists.
            #
            # Generation writes no light, and the two places that queue a
            # chunk's first pass (`ChunkLifecycle.on_chunk_ready` and its
            # `_load_missing` scan) belong to lifecycles that `_start_ticking`
            # has not built yet. So this is the last point at which the spawn
            # area can be lit without an actor to carry the work: done here, a
            # joining player is sent real light instead of the encoder's
            # heightmap approximation. `skipped` is mostly chunks that came
            # back from disk already lit -- the whole disc on a re-boot.
            try:
                light_started = time.time()
                lit = self._world.light_chunks_blocking(
                    spawn_coords, DimensionType.OVERWORLD
                )
                logger.info(
                    f"Spawn light: {lit['lit']} lit, {lit['skipped']} already "
                    f"lit, {len(lit['deferred'])} awaiting neighbours "
                    f"in {time.time() - light_started:.2f}s"
                )
                if lit["engine_errors"]:
                    # Never silent: chunks left unlit here are chunks the
                    # client draws wrong until someone runs `/light relight`,
                    # which is the bug this pass exists to remove.
                    logger.warning(
                        f"Spawn lighting hit {lit['engine_errors']} kernel "
                        f"error(s); {lit['pending']} chunk(s) still unlit"
                    )
                # The rim of the disc cannot be patched until something loads
                # outside it. Hand those to the actor's engine in
                # `_start_ticking`, which retries them until then -- the boot
                # pass's own engine dies with the call.
                self._spawn_light_backlog = list(lit["deferred"])
            except Exception as error:  # noqa: BLE001
                # Dark spawn beats no world. This pass is an improvement over
                # what came before, and letting it fail the whole
                # `_init_world` would throw away a generated, saved spawn area.
                logger.warning(f"Spawn lighting skipped: {error}")

            spawn_x, spawn_y, spawn_z = self._world.get_spawn_point()
            self._spawn_point = (spawn_x, spawn_y, spawn_z)
            logger.info(f"World spawn: ({spawn_x}, {spawn_y}, {spawn_z})")
            
        except ImportError as e:
            logger.warning(f"World module not available: {e}")
            self._world = None
        except Exception as e:
            logger.error(f"Failed to initialize world: {e}")
            self._world = None
    
    def _parse_seed(self, seed_str: str) -> int:
        """Parse world seed from string."""
        if not seed_str:
            import random
            return random.randint(0, 2**31 - 1)
        
        try:
            return int(seed_str)
        except ValueError:
            # Hash string to int
            return hash(seed_str) & 0xFFFFFFFF

    def _start_ticking(self) -> None:
        """Start the tick pipeline: Ticker paces, one actor per dimension executes."""
        if not self._world:
            return

        from core.ticker import Ticker
        from world.dimension_actor import DimensionActor
        from world.chunk_gen_thread import chunk_generator

        self._ticker = Ticker()
        # How far entity broadcast reaches, as a percentage of the range the
        # entity type declares. The key has shipped in config.toml since the
        # config was first written and nothing ever read it; the tracker is
        # its first consumer. Clamped to something sane rather than trusted:
        # a typo'd 0 makes every other player invisible, and 5000 would try
        # to hold the whole map in every client's entity list.
        broadcast_pct = config.get("advanced.entity-broadcast-range-percentage", 100)
        try:
            broadcast_pct = max(1, min(int(broadcast_pct), 500))
        except (TypeError, ValueError):
            logger.warning(
                f"advanced.entity-broadcast-range-percentage is "
                f"{broadcast_pct!r}, not a number; using 100"
            )
            broadcast_pct = 100

        # Natural mob spawning. Read once here and handed to every actor
        # rather than re-read per dimension: a config change needs a restart,
        # which is already true of every other key in the file.
        natural_spawning = bool(config.get("spawning.natural-mob-spawning", True))
        spawn_interval = config.get("spawning.spawn-interval-ticks", 20)
        try:
            spawn_interval = max(1, int(spawn_interval))
        except (TypeError, ValueError):
            logger.warning(
                f"spawning.spawn-interval-ticks is {spawn_interval!r}, not a "
                f"number; using 20"
            )
            spawn_interval = 20
        if not natural_spawning:
            logger.info("Natural mob spawning is disabled by config")

        # Player movement authority, read once here for the same reason: a
        # config change needs a restart, which is already true of the file. The
        # threshold is clamped rather than rejected -- a nonsense value falls
        # back to the default with a warning, because the alternative is a
        # server that refuses to boot over a typo in a tuning knob.
        authoritative_movement = config.server_authoritative_movement
        correction_threshold = config.movement_correction_threshold
        if not (0.0 <= correction_threshold <= 16.0):
            logger.warning(
                f"advanced.movement-correction-threshold is "
                f"{correction_threshold!r}; using 0.5"
            )
            correction_threshold = 0.5
        if not authoritative_movement:
            logger.info(
                "Server-authoritative player movement is disabled by config: "
                "movement packets are taken at face value"
            )

        for dim_type, dimension in self._world.dimensions.items():
            # The lifecycle is created by `World` (so tickets can be added by
            # anything holding the world); installing the request handler here
            # is what makes it load chunks instead of generating them inline on
            # the actor thread.
            lifecycle = self._world.lifecycle(dim_type)
            on_missing = self._missing_chunks_for(dim_type)
            if lifecycle is not None:
                lifecycle.set_missing_handler(on_missing)
                lifecycle.sync_fallback = False
            actor = DimensionActor(
                dimension,
                self._ticker.done_inbox,
                on_missing_chunks=on_missing,
                entity_broadcast_percentage=broadcast_pct,
                lifecycle=lifecycle,
                natural_spawning=natural_spawning,
                spawn_interval_ticks=spawn_interval,
                server_authoritative_movement=authoritative_movement,
                movement_correction_threshold=correction_threshold,
            )
            self._actors.append(actor)
            self._actors_by_dim[dim_type] = actor
            self._ticker.register_actor(actor.name, actor.inbox)
            actor.start()

        # While the actors are up, finished chunks are handed to the owning
        # actor's inbox, so the chunk manager is only ever mutated on the
        # thread that owns it. Workers themselves never touch dimension
        # state. If the tick pipeline stops (shutdown, or no actor for a
        # dimension), the sinks fall back to the world's direct insert.
        chunk_generator.set_completion_sink(self._route_finished_chunk)
        chunk_generator.set_failure_sink(self._route_failed_chunk)

        self._queue_spawn_light_backlog()

        # Ticket state belongs to the actor thread, so the console and the world
        # API reach it through the actor's inbox rather than touching it from
        # wherever the operator happens to be. Cleared again in _stop_ticking,
        # after which a direct call is safe because no actor is running.
        self._world.chunk_actor_inbox = self._actor_inbox_for
        self._ticker.start()

    def _queue_spawn_light_backlog(self) -> None:
        """Hand the spawn disc's rim to the overworld's light engine.

        `World.light_chunks_blocking` lit the disc's interior at boot with a
        throwaway engine, because that was the last moment before the listener
        opens. It deliberately left the rim: a patch is one chunk's 3x3, and the
        rim's outer neighbours do not exist until a player walks out there.
        Those coordinates come back as `deferred`, and this is where they go
        onto the engine that will actually finish them.

        The engine re-queues a chunk whose 3x3 is incomplete and retries it on
        later ticks, so this is a promise rather than a poll: they light as soon
        as the streamer loads their outer neighbours, with no further wiring.

        Queued here rather than in the actor's constructor because the actor
        starts above and starts ticking immediately -- this runs after
        `actor.start()`, on the main thread, before the first tick has been
        paced. Nothing has ticked yet, so nothing races the enqueue, and the
        streamer's own `request_chunk` calls are idempotent against it.
        """
        backlog = self._spawn_light_backlog
        self._spawn_light_backlog = []
        if not backlog:
            return

        from world.dimension import DimensionType

        actor = self._actors_by_dim.get(DimensionType.OVERWORLD)
        if actor is None:
            # No overworld actor: there is no tick to retry these, so say so
            # rather than leaving the operator to find black chunks by walking.
            logger.warning(
                f"{len(backlog)} spawn chunk(s) still await their outer "
                f"neighbours and no light engine will retry them"
            )
            return

        for x, z in backlog:
            actor.light.request_chunk((x, z))
        logger.info(
            f"{len(backlog)} spawn chunk(s) queued for light once their "
            f"neighbours arrive"
        )

    def _actor_inbox_for(self, dim_type):
        """The inbox of the actor that owns this dimension, or None."""
        actor = self._actors_by_dim.get(dim_type)
        return actor.inbox if actor else None

    def _missing_chunks_for(self, dim_type):
        """Bind a missing-chunk callback to one dimension.

        Called from that dimension's actor thread, which is exactly where the
        claim has to happen: `claim_chunk` mutates the chunk manager the
        actor owns. The blank chunk it leaves is the streamer's dedupe mark
        ("claimed, being filled right now"), so two players whose views share
        a chunk produce one generation, not two. The lifecycle asks through
        the same callback, so a chunk that is in someone's view and a chunk a
        ticket covers are one request, not two.

        The dimension has to be captured per actor. The generator's workers
        are shared, and a request carries the dimension it is for -- routing
        every request to the overworld would fill a nether player's view
        with overworld terrain and leave their own view empty.
        """
        dimension = self._world.dimensions.get(dim_type)
        lifecycle = self._world.lifecycle(dim_type)

        def on_missing_chunks(cx: int, cz: int, priority: int) -> bool:
            try:
                claimed = self._world.claim_chunk(cx, cz, dim_type)
            except Exception:
                logger.exception(f"Failed to claim chunk ({cx}, {cz}) for {dim_type}")
                return False

            # The claim now sits in the manager whoever asked for it -- the
            # lifecycle's own `_request` (which records ownership itself) or
            # the streamer's view pump (which does not). Adoption is what
            # makes a streamer-created claim unloadable when the last viewer
            # leaves; without it the coordinate is invisible to the unload
            # pass and strands in memory.
            if lifecycle is not None:
                lifecycle.adopt_claim(ChunkPos(cx, cz))

            if not claimed:
                # Already in the manager: either finished (nothing to do) or a
                # blank claim another request is filling. The blank case is
                # the interesting one -- a worker that died leaves the claim
                # behind forever, and the lifecycle only re-asks after its
                # stall timeout, so answering "satisfied" here would strand the
                # coordinate. Queue a request instead: a duplicate for a chunk
                # already in flight is harmless (the losing insert is
                # discarded), while a missing one is a hole in someone's view.
                chunk = (
                    dimension.chunk_manager.peek(ChunkPos(cx, cz))
                    if dimension is not None
                    else None
                )
                if chunk is None or chunk.is_generated:
                    return True

            # Returns False when the request was refused (queue full, or the
            # generator is stopped). The caller retries refused chunks, so this
            # must not be swallowed.
            return self._world.request_chunk(cx, cz, dim_type, priority=priority)

        return on_missing_chunks

    def _route_finished_chunk(self, dimension: str, chunk) -> None:
        """Completion sink while actors are up: post the chunk to its actor.

        Runs on the generator pool's result thread. The insert itself happens
        when the actor processes the message, on the actor's own thread.
        """
        from world.dimension import DimensionType
        from world.chunk_gen_thread import ChunkReadyMsg

        actor = self._actors_by_dim.get(DimensionType(dimension))
        if actor is not None:
            actor.inbox.put(ChunkReadyMsg(chunk=chunk))
        else:
            # No actor for this dimension (tick pipeline down): fall back to
            # the world's direct insert so finished work is not wasted.
            self._world.insert_generated_chunk(dimension, chunk)

    def _route_failed_chunk(self, dimension: str, chunk_x: int, chunk_z: int) -> None:
        """Failure sink while actors are up: let the actor drop the claim."""
        from world.dimension import DimensionType
        from world.chunk_gen_thread import ChunkFailedMsg

        actor = self._actors_by_dim.get(DimensionType(dimension))
        if actor is not None:
            actor.inbox.put(ChunkFailedMsg(chunk_x=chunk_x, chunk_z=chunk_z))
        else:
            self._world.discard_failed_chunk(dimension, chunk_x, chunk_z)

    def _stop_ticking(self) -> None:
        """Stop the tick pipeline: no new ticks, then actors drain and stop."""
        if self._world is not None:
            # Actors are going away, so the console may touch the lifecycles
            # directly again (nothing else owns them).
            self._world.chunk_actor_inbox = None
        if self._ticker:
            self._ticker.stop()
            self._ticker = None
        for actor in self._actors:
            actor.stop()
        self._actors.clear()
        self._actors_by_dim.clear()

        # Actors are gone; completions must not be posted into dead inboxes.
        # Back to the world's direct insert (used by the shutdown drain, if
        # any in-flight generations land between here and the pool stop).
        from world.chunk_gen_thread import chunk_generator

        if self._world:
            chunk_generator.set_completion_sink(self._world.insert_generated_chunk)
            chunk_generator.set_failure_sink(self._world.discard_failed_chunk)

    def _player_actor_inbox(self) -> Optional["queue.Queue"]:
        """The overworld actor's inbox, where player view updates are posted.

        Players start in the overworld, so that is the only dimension that
        needs a wired path for now. A nether transfer would re-post the view
        message to a different actor.
        """
        from world.dimension import DimensionType

        actor = self._actors_by_dim.get(DimensionType.OVERWORLD)
        return actor.inbox if actor else None

    async def broadcast_player_info_add(
        self, uuid: str, name: str, latency: int = 0
    ) -> None:
        """Put a player in everyone's tab list, including their own.

        This is not cosmetic. The client resolves the sender of an incoming
        Player Chat Message against its tab list before it will render the
        line, and a miss surfaces to the player as a "chat validation error"
        toast with the message discarded. So this has to reach every loaded
        connection, and it has to land before any chat is relayed.
        """
        from network.packets import CPlayerInfoUpdate, PlayerInfoEntry

        packet = CPlayerInfoUpdate.add_player(
            PlayerInfoEntry(uuid, name, latency=latency)
        )
        async with self._lock:
            targets = [c for c in self._connections.values() if c.in_game]
        for conn in targets:
            try:
                await conn.send_packet(packet)
            except Exception as e:
                logger.debug(f"Broadcast to {conn.remote_address} failed: {e}")

    async def broadcast_player_info_remove(self, uuid: str) -> None:
        """Drop a player from every tab list.

        The client only ever adds entries, so this is the only thing that
        takes one out; without it a disconnected player stays listed forever,
        and their UUID is still resolvable for chat.
        """
        from network.packets import CPlayerInfoRemove

        packet = CPlayerInfoRemove([uuid])
        async with self._lock:
            targets = [c for c in self._connections.values() if c.in_game]
        for conn in targets:
            try:
                await conn.send_packet(packet)
            except Exception as e:
                logger.debug(f"Broadcast to {conn.remote_address} failed: {e}")

    async def broadcast_player_info_gamemode(self, uuid: str, gamemode: int) -> None:
        """Refresh a player's gamemode icon in every tab list.

        Sent on a gamemode change; without it the other clients keep showing
        the icon the join add set, which quietly drifts from reality.
        """
        from network.packets import CPlayerInfoUpdate, PlayerInfoEntry, ACTION_UPDATE_GAME_MODE

        packet = CPlayerInfoUpdate(
            [PlayerInfoEntry(uuid, game_mode=gamemode)],
            actions=ACTION_UPDATE_GAME_MODE,
        )
        async with self._lock:
            targets = [c for c in self._connections.values() if c.in_game]
        for conn in targets:
            try:
                await conn.send_packet(packet)
            except Exception as e:
                logger.debug(f"Broadcast to {conn.remote_address} failed: {e}")

    async def broadcast_system_message(self, message: str) -> None:
        """Send a chat line to every loaded player.

        Takes a snapshot of the connections under the lock, then sends outside
        it: send_packet awaits the writer, so holding the lock across the loop
        would let one slow client stall every join and leave.
        """
        from network.packets import CSystemChatMessage

        packet = CSystemChatMessage(message)
        async with self._lock:
            targets = [c for c in self._connections.values() if c.in_game]
        for conn in targets:
            try:
                await conn.send_packet(packet)
            except Exception as e:
                # A dying connection must not take the rest of the broadcast
                # with it; its own close path cleans it up.
                logger.debug(f"Broadcast to {conn.remote_address} failed: {e}")

    async def broadcast_player_chat(
        self, sender_name: str, sender_uuid: Optional[str], sender_index: int,
        message: str
    ) -> None:
        """Relay a line a player typed, to every loaded player, as Player Chat.

        Unlike broadcast_system_message, this cannot build one packet and
        reuse it: the Global Index counts what a *single* client has been
        sent, so each connection claims its own off its own player and gets
        its own packet.
        """
        from network.packets import CPlayerChatMessage

        if not config.get("server.player-chat-packet", True):
            await self.broadcast_system_message(f"{sender_name}: {message}")
            return

        async with self._lock:
            targets = [c for c in self._connections.values() if c.in_game]
        for conn in targets:
            try:
                await conn.send_packet(
                    CPlayerChatMessage(
                        sender_uuid,
                        sender_name,
                        message,
                        index=sender_index,
                        global_index=conn.player.claim_chat_index(),
                    )
                )
            except Exception as e:
                # A dying connection must not take the rest of the broadcast
                # with it; its own close path cleans it up.
                logger.debug(f"Broadcast to {conn.remote_address} failed: {e}")

    async def _handle_client(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter
    ) -> None:
        """Handle a new client connection."""
        # Import here to avoid circular import
        from network.client_handler import ClientHandler

        addr = writer.get_extra_info("peername")
        logger.info(f"Client connected: {addr}")

        # Create client handler
        conn = ClientHandler(
            reader,
            writer,
            actor_inbox=self._player_actor_inbox(),
            spawn_point=self._spawn_point,
            broadcast=self.broadcast_player_chat,
            system_broadcast=self.broadcast_system_message,
            player_info_add=self.broadcast_player_info_add,
            player_info_remove=self.broadcast_player_info_remove,
            player_info_gamemode=self.broadcast_player_info_gamemode,
        )
        
        # Register in connection list
        async with self._lock:
            self._connections[str(addr)] = conn
        
        try:
            # Start handling the connection
            await conn.start()
        finally:
            # Remove from connection list
            async with self._lock:
                self._connections.pop(str(addr), None)
            
            logger.info(f"Client disconnected: {addr}")
    
    async def stop(self) -> None:
        """Stop the server."""
        if not self._running:
            return
        
        logger.info("Stopping server...")
        self._running = False

        # 先停命令线程（不再执行新的命令）。若本次 stop 是 /stop 命令
        # 发起的，dispatcher.stop() 会识别出调用方就是命令线程自己并
        # 跳过 —— 否则会把自己正跑着关停协程的 loop 停掉。
        from console.dispatcher import command_dispatcher
        command_dispatcher.stop(timeout=3.0)

        # Disconnect every player ("Server closed" on their screen), then
        # stop the listener. Teardown must run on the network loop: the
        # console thread executes stop() on its own throwaway loop, and
        # awaiting a connection's writer/tasks from there hangs shutdown.
        if self._loop and self._loop.is_running():
            if self._loop is asyncio.get_running_loop():
                await self._shutdown_connections()
            else:
                await asyncio.wrap_future(
                    asyncio.run_coroutine_threadsafe(self._shutdown_connections(), self._loop)
                )

        # Stop ticking before saving so nothing mutates the world mid-save
        self._stop_ticking()

        # Save and unload world
        await self._shutdown_world()
        
        # Stop server
        if self._server:
            self._server.close()
            await self._server.wait_closed()

        # Stop threads
        thread_manager.stop_all()

        logger.info("Server stopped")

    async def _shutdown_connections(self) -> None:
        """Disconnect and close every live connection. Network loop only.

        The lock guards the dict, not the connections: snapshot under it and
        release before awaiting anything. ClientHandler.disconnect() fires
        the leave broadcast, which re-enters broadcast_* -> self._lock, and
        asyncio.Lock is not reentrant — holding it here deadlocked shutdown
        right after "left the game." was logged.
        """
        async with self._lock:
            conns = list(self._connections.values())
        for conn in conns:
            try:
                await conn.disconnect("Server closed")
            except Exception as e:
                logger.exception(f"Error disconnecting {conn.remote_address}: {e}")

    def _start_console(self) -> None:
        """Start the console in a separate thread."""
        try:
            from console import SimpleConsoleInterface, register_builtin_commands
            from console.command import cmd_manager

            # 注册内置命令
            register_builtin_commands(
                stop_callback=self.stop,
                world_getter=lambda: self._world,
            )

            # 创建控制台
            self._console = SimpleConsoleInterface(
                command_manager=cmd_manager,
                title=f"{self._server_version} Console",
                prompt="> ",
            )

            # 在单独线程中运行控制台
            self._console_thread = threading.Thread(
                target=self._run_console,
                daemon=True,
                name="ConsoleThread"
            )
            self._console_thread.start()

            logger.info("Console started in background thread")

        except ImportError as e:
            logger.warning(f"Console module not available: {e}")
        except Exception as e:
            logger.exception(f"Failed to start console: {e}")

    def _run_console(self) -> None:
        """Run the console (called in separate thread)."""
        import asyncio
        if self._console:
            # 设置事件循环以便在主事件循环中执行协程
            try:
                loop = asyncio.get_event_loop()
                self._console.set_event_loop(loop)
            except RuntimeError:
                # 如果没有事件循环，创建一个
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                self._console.set_event_loop(loop)
            self._console.run()

    async def _shutdown_world(self) -> None:
        """Shutdown and save the world."""
        if not self._world:
            return

        try:
            # Generation stops BEFORE the save, not after: a chunk landing in
            # the chunk manager after the save pass has iterated would stay
            # dirty forever, and concurrent inserts can break the save's dict
            # iteration outright. The drain is bounded; whatever misses it is
            # regenerated deterministically next boot.
            from world.chunk_gen_thread import chunk_generator

            logger.info("Stopping chunk generation...")
            chunk_generator.stop()

            logger.info("Saving world...")

            # Save all chunks
            self._world.save()

            # Shutdown world
            if self._world_manager:
                self._world_manager.shutdown()

            logger.info("World saved successfully")
        except Exception as e:
            logger.exception(f"Failed to save world: {e}")
    
    @property
    def world(self):
        """Get the world instance."""
        return self._world

    @property
    def registry(self):
        """The loaded game registry, or None before startup reaches it."""
        return self._registry

    @property
    def ticker(self):
        """Get the Ticker instance, or None while the tick pipeline is stopped."""
        return self._ticker

    def get_status(self) -> dict:
        """Get server status."""
        thread_status = thread_manager.get_status()
        
        status = {
            "running": self._running,
            "version": self._server_version,
            "protocol": self._protocol_version,
            "players": {
                "max": config.max_players,
                "online": len(self._connections),
            },
            "threads": thread_status,
            "ticker": {
                "tps": self._ticker.tps if self._ticker else 0,
                "tick": self._ticker.tick_count if self._ticker else 0,
                "actors": len(self._actors),
            },
        }
        
        if self._world:
            status["world"] = self._world.get_world_info()
        
        return status


# Global server instance
_server: Optional[CrotalusServer] = None


def get_server() -> CrotalusServer:
    """Get the global server instance."""
    global _server
    if _server is None:
        _server = CrotalusServer()
    return _server


async def run():
    """Run the server."""
    # Setup logging
    logger.remove()
    logger.add(
        sys.stdout,
        level="DEBUG" if config.debug else "INFO",
        format="<green>[{time:HH:mm:ss}]</green> <level>[{thread.name}/{level: <8}]</level> | <level>{message}</level>"
    )
    log_path = Path(f"logs/Crotalus_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log")
    logger.add(
        log_path,
        level="DEBUG", # default to DEBUG
        format="[{time:HH:mm:ss}] [{thread.name}/{level: <8}] | {message}",
        enqueue=True,
        backtrace=True,
        diagnose=True,
    )

    # Vanilla archives previous runs once the new log is open, so the
    # "Compressing ..." lines land in this run's log too.
    compress_old_logs(log_path.parent, log_path.name)
    
    # Get server instance
    server = get_server()
    
    # Setup signal handlers
    #
    # The handler itself must not call logger.*: a signal can interrupt the
    # main thread in the middle of a loguru emit, while it still holds the
    # handler's internal lock, and logging again from the handler then aborts
    # with "Could not acquire internal lock (deadlock avoided)". The same
    # goes for asyncio.create_task, which needs a running loop at that exact
    # instant. Keep the handler dumb: defer everything onto the event loop
    # via call_soon_threadsafe and log/stop from a safe point in the loop.
    loop = asyncio.get_running_loop()
    shutdown_started = False

    def signal_handler(sig, frame):
        if loop.is_closed():
            return  # Process is already exiting; nothing left to stop.
        loop.call_soon_threadsafe(initiate_shutdown, sig)

    def initiate_shutdown(sig):
        nonlocal shutdown_started
        if shutdown_started:
            logger.warning("Shutdown already in progress; forcing immediate exit (unsaved changes may be lost).")
            os._exit(1)
        shutdown_started = True
        logger.info(f"Received signal {sig}, shutting down...")
        asyncio.create_task(server.stop())
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Start server
    try:
        await server.start()
    except (asyncio.CancelledError, KeyboardInterrupt):
        logger.info("Shutting down Python...")
    except Exception as e:
        logger.exception(f"Server error: {e}")
        await server.stop()


if __name__ == "__main__":
    asyncio.run(run())
