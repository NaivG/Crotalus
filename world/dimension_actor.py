"""
Dimension actor.

One OS thread per dimension, owning every bit of that dimension's mutable
state. The inbound queue is the only way anything reaches a dimension:
packets, ticks from the Ticker, and (later) handoffs from other dimensions
all arrive as messages and are handled in order, on this thread. No locks
needed — single ownership is the concurrency control.

This is also where a dimension's chunk streaming lives. `ChunkStreamer` reads
chunks and keeps its own bookkeeping here, on the owning thread, so streaming
a chunk and mutating it in the same tick can never interleave. All it hands
back to the event loop is a list of finished packets.
"""

import queue
import zlib
from typing import Any, Callable, Optional
from loguru import logger

from core.threads import ServerThread
from core.ticker import TickDone, TickMsg
from entity.entity_tracker import EntityTracker
from entity.spawner import DEFAULT_INTERVAL_TICKS, NaturalSpawner
from network.packets.clientbound.play.update_time import CUpdateTime
from physics.block_shapes import block_shapes
from physics.player import PlayerMoveValidator
from physics.solver import PhysicsSolver
from physics.world_view import WorldView
from world.chunk_gen_thread import ChunkFailedMsg, ChunkReadyMsg
from world.chunk_lifecycle import ChunkLifecycle, StatsMsg, TicketMsg
from world.chunk_streamer import ChunkStreamer, PlayerViewMsg
from world.dimension import Dimension
from world.light_engine import DimensionLight
from world.perf import PerfMsg, TickProfiler
from world.light_query import (
    LightQueryMsg,
    LightRelightMsg,
    read_light,
    relight_all,
    relight_area,
)


#: How often the server tells its clients what time it is, in ticks. Vanilla's
#: `ServerLevel.tickTime` broadcasts every 20 (one second): the client advances
#: its own clock in between, so this is a drift correction, not the clock.
TIME_SYNC_TICKS = 20


def _terrain_ready(chunk) -> bool:
    """Whether a chunk's blocks are real yet, for the physics view.

    `ChunkManager.chunks` holds *claims* as well as finished chunks: a chunk the
    generator has been asked for is present but all air until `ChunkReadyMsg`
    replaces it. Physics must not see that state -- its rule for a chunk it
    cannot answer for is "freeze the body", and reading a not-yet-filled chunk
    as air would drop every mob standing on it through the world for one tick,
    which is exactly the kind of one-tick hole that never gets reproduced.
    """
    return chunk.is_generated


def _advanced_int(name: str, default: int) -> int:
    """One `[advanced]` performance valve, config or the default.

    Same shape as the light engine's `[world]` reader: a missing config
    module (tools) or a mistyped value falls back to the default rather than
    disabling the valve.
    """
    try:
        from core.config import config
    except ImportError:  # pragma: no cover - tools without the config module
        return default
    try:
        return int(config.get(f"advanced.{name}", default))
    except (TypeError, ValueError):
        return default


class DimensionActor(ServerThread):
    """Runs one Dimension: drains its inbox, executes its ticks."""

    def __init__(
        self,
        dimension: Dimension,
        done_inbox: "queue.Queue",
        on_missing_chunks: Optional[Callable[[int, int, int], bool]] = None,
        entity_broadcast_percentage: int = 100,
        lifecycle: Optional[ChunkLifecycle] = None,
        natural_spawning: bool = True,
        spawn_interval_ticks: int = DEFAULT_INTERVAL_TICKS,
        server_authoritative_movement: bool = True,
        movement_correction_threshold: float = 0.5,
    ):
        super().__init__(f"Dimension-{dimension.name}")
        self.dimension = dimension
        self.inbox: queue.Queue = queue.Queue()

        #: Whether clients should keep their own clock running between
        #: `CUpdateTime` packets. The day/night cycle runs unless this
        #: dimension's time is pinned (`DimensionSettings.fixed_time`, the
        #: End's twilight) -- the same rule `Dimension.advance_clock` applies to
        #: its own clock, so client and server cannot disagree about whether
        #: time passes.
        self._day_cycle_runs = getattr(dimension.settings, "fixed_time", None) is None

        self._done_inbox = done_inbox
        self._last_tick_no = 0

        # Load/unload driver. `World` creates one per dimension (so tickets can
        # be added by anything holding the world); the actor is what ticks it,
        # on this thread, next to the streamer that sends what it loaded.
        #
        # The light engine is created first and handed to the lifecycle, because
        # the lifecycle is what knows a coordinate became *readable* -- the one
        # moment light has to be queued by, since the streamer and the spawner
        # both read chunks this pass just finished.
        self.light = DimensionLight(dimension)
        self.lifecycle = lifecycle or ChunkLifecycle(
            dimension, on_missing_chunks=on_missing_chunks, light=self.light
        )
        if lifecycle is not None:
            # A caller that brought its own lifecycle (World does) still has to
            # be told about the engine: without this hook the chunks it loads
            # are never queued for light and the whole pipeline stays dark.
            lifecycle.light = self.light

        self.streamer = ChunkStreamer(
            dimension, on_missing_chunks=on_missing_chunks, light=self.light
        )
        # The entity tracker lives on this thread next to the streamer and
        # is fed by the same PlayerViewMsg stream (see _handle); the two
        # never disagree about who is attached because they read the same
        # messages in the same order.
        self.entity_tracker = EntityTracker(
            dimension.name, broadcast_percentage=entity_broadcast_percentage
        )
        # The natural spawner owns the mobs it places - it holds the chunk ->
        # entity-id index, decides residency, and drives the tracker through
        # `add_entity` / `remove_entity`. It lives here for the same reason
        # the tracker does: every table it touches belongs to this thread.
        #
        # Its RNG is seeded from the world seed mixed with the dimension's
        # name. Vanilla gives each ServerLevel its own `WorldgenRandom`; two
        # dimensions sharing one stream would draw the same column offsets in
        # the same order, and `crc32` rather than `hash()` because Python
        # randomizes string hashes per process - a spawn pattern that changed
        # on every restart would be a bug that only shows up as "the cows
        # moved".
        spawner_seed = (dimension.seed
                        ^ zlib.crc32(dimension.name.encode())) & 0xFFFFFFFF
        # Physics: one `PhysicsSolver` per dimension, on this thread, next to
        # everything it reads and writes. `WorldView` is a mirror of the chunk
        # manager's *generated* chunks -- a blank chunk whose terrain has not
        # been filled in yet is deliberately absent, because the solver's rule
        # for a chunk it cannot see is "freeze the body", and treating a
        # not-yet-generated chunk as air would drop every mob standing in it
        # through the world for a tick.
        #
        # `ticks_entities` is the second half of that rule and the one the
        # solver used to be missing: residency says whether the blocks can be
        # read, and the load level says whether the chunk runs anything at all.
        # `Chunk.mdx` puts entity ticking at level <= 31, so a mob in a lazy
        # (32) or border (33) chunk is paused -- no gravity, no sweep, no
        # sleep/wake -- while staying in the world, the store and the tracker.
        # The predicate is the dimension's own, the one the spawner already
        # gates its candidate chunks on, so "where mobs may spawn" and "where
        # mobs move" cannot drift apart.
        settings = dimension.settings
        self.physics_view = WorldView(settings.min_y, settings.height)
        self.physics = PhysicsSolver(
            self.physics_view,
            block_shapes(),
            settings.min_y,
            settings.height,
            entities=dimension.entities,
            tracker=self.entity_tracker,
            on_removed=self._on_physics_removed,
            ticks_entities=dimension.ticks_entities,
        )
        self.spawner = NaturalSpawner(
            dimension,
            lifecycle=self.lifecycle,
            tracker=self.entity_tracker,
            seed=spawner_seed,
            interval_ticks=spawn_interval_ticks,
            enabled=natural_spawning,
            physics=self.physics,
        )
        # Compile the kernel now, on the setup path, where a one-off stall is
        # invisible. The alternative is the first tick after boot paying it,
        # which is a dozen ticks of stall the ticker cannot see coming.
        self.physics.warm_up()

        # Player movement authority. The validator reads the *same* view and
        # shape tables the mob solver does, on this thread; the only thing that
        # crosses a thread boundary is the correction packet, and it crosses
        # through `ChunkSink.send` (see `physics/player.py`). It is built even
        # when the switch is off so `stats()` still reports, but nothing calls
        # it, which is what "off" means here: the movement packet is taken at
        # face value, exactly as before physics existed.
        self.authoritative_movement = bool(server_authoritative_movement)
        self.player_validator = PlayerMoveValidator(
            self.physics_view, block_shapes(),
            correction_threshold=movement_correction_threshold,
        )

        # Always-on per-stage tick timing for /perf. Actor-owned state like
        # everything else here; the console reads it through PerfMsg.
        self._prof = TickProfiler()

        # The physics view's storm valve (`advanced.physics-view-adds-per-tick`).
        # 0 = uncapped, which is what a tool that wants the old behavior passes;
        # a server keeps the cap so a batch arrival costs a bounded slice of
        # the tick instead of the whole thing.
        adds = _advanced_int("physics-view-adds-per-tick", 32)
        self._view_add_cap = None if adds <= 0 else adds

    def _run(self) -> None:
        logger.info(f"{self.name} actor started")
        while self._running:
            try:
                msg = self.inbox.get(timeout=0.1)
            except queue.Empty:
                continue

            try:
                self._handle(msg)
            except Exception as e:
                logger.exception(f"Error handling message in {self.name}: {e}")

        # Tell anyone still watching a player that their socket is finished.
        self.streamer.detach_all()
        self.entity_tracker.detach_all()
        # Take the dimension's chunks out of memory, saving what has to be
        # saved. The world's own save runs after the actors stop, so whatever
        # this hands to the streaming saver is flushed there rather than lost.
        dropped = self.lifecycle.unload_all()
        if dropped:
            logger.info(f"{self.name} unloaded {dropped} chunk(s) on shutdown")
        logger.info(f"{self.name} actor stopped")

    def _handle(self, msg: Any) -> None:
        if isinstance(msg, TickMsg):
            self._on_tick(msg)
        elif isinstance(msg, PlayerViewMsg):
            # Sent from the event loop when a player attaches, moves, or
            # leaves (sink=None). Both consumers see the same ordered
            # stream: the streamer reconfigures chunk sending, the tracker
            # refreshes the player's position snapshot and visibility.
            # Cheap on both ends: a queue put here, and a few field writes.
            #
            # The spawner is deliberately NOT a third consumer. It reads
            # `tracker.viewer_positions()` once per round instead, so the
            # tracker is the single authority on who is present and the two
            # cannot disagree; a second copy of the message stream would be
            # a second place for a leave to be missed.
            known = msg.player_id in self.streamer.views
            self.streamer.attach(msg)
            self.entity_tracker.attach(msg)
            self._handle_player_tickets(msg)
            # And a fourth consumer, in the same ordered stream: the validator
            # judges the claim this message carries. It is not a fourth *copy*
            # of the position -- the ghost is the server's opinion of it, and
            # the message is the client's claim about the same tick.
            if self.authoritative_movement:
                self.player_validator.on_view_message(msg)
            if msg.sink is not None and not known:
                # A player who just entered the world gets the clock now rather
                # than at the next broadcast: until it arrives the client runs
                # on its own default, which is a different time of day (and its
                # sky is not something the chunk stream can hide for long).
                self._send_time(msg.sink)
        elif isinstance(msg, ChunkReadyMsg):
            self._on_chunk_ready(msg)
        elif isinstance(msg, ChunkFailedMsg):
            self._on_chunk_failed(msg)
        elif isinstance(msg, TicketMsg):
            # A ticket change from the console or the world API. Applying it
            # here is the whole point: this thread owns the ticket table.
            result = msg.apply(self.lifecycle)
            logger.info(
                f"{self.name} {msg.action} ({msg.x}, {msg.z}) -> {result}"
            )
        elif isinstance(msg, StatsMsg):
            msg.box["stats"] = self.lifecycle.stats()
            msg.box["streamer"] = self.streamer.stats()
            msg.box["spawner"] = self.spawner.stats()
            msg.box["physics"] = self.physics.stats()
            msg.box["player_movement"] = self.player_validator.stats()
            msg.box["light"] = self.light.stats()
            msg.done.set()
        elif isinstance(msg, LightQueryMsg):
            # `/light <x> <y> <z>`. Reading light is a plain array index, so
            # this is cheap enough to answer inline; what it must not do is run
            # on the console's thread, which is why it is a message at all.
            try:
                msg.box.update(read_light(self.dimension, msg.x, msg.y, msg.z))
            except Exception as error:  # pragma: no cover - defensive
                logger.exception(f"{self.name} light query failed")
                msg.box["error"] = str(error)
            msg.done.set()
        elif isinstance(msg, LightRelightMsg):
            # `/light relight <cx> <cz> [<radius>]` and `/light relight all`.
            # Queueing only: the patches run on this thread's next
            # `light.tick()`, inside the budget, so a busy actor answers
            # "queued" instead of timing the console out -- and queueing a whole
            # dimension is a few hundred deque appends, not a few hundred
            # patches.
            try:
                if msg.all_loaded:
                    msg.box.update(relight_all(self.dimension, self.light))
                else:
                    msg.box.update(relight_area(self.dimension, self.light,
                                                msg.chunk_x, msg.chunk_z,
                                                msg.radius))
            except Exception as error:  # pragma: no cover - defensive
                logger.exception(f"{self.name} relight request failed")
                msg.box["error"] = str(error)
            msg.done.set()
        elif isinstance(msg, PerfMsg):
            # `/perf`. The snapshot is a plain dict of numbers built on this
            # thread, so the console gets a copy, never a reference into
            # actor state. `reset` lands before the snapshot: /perf reset
            # answers with the (fresh) empty window it just opened.
            if msg.reset:
                self._prof.reset()
            msg.box.update(self._prof.snapshot())
            msg.box["dimension"] = self.dimension.name
            msg.box["queue_depth"] = self.inbox.qsize()
            msg.done.set()
        else:
            # PacketMsg / HandoffMsg / CommandMsg land here once packet
            # handling migrates onto the actors.
            logger.debug(f"{self.name} received unhandled message: {msg}")

    def _handle_player_tickets(self, msg: PlayerViewMsg) -> None:
        """Players are what makes chunks load and unload automatically.

        The position on the message places two tickets (loading at the view
        distance, simulation at the simulation distance); the lifecycle diffs
        the resulting level map against what is in memory on its next tick.
        `sink=None` means the player left, which drops both tickets and lets
        their chunks unload -- this is the half of "automatic" that a player
        quitting depends on.
        """
        if msg.sink is None:
            self.lifecycle.remove_player(msg.player_id)
            return
        self.lifecycle.set_player_view(
            msg.player_id,
            msg.x,
            msg.y,
            msg.z,
            msg.radius,
            msg.simulation_radius or msg.radius,
        )

    def _on_chunk_ready(self, msg: ChunkReadyMsg) -> None:
        """Insert a chunk that a generator process finished.

        This is the chunk manager's owning thread doing the insert, so the
        claim made by `claim_chunk` and the finished chunk arriving here are
        ordered on one thread. The double-check keeps the same guarantee the
        old in-place generation had: a chunk loaded from disk (or filled by
        the spawn preload) is never overwritten by a worker's copy.

        A finished chunk with *no* entry at all is a late arrival: its claim
        was removed while the worker ran (the player left and the unload pass
        dropped it, or the LRU valve evicted it). Inserting it would strand an
        unowned, undesired copy in the manager that no unload pass will ever
        visit again. The terrain is already on disk -- the unload that removed
        the claim offered the chunk to the save pipeline first, and a second
        worker's copy of the same seed is identical -- so the arrival is
        dropped rather than inserted.
        """
        chunk = msg.chunk
        existing = self.dimension.chunk_manager.get_chunk(chunk.position)
        if existing is None:
            return
        if existing.is_generated:
            return
        self.dimension.chunk_manager.set_chunk(chunk)
        self.streamer.discard_request(chunk.position.x, chunk.position.z)
        self.lifecycle.on_chunk_ready(chunk.position)

    def _on_chunk_failed(self, msg: ChunkFailedMsg) -> None:
        """Drop a blank claim whose generation failed, and forget the request.

        Two halves, both needed: without the claim removal the coord looks
        "being filled" forever; without `discard_request` the streamer would
        keep treating the coord as already requested and never re-ask. The
        lifecycle is told too, so its stall timer starts from a clean slate
        instead of re-requesting a coordinate that just failed.
        """
        from world.chunk import ChunkPos

        pos = ChunkPos(msg.chunk_x, msg.chunk_z)
        chunk = self.dimension.chunk_manager.get_chunk(pos)
        if chunk is not None and not chunk.is_generated and not chunk.is_dirty:
            self.dimension.chunk_manager.remove(pos)
        self.streamer.discard_request(msg.chunk_x, msg.chunk_z)
        self.lifecycle.discard_request(msg.chunk_x, msg.chunk_z)

    def _on_physics_removed(self, entity_ids: list) -> None:
        """A body fell out of the world; take the mob with it.

        The same three tables the spawner's own residency pass clears, in the
        same order: the tracker first, while the entity is still reachable and
        its viewers known, then the dimension. Vanilla would have applied void
        damage first; there is no damage path yet (`PHYSICS_PLAN.md` §七), so a
        mob that falls past `min_y - 64` is simply gone.
        """
        for entity_id in entity_ids:
            entity = self.dimension.get_entity(entity_id)
            if entity is None:
                continue
            self.spawner.forget_entity(entity)

    # ---------------- the world clock (actor thread) ----------------

    def _send_time(self, sink) -> None:
        """Hand one client this dimension's clock.

        Actor thread only, like every other read of `Dimension.day_time`: the
        clock advances inside `Dimension.tick` on this thread, so the value here
        is the one the rest of the tick sees.
        """
        sink.send([CUpdateTime(
            world_age=self.dimension.world_time,
            time_of_day=self.dimension.day_time,
            increasing=self._day_cycle_runs,
        )])

    def _broadcast_time(self) -> None:
        """Tell everyone watching this dimension what time it is.

        One packet per view, built here: the sink's contract is a list of
        packets per player, and a shared instance would be one object written by
        several connections.
        """
        for view in self.streamer.views.values():
            if view.sink.closed:
                continue
            self._send_time(view.sink)

    def _on_tick(self, msg: TickMsg) -> None:
        # Ticks discarded while we were busy skipped some tick numbers.
        # Fast-forward the clock by the gap so world time tracks the wall
        # clock instead of the number of ticks we actually ran.
        gap = msg.tick_no - self._last_tick_no
        if gap > 1:
            self.dimension.advance_clock(gap - 1)
        self._last_tick_no = msg.tick_no

        # One `perf_counter()` between stages (~1 µs/tick at 20 TPS) buys the
        # /perf breakdown; the marks land inside the measured window, so
        # TickDone.duration still covers the whole tick, profiler included.
        prof = self._prof
        prof.start_tick()
        self.dimension.tick()
        prof.mark("dimension")
        # The clock just moved (`Dimension.tick` -> `advance_clock`), so this is
        # the moment its new value is the tick's own. One packet per second per
        # player: the client advances its own clock in between, and vanilla's
        # `ServerLevel.tickTime` corrects it on the same 20-tick beat.
        if msg.tick_no % TIME_SYNC_TICKS == 0:
            self._broadcast_time()
        prof.mark("time_sync")
        self.lifecycle.tick(msg.tick_no)
        prof.mark("lifecycle")
        # Physics runs *after* the lifecycle, not before it as the plan first
        # sketched. The reason is one line long: physics reads the world, and
        # the lifecycle is the pass that decides which parts of the world exist.
        # Reconciling the view before it would expose a chunk the lifecycle is
        # about to unload in this same tick, and hide one it just loaded.
        self.physics_view.sync(self.dimension.chunk_manager.chunks.values(),
                               ready=_terrain_ready,
                               max_adds=self._view_add_cap)
        prof.mark("view_sync")
        # Bodies move before the spawner: a mob that spawned last tick falls
        # this tick, and one this tick's spawner places gets its body with the
        # spawn's own `on_ground`, so it does not fall a tick before it exists.
        #
        # Before the tracker's pump, always: a mob placed in this tick has to
        # be visible to the visibility pass that runs in the same tick, or a
        # player standing next to it waits a tick to see it.
        self.physics.tick()
        prof.mark("physics")
        self.player_validator.tick()
        prof.mark("player_validator")
        self.spawner.tick(msg.tick_no)
        prof.mark("spawner")
        # Light runs after the spawner and before the tracker's pump, which is
        # the one place it can be and satisfy both of its readers:
        #
        # * after the spawner, because the spawner's brightness test should see
        #   this tick's light -- otherwise a torch placed this tick is invisible
        #   to the spawn check until the next one;
        # * before the tracker and the streamer, because both put this tick's
        #   world state on the wire, and a `CLightUpdate` that arrived before
        #   the chunk it belongs to would be pointless.
        #
        # The cost lands inside the measured window on purpose: light is
        # budgeted (`light-budget-ms`), and if it ever eats the tick,
        # `TickDone.duration` should say so rather than hide it.
        self.light.tick()
        prof.mark("light")
        # Entities move before the wire flush: the tracker's pump turns
        # this tick's snapshots into spawn/despawn/delta packets. Running
        # it before the streamer keeps the per-tick order "world state,
        # then entity wire, then chunk wire" — a viewer who just came into
        # range gets their spawn in this tick's tracker pass, not the next.
        # Like the streamer below, the encode cost lands inside the
        # measured window on purpose.
        self.entity_tracker.pump()
        prof.mark("entity_tracker")
        # Stream *after* the tick so a player sees the state this tick
        # produced. The encode cost lands inside the measured window on
        # purpose: if streaming eats the tick budget, TickDone.duration
        # should say so rather than hide it.
        self.streamer.pump()
        duration = prof.end_tick(discarded=max(gap - 1, 0))

        self._done_inbox.put(TickDone(
            actor=self.name,
            tick_no=msg.tick_no,
            duration=duration,
            queue_depth=self.inbox.qsize(),
        ))
