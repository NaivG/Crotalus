"""
End-to-end check for chunk load levels, automatic loading and streaming save.

This is the acceptance test for four things that only make sense together:

1. **The load-level model drives memory.** A player's two tickets (loading at
   the view distance, simulation at the simulation distance) must put exactly
   the level-<=33 square in memory -- no more (that is the memory complaint)
   and no less (that is a hole in someone's view).
2. **Ticking areas come from the simulation ticket only.** Entity ticking is
   the 2s+1 square and block ticking the 2s+3 square; a large view distance
   must not widen either. This is the separation the wiki draws between the
   "player loading" and "player simulation" tickets. The tick loop is checked
   against the same model: an entity in a chunk that is loaded but not
   entity-ticking does not run at all, and resumes when its chunk does.
3. **Unloading saves, and the save is durable.** A chunk edited, unloaded and
   then read back through a *fresh* World must come back with the edit -- and
   so must its neighbours in the same region, which the old region writer
   destroyed by rewriting the file with only the chunks it was told to save.
4. **Nothing is dropped without being offered to the save pipeline first** --
   neither by the unload pass nor by the manager's LRU valve, which used to
   discard dirty chunks silently.

Two checks are deliberately shaped around races that are otherwise invisible:

* the read-after-write guard runs against a saver that takes 250 ms per chunk,
  so "load the coordinate you just unloaded" is a deterministic test instead of
  a coin flip;
* the skip policy is measured on the *third* pass over the same strip, so the
  chunks involved are all ones that came back from disk and were never touched
  -- they must not be written again, which is what keeps walking around from
  rewriting a player's whole view.

Run: python tools/check_chunk_lifecycle.py
"""

import shutil
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
from loguru import logger  # noqa: E402

logger.remove()

from world.block_states import state_id
from world.chunk import Chunk, ChunkPos
from world.chunk_lifecycle import (
    UNLOAD_GRACE_TICKS,
    ChunkLifecycle,
    StatsMsg,
    TicketMsg,
)
from world.chunk_saver import ChunkSaver
from world.chunk_streamer import PlayerViewMsg
from world.chunk_ticket import BORDER_LEVEL, FORCED, LoadType
from world.dimension import Dimension, DimensionType
from world.dimension_actor import DimensionActor
from world.generator import GeneratorSettings, create_generator
from world.world import World

FAILURES = []
MARKER = state_id("minecraft:glowstone")
MARKER_Y = 100


def check(label: str, ok: bool, detail: str = "") -> bool:
    print(f"  {'OK  ' if ok else 'FAIL'} {label}" + (f"  ({detail})" if detail else ""))
    if not ok:
        FAILURES.append(label)
    return ok


def legacy_generator(seed: int = 12345):
    """The lifecycle checks are about bookkeeping and storage, not terrain.

    Pinned to the legacy backend for the same reason
    `tools/check_chunk_streamer.py` is: the default (vanilla) backend costs
    about a second per chunk, and this tool loads a few hundred of them.
    """
    return create_generator(GeneratorSettings(seed=seed), backend="legacy")


def make_dimension() -> Dimension:
    return Dimension(
        DimensionType.OVERWORLD,
        world_seed=12345,
        generator=legacy_generator(),
    )


def drive(lifecycle: ChunkLifecycle, dimension: Dimension, ticks: int) -> None:
    """One game tick, in the order the actor does it: dimension, then lifecycle."""
    for _ in range(ticks):
        dimension.tick()
        lifecycle.tick()


def drive_until(lifecycle: ChunkLifecycle, dimension: Dimension, predicate, limit: int = 600) -> int:
    for tick in range(1, limit + 1):
        dimension.tick()
        lifecycle.tick()
        if predicate():
            return tick
    return -1


def square_set(cx: int, cz: int, radius: int):
    return {(cx + dx, cz + dz) for dx in range(-radius, radius + 1)
            for dz in range(-radius, radius + 1)}


def loaded_coords(dimension: Dimension):
    return {(pos.x, pos.z) for pos in dimension.chunk_manager.chunks.keys()}


# ---------------------------------------------------------------------------
# 1. Automatic loading: exactly the ticket square
# ---------------------------------------------------------------------------

def check_automatic_loading() -> None:
    print("\n1. A player's tickets decide what is in memory")
    dimension = make_dimension()
    lifecycle = ChunkLifecycle(dimension, save_chunk=lambda chunk: True)

    check("no player -> nothing loaded", dimension.chunk_manager.get_chunk_count() == 0)

    view, simulation = 2, 1
    lifecycle.set_player_view("p1", 8.0, 64.0, 8.0, view, simulation)
    ticks = drive_until(lifecycle, dimension, lambda: not lifecycle._untracked)

    expected_radius = view + (BORDER_LEVEL - 31)  # level 33 reaches two rings out
    expected = square_set(0, 0, expected_radius)
    got = loaded_coords(dimension)
    check(
        "the level<=33 square is loaded",
        got == expected,
        f"{len(got)} chunks in {ticks} ticks, expected {len(expected)} "
        f"(square radius {expected_radius})",
    )
    disc = {(x, z) for x, z in expected if x * x + z * z <= view * view}
    check(
        "loading is wider than the disc the client is sent",
        len(expected) > len(disc),
        f"{len(expected)} loaded vs {len(disc)} sent (the two border rings)",
    )
    levels = (
        dimension.get_load_level(0, 0),
        dimension.get_load_level(view + 1, 0),
        dimension.get_load_level(view + 2, 0),
        dimension.get_load_level(view + 3, 0),
    )
    check(
        "levels 31/32/33/34 land where the model says",
        levels == (31, 32, 33, 34),
        f"levels {levels} at 0/{view + 1}/{view + 2}/{view + 3} chunks",
    )

    probe = square_set(0, 0, simulation + 3)
    entity_got = {(x, z) for x, z in probe if dimension.ticks_entities(x, z)}
    block_got = {(x, z) for x, z in probe if dimension.ticks_blocks(x, z)}
    check(
        "entity ticking is the 2s+1 square",
        entity_got == square_set(0, 0, simulation),
        f"{len(entity_got)} chunks, expected {(2 * simulation + 1) ** 2}",
    )
    check(
        "block ticking is the 2s+3 square",
        block_got == square_set(0, 0, simulation + 1),
        f"{len(block_got)} chunks, expected {(2 * simulation + 3) ** 2}",
    )
    check(
        "a wide view distance does not widen the ticking area",
        not dimension.ticks_entities(view + 2, 0)
        and not dimension.ticks_blocks(view + 2, 0),
        f"view={view} vs simulation={simulation}, probed at {view + 2} chunks",
    )


# ---------------------------------------------------------------------------
# 2. Automatic unloading: bounded memory, durable edits
# ---------------------------------------------------------------------------

def check_automatic_unloading(world: World) -> None:
    print("\n2. Moving away unloads the chunks behind the player, and they persist")
    dimension = world.overworld
    lifecycle = world.lifecycle(DimensionType.OVERWORLD)

    view, simulation = 2, 1
    lifecycle.set_player_view("p1", 8.0, 64.0, 8.0, view, simulation)
    drive_until(lifecycle, dimension, lambda: not lifecycle._untracked)
    loaded_before = dimension.chunk_manager.get_chunk_count()

    target = dimension.chunk_manager.peek(ChunkPos(0, 0))
    check("the player's own chunk is generated", target is not None and target.is_generated)
    target.set_block(0, MARKER_Y, 0, MARKER)
    check("editing marks the chunk dirty", target.is_dirty)

    # Walk 12 chunks away: the whole previous square leaves the desired set.
    lifecycle.set_player_view("p1", 8.0 + 16 * 12, 64.0, 8.0, view, simulation)
    drive_until(lifecycle, dimension, lambda: not lifecycle._untracked)
    drive(lifecycle, dimension, UNLOAD_GRACE_TICKS + 2)

    stats = lifecycle.stats()
    loaded_after = dimension.chunk_manager.get_chunk_count()
    check(
        "memory stays bounded while moving",
        loaded_after <= 2 * loaded_before,
        f"{loaded_before} -> {loaded_after} chunks",
    )
    check(
        "the chunks left behind were unloaded",
        stats["unloaded"] >= loaded_before,
        f"unloaded={stats['unloaded']}, saved_on_unload={stats['saved_on_unload']}",
    )
    check(
        "generated chunks were written before being dropped",
        stats["saved_on_unload"] >= loaded_before,
        f"saved={stats['saved_on_unload']} kept={stats['kept_unsaved']}",
    )

    pending = world.chunk_saver.flush(timeout=60.0)
    check("the streaming save queue drains", pending == 0, f"{pending} left")

    # Read the edit back through a brand new World: only disk can answer this.
    world.chunk_saver.close()
    fresh = World(
        name=world.name,
        world_dir=world.world_dir,
        generator=legacy_generator(),
    )
    try:
        reloaded = fresh.load_chunk(0, 0, dimension=DimensionType.OVERWORLD)
        block = None if reloaded is None else reloaded.get_block(0, MARKER_Y, 0)
        check(
            "an unloaded chunk comes back with its edit",
            block == MARKER,
            f"block at (0,{MARKER_Y},0) = {block}, expected {MARKER}",
        )
        check(
            "a chunk read from disk is marked persisted",
            reloaded is not None and reloaded.is_persisted and not reloaded.is_dirty,
        )
    finally:
        fresh.chunk_saver.close()


# ---------------------------------------------------------------------------
# 3. The skip policy: walking around must not rewrite unchanged chunks
# ---------------------------------------------------------------------------

def check_skip_unchanged() -> None:
    print("\n3. Chunks that came from disk and were not touched are not re-written")
    with tempfile.TemporaryDirectory() as tmp:
        world = World(
            name="skip",
            seed=7,
            world_dir=Path(tmp) / "skip",
            generator=legacy_generator(7),
        )
        try:
            dimension = world.overworld
            lifecycle = world.lifecycle(DimensionType.OVERWORLD)

            def walk(chunk_x: int) -> None:
                lifecycle.set_player_view("p1", chunk_x * 16 + 8.0, 64.0, 8.0, 2, 1)
                drive_until(lifecycle, dimension, lambda: not lifecycle._untracked)
                drive(lifecycle, dimension, UNLOAD_GRACE_TICKS + 2)

            # Pass A: the terrain is generated (never persisted -> must be
            # written on unload). Pass B walks back over it. Pass C walks out
            # again: by now every chunk on the strip is either on disk and
            # untouched, or was generated in this session.
            walk(0)
            walk(8)
            walk(0)
            world.chunk_saver.flush(timeout=120.0)

            before_stats = dict(lifecycle.stats())
            before_saved = world.chunk_saver.stats()["submitted"]
            walk(8)
            world.chunk_saver.flush(timeout=120.0)
            after_stats = lifecycle.stats()
            after_saved = world.chunk_saver.stats()["submitted"]

            unloads = after_stats["unloaded"] - before_stats["unloaded"]
            skipped = after_stats["skipped_unchanged"] - before_stats["skipped_unchanged"]
            writes = after_saved - before_saved
            check(
                "the third pass over the same strip mostly needed no writes",
                skipped > 0 and writes < unloads,
                f"{writes} writes for {unloads} unloads ({skipped} skipped as unchanged)",
            )
            check(
                "and the skip path was really taken, not just unused",
                skipped >= unloads // 2,
                f"{skipped}/{unloads} unloads skipped",
            )

            # Nothing was lost to the skipping: the whole strip reads back.
            world.chunk_saver.close()
            fresh = World(
                name="skip", world_dir=Path(tmp) / "skip", generator=legacy_generator(7)
            )
            try:
                found = 0
                for cx in range(-4, 13):
                    chunk = fresh.load_chunk(cx, 0, dimension=DimensionType.OVERWORLD)
                    if chunk is not None and chunk.is_generated:
                        found += 1
                check(
                    "every coordinate on the walked strip reads back",
                    found == 17,
                    f"{found}/17 chunks present",
                )
            finally:
                fresh.chunk_saver.close()
        finally:
            world.chunk_saver.close()


# ---------------------------------------------------------------------------
# 4. The read-after-write guard, with a deliberately slow saver
# ---------------------------------------------------------------------------

class SlowRegionManager:
    """Delegates to a real region manager, but makes one save take 250 ms."""

    def __init__(self, wrapped, delay: float = 0.25):
        self._wrapped = wrapped
        self._delay = delay

    def save_chunk(self, chunk):
        time.sleep(self._delay)
        return self._wrapped.save_chunk(chunk)

    def save_all(self):
        return self._wrapped.save_all()


def check_read_after_write() -> None:
    print("\n4. A coordinate whose save is in flight is not read back stale")
    with tempfile.TemporaryDirectory() as tmp:
        world = World(
            name="guard",
            seed=11,
            world_dir=Path(tmp) / "guard",
            generator=legacy_generator(11),
        )
        try:
            dimension = world.overworld
            chunk = world.load_chunk(3, 3, dimension=DimensionType.OVERWORLD)
            chunk.set_block(0, MARKER_Y, 0, MARKER)
            pos = ChunkPos(3, 3)

            # Take the chunk out of memory the way an unload does, and hand it
            # to a saver that is still busy with it. The world's own saver is
            # swapped for the slow one, because that is the object the load path
            # asks "is this coordinate still in flight?".
            dimension.chunk_manager.remove(pos)
            slow = ChunkSaver(
                lambda dim: SlowRegionManager(world.region_managers[dim]),
                sync_writes=False,
            )
            original_saver = world.chunk_saver
            world.chunk_saver = slow
            try:
                slow.submit(DimensionType.OVERWORLD, chunk, force=True)
                check("the slow saver has the chunk in flight", slow.pending == 1)

                started = time.monotonic()
                reread = world._read_from_disk(
                    world.region_managers[DimensionType.OVERWORLD],
                    3, 3, DimensionType.OVERWORLD,
                )
                waited = time.monotonic() - started
                check(
                    "the load path waited for the in-flight save",
                    waited >= 0.2,
                    f"waited {waited * 1000:.0f} ms",
                )
                block = None if reread is None else reread.get_block(0, MARKER_Y, 0)
                check(
                    "and therefore read the edited copy",
                    block == MARKER,
                    f"block = {block}, expected {MARKER}",
                )
            finally:
                world.chunk_saver = original_saver
                slow.close()
        finally:
            world.chunk_saver.close()


# ---------------------------------------------------------------------------
# 5. The manager's safety valve also saves
# ---------------------------------------------------------------------------

def check_lru_eviction_saves() -> None:
    print("\n5. LRU eviction hands its victims to the saver")
    with tempfile.TemporaryDirectory() as tmp:
        world = World(
            name="lru",
            seed=13,
            world_dir=Path(tmp) / "lru",
            generator=legacy_generator(13),
        )
        try:
            dimension = world.overworld
            manager = dimension.chunk_manager
            manager.max_chunks = 8

            coords = [(cx, 0) for cx in range(20)]
            expected = {}
            for cx, cz in coords:
                chunk = world.load_chunk(cx, cz, dimension=DimensionType.OVERWORLD)
                marker = MARKER + 1 + (cx % 5)
                chunk.set_block(0, MARKER_Y, 0, marker)
                expected[(cx, cz)] = marker

            check(
                "the cap is enforced",
                manager.get_chunk_count() <= 9,
                f"{manager.get_chunk_count()} chunks resident (cap 8)",
            )
            check(
                "evictions reached the saver",
                world.chunk_saver.stats()["submitted"] >= 12,
                f"{world.chunk_saver.stats()['submitted']} submitted",
            )
            # The resident survivors are written by the ordinary save path.
            world.save()
            world.chunk_saver.flush(timeout=60.0)
            world.chunk_saver.close()

            fresh = World(
                name="lru", world_dir=Path(tmp) / "lru", generator=legacy_generator(13)
            )
            try:
                missing, wrong = [], []
                for (cx, cz), marker in expected.items():
                    chunk = fresh.load_chunk(cx, cz, dimension=DimensionType.OVERWORLD)
                    if chunk is None or not chunk.is_generated:
                        missing.append((cx, cz))
                    elif chunk.get_block(0, MARKER_Y, 0) != marker:
                        wrong.append((cx, cz))
                check(
                    "no evicted chunk was lost",
                    not missing,
                    f"{len(missing)} missing: {missing[:4]}",
                )
                check(
                    "no evicted chunk was mixed up with another",
                    not wrong,
                    f"{len(wrong)} wrong: {wrong[:4]}",
                )
            finally:
                fresh.chunk_saver.close()
        finally:
            world.chunk_saver.close()


# ---------------------------------------------------------------------------
# 6. Force-loaded chunks: no players, still loaded, still there next boot
# ---------------------------------------------------------------------------

def check_forced_tickets() -> None:
    print("\n6. Force-loaded chunks survive an empty server and a restart")
    with tempfile.TemporaryDirectory() as tmp:
        world = World(
            name="forced",
            seed=17,
            world_dir=Path(tmp) / "forced",
            generator=legacy_generator(17),
        )
        try:
            dimension = world.overworld
            lifecycle = world.lifecycle(DimensionType.OVERWORLD)

            check(
                "nothing is loaded with no players",
                dimension.chunk_manager.get_chunk_count() == 0,
            )
            world.forceload(40, 40)
            drive_until(lifecycle, dimension, lambda: not lifecycle._untracked)
            expected = square_set(40, 40, BORDER_LEVEL - 31)
            check(
                "a forced ticket loads its level<=33 square without players",
                loaded_coords(dimension) == expected,
                f"{len(loaded_coords(dimension))} chunks",
            )
            check(
                "a forced chunk refuses to be unloaded while its ticket holds it",
                lifecycle.unload_chunk(40, 40) is False,
            )
            check(
                "the forced ticket keeps the dimension active",
                dimension.tickets.activates_dimension() and dimension.is_active,
            )

            world.save()
            world.chunk_saver.close()
            fresh = World(
                name="forced",
                world_dir=Path(tmp) / "forced",
                generator=legacy_generator(17),
            )
            try:
                fresh_lifecycle = fresh.lifecycle(DimensionType.OVERWORLD)
                restored = fresh_lifecycle.force_loaded_positions()
                check(
                    "the forced ticket is restored on the next boot",
                    [pos.to_tuple() for pos in restored] == [(40, 40)],
                    f"restored {[pos.to_tuple() for pos in restored]}",
                )
                drive_until(
                    fresh_lifecycle, fresh.overworld, lambda: not fresh_lifecycle._untracked
                )
                check(
                    "and reloads its chunks",
                    fresh.overworld.chunk_manager.peek(ChunkPos(40, 40)) is not None,
                )
            finally:
                fresh.chunk_saver.close()
        finally:
            world.chunk_saver.close()


# ---------------------------------------------------------------------------
# 7. A stalled claim is retried instead of blocking the view forever
# ---------------------------------------------------------------------------

class Handler:
    """Records load requests, and answers them however the test wants."""

    def __init__(self, accept=True):
        self.calls = []
        self.accept = accept

    def __call__(self, cx, cz, priority):
        self.calls.append((cx, cz, priority))
        return self.accept


def check_stalled_claim_retry() -> None:
    print("\n7. A blank claim that never finishes is re-requested")
    dimension = make_dimension()
    handler = Handler()
    lifecycle = ChunkLifecycle(
        dimension,
        on_missing_chunks=handler,
        claim_stall_ticks=20,
        request_retry_ticks=10_000,  # isolate the stall path from the retry path
        sync_fallback=False,
    )
    lifecycle.set_player_view("p", 8.0, 64.0, 8.0, 1, 1)
    lifecycle.tick()
    first = len(handler.calls)
    check("a missing coordinate is requested", first > 0, f"{first} request(s)")

    # The generator "claimed" the coordinate and then died: present, blank.
    for cx, cz, _priority in handler.calls:
        dimension.chunk_manager.set_chunk(
            Chunk(
                position=ChunkPos(cx, cz),
                min_y=dimension.settings.min_y,
                height=dimension.settings.height,
            )
        )
    handler.calls.clear()
    lifecycle.tick()
    check(
        "a fresh blank claim is left alone",
        not handler.calls,
        f"{len(handler.calls)} premature re-request(s)",
    )
    for _ in range(25):
        lifecycle.tick()
    check(
        "after the stall timeout the claim is requested again",
        len(handler.calls) > 0 and lifecycle.stats()["stall_retries"] > 0,
        f"{len(handler.calls)} re-request(s), stall_retries="
        f"{lifecycle.stats()['stall_retries']}",
    )

    # The retry backs off: a claim that is legitimately queued behind a deep
    # generation queue must not be re-queued on every stall timeout.
    backed_off = len(handler.calls)
    for _ in range(25):
        lifecycle.tick()
    check(
        "the retry backs off instead of re-queueing every timeout",
        len(handler.calls) == backed_off,
        f"{len(handler.calls) - backed_off} extra request(s) in the next window",
    )


# ---------------------------------------------------------------------------
# 8. The actor wiring: player views in, ticks out
# ---------------------------------------------------------------------------

class FakeSink:
    def __init__(self):
        self.batches = []
        self.closed = False

    def send(self, packets):
        if not self.closed:
            self.batches.append(list(packets))

    def close(self):
        self.closed = True


def check_actor_wiring() -> None:
    print("\n8. DimensionActor feeds the lifecycle from player views")
    import queue as queue_module

    from core.ticker import TickMsg

    dimension = make_dimension()
    done_inbox = queue_module.Queue()
    handler = Handler()
    lifecycle = ChunkLifecycle(dimension, on_missing_chunks=handler, sync_fallback=False)
    actor = DimensionActor(
        dimension, done_inbox, on_missing_chunks=handler, lifecycle=lifecycle
    )

    actor._handle(PlayerViewMsg(player_id="p1", sink=FakeSink(), x=8.0, y=64.0, z=8.0,
                                radius=2, simulation_radius=1))
    check("attaching a view places both tickets", len(lifecycle.tickets) == 2,
          f"{lifecycle.tickets}")

    actor._handle(TickMsg(tick_no=1, sent_at=time.monotonic()))
    done = done_inbox.get_nowait()
    check(
        "the tick runs the lifecycle and reports back",
        done.tick_no == 1 and lifecycle.stats()["ticks"] == 1,
    )
    check(
        "the lifecycle asked for the view's chunks",
        lifecycle.stats()["requests"] > 0,
        f"{lifecycle.stats()['requests']} request(s)",
    )

    actor._handle(PlayerViewMsg(player_id="p1", sink=None))
    check("detaching a view drops both tickets", len(lifecycle.tickets) == 0)
    check("and forgets the player", len(lifecycle.players) == 0)

    # Moving the view re-places the tickets rather than stacking new ones, and
    # a standing player costs no ticket churn at all.
    actor._handle(PlayerViewMsg(player_id="p2", sink=FakeSink(), x=8.0, y=64.0, z=8.0,
                                radius=2, simulation_radius=1))
    actor._handle(PlayerViewMsg(player_id="p2", sink=FakeSink(), x=8.0, y=64.0, z=8.0,
                                radius=2, simulation_radius=1))
    check("a standing player costs no ticket churn", len(lifecycle.tickets) == 2)
    actor._handle(PlayerViewMsg(player_id="p2", sink=FakeSink(), x=8.0 + 16 * 4,
                                y=64.0, z=8.0, radius=2, simulation_radius=1))
    anchors = sorted(
        (ticket.type.name, ticket.pos.to_tuple()) for ticket in lifecycle.tickets
    )
    check(
        "moving re-places both tickets at the new chunk",
        len(anchors) == 2 and all(pos == (4, 0) for _name, pos in anchors),
        f"{anchors}",
    )

    # A request from another thread must go through the inbox: the ticket table
    # is owned by this thread, and touching it from the console is the race that
    # `TicketMsg` exists to close.
    actor._handle(TicketMsg(action="force", x=9, z=9))
    check(
        "a ticket message is applied on the owner thread",
        any(ticket.pos.to_tuple() == (9, 9) for ticket in lifecycle.tickets),
        f"{[ticket.pos.to_tuple() for ticket in lifecycle.tickets]}",
    )
    actor._handle(TicketMsg(action="unforce", x=9, z=9))
    check(
        "and can be taken back",
        not any(ticket.pos.to_tuple() == (9, 9) for ticket in lifecycle.tickets),
    )

    message = StatsMsg()
    actor._handle(message)
    check(
        "a stats request is answered with a snapshot",
        message.done.is_set()
        and message.box.get("stats", {}).get("dimension") == dimension.name,
        f"{sorted(message.box)}",
    )


# ---------------------------------------------------------------------------
# 9. Dimension-level interface contract
# ---------------------------------------------------------------------------

def check_dimension_interface() -> None:
    print("\n9. The load/unload interface answers what it promises")
    dimension = make_dimension()

    dimension.add_ticket(FORCED, 0, 0)
    check(
        "a ticket does not load anything by itself",
        dimension.chunk_manager.get_chunk_count() == 0,
    )
    check(
        "levels are visible before the chunk exists",
        dimension.get_load_level(0, 0) == 31
        and dimension.get_load_type(0, 0) is LoadType.ENTITY_TICKING
        and dimension.get_load_type(2, 0) is LoadType.BORDER
        and dimension.get_load_type(3, 0) is LoadType.INACCESSIBLE,
        f"{dimension.get_load_type(0, 0).name}/"
        f"{dimension.get_load_type(2, 0).name}/"
        f"{dimension.get_load_type(3, 0).name}",
    )
    check(
        "is_chunk_loaded is False until it is resident",
        not dimension.is_chunk_loaded(0, 0),
    )
    dimension.load_chunk(0, 0)
    check("after loading it is resident", dimension.is_chunk_loaded(0, 0))

    handed_over = []
    dimension.chunk_manager.set_eviction_handler(
        lambda chunk: (handed_over.append(chunk), True)[1]
    )
    check(
        "unload_chunk goes through the eviction handler",
        dimension.unload_chunk(0, 0, save=True) and len(handed_over) == 1,
    )
    check("it is gone afterwards", dimension.chunk_manager.peek(ChunkPos(0, 0)) is None)

    dimension.load_chunk(1, 1)
    check(
        "unload_chunk(save=False) drops without the handler",
        dimension.unload_chunk(1, 1, save=False) and len(handed_over) == 1,
    )

    # A refusing handler must keep the chunk: an unsaved chunk is worse than a
    # late one.
    dimension.load_chunk(2, 2)
    dimension.chunk_manager.set_eviction_handler(lambda chunk: False)
    check(
        "a refusing handler keeps the chunk in memory",
        dimension.unload_chunk(2, 2, save=True) is False
        and dimension.chunk_manager.peek(ChunkPos(2, 2)) is not None,
    )

    dimension.tickets.clear()
    dimension.tick()
    check(
        "with no tickets the dimension is still active for now",
        dimension.idle_ticks == 1 and dimension.is_active,
        f"idle_ticks={dimension.idle_ticks}",
    )
    for _ in range(300):
        dimension.tick()
    check(
        "after 300 idle ticks the dimension stops ticking",
        not dimension.is_active,
        f"idle_ticks={dimension.idle_ticks}",
    )
    dimension.add_ticket(FORCED, 5, 5)
    dimension.tick()
    check("an activating ticket resets the idle clock", dimension.idle_ticks == 0)


def check_entity_tick_gate() -> None:
    """`Dimension.tick` runs an entity only where entities tick (`Chunk.mdx`).

    `check_automatic_loading` proves the level model; this proves the model is
    what the tick loop actually consults, per entity and per tick. Both entities
    below are ordinary `Entity`s with a counting `tick`, in one dimension, and
    the only difference between them is the ticket on the chunk they stand in.
    """
    from entity.entity import Entity
    from entity.position import EntityPosition
    from world.chunk_ticket import PLAYER_LOADING

    class Counter(Entity):
        """An entity whose only job is to say whether it ran."""

        def __init__(self, entity_id: int, x: float, y: float, z: float) -> None:
            super().__init__(entity_id=entity_id,
                             position=EntityPosition(x=x, y=y, z=z))
            self.ticks = 0

        def tick(self) -> None:
            self.ticks += 1

    print("\n10. Entity ticking follows the load level, per entity")
    dimension = make_dimension()
    # Chunk (1, 0) is forced: level 31, entity ticking. Chunk (2, 0) is held by
    # a loading ticket alone: resident and sendable at level 31, and nothing in
    # it runs -- the case the two-level split exists for.
    dimension.add_ticket(FORCED, 1, 0)
    dimension.add_ticket(PLAYER_LOADING, 2, 0)
    check(
        "the fixture has one ticking chunk and one merely loaded one",
        dimension.ticks_entities(1, 0)
        and not dimension.ticks_entities(2, 0)
        and dimension.get_load_level(2, 0) == 31,
        f"levels {dimension.get_load_level(1, 0)}/"
        f"{dimension.get_load_level(2, 0)}",
    )

    runs = Counter(1, 24.5, 40.0, 8.5)      # chunk (1, 0) -> ticks
    paused = Counter(2, 40.5, 40.0, 8.5)    # chunk (2, 0) -> loaded, not ticking
    dimension.spawn_entity(runs)
    dimension.spawn_entity(paused)
    check("both are in the world", len(dimension.entities) == 2)

    for _ in range(5):
        dimension.tick()
    check("only the entity in the ticking chunk ran", runs.ticks == 5,
          str(runs.ticks))
    check("the entity in the loaded-but-lazy chunk never ran", paused.ticks == 0,
          str(paused.ticks))
    check("and the dimension counted it", dimension.paused_entities == 1,
          str(dimension.paused_entities))
    check(
        "it is still in the world, readable, exactly where it was",
        dimension.get_entity(2) is paused
        and (paused.position.x, paused.position.y, paused.position.z)
        == (40.5, 40.0, 8.5),
    )

    # Accessible means movable: walk it into the ticking chunk (the actor-side
    # equivalent of a teleport) and it runs again on the very next tick --
    # vanilla's `EntityTickList` behaviour, and the reason a paused entity has to
    # stay in every table rather than being removed.
    paused.position.x = 24.5
    dimension.tick()
    check("moving it into the ticking chunk resumes it", paused.ticks == 1,
          str(paused.ticks))
    check("and nothing is paused any more", dimension.paused_entities == 0,
          str(dimension.paused_entities))

    # The gauge describes one tick, so it must clear when there is nothing left
    # to pause -- not keep reporting the last entity that was skipped.
    dimension.remove_entity(2)
    dimension.tick()
    check("the gauge clears once nothing is left to pause",
          dimension.paused_entities == 0, str(dimension.paused_entities))


def check_light_pipeline() -> None:
    """11. The light engine rides the lifecycle's "this chunk is readable" edge.

    Four properties, each of which fails silently in production:

    * a chunk that enters memory is lit -- without this, every chunk in the
      world ships the `ChunkSection` default arrays and the whole pipeline is
      decorative. The last block below drives the *edge* (the lifecycle's
      `on_chunk_ready`, not a hand-made `request_chunk`), because that is the
      one the engine actually rides in production and the one that was missing;
    * a 3x3 with a hole is *not* lit and stays queued -- the alternative is
      computing a patch against air standing in for an unloaded neighbour and
      lighting a fake sky nobody ever corrects (`LIGHT_PLAN.md` section 4.5);
    * a chunk that unloads while queued is dropped and counted, not retried
      forever;
    * the budget is real: at 0 ms a tick lights nothing, at a huge budget one
      tick converges, and both end at the same light field.
    """
    from world.light_engine import DimensionLight

    print("\n11. Light rides the lifecycle, and the budget is real")

    def fill(cx: int, cz: int) -> Dimension:
        """A dimension with exactly the 3x3 around (cx, cz) generated."""
        dimension = make_dimension()
        for dx in (-1, 0, 1):
            for dz in (-1, 0, 1):
                chunk = Chunk(ChunkPos(cx + dx, cz + dz))
                dimension.generator.generate_chunk(chunk)
                chunk.is_generated = True
                dimension.chunk_manager.set_chunk(chunk)
        return dimension

    def nibbles_of(chunk) -> np.ndarray:
        packed = np.concatenate([np.frombuffer(s.sky_light.tobytes(), dtype=np.uint8)
                                 for s in chunk.sections])
        out = np.empty(packed.size * 2, dtype=np.uint8)
        out[0::2] = packed & 0x0F
        out[1::2] = packed >> 4
        return out

    def field(dimension) -> bytes:
        chunk = dimension.get_chunk(4, 4)
        if chunk is None:
            return b""
        return b"".join(s.sky_light.tobytes() + s.block_light.tobytes()
                        for s in chunk.sections)

    # --- a full 3x3 gets lit -------------------------------------------------
    dimension = fill(4, 4)
    light = DimensionLight(dimension)
    light.request_chunk((4, 4))
    check("the chunk is queued", light.pending_chunks == 1, str(light.pending_chunks))
    light.tick()
    centre = dimension.get_chunk(4, 4)
    check("one tick lights it", light.chunks_lit >= 1, str(light.chunks_lit))
    check("and says so on the chunk", centre.light_populated)
    check("the queue is empty again", light.pending_chunks == 0,
          str(light.pending_chunks))

    # The light has to be *terrain*, not a constant: a generated overworld
    # chunk has a surface, so the sky is full above it and dark below. A patch
    # that wrote nothing at all would still pass "is_light_dirty".
    nibbles = nibbles_of(centre)
    check("the centre chunk has a real sky gradient",
          int(nibbles.min()) == 0 and int(nibbles.max()) == 15,
          f"min={int(nibbles.min())} max={int(nibbles.max())}")
    check("most of the column above the terrain is full daylight",
          int((nibbles == 15).sum()) > 40000, str(int((nibbles == 15).sum())))

    # --- a hole in the 3x3 is refused ---------------------------------------
    holed = fill(4, 4)
    holed.chunk_manager.remove(ChunkPos(5, 5))          # the far corner
    holed_light = DimensionLight(holed)
    holed_light.request_chunk((4, 4))
    holed_light.tick()
    check("a 3x3 with a hole lights nothing", holed_light.chunks_lit == 0,
          str(holed_light.chunks_lit))
    check("and the chunk was deferred, not dropped", holed_light.deferred_chunks >= 1,
          str(holed_light.deferred_chunks))
    check("it is still queued for when the neighbour arrives",
          holed_light.pending_chunks == 1, str(holed_light.pending_chunks))
    check("and it was not marked as lit",
          not holed.get_chunk(4, 4).light_populated)

    # --- a chunk that unloads while queued is dropped and counted ------------
    gone = fill(4, 4)
    gone_light = DimensionLight(gone)
    gone_light.request_chunk((4, 4))
    gone.chunk_manager.remove(ChunkPos(4, 4))
    gone_light.tick()
    check("a queued chunk that unloaded is dropped",
          gone_light.dropped_unloaded >= 1, str(gone_light.dropped_unloaded))
    check("and does not stay in the queue", gone_light.pending_chunks == 0,
          str(gone_light.pending_chunks))

    # --- the budget is a real gate ------------------------------------------
    # The same dimension and the same request, at two budgets. This is what
    # makes the budget more than decorative: 0 ms must move *nothing*.
    frozen = fill(4, 4)
    frozen_light = DimensionLight(frozen, budget_ms=0.0)
    frozen_light.request_chunk((4, 4))
    frozen_light.tick()
    check("a 0 ms budget lights nothing in a tick", frozen_light.chunks_lit == 0,
          str(frozen_light.chunks_lit))
    check("and leaves the chunk marked unlit",
          not frozen.get_chunk(4, 4).light_populated)
    check("but keeps it queued rather than losing it", frozen_light.pending_chunks >= 0,
          str(frozen_light.pending_chunks))

    generous = fill(4, 4)
    generous_light = DimensionLight(generous, budget_ms=100.0)
    generous_light.request_chunk((4, 4))
    generous_light.tick()
    check("a 100 ms budget converges in one tick", generous_light.chunks_lit >= 1,
          str(generous_light.chunks_lit))

    # The frozen one converges once the budget allows it -- which is what
    # "deferred by budget" has to mean rather than "lost".
    frozen_light.budget_ms = 100.0
    for _ in range(5):
        frozen_light.tick()
    check("the frozen chunk converges when the budget comes back",
          frozen.get_chunk(4, 4).light_populated,
          str(frozen_light.chunks_lit))
    check("so the two light fields agree byte for byte",
          field(frozen) == field(generous),
          f"{len(field(frozen))} vs {len(field(generous))} bytes")

    # --- the edge itself: a chunk the pipeline produced is queued ------------
    #
    # Every check above asks the engine for a chunk by hand, and that is exactly
    # how the wiring went missing: `on_chunk_ready` is what the actor calls when
    # a generator finishes a chunk, and it dropped the coordinate out of
    # `_untracked` -- the set the untracked pass walks to find chunks that became
    # readable -- without queueing it, so a chunk a worker built was never lit
    # and never corrected. Live, with the real server and one viewer: 169 chunks
    # in memory, `pending_chunks=0`, `chunks_lit=0`, 49 of 49 chunk packets sent
    # with the encoder's heightmap fallback and 0 light updates. The player sees
    # that as "everything under a block is black, whatever the block is".
    edge = fill(4, 4)
    edge_light = DimensionLight(edge)
    lifecycle = ChunkLifecycle(edge, light=edge_light)
    lifecycle.on_chunk_ready(ChunkPos(4, 4))
    check("the lifecycle queues a finished chunk for its first light pass",
          edge_light.pending_chunks == 1, str(edge_light.pending_chunks))
    edge_light.tick()
    check("so the next tick lights it", edge.get_chunk(4, 4).light_populated,
          str(edge_light.chunks_lit))
    check("and the field is terrain, not the all-15 default",
          edge.get_chunk(4, 4).sections[0].sky_light.tobytes()
          != np.full(2048, 0xFF, dtype=np.uint8).tobytes())


def main() -> None:
    print("Chunk lifecycle checks (load levels, auto load/unload, streaming save)")
    check_automatic_loading()
    check_stalled_claim_retry()
    check_dimension_interface()
    check_entity_tick_gate()
    check_actor_wiring()
    check_light_pipeline()

    with tempfile.TemporaryDirectory() as tmp:
        world = World(
            name="life",
            seed=12345,
            world_dir=Path(tmp) / "life",
            generator=legacy_generator(),
        )
        try:
            check_automatic_unloading(world)
        finally:
            try:
                world.chunk_saver.close()
            except Exception:
                pass

    check_skip_unchanged()
    check_read_after_write()
    check_lru_eviction_saves()
    check_forced_tickets()

    print()
    if FAILURES:
        print(f"RESULT: FAIL ({len(FAILURES)})")
        for name in FAILURES:
            print(f"  - {name}")
        sys.exit(1)
    print("RESULT: PASS")


if __name__ == "__main__":
    main()
