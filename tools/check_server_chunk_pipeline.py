"""
End-to-end check of the server's chunk pipeline: join -> load -> stream -> leave.

Every other chunk tool drives one subsystem directly. This one drives the real
wiring instead: a real `Ticker`, real `DimensionActor` threads, the real
background generator pool, and the same `PlayerViewMsg` the network handler
posts. It is the check that would have caught a lifecycle wired to the wrong
callback, a tick order that reads the level map before the tickets were moved,
or an actor that streams chunks the lifecycle never asked for.

The player is simulated with a view message and a recording sink instead of a
socket, so the test needs no port, no client and no login sequence:

* post `PlayerViewMsg` (a join at radius 2) into the overworld actor's inbox;
* wait for `LevelChunkWithLight` packets to reach the sink and for the loaded
  set to match the ticket square;
* post `PlayerViewMsg(sink=None)` (a disconnect) and wait for the unload pass;
* stop the tick pipeline the way the server does and check that what was in
  memory is on disk -- the shutdown path is where "unload without saving"
  would hurt most.

Run: python tools/check_server_chunk_pipeline.py
"""

import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from loguru import logger

logger.remove()

from core.server import CrotalusServer
from network.packets import CLevelChunkWithLight
from world.chunk import ChunkPos
from world.chunk_streamer import PlayerViewMsg, _chunk_disc
from world.dimension import DimensionType
from world.generator import GeneratorSettings, create_generator
from world.world import World

FAILURES = []
VIEW = 2
SIMULATION = 1
SQUARE_RADIUS = VIEW + 2  # level <= 33 reaches two rings past the ticket


def check(label: str, ok: bool, detail: str = "") -> bool:
    print(f"  {'OK  ' if ok else 'FAIL'} {label}" + (f"  ({detail})" if detail else ""))
    if not ok:
        FAILURES.append(label)
    return ok


class RecordingSink:
    """Stands in for ChunkSink: counts what would have gone to the socket.

    It also watches the light edge, because that is the half of the pipeline a
    counter of chunk packets cannot see: a chunk goes out as soon as its terrain
    exists, and the light engine fills its light afterwards. `lit_at_send` is
    the question "did the client get real light in the chunk packet", and
    `unlit_coords`/`updated` are the two halves of "and was it corrected if
    not". This sink runs on the actor thread, so reading the chunk here is
    reading it on the thread that owns it.
    """

    def __init__(self, dimension=None):
        self.dimension = dimension
        self.chunks = 0
        self.batches = 0
        self.unloads = 0
        self.updates = 0
        self.lit_at_send = 0
        self.unlit_at_send = 0
        self.unlit_coords = set()
        self.updated = set()
        self.closed = False

    def send(self, packets):
        if self.closed:
            return
        for packet in packets:
            if isinstance(packet, CLevelChunkWithLight):
                self.chunks += 1
                payload = packet.payload
                coord = (int.from_bytes(payload[0:4], "big", signed=True),
                         int.from_bytes(payload[4:8], "big", signed=True))
                chunk = (self.dimension.chunk_manager.peek(ChunkPos(*coord))
                         if self.dimension is not None else None)
                if chunk is not None and chunk.light_populated:
                    self.lit_at_send += 1
                else:
                    self.unlit_at_send += 1
                    self.unlit_coords.add(coord)
            elif type(packet).__name__ == "CLightUpdate":
                self.updates += 1
                self.updated.add((packet.chunk_x, packet.chunk_z))
            elif type(packet).__name__ == "CUnloadChunk":
                self.unloads += 1

    def close(self):
        self.closed = True


def wait_for(predicate, timeout: float, interval: float = 0.1) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return predicate()


def main() -> None:
    print("Server chunk pipeline check (join -> stream -> leave -> shutdown)")
    tmp = Path(tempfile.mkdtemp())
    server = CrotalusServer()
    world = World(
        name="pipeline",
        seed=4242,
        world_dir=tmp / "pipeline",
        generator=create_generator(GeneratorSettings(seed=4242), backend="legacy"),
    )
    server._world = world

    try:
        server._start_ticking()
        actors = list(server._actors_by_dim.items())
        check("one actor per dimension", len(actors) == 3, f"{[name.value for name, _ in actors]}")

        dimension = world.overworld
        lifecycle = world.lifecycle(DimensionType.OVERWORLD)
        lifecycle.save_chunk = world._save_hook(DimensionType.OVERWORLD)
        check(
            "the lifecycle is wired to the server's request handler",
            lifecycle._on_missing is not None and not lifecycle.sync_fallback,
        )
        check("nothing is loaded before anyone joins", dimension.chunk_manager.get_chunk_count() == 0)

        inbox = server._player_actor_inbox()
        sink = RecordingSink(dimension)
        inbox.put(
            PlayerViewMsg(
                player_id="itest",
                sink=sink,
                x=8.0,
                y=64.0,
                z=8.0,
                radius=VIEW,
                simulation_radius=SIMULATION,
            )
        )

        expected = (2 * SQUARE_RADIUS + 1) ** 2
        # What the *client* is sent is the view disc, not the loaded square:
        # the two border rings exist so edge rendering is correct and are never
        # streamed. Asserting the loaded count against the packet count would
        # be asserting the wrong thing.
        disc_size = len(_chunk_disc(0, 0, VIEW))
        loaded = wait_for(
            lambda: dimension.chunk_manager.get_chunk_count() >= expected
            and len(lifecycle._untracked) == 0,
            timeout=90.0,
        )
        check(
            "the join loads the ticket square through the real generator pool",
            loaded,
            f"{dimension.chunk_manager.get_chunk_count()}/{expected} chunks in memory",
        )

        streamed = wait_for(lambda: sink.chunks >= disc_size, timeout=60.0)
        check(
            "the streamer sends the client's view disc",
            streamed and sink.chunks == disc_size,
            f"{sink.chunks} chunk packets in {sink.batches} batch(es), "
            f"expected the {disc_size}-chunk disc of radius {VIEW}",
        )
        check(
            "the client gets less than the server keeps loaded",
            sink.chunks < expected,
            f"sent {sink.chunks} of {expected} loaded (the border rings stay server-side)",
        )

        # --- the light edge --------------------------------------------------
        #
        # A chunk packet carries the chunk's light, and the streamer sends one
        # as soon as `is_generated` is set -- so the light has to reach the
        # chunk, without being asked for, or the client is handed the encoder's
        # heightmap fallback (15 above the surface, 0 below, no block light).
        # That is what a whole session looked like when the lifecycle did not
        # queue a finished chunk: measured live with the real server, 49 of 49
        # chunk packets went out with `light_populated` False, 0 chunks were
        # ever lit and 0 light updates followed. Both halves are checked here:
        # the view ends up lit, and anything sent before its light existed is
        # corrected by a `CLightUpdate` (the streamer sends those before chunk
        # sends and only to players who already have the chunk, so an update
        # implies its packet went first).
        lit = wait_for(
            lambda: all(
                (chunk := dimension.get_chunk(*coord)) is not None
                and chunk.light_populated
                for coord in _chunk_disc(0, 0, VIEW)
            ),
            timeout=90.0,
        )
        check(
            "every chunk in the client's view was lit by the engine",
            lit,
            f"{sink.lit_at_send} of {sink.chunks} packets carried real light",
        )
        check(
            "and the ones sent before their light existed were corrected",
            sink.unlit_coords <= sink.updated,
            f"{sink.unlit_at_send} sent unlit, {sink.updates} updates, "
            f"{len(sink.unlit_coords - sink.updated)} never corrected",
        )
        check(
            "the ticking squares came from the simulation ticket",
            dimension.ticks_entities(0, 0)
            and not dimension.ticks_entities(SIMULATION + 1, 0)
            and dimension.ticks_blocks(SIMULATION + 1, 0)
            and not dimension.ticks_blocks(SIMULATION + 2, 0),
        )

        # Edit a block so the shutdown save has something only memory knows.
        chunk = dimension.chunk_manager.peek(ChunkPos(0, 0))
        chunk.set_block(0, 100, 0, 1)

        # --- the player disconnects ---
        inbox.put(PlayerViewMsg(player_id="itest", sink=None))
        drained = wait_for(lambda: dimension.chunk_manager.get_chunk_count() == 0, timeout=30.0)
        check(
            "leaving unloads the chunks the player was holding",
            drained,
            f"{dimension.chunk_manager.get_chunk_count()} chunks still resident",
        )
        check(
            "the tick kept running throughout",
            lifecycle.stats()["ticks"] > 0,
            f"{lifecycle.stats()['ticks']} ticks",
        )

        # --- shutdown, the way core/server.py does it ---
        server._stop_ticking()
        pending = world.chunk_saver.flush(timeout=60.0)
        check("the save queue drains on shutdown", pending == 0, f"{pending} left")
        check(
            "chunks unloaded before the disconnect were written",
            world.chunk_saver.stats()["saved"] > 0,
            f"{world.chunk_saver.stats()['saved']} saved, "
            f"{world.chunk_saver.stats()['failed']} failed",
        )
        world.chunk_saver.close()

        region_dir = world.world_dir / "region"
        region_files = list(region_dir.glob("*.mca"))
        check("region files exist after shutdown", len(region_files) > 0,
              f"{[p.name for p in region_files]}")

        fresh = World(
            name="pipeline",
            world_dir=tmp / "pipeline",
            generator=create_generator(GeneratorSettings(seed=4242), backend="legacy"),
        )
        try:
            reloaded = fresh.load_chunk(0, 0, dimension=DimensionType.OVERWORLD)
            check(
                "a chunk the player edited is on disk after the pipeline stopped",
                reloaded is not None and reloaded.get_block(0, 100, 0) == 1,
                f"block at (0,100,0) = "
                f"{None if reloaded is None else reloaded.get_block(0, 100, 0)}",
            )
        finally:
            fresh.chunk_saver.close()
    finally:
        try:
            server._stop_ticking()
        except Exception:
            pass
        try:
            from world.chunk_gen_thread import chunk_generator

            chunk_generator.stop()
        except Exception:
            pass
        try:
            world.chunk_saver.close()
        except Exception:
            pass
        shutil.rmtree(tmp, ignore_errors=True)

    print()
    if FAILURES:
        print(f"RESULT: FAIL ({len(FAILURES)})")
        for name in FAILURES:
            print(f"  - {name}")
        _exit(1)
    print("RESULT: PASS")
    _exit(0)


def _exit(code: int) -> None:
    """Leave without waiting for the generator pool's threads.

    This tool is the only one that starts the real background generator (and
    `chunk_generator` is a process-wide singleton). Its pool threads can still
    be parked on their work queue when the checks are done, and a non-daemon
    thread at interpreter exit turns "RESULT: PASS" into a job that hangs for
    half an hour and then reports failure. Every result has been printed by the
    time this runs, so flushing and leaving is strictly better than waiting.
    """
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)


if __name__ == "__main__":
    main()
