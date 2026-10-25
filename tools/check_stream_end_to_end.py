"""End-to-end: a player's whole view drains through the real pipeline.

Everything here is the production path -- a real `World`, the real
`ChunkGeneratorThread` worker pool, a real `ChunkStreamer` pumped at the
dimension actor's tick rate. Only the socket is a stub that counts packets
instead of writing them.

This is the shape of the bug that stalled clients on their loading screen:
the streamer holds strict send order, so it cannot skip a chunk that is not
ready, and a generation request that quietly did nothing left the nearest
queued chunk permanently unavailable.

Run: python tools/check_stream_end_to_end.py
"""

import sys, tempfile, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from loguru import logger
logger.remove()

from world.chunk_gen_thread import chunk_generator
from world.chunk_streamer import ChunkStreamer, PlayerViewMsg, _chunk_disc
from world.dimension import DimensionType
from world.world import World

RADIUS = 10
#: The view is 317 chunks and the production generator runs the four
#: `world/vanilla` layers - biomes, surface rules, carvers, features - across
#: a pool of worker processes. `chunks/second` below is the headline.
TIMEOUT_S = 300.0
TICK = 1.0 / 20          # 20 TPS, like the dimension actor


class CountingSink:
    """Duck-typed ChunkSink: counts chunks instead of writing to a socket."""

    def __init__(self):
        self.chunks = 0
        self.closed = False

    def send(self, packets):
        if self.closed:
            return
        self.chunks += sum(1 for p in packets
                           if type(p).__name__ == "CLevelChunkWithLight")

    def close(self):
        self.closed = True


def main() -> None:
    tmp = Path(tempfile.mkdtemp())
    world = World(name="e2e", seed=1234, world_dir=tmp / "e2e")

    streamer = ChunkStreamer(
        world.overworld,
        on_missing_chunks=lambda cx, cz, p: world.request_chunk(
            cx, cz, DimensionType.OVERWORLD, priority=p
        ),
    )

    sink = CountingSink()
    streamer.attach(PlayerViewMsg("tester", sink, x=0.5, y=80.0, z=0.5, radius=RADIUS))

    expected = len(_chunk_disc(0, 0, RADIUS))
    print(f"view disc: {expected} chunks, budget {streamer.chunks_per_tick}/tick")

    start = time.time()
    last_report = 0.0
    while time.time() - start < TIMEOUT_S:
        streamer.pump()
        stats = streamer.stats()["tester"]
        if time.time() - last_report > 2:
            last_report = time.time()
            print(f"  t={time.time()-start:5.1f}s sent={stats['sent']:3d} "
                  f"queued={stats['queued']:3d} requested={stats['requested']:3d}")
        if sink.chunks >= expected:
            break
        time.sleep(TICK)

    elapsed = time.time() - start
    stats = streamer.stats()["tester"]
    ok = sink.chunks == expected and stats["queued"] == 0
    print(f"\nstreamed {sink.chunks}/{expected} chunks in {elapsed:.1f}s, "
          f"{stats['queued']} still queued")
    print(f"throughput: {sink.chunks / max(elapsed, 1e-9):.2f} chunks/s "
          f"({elapsed / max(sink.chunks, 1) * 1000:.0f} ms/chunk end to end)")
    print("RESULT:", "PASS" if ok else "FAIL")

    streamer.detach_all()
    chunk_generator.stop()
    world.unload(save=False)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
