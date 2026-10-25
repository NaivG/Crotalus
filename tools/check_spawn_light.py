"""The spawn preload must leave its chunks LIT, not merely generated.

Generation writes no light -- a worker ships blocks and the receiving side's
sections carry `ChunkSection`'s defaults (`world/chunk_gen_worker.py`), so a
chunk is dark until an engine patches it. Two places queue that patch once the
actors are up: `ChunkLifecycle.on_chunk_ready` (a worker finished) and its
`_load_missing` scan (a chunk came back from disk). The spawn preload runs
*before* `_start_ticking`, so neither existed and every spawn chunk used to
reach memory unlit and stay that way: `pending_chunks=0`, every chunk packet
sent with the encoder's heightmap approximation, and the client drew that
approximation until an operator ran `/light relight`. Chunks loaded later were
fine -- which is the asymmetry this guards.

A patch is one chunk's 3x3, so the rim of any finite disc cannot be lit until
something loads outside it. The checks below therefore use a 5x5 and split
the disc: the 3x3 interior must light at boot, the 16-chunk rim must come back
as `deferred` (and must NOT be queued -- the engine re-queues an incomplete
chunk, which would turn the boot pass into a full-timeout spin), and the rim
must actually light once its neighbours exist.

Run: py -3.14t tools/check_spawn_light.py
"""

import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from loguru import logger

logger.remove()

from world.chunk import ChunkPos
from world.chunk_gen_thread import chunk_generator
from world.dimension import DimensionType
from world.light_engine import DimensionLight
from world.world import World

SEED = 424242
RADIUS = 2  # 5x5: a real interior (3x3) plus a rim that genuinely cannot light

fails = 0


def check(name: str, ok: bool, detail: str = "") -> None:
    global fails
    tag = "ok  " if ok else "FAIL"
    print(f"[{tag}] {name}" + (f"  ({detail})" if detail else ""))
    if not ok:
        fails += 1


def disc(cx: int = 0, cz: int = 0, radius: int = RADIUS):
    return [(cx + dx, cz + dz)
            for dx in range(-radius, radius + 1)
            for dz in range(-radius, radius + 1)]


def interior(radius: int = RADIUS - 1):
    """The coords whose whole 3x3 is inside the disc -- the ones that can light."""
    return disc(radius=radius)


def chunk_at(world, coord):
    return world.overworld.chunk_manager.get_chunk(ChunkPos(*coord))


def light_bytes(world, coords) -> dict:
    out = {}
    for coord in coords:
        chunk = chunk_at(world, coord)
        out[coord] = b"".join(
            section.sky_light.tobytes() + section.block_light.tobytes()
            for section in chunk.sections
        )
    return out


def drain(engine, want: int, timeout: float = 60.0) -> None:
    """Tick until `want` chunks have been lit, or the deadline passes."""
    deadline = time.monotonic() + timeout
    while engine.chunks_lit < want and time.monotonic() < deadline:
        engine.tick()


def main() -> None:
    tmp = Path(tempfile.mkdtemp())
    world = World(name="spawnlight", seed=SEED, world_dir=tmp / "spawnlight")

    chunk_generator.configure(workers=4, backend="vanilla")
    # The sink the server actually has at this point in boot: `core/server.py`
    # only swaps in the actor router during `_start_ticking`, after the preload.
    # Getting this wrong would make the whole check vacuous.
    chunk_generator.set_completion_sink(world.insert_generated_chunk)
    chunk_generator.set_failure_sink(world.discard_failed_chunk)
    chunk_generator.start()

    coords = disc()
    inner = interior()
    rim = [c for c in coords if c not in set(inner)]
    try:
        # -- the pre-condition this bug lived in ----------------------------
        done = world.generate_chunks_blocking(coords, DimensionType.OVERWORLD,
                                              timeout=600)
        check("spawn batch settled", done == len(coords), f"{done}/{len(coords)}")

        unlit = [c for c in coords if not chunk_at(world, c).light_populated]
        check("generation alone leaves the disc unlit (the bug's shape)",
              len(unlit) == len(coords), f"{len(unlit)}/{len(coords)} unlit")

        # -- 1. the boot pass lights the interior ---------------------------
        started = time.monotonic()
        answer = world.light_chunks_blocking(coords, DimensionType.OVERWORLD)
        elapsed = time.monotonic() - started
        check("queued exactly the interior", answer["queued"] == len(inner),
              f"{answer['queued']} queued, want {len(inner)}")
        check("the rim came back as deferred, not queued",
              sorted(answer["deferred"]) == sorted(rim),
              f"{len(answer['deferred'])} deferred, want {len(rim)}")
        check("the drain finished", answer["pending"] == 0,
              f"{answer['pending']} pending")
        check("no kernel errors", answer["engine_errors"] == 0,
              str(answer["engine_errors"]))
        check("and it did not spin on the rim", elapsed < 20.0, f"{elapsed:.1f}s")

        still = [c for c in inner if not chunk_at(world, c).light_populated]
        check("every interior chunk is lit", not still, str(still))

        # -- 2. the field is terrain, not the all-15 default ----------------
        chunk = chunk_at(world, (0, 0))
        default_sky = np.full(2048, 0xFF, dtype=np.uint8).tobytes()
        solid = [s for s in chunk.sections
                 if s.sky_light.tobytes() != default_sky]
        check("sky light is not the 'nobody computed this' default",
              bool(solid), f"{len(solid)}/{len(chunk.sections)} sections differ")

        underground = [s for s in chunk.sections[1:]
                       if s.block_light.any() or (s.sky_light == 0).any()]
        check("something below the top section is dark or torch-lit",
              bool(underground),
              f"{len(underground)}/{max(0, len(chunk.sections) - 1)} sections")

        # -- 3. agrees with the repair path, byte for byte ------------------
        boot_light = light_bytes(world, inner)

        repair = DimensionLight(world.overworld,
                                chunks_per_tick=max(1, len(coords)))
        for coord in coords:
            repair.queue_chunk(*coord)
        drain(repair, len(inner))

        mismatched = [c for c in inner
                      if boot_light[c] != light_bytes(world, [c])[c]]
        check("boot light == /light relight light, byte for byte",
              not mismatched, str(mismatched))

        # -- 4. re-running costs nothing -------------------------------------
        again = world.light_chunks_blocking(coords, DimensionType.OVERWORLD)
        check("a second pass queues nothing", again["queued"] == 0, str(again))
        check("skips the already-lit interior", again["skipped"] == len(inner),
              str(again["skipped"]))
        check("still reports the rim", len(again["deferred"]) == len(rim),
              str(len(again["deferred"])))
        check("and the field did not move",
              light_bytes(world, inner) == boot_light)

        # -- 5. absent coordinates are skipped, not fatal --------------------
        absent = world.light_chunks_blocking(
            [(40, 40)] + coords, DimensionType.OVERWORLD)
        check("an absent coordinate is skipped, not queued",
              absent["queued"] == 0 and absent["skipped"] == len(inner) + 1,
              str(absent["skipped"]))

        # -- 6. an isolated chunk defers instead of spinning ----------------
        lonely = world.generate_chunks_blocking([(60, 60)], DimensionType.OVERWORLD,
                                                timeout=600)
        check("isolated chunk generated", lonely == 1, str(lonely))
        started = time.monotonic()
        stuck = world.light_chunks_blocking([(60, 60)], DimensionType.OVERWORLD)
        check("its incomplete 3x3 defers it, and returns at once",
              stuck["queued"] == 0 and stuck["deferred"] == [(60, 60)]
              and time.monotonic() - started < 1.0, str(stuck))
        check("and it stays honestly unlit",
              not chunk_at(world, (60, 60)).light_populated)

        # -- 7. the rim's promise: it lights once its neighbours arrive -----
        # What `_queue_spawn_light_backlog` sets up at boot, replayed here on
        # a live engine. The engine re-queues an incomplete chunk, so this is a
        # promise rather than a poll.
        live = DimensionLight(world.overworld, budget_ms=50.0,
                              chunks_per_tick=max(1, len(rim) + 1))
        for coord in rim:
            live.request_chunk(coord)
        check("the rim starts queued on a live engine",
              live.pending_chunks == len(rim), str(live.pending_chunks))

        # Nothing outside the disc exists yet, so it must hold: not lit, not
        # dropped.
        live.tick()
        check("it holds while its outer neighbours are missing",
              live.chunks_lit == 0 and live.pending_chunks == len(rim),
              f"lit={live.chunks_lit} pending={live.pending_chunks}")

        # Now supply them, the way the streamer would.
        outside = [c for c in disc(radius=RADIUS + 1) if c not in set(coords)]
        world.generate_chunks_blocking(outside, DimensionType.OVERWORLD,
                                       timeout=600)
        for coord in outside:
            live.request_chunk(coord)
        drain(live, len(rim) + len(outside), timeout=180.0)

        dark = [c for c in rim if not chunk_at(world, c).light_populated]
        check("and lights every one of them once they arrive", not dark,
              f"{len(dark)}/{len(rim)} still dark")
        # What is left pending is the *new* ring's own rim. `outside` is exactly
        # the radius-3 ring (the 7x7 minus the 5x5), so every one of them needs
        # an eighth ring that does not exist. Deferral is a property of any
        # finite area, not something the spawn pass introduced.
        check("and only the next rim out is still waiting",
              live.pending_chunks == len(outside),
              f"{live.pending_chunks} pending, want {len(outside)}")

    finally:
        chunk_generator.stop()
        world.unload(save=False)

    print()
    if fails:
        print(f"{fails} check(s) FAILED")
        sys.exit(1)
    print("all checks passed")


if __name__ == "__main__":
    main()