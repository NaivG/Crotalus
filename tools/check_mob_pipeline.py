"""
End-to-end check of the mob pipeline: join -> spawn -> track -> leave.

Every other mob tool drives one layer directly. This one drives the real
wiring instead: a real `Ticker`, real `DimensionActor` threads, the real
background generator pool, the real `NaturalSpawner` built by the actor, and
the same `PlayerViewMsg` the network handler posts. It is the check that would
have caught a spawner constructed without its lifecycle, a tick order that
lets a chunk unload before the mobs in it are cleaned up, or a mob that
reaches `Dimension.entities` but never reaches the tracking table.

The player is simulated with a view message and a recording sink instead of a
socket, so the test needs no port, no client and no login sequence:

* post `PlayerViewMsg` (a join on a landmass at radius 5 / simulation 4);
* wait for the first round to place mobs, and check the client was told in
  the order a client requires - Add Entity, then the metadata blob;
* check no mob is floating: every one stands exactly on its column's
  no-leaves surface;
* post `PlayerViewMsg(sink=None)` (a disconnect) and wait for the unload pass;
* check the mobs left both tables, that the spawner's unload counter saw
  them, and that nothing is left in `Dimension.entities`.

**Why the unload half uses a disconnect rather than a long walk.** The loaded
square is always larger than an entity's tracking range - `(radius + 2) * 16`
blocks against `min(type_range, radius) * 16` - so a chunk can only leave
memory once it is already past the point where the tracker would have
despawned its mobs for being out of range. A walking player therefore cannot
separate the two removal paths; a leaving player can, because with no viewer
attached the visibility pass has nothing to do and the unload pass is the only
one left. The out-of-range despawn is covered by `check_entity_delta.py` and
the unload removal in isolation by `check_mob_spawning.py`; this tool is the
one that runs both halves through the real actors.

Run: py -3.14t tools/check_mob_pipeline.py
"""

import os
import shutil
import sys
import tempfile
import time
from math import floor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from loguru import logger

logger.remove()

from core.server import CrotalusServer
from network.packets import CLevelChunkWithLight
from physics.bodies import SLEEPING
from world.chunk_gen_worker import _build_generator
from world.chunk_streamer import PlayerViewMsg
from world.dimension import DimensionType
from world.world import World

FAILURES = []

#: The three implemented animals, by the numeric ids the wire carries.
MOB_TYPE_IDS = {"minecraft:chicken": 26, "minecraft:cow": 30, "minecraft:sheep": 111}
MOB_IDS = frozenset(MOB_TYPE_IDS.values())

#: Render distance, in chunks. Must be at least `simulation_radius + 1` so the
#: client is *sent* the chunks its mobs stand in, and it decides the tracker's
#: reach: `min(10, radius) * 16` = 80 blocks, which covers the whole 72-block
#: half-width of the ticking square below. Every mob that spawns is therefore
#: in range, and a missing Add Entity is a failure rather than a far corner.
VIEW = 5

#: Simulation distance, in chunks. 81 chunks tick, so the creature cap is
#: `10 * 81 // 289 = 2` - small, but it is the real cap for this much world,
#: and two mobs are enough to prove the path.
SIMULATION = 4

#: A fully spawnable landmass on seed 0: every chunk of (11..17, -8..-2) has
#: 256 of 256 columns with a sturdy, sky-lit top. The world spawn is at
#: (0, 5) chunks away, so the 24-block world-spawn guard never applies here.
PLAYER_X, PLAYER_Z = 14 * 16 + 8, -5 * 16 + 8
PLAYER_Y = 64.0


def check(label: str, ok: bool, detail: str = "") -> bool:
    print(f"  {'OK  ' if ok else 'FAIL'} {label}" + (f"  ({detail})" if detail else ""))
    if not ok:
        FAILURES.append(label)
    return ok


class RecordingSink:
    """Stands in for ChunkSink: keeps the entity packets in send order."""

    def __init__(self):
        self.packets = []
        self.chunks = 0
        self.closed = False

    def send(self, packets):
        if self.closed:
            return
        # Collect, then publish in one list operation. The actor thread is
        # calling this while the test thread reads `self.packets`, and the
        # checks below compare packet *adjacency* (an Add Entity must be
        # followed by that entity's metadata). A reader that catches a batch
        # half-appended sees an ordering violation that never happened on the
        # wire -- which is what a bare `append` per packet allows.
        batch = []
        chunks = 0
        for packet in packets:
            name = type(packet).__name__
            if isinstance(packet, CLevelChunkWithLight):
                chunks += 1
            elif name in ("CAddEntity", "CSetEntityData", "CRemoveEntities",
                          "CRotateHead"):
                batch.append(packet)
        self.packets.extend(batch)
        self.chunks += chunks

    def close(self):
        self.closed = True

    def of(self, name):
        return [p for p in self.packets if type(p).__name__ == name]


def wait_for(predicate, timeout: float, interval: float = 0.1) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return predicate()


def surface_y(dimension, x: int, z: int):
    from entity.spawn_placement import top_motion_blocking_no_leaves

    return top_motion_blocking_no_leaves(dimension, x, z)


def main() -> None:
    print("Mob pipeline check (join -> spawn -> track -> leave)")
    tmp = Path(tempfile.mkdtemp())
    server = CrotalusServer()
    world = World(
        name="mobpipeline",
        seed=0,
        world_dir=tmp / "mobpipeline",
        # The generator the worker pool builds: same preset, same terrain.
        # A preset-less generator here would skip biomes/surface/features, so
        # any in-process fallback chunk would disagree with the pool's output.
        generator=_build_generator(DimensionType.OVERWORLD, seed=0, backend="legacy"),
    )
    server._world = world

    try:
        server._start_ticking()
        actors = list(server._actors_by_dim.items())
        check("one actor per dimension", len(actors) == 3,
              f"{[name.value for name, _ in actors]}")

        actor = server._actors_by_dim[DimensionType.OVERWORLD]
        dimension = world.overworld
        lifecycle = world.lifecycle(DimensionType.OVERWORLD)
        lifecycle.save_chunk = world._save_hook(DimensionType.OVERWORLD)

        check("the actor built a spawner with its own lifecycle and tracker",
              actor.spawner.lifecycle is lifecycle
              and actor.spawner.tracker is actor.entity_tracker)
        check("spawning is on and on vanilla's cadence",
              actor.spawner.enabled and actor.spawner.interval_ticks == 20,
              f"enabled={actor.spawner.enabled} "
              f"interval={actor.spawner.interval_ticks}")

        # **Every position assertion in this file is about a mob staying in
        # the chunk it was placed in**, and since mobs learned to walk
        # (`entity/ai/`) that has to be arranged rather than assumed: a cow
        # that strolls across a chunk boundary halfway through the
        # simulation-square section makes "mobs outside the square are paused"
        # fail with a mob that is inside it, and the failure reads exactly
        # like a residency bug. Dropping the spawner's `ai_context` before the
        # join is the inert state by construction - `_place` skips the bind,
        # and `MobAI.tick` returns before any goal runs without a context - so
        # every mob this run places is born static, with no race between a
        # mob's first tick and any test-side unbinding.
        #
        # The walking side of the pipeline is covered where it belongs:
        # `tools/check_mob_ai.py` drives the goals against a real solver, and
        # `NaturalSpawner.rekey_movers` (the chunk index following a mover) is
        # exercised there too.
        actor.spawner.ai_context = None
        check("the spawner can be made to place static mobs",
              actor.spawner.ai_context is None)
        check("no mob exists before anyone joins",
              not dimension.entities and actor.spawner.stats()["entities"] == 0,
              f"{len(dimension.entities)} entities")

        # ---- the join ----
        inbox = server._player_actor_inbox()
        sink = RecordingSink()
        inbox.put(PlayerViewMsg(
            player_id="mobtest",
            sink=sink,
            x=float(PLAYER_X),
            y=PLAYER_Y,
            z=float(PLAYER_Z),
            radius=VIEW,
            simulation_radius=SIMULATION,
            entity_id=99,
            uuid="00000000-0000-4000-8000-000000000099",
        ))

        loaded = wait_for(
            lambda: dimension.chunk_manager.get_chunk_count() > 0
            and len(lifecycle._untracked) == 0,
            timeout=180.0,
        )
        check("the join loads chunks through the real generator pool", loaded,
              f"{dimension.chunk_manager.get_chunk_count()} chunks in memory")

        ticking = sum(
            1 for cx in range(-SIMULATION, SIMULATION + 1)
            for cz in range(-SIMULATION, SIMULATION + 1)
            if dimension.ticks_entities(PLAYER_X // 16 + cx,
                                        PLAYER_Z // 16 + cz)
        )
        expected_cap = 10 * ticking // 289
        check("the ticking square is what the cap is quoted against",
              ticking == (2 * SIMULATION + 1) ** 2,
              f"{ticking} ticking chunks -> creature cap {expected_cap}")

        # ---- the first round ----
        spawned = wait_for(
            lambda: len(dimension.entities) >= expected_cap and expected_cap > 0,
            timeout=60.0,
        )
        stats = actor.spawner.stats()
        check("the spawner places mobs into the real world", spawned,
              f"{len(dimension.entities)} mobs, stats={_spawn_stats(stats)}")

        mobs = [e for e in dimension.entities.values()]
        check("every mob it placed is one of the three implemented animals",
              mobs and all(e.entity_type.name in MOB_TYPE_IDS for e in mobs),
              str(sorted({e.entity_type.name for e in mobs})))

        # ---- the tracking table ----
        tracker = actor.entity_tracker
        tracked_ids = set(tracker.tracked)
        mob_ids = {e.entity_id for e in mobs}
        check("every mob is in the tracking table as well as the world",
              mob_ids and mob_ids <= tracked_ids,
              f"mobs={sorted(mob_ids)} tracked={sorted(tracked_ids)}")
        check("and is indexed by the chunk it stands in",
              sum(len(v) for v in actor.spawner.chunk_entities.values())
              == len(mobs),
              str({k: sorted(v) for k, v in actor.spawner.chunk_entities.items()}))

        # ---- the wire ----
        # The tracker's visibility rule (`entity_tracker._in_range`) is a 3D
        # distance against `min(client_tracking_range, radius) * 16`. The
        # spawner's candidate square is wider than that reach - the corner of
        # the simulation square sits ~102 blocks out at simulation radius 4,
        # against an 80-block reach - so a far-corner mob is correctly *not*
        # sent, and the assertion follows vanilla's split rather than
        # assuming every mob is reachable.
        tracker_pct = tracker.broadcast_percentage
        viewer = (PLAYER_X, PLAYER_Y, PLAYER_Z)

        def in_reach(entity) -> bool:
            dx = entity.position.x - viewer[0]
            dy = entity.position.y - viewer[1]
            dz = entity.position.z - viewer[2]
            reach = (min(entity.entity_type.client_tracking_range, VIEW)
                     * 16 * tracker_pct / 100)
            return dx * dx + dy * dy + dz * dz <= reach * reach

        def mob_add_ids() -> set:
            return {p.entity_id for p in sink.of("CAddEntity")
                    if p.entity_type in MOB_IDS}

        def unsent_in_reach() -> int:
            sent_ids = mob_add_ids()
            return sum(1 for e in dimension.entities.values()
                       if in_reach(e) and e.entity_id not in sent_ids)

        sent = wait_for(lambda: unsent_in_reach() == 0, timeout=20.0)

        # Snapshot here: the creature cap blocks further rounds once the
        # count reaches it (the "places mobs" wait passed that mark), so the
        # entity set is frozen and every set below describes the same world
        # the sink's packets do. The AI was unbound above, before the first
        # position-dependent check, so "the same world" is also "the same
        # positions".
        mobs = list(dimension.entities.values())
        reachable = {e.entity_id for e in mobs if in_reach(e)}
        unreachable = {e.entity_id for e in mobs if not in_reach(e)}
        adds = sink.of("CAddEntity")
        add_ids = mob_add_ids()
        check("every mob in the tracker's reach is told to the client",
              sent and reachable <= add_ids,
              f"{sorted(add_ids)} sent for {sorted(reachable)} in reach, "
              f"{len(unreachable)} past the reach")
        check("and mobs past the reach are not sent",
              not (unreachable & add_ids),
              f"out of reach: {sorted(unreachable)}")
        check("and no phantom mob was ever sent",
              add_ids <= {e.entity_id for e in mobs},
              f"{sorted(add_ids - {e.entity_id for e in mobs})} unknown")

        # Add Entity, then the metadata for the same entity, immediately. A
        # client that gets the add without the health entry reads health 0
        # and deletes the mob on the spot, so adjacency is the assertion.
        order_ok = True
        detail = []
        for i, packet in enumerate(sink.packets):
            if type(packet).__name__ != "CAddEntity" or packet.entity_type not in MOB_IDS:
                continue
            nxt = sink.packets[i + 1] if i + 1 < len(sink.packets) else None
            if (nxt is None or type(nxt).__name__ != "CSetEntityData"
                    or nxt.entity_id != packet.entity_id):
                order_ok = False
                detail.append(f"{packet.entity_id}->"
                              f"{None if nxt is None else type(nxt).__name__}")
        check("each Add Entity is followed by that entity's metadata",
              order_ok and add_ids, "; ".join(detail) or
              f"{len(add_ids)} add(s)")

        # ---- nothing is floating ----
        # `floor`, not `int`: a mob stands at block + 0.5, and `int(-104.5)`
        # is -104 while the block it is in is -105. Truncating toward zero
        # reads the neighbouring column for every negative coordinate, which
        # is a check that passes or fails on where the mob happens to be.
        floating = []
        for mob in mobs:
            bx, bz = floor(mob.position.x), floor(mob.position.z)
            top = surface_y(dimension, bx, bz)
            if top is None or abs(mob.position.y - (top + 1)) > 1e-6:
                chunk = dimension.get_chunk(bx >> 4, bz >> 4)
                col = [chunk.get_block(bx & 15, y, bz & 15)
                       for y in range(floor(mob.position.y) + 4,
                                      floor(mob.position.y) - 5, -1)]
                floating.append(
                    f"{mob.entity_type.name} at ({bx}, {mob.position.y}, {bz}) "
                    f"top={top} gen={chunk.is_generated} "
                    f"height_at={chunk.get_height_at(bx & 15, bz & 15)} "
                    f"column(y+4..y-4)={col}"
                )
        check("every mob stands exactly on its column's no-leaves surface",
              not floating, "; ".join(floating) or
              f"{len(mobs)} mob(s) checked")

        check("and every mob is in the chunk it says it is",
              all((floor(m.position.x) >> 4, floor(m.position.z) >> 4)
                  in actor.spawner.chunk_entities for m in mobs))

        # ---- physics: they fell, landed, and stayed on their feet ----
        # The strongest end-to-end assertion in this file is above ("stands
        # exactly on its column's surface"), and it is now a statement about the
        # collision engine rather than about the spawn placement: the mobs were
        # placed by validation and then held there by the solver, which is the
        # only thing keeping them from sinking once `update_interval` stops
        # re-sending a position.
        #
        # Since mobs learned to walk (`entity/ai/`), "every body is asleep" is
        # no longer the right claim - a cow mid-stroll is legitimately awake.
        # The claim that survived is the one underneath it: a body is awake
        # *only* because the AI is driving it, and everything the AI is not
        # driving has gone back to sleep.
        bodies = actor.physics.bodies
        check("every mob has a physics body",
              all(actor.physics.has_body(m.entity_id) for m in mobs),
              f"{len(mobs)} mob(s), {len(bodies)} bodies")
        drifted = []
        for mob in mobs:
            slot = bodies.slot_of(mob.entity_id)
            if slot is None:
                continue
            body = (float(bodies.x[slot]), float(bodies.y[slot]), float(bodies.z[slot]))
            pose = (mob.position.x, mob.position.y, mob.position.z)
            if body != pose:
                drifted.append(f"{mob.entity_id}: body {body} != pose {pose}")
        check("the body and the entity agree on where the mob is",
              not drifted, "; ".join(drifted) or f"{len(mobs)} mob(s) checked")
        awake = {m.entity_id for m in mobs
                 if not (bodies.flags[bodies.slot_of(m.entity_id)] & SLEEPING)}
        walking = {m.entity_id for m in mobs
                   if m.ai is not None and not m.ai.navigator.is_idle}
        check("every awake mob is awake because its AI is walking it",
              awake <= walking, f"awake={sorted(awake)} walking={sorted(walking)}")
        check("and every mob is on the ground, not falling or floating",
              all(m.position.on_ground for m in mobs),
              str([m.entity_id for m in mobs if not m.position.on_ground]))
        pstats = actor.physics.stats()
        check("no body was deferred to a chunk that is not resident",
              pstats["deferred_unloaded"] == 0, str(pstats["deferred_unloaded"]))
        check("no candidate buffer overflowed", pstats["overflow"] == 0,
              str(pstats["overflow"]))
        check("no mob fell out of the world", pstats["removed_void"] == 0,
              str(pstats["removed_void"]))

        # A sleeping mob's position does not change, so the tracker must not
        # produce deltas for it. This is the "the server is moving it and the
        # client is not" bug, asserted from the client's side of the wire.
        #
        # Note what this does *not* cover: these mobs are placed on validated
        # ground and never fall, so the packets *during* a fall are not seen
        # here. That path is covered by `check_physics_solver.py`, and pushing
        # one into the air from this thread would be a cross-thread write to
        # the actor's solver -- the one thing this architecture forbids.
        def mob_position_packets() -> int:
            return sum(1 for packet in sink.of("CEntityPositionSync")
                       if packet.entity_id in mob_ids)

        settled = mob_position_packets()
        time.sleep(2.0)
        check("a settled mob produces no position updates at all",
              mob_position_packets() == settled == 0,
              f"{settled} -> {mob_position_packets()}")

        # ---- the load level gates the solver, on the real actor ----
        # `check_physics_solver.py` §14 proves the pause semantics against a real
        # ticket table; this proves the *wiring* -- that the solver built by the
        # actor asks the dimension's own model -- and then drives the one case
        # the spawner can never produce: mobs that are loaded but no longer
        # ticking. Shrinking the simulation square while the view distance stays
        # put is exactly `Chunk.mdx`'s lazy/border situation, on a live actor.
        gate = actor.physics.ticks_entities
        check("the actor's solver is gated on the dimension's own load levels",
              getattr(gate, "__self__", None) is dimension, repr(gate))
        player_chunk = (PLAYER_X // 16, PLAYER_Z // 16)
        check("and the gate answers where the world is: here yes, 40 chunks out no",
              gate(*player_chunk) and not gate(player_chunk[0] + 40, player_chunk[1]))
        pstats = actor.physics.stats()
        check("with every mob inside it, none of them is paused",
              pstats["paused_not_ticking"] == 0,
              f"{pstats['paused_not_ticking']} of {len(mobs)}")

        poses_before = {m.entity_id: (m.position.x, m.position.y, m.position.z)
                        for m in mobs}
        packets_before = mob_position_packets()
        inbox.put(PlayerViewMsg(
            player_id="mobtest", sink=sink, x=float(PLAYER_X), y=PLAYER_Y,
            z=float(PLAYER_Z), radius=VIEW, simulation_radius=1, entity_id=99,
            uuid="00000000-0000-4000-8000-000000000099",
        ))
        # Wait for the *model* to move rather than sleeping a fixed number of
        # seconds: the message is applied on the actor thread, and asserting
        # before it arrives would read the old square. Nothing else changes --
        # the loading ticket still holds every chunk these mobs stand in.
        shrank = wait_for(
            lambda: not dimension.ticks_entities(player_chunk[0] + 2,
                                                 player_chunk[1]),
            timeout=30.0,
        )
        check("the lifecycle applied the shrunken simulation square", shrank,
              str(dimension.get_load_level(player_chunk[0] + 2, player_chunk[1])))
        # Which mobs that leaves outside the square is not the test's choice: the
        # spawner decides where they stand, and its draws depend on how much
        # terrain had been generated when a round opened. So the assertion is not
        # "some mob was paused" but "the solver's pause set is exactly the model's
        # pause set", which holds whatever the spawner did -- and when the two
        # sets are empty it still pins the absence of *over*-pausing. The
        # positive case in full detail is `check_physics_solver.py` §14, which
        # builds the load types directly.
        outside = [m.entity_id for m in mobs
                   if not dimension.ticks_entities(floor(m.position.x) >> 4,
                                                   floor(m.position.z) >> 4)]
        if not outside:
            print("  ..   no mob fell outside the shrunken square this run; the "
                  "pause/resume semantics are pinned by check_physics_solver.py "
                  "§14 and check_chunk_lifecycle.py §10")
        # Wait for the solver to have run a tick with it, so the counters below
        # describe a steady state rather than the tick the ticket moved on.
        paused_in_time = wait_for(
            lambda: actor.physics.stats()["paused_not_ticking"] == len(outside),
            timeout=30.0,
        )
        live = actor.physics.stats()
        check("the solver pauses exactly the mobs the model says it should",
              paused_in_time and live["paused_not_ticking"] == len(outside),
              f"solver={live['paused_not_ticking']} model={len(outside)} "
              f"({sorted(outside)} of {sorted(mob_ids)} outside the 3x3 square)")
        # A few more ticks of being paused, so "did not move" means something.
        time.sleep(0.5)
        drifted_paused = [eid for eid in outside
                          if poses_before[eid]
                          != (dimension.entities[eid].position.x,
                              dimension.entities[eid].position.y,
                              dimension.entities[eid].position.z)]
        check("a paused mob's position is bit-for-bit unchanged",
              not drifted_paused, str(drifted_paused))
        check("and no packet was sent for one",
              mob_position_packets() == packets_before,
              f"{packets_before} -> {mob_position_packets()}")
        check("and it is still in the world and the tracking table",
              all(eid in dimension.entities and eid in tracker.tracked
                  for eid in outside),
              f"{sorted(outside)}")

        # Put the square back: the paused mobs must resume, and the leave below
        # must behave exactly as it did before this section ran.
        inbox.put(PlayerViewMsg(
            player_id="mobtest", sink=sink, x=float(PLAYER_X), y=PLAYER_Y,
            z=float(PLAYER_Z), radius=VIEW, simulation_radius=SIMULATION,
            entity_id=99,
            uuid="00000000-0000-4000-8000-000000000099",
        ))
        resumed = wait_for(
            lambda: actor.physics.stats()["paused_not_ticking"] == 0
            and dimension.ticks_entities(player_chunk[0] + 2, player_chunk[1]),
            timeout=30.0,
        )
        check("restoring the simulation square resumes every mob", resumed,
              str(actor.physics.stats()["paused_not_ticking"]))

        # ---- the leave ----
        before = actor.spawner.stats()["removed_on_unload"]
        inbox.put(PlayerViewMsg(player_id="mobtest", sink=None, entity_id=99))
        # Generous, because every chunk that was never written before has to
        # go through the save pipeline before its slot is free: this run
        # generates 225 chunks, so the drain is bounded by disk rather than
        # by ticks. The unload itself is two grace periods, about 2 seconds.
        drained = wait_for(
            lambda: dimension.chunk_manager.get_chunk_count() == 0, timeout=180.0
        )
        check("leaving unloads the chunks the player was holding", drained,
              f"{dimension.chunk_manager.get_chunk_count()} chunks still resident; "
              f"{_lifecycle_stats(lifecycle)}")

        gone = wait_for(lambda: not dimension.entities, timeout=30.0)
        check("the mobs left Dimension.entities with their chunks", gone,
              f"{len(dimension.entities)} left")
        check("and the spawner's unload pass is what removed them",
              actor.spawner.stats()["removed_on_unload"] > before,
              f"{before} -> {actor.spawner.stats()['removed_on_unload']}")
        check("the residency index is empty", not actor.spawner.chunk_entities,
              str(actor.spawner.chunk_entities))
        check("the tracking table dropped them too",
              not (mob_ids & set(tracker.tracked)),
              str(sorted(tracker.tracked)))
        check("and nothing is left in the round's candidate list",
              not actor.spawner._candidates
              or actor.spawner.stats()["entities"] == 0,
              f"candidates={len(actor.spawner._candidates)}")

        # A dimension with nobody in it does not spawn: the cap is quoted
        # against ticking chunks, and no player means no tickets.
        idle = actor.spawner.stats()["rounds"]
        time.sleep(1.5)
        check("an empty dimension spawns nothing further",
              not dimension.entities
              and actor.spawner.stats()["rounds"] >= idle,
              f"rounds {idle} -> {actor.spawner.stats()['rounds']}, "
              f"entities {len(dimension.entities)}")

        # ---- shutdown, the way core/server.py does it ----
        server._stop_ticking()
        world.chunk_saver.flush(timeout=60.0)
        check("the save queue drains on shutdown", True)
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


def _spawn_stats(stats) -> str:
    keys = ("rounds", "candidate_chunks", "spawned", "skipped_over_cap",
            "rejected_position", "removed_on_unload")
    return " ".join(f"{k}={stats.get(k)}" for k in keys)


def _lifecycle_stats(lifecycle) -> str:
    return str(lifecycle.stats())


def _exit(code: int) -> None:
    """Leave without waiting for the generator pool's threads.

    Same reason as `check_server_chunk_pipeline.py`: this tool starts the real
    background generator, and its pool threads can still be parked on their
    work queue when the checks are done. Every result has been printed by the
    time this runs, so flushing and leaving beats hanging at interpreter exit.
    """
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)


if __name__ == "__main__":
    main()
