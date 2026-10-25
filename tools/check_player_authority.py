"""Acceptance check for server-authoritative player movement validation.

`physics/player.py` on a synthetic world this file builds itself -- no server,
no client, no network, no worldgen. Every case is "given this position, this
reported position and this world, assert what the validator did about it":

1. the first pose from a player is seeded, not judged, and a legal walk on flat
   ground produces **zero** correction packets;
2. a legal path that goes *around* a wall produces zero corrections -- the gate
   must not fire on movement that never passed through anything;
3. a path that crosses a wall in one packet produces **exactly one** correction,
   its target is in front of the wall, and the packet on the wire is the
   synchronize-position packet the join path uses (`flags = 0`, zero velocity,
   the client's own rotation);
4. a deviation inside `correction_threshold` (0.2 blocks) produces **zero**
   corrections, and repeating it cannot accumulate: the server's ghost stays at
   the wall face until the claim passes the threshold;
5. a single-tick displacement of 100 blocks produces one correction, back to the
   last position the server believed -- and a 10^6-block claim is answered
   without walking the kernel's block range at all;
6. two violations in a row: the second is suppressed by the cooldown and
   `corrections_suppressed` counts it, and the correction after the cooldown
   carries a strictly greater `teleport_id`;
7. the teleport ids are strictly increasing and the id a correction uses is
   accepted by the **real** `ClientHandler._handle_teleport_confirm`, which
   mirrors it into `_pending_teleport_id`; an id below the pending one is still
   refused;
8. rotation is never corrected: a rotation-only report changes nothing, and the
   correction carries the client's yaw/pitch verbatim;
9. a player walking up a half slab (the client's own step-up, i.e. a packet with
   a vertical component) is not corrected;
10. a still player in a world with **no chunks at all** is not judged and not
    deferred -- the zero-motion fast path reads no world;
11. a chunk in the swept span that is not resident skips the step, counts it,
    and does not move the ghost -- unloaded is neither air nor solid;
12. a spectator's report is passed through unjudged (they fly through blocks);
13. a disconnect (`sink=None`) drops the ghost, and the next join under the same
    id is seeded rather than judged;
14. the counters add up: every judged view lands in exactly one bucket, and
    every violation is either sent, suppressed or undeliverable.

Run: py -3.14t tools/check_player_authority.py

Exit code 0 when everything passes, 1 otherwise.
"""

from __future__ import annotations

import asyncio
import math
import queue
import sys
from pathlib import Path
from typing import List, Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from loguru import logger  # noqa: E402

# The stale-confirm case below deliberately logs a warning; the check should
# report its own results, not a wall of expected log lines.
logger.remove()

from network.client_handler import ClientHandler  # noqa: E402
from network.packets import CSynchronizePlayerPosition, STeleportConfirm  # noqa: E402
from physics.block_shapes import block_shapes  # noqa: E402
from physics.player import (DEFAULT_CORRECTION_THRESHOLD,  # noqa: E402
                           DEFAULT_SPEED_CAP, SPEED_MARGIN, SPRINT_SPEED,
                           Correction, PlayerMoveValidator)
from physics.world_view import WorldView  # noqa: E402
from player.player import Player  # noqa: E402
from world import block_states  # noqa: E402
from world.chunk import Chunk, ChunkPos  # noqa: E402
from world.chunk_streamer import PlayerViewMsg  # noqa: E402

#: The real overworld window, so the section arithmetic is the live one.
MIN_Y, HEIGHT = -64, 384
FLOOR_SECTION = (0 - MIN_Y) // 16      # the section holding world Y 0..15
FLOOR_TOP = 1.0                        # a stone layer at y = 0 ends at y = 1
HALF = 0.3                             # half of the player's 0.6-wide box

#: The wall the path checks use: one block thick at x = 2, two tall, from z = 4
#: to z = 7. Short enough to walk around, solid enough not to step over.
WALL_X = 2
WALL_Z0, WALL_Z1 = 4, 7

failures: List[str] = []
checks = 0


def check(label: str, condition: bool, detail: str = "") -> None:
    global checks
    checks += 1
    marker = "OK  " if condition else "FAIL"
    print(f"  {marker} {label}" + (f"  ({detail})" if detail else ""))
    if not condition:
        failures.append(label)


def near(a: float, b: float, tol: float = 1e-9) -> bool:
    return abs(a - b) <= tol


# ==================== the synthetic world ====================


class World:
    """Flat stone at y = 0, plus whatever a case puts on top of it."""

    def __init__(self, radius: int = 1) -> None:
        self.chunks = {}
        self.view = WorldView(MIN_Y, HEIGHT)
        for cx in range(-radius, radius + 1):
            for cz in range(-radius, radius + 1):
                self.ensure_chunk(cx, cz)

    def ensure_chunk(self, cx: int, cz: int) -> Chunk:
        chunk = self.chunks.get((cx, cz))
        if chunk is not None:
            return chunk
        chunk = Chunk(position=ChunkPos(cx, cz), min_y=MIN_Y, height=HEIGHT)
        section = chunk.sections[FLOOR_SECTION]
        section.blocks.reshape(16, 16, 16)[0, :, :] = STONE
        section.recompute_non_air_count()
        self.chunks[(cx, cz)] = chunk
        self.view.add(chunk)
        return chunk

    def put(self, x: int, y: int, z: int, state: int) -> None:
        chunk = self.ensure_chunk(x >> 4, z >> 4)
        chunk.set_block(x & 15, y, z & 15, state)

    def unload(self, cx: int, cz: int) -> None:
        self.chunks.pop((cx, cz), None)
        self.view.remove(cx, cz)


def wall_world() -> World:
    world = World()
    for z in range(WALL_Z0, WALL_Z1 + 1):
        world.put(WALL_X, 1, z, STONE)
        world.put(WALL_X, 2, z, STONE)
    return world


# ==================== harness ====================


class RecordingSink:
    """Stands in for `ChunkSink`: keeps what would have gone to the socket."""

    def __init__(self) -> None:
        self.batches: List[list] = []
        self.closed = False

    def send(self, packets) -> None:
        if self.closed:
            return
        self.batches.append(list(packets))

    def close(self) -> None:
        self.closed = True

    def packets(self) -> list:
        return [packet for batch in self.batches for packet in batch]


class Rig:
    """One player, one validator, one recorded socket, one ghost."""

    def __init__(self, world: World, player_id: str = "p1",
                 spectator: bool = False, **kwargs) -> None:
        self.world = world
        self.player_id = player_id
        self.spectator = spectator
        self.sink = RecordingSink()
        self.validator = PlayerMoveValidator(world.view, block_shapes(), **kwargs)
        self.validator.warm_up()
        self.sequence = 0
        #: The handler's `_pending_teleport_id`: 1 is the join teleport, which
        #: is the only one the handler itself sends today.
        self.teleport_id = 1
        self.yaw = 0.0
        self.pitch = 0.0

    def move(self, x: float, y: float, z: float, yaw: Optional[float] = None,
             pitch: Optional[float] = None) -> Optional[Correction]:
        self.sequence += 1
        if yaw is not None:
            self.yaw = yaw
        if pitch is not None:
            self.pitch = pitch
        return self.validator.on_view_message(PlayerViewMsg(
            player_id=self.player_id,
            sink=self.sink,
            x=x, y=y, z=z,
            yaw=self.yaw, pitch=self.pitch, on_ground=True,
            spectator=self.spectator,
            sequence=self.sequence,
            teleport_id=self.teleport_id,
        ))

    def hide(self) -> None:
        """A disconnect: the same message the streamer and tracker see."""
        self.validator.on_view_message(
            PlayerViewMsg(player_id=self.player_id, sink=None))

    def walk(self, points, step: float = 0.25) -> List[Optional[Correction]]:
        """Follow a polyline in `step`-sized reports, returning every answer."""
        answers: List[Optional[Correction]] = []
        current = self.validator.for_player(self.player_id).position
        for target in points:
            distance = math.dist(current, target)
            legs = max(1, int(math.ceil(distance / step)))
            for index in range(1, legs + 1):
                fraction = index / legs
                answers.append(self.move(
                    current[0] + (target[0] - current[0]) * fraction,
                    current[1] + (target[1] - current[1]) * fraction,
                    current[2] + (target[2] - current[2]) * fraction,
                ))
            current = target
        return answers

    @property
    def ghost(self):
        return self.validator.for_player(self.player_id)

    def corrections(self) -> List[CSynchronizePlayerPosition]:
        return [packet for packet in self.sink.packets()
                if isinstance(packet, CSynchronizePlayerPosition)]

    def stats(self) -> dict:
        return self.validator.stats()


def make_stub_handler() -> ClientHandler:
    """A `ClientHandler` with just enough state for the teleport handshake.

    Built via `__new__` so no socket is needed; the fields are the ones
    `_handle_teleport_confirm` and `_publish_view` read. `_pending_teleport_id`
    starts at 1, which is where the join teleport leaves it.
    """
    handler = ClientHandler.__new__(ClientHandler)
    handler.player = Player(uuid="00000000-0000-0000-0000-000000000000",
                            name="Tester")
    handler._actor_inbox = queue.Queue()
    handler._chunk_sink = None
    handler._spawn_blob = None
    handler._teleport_confirmed = False
    handler._pending_teleport_id = 1
    handler._view_sequence = 0
    return handler


def expect_one(label: str, answers) -> Optional[Correction]:
    """Assert an answer list holds exactly one correction; return it."""
    found = [answer for answer in answers if answer is not None]
    check(label, len(found) == 1, f"{len(found)} correction(s)")
    return found[0] if len(found) == 1 else None


# ==================== the checks ====================


def check_legal_walk() -> None:
    print("\n1. Seeding, and a legal walk on flat ground")
    world = World()
    rig = Rig(world)
    check("the first pose is seeded, not judged",
          rig.move(0.5, FLOOR_TOP, 8.5) is None)
    check("it is counted as seeded", rig.stats()["seeded"] == 1,
          str(rig.stats()["seeded"]))
    check("the ghost is where the client says it is",
          rig.ghost is not None and rig.ghost.position == (0.5, FLOOR_TOP, 8.5),
          str(rig.ghost))

    answers = rig.walk([(8.5, FLOOR_TOP, 8.5)])
    check("a 32-step walk answers nothing", all(a is None for a in answers),
          f"{len(answers)} steps")
    stats = rig.stats()
    check("zero corrections", stats["corrections"] == 0, str(stats["corrections"]))
    check("zero violations", stats["violations"] == 0, str(stats["violations"]))
    check("every step was swept", stats["judged"] == len(answers),
          f"judged={stats['judged']} steps={len(answers)}")
    check("the ghost followed the client to x = 8.5",
          near(rig.ghost.x, 8.5), repr(rig.ghost.x))
    check("and nothing was sent", rig.corrections() == [])


def check_path_around_wall() -> None:
    print("\n2. A legal path around a wall")
    world = wall_world()
    rig = Rig(world)
    rig.move(0.5, FLOOR_TOP, 5.5)
    answers = rig.walk([
        (0.5, FLOOR_TOP, 8.5),      # +z, past the end of the wall
        (3.5, FLOOR_TOP, 8.5),      # +x, in the clear past x = 3
        (3.5, FLOOR_TOP, 5.5),      # -z, behind the wall
    ])
    check("the path never answers with a correction",
          all(a is None for a in answers), f"{len(answers)} steps")
    check("zero corrections", rig.stats()["corrections"] == 0,
          str(rig.stats()["corrections"]))
    check("the ghost ended behind the wall",
          near(rig.ghost.x, 3.5) and near(rig.ghost.z, 5.5), str(rig.ghost))


def check_wall_crossing() -> None:
    print("\n3. A path that crosses the wall in one packet")
    world = wall_world()
    rig = Rig(world)
    rig.move(0.5, FLOOR_TOP, 5.5)
    correction = rig.move(3.5, FLOOR_TOP, 5.5, yaw=45.0, pitch=-30.0)
    check("exactly one correction came back", correction is not None)
    if correction is None:
        return
    check("it is a path violation, not a speed one", correction.reason == "path",
          correction.reason)
    check("it is pulled back to the wall face",
          near(correction.x, WALL_X - HALF), repr(correction.x))
    check("that is in front of the wall", correction.x < WALL_X)
    check("the other axes are untouched",
          near(correction.y, FLOOR_TOP) and near(correction.z, 5.5))
    check("the deviation is the whole claimed distance",
          near(correction.deviation, 3.5 - (WALL_X - HALF), 1e-9),
          repr(correction.deviation))
    check("the ghost is at the wall too", near(rig.ghost.x, WALL_X - HALF),
          repr(rig.ghost.x))
    check("the server believes what it just told the client",
          (correction.x, correction.y, correction.z) == rig.ghost.position,
          f"{correction} vs {rig.ghost}")

    packets = rig.corrections()
    check("one packet on the wire", len(packets) == 1, str(len(packets)))
    if not packets:
        return
    packet = packets[0]
    check("it is Synchronize Player Position",
          isinstance(packet, CSynchronizePlayerPosition))
    check("with absolute coordinates", packet.flags == 0, str(packet.flags))
    check("zero velocity", (packet.velocity_x, packet.velocity_y,
                            packet.velocity_z) == (0.0, 0.0, 0.0))
    check("the client's own rotation, verbatim",
          (packet.yaw, packet.pitch) == (45.0, -30.0),
          f"yaw={packet.yaw} pitch={packet.pitch}")
    check("the correction's id", packet.teleport_id == correction.teleport_id,
          f"{packet.teleport_id} vs {correction.teleport_id}")
    check("the correction's position",
          (packet.x, packet.y, packet.z)
          == (correction.x, correction.y, correction.z))


def check_threshold() -> None:
    print("\n4. A deviation below the threshold, and that it cannot accumulate")
    world = wall_world()
    rig = Rig(world)
    rig.move(WALL_X - HALF, FLOOR_TOP, 5.5)     # standing against the wall face
    check("a claimed 0.2 blocks inside the wall is not corrected",
          rig.move(WALL_X - HALF + 0.2, FLOOR_TOP, 5.5) is None)
    check("nor is 0.4", rig.move(WALL_X - HALF + 0.4, FLOOR_TOP, 5.5) is None)
    check("zero corrections so far", rig.stats()["corrections"] == 0,
          str(rig.stats()["corrections"]))
    check("the ghost never entered the wall", near(rig.ghost.x, WALL_X - HALF),
          repr(rig.ghost.x))

    correction = rig.move(WALL_X - HALF + 0.6, FLOOR_TOP, 5.5)
    check("0.6 blocks in does cross the threshold", correction is not None)
    if correction is not None:
        check("and is pulled back to the face",
              near(correction.x, WALL_X - HALF), repr(correction.x))
        check("the threshold is what held the first two back",
              near(correction.deviation, 0.6, 1e-9), repr(correction.deviation))
        check("the default threshold is 0.5",
              near(DEFAULT_CORRECTION_THRESHOLD, 0.5))

    rig.hide()
    rig2 = Rig(world, correction_threshold=2.0)
    rig2.move(WALL_X - HALF, FLOOR_TOP, 5.5)
    check("a raised threshold tolerates the same 0.6-block claim",
          rig2.move(WALL_X - HALF + 0.6, FLOOR_TOP, 5.5) is None)


def check_speed_gate() -> None:
    print("\n5. A single-tick displacement of 100 blocks")
    world = World()
    rig = Rig(world)
    rig.move(0.5, FLOOR_TOP, 0.5)
    correction = rig.move(100.5, FLOOR_TOP, 0.5)
    check("exactly one correction came back", correction is not None)
    if correction is not None:
        check("it is a speed violation", correction.reason == "speed",
              correction.reason)
        check("the player is put back where the server had them",
              (correction.x, correction.y, correction.z)
              == (0.5, FLOOR_TOP, 0.5), str(correction))
        check("the claimed displacement is reported",
              near(correction.horizontal, 100.0, 1e-9),
              repr(correction.horizontal))
        check("and the ghost did not follow the claim",
              rig.ghost.position == (correction.x, correction.y, correction.z),
              f"{correction} vs {rig.ghost}")
    stats = rig.stats()
    check("it was never swept (past max_sweep)", stats["too_far"] == 1,
          str(stats["too_far"]))
    check("and nothing is left pending on the actor",
          stats["judged"] == 0, str(stats["judged"]))
    check("the gate is above a sprint",
          DEFAULT_SPEED_CAP * SPEED_MARGIN > SPRINT_SPEED,
          f"gate={DEFAULT_SPEED_CAP * SPEED_MARGIN} sprint={SPRINT_SPEED}")

    print("\n5b. A 10^6-block claim is answered without sweeping")
    rig2 = Rig(world)
    rig2.move(0.5, FLOOR_TOP, 0.5)
    huge = rig2.move(1_000_000.5, FLOOR_TOP, 0.5)
    check("one correction", huge is not None and huge.reason == "speed",
          str(huge))
    check("no sweep was attempted", rig2.stats()["too_far"] == 1,
          str(rig2.stats()["too_far"]))
    check("the ghost did not move", rig2.ghost.position == (0.5, FLOOR_TOP, 0.5),
          str(rig2.ghost))


def check_cooldown() -> None:
    print("\n6. Two violations in a row: the cooldown suppresses the second")
    world = wall_world()
    rig = Rig(world)
    rig.move(0.5, FLOOR_TOP, 5.5)
    first = rig.move(3.5, FLOOR_TOP, 5.5)
    check("the first violation is corrected", first is not None)
    check("the cooldown started",
          rig.ghost.cooldown == 10, str(rig.ghost.cooldown))
    second = rig.move(5.5, FLOOR_TOP, 5.5)
    check("the second is suppressed", second is None)
    stats = rig.stats()
    check("one correction on the wire", stats["corrections"] == 1,
          str(stats["corrections"]))
    check("corrections_suppressed == 1", stats["corrections_suppressed"] == 1,
          str(stats["corrections_suppressed"]))
    check("both were counted as violations", stats["violations"] == 2,
          str(stats["violations"]))
    check("the ghost still took the physical answer",
          near(rig.ghost.x, WALL_X - HALF), repr(rig.ghost.x))

    for _ in range(10):
        rig.validator.tick()
    check("ten ticks clear the cooldown", rig.ghost.cooldown == 0,
          str(rig.ghost.cooldown))
    third = rig.move(3.5, FLOOR_TOP, 5.5)
    check("the next violation is corrected again", third is not None)
    if first is not None and third is not None:
        check("with a strictly greater teleport id",
              third.teleport_id > first.teleport_id,
              f"{first.teleport_id} -> {third.teleport_id}")
    check("one more correction on the wire", rig.stats()["corrections"] == 2,
          str(rig.stats()["corrections"]))
    return first, third


def check_handshake() -> None:
    print("\n7. teleport_id is strictly increasing and the real handler accepts it")
    world = wall_world()
    rig = Rig(world, player_id="p-join")
    rig.move(0.5, FLOOR_TOP, 5.5)
    first = rig.move(3.5, FLOOR_TOP, 5.5)
    check("the first correction exists", first is not None)
    if first is None:
        return
    check("its id is past the join teleport's 1", first.teleport_id == 2,
          str(first.teleport_id))

    handler = make_stub_handler()

    async def drive() -> None:
        # The real confirm path, on the real handler.
        await handler._handle_teleport_confirm(
            STeleportConfirm(teleport_id=first.teleport_id))
        check("the handler accepts the correction's id",
              handler._pending_teleport_id == first.teleport_id,
              str(handler._pending_teleport_id))
        check("a matching confirm publishes a view",
              handler._actor_inbox.qsize() == 1,
              str(handler._actor_inbox.qsize()))
        published = handler._actor_inbox.get()
        check("the published view carries the id the client is waiting on",
              published.teleport_id == first.teleport_id,
              str(published.teleport_id))
        check("and is sequenced", published.sequence == 1,
              str(published.sequence))

        # An id below the pending one is still a mismatch: the check was
        # widened to "newer", not removed.
        await handler._handle_teleport_confirm(
            STeleportConfirm(teleport_id=first.teleport_id - 1))
        check("a stale confirm does not move the pending id",
              handler._pending_teleport_id == first.teleport_id,
              str(handler._pending_teleport_id))
        check("and publishes nothing", handler._actor_inbox.qsize() == 0,
              str(handler._actor_inbox.qsize()))

    asyncio.run(drive())

    # The actor learns the new id from the published message and must allocate
    # above it, never on top of it.
    rig.teleport_id = first.teleport_id
    rig.move(3.5, FLOOR_TOP, 5.5)          # a fresh pose at the same place
    for _ in range(10):
        rig.validator.tick()
    second = rig.move(5.5, FLOOR_TOP, 5.5)
    check("the next correction is one past the confirmed id",
          second is not None and second.teleport_id == first.teleport_id + 1,
          str(second))
    ids = [packet.teleport_id for packet in rig.corrections()]
    check("the ids are strictly increasing", all(
        earlier < later for earlier, later in zip(ids, ids[1:])), str(ids))


def check_rotation() -> None:
    print("\n8. Rotation is never corrected")
    world = wall_world()
    rig = Rig(world)
    rig.move(0.5, FLOOR_TOP, 5.5, yaw=0.0, pitch=0.0)
    check("turning in place is not a movement",
          rig.move(0.5, FLOOR_TOP, 5.5, yaw=90.0, pitch=10.0) is None)
    check("nothing was sent", rig.stats()["corrections"] == 0,
          str(rig.stats()["corrections"]))
    check("the ghost is exactly where it was",
          rig.ghost.position == (0.5, FLOOR_TOP, 5.5), str(rig.ghost))

    correction = rig.move(3.5, FLOOR_TOP, 5.5, yaw=-135.0, pitch=42.5)
    check("the position violation is corrected", correction is not None)
    if correction is not None:
        check("yaw is the client's, untouched", correction.yaw == -135.0,
              repr(correction.yaw))
        check("pitch is the client's, untouched", correction.pitch == 42.5,
              repr(correction.pitch))
    packets = rig.corrections()
    if packets:
        check("the packet's rotation is the client's too",
              (packets[0].yaw, packets[0].pitch) == (-135.0, 42.5),
              f"yaw={packets[0].yaw} pitch={packets[0].pitch}")


def check_step_up() -> None:
    print("\n9. Walking up a half slab is not corrected")
    world = World()
    world.put(2, 1, 0, SLAB)
    rig = Rig(world)
    rig.move(0.5, FLOOR_TOP, 0.5)
    answers = [
        rig.move(0.75, FLOOR_TOP, 0.5),
        rig.move(1.0, FLOOR_TOP, 0.5),
        rig.move(1.25, FLOOR_TOP, 0.5),
        rig.move(1.5, FLOOR_TOP, 0.5),
        rig.move(1.7, FLOOR_TOP, 0.5),          # up against the slab's face
        rig.move(1.95, FLOOR_TOP + 0.5, 0.5),   # the client's own step-up
        rig.move(2.2, FLOOR_TOP + 0.5, 0.5),    # on top of the slab
    ]
    check("no step of the climb is corrected",
          all(answer is None for answer in answers),
          str([a is not None for a in answers]))
    check("zero corrections", rig.stats()["corrections"] == 0,
          str(rig.stats()["corrections"]))
    check("the ghost climbed with the client",
          near(rig.ghost.y, FLOOR_TOP + 0.5) and near(rig.ghost.x, 2.2),
          str(rig.ghost))


def check_still_and_unloaded() -> None:
    print("\n10. A still player in an empty world is not judged")
    empty = World(radius=0)
    empty.unload(0, 0)
    rig = Rig(empty)
    rig.move(0.5, FLOOR_TOP, 0.5)
    check("the same pose again is not a movement",
          rig.move(0.5, FLOOR_TOP, 0.5) is None)
    stats = rig.stats()
    check("it is counted as still", stats["still"] == 1, str(stats["still"]))
    check("no chunk was asked for", stats["deferred_unloaded"] == 0,
          str(stats["deferred_unloaded"]))
    check("nothing was judged", stats["judged"] == 0, str(stats["judged"]))
    check("zero corrections", stats["corrections"] == 0)

    check("a real step in an unloaded world is skipped, not guessed",
          rig.move(1.0, FLOOR_TOP, 0.5) is None)
    stats = rig.stats()
    check("deferred_unloaded == 1", stats["deferred_unloaded"] == 1,
          str(stats["deferred_unloaded"]))
    check("zero corrections", stats["corrections"] == 0,
          str(stats["corrections"]))
    check("the ghost did not move", rig.ghost.position == (0.5, FLOOR_TOP, 0.5),
          str(rig.ghost))

    print("\n11. An unloaded chunk in the swept span defers the step")
    world = World(radius=0)           # chunk (0, 0) only
    rig2 = Rig(world)
    rig2.move(15.5, FLOOR_TOP, 8.5)
    # A walking-sized step: the box sweeps into chunk 1 (x >= 16), which is not
    # resident, so the geometry cannot answer and the step is skipped.
    check("a step into chunk 1 is skipped",
          rig2.move(15.9, FLOOR_TOP, 8.5) is None)
    stats = rig2.stats()
    check("deferred_unloaded == 1", stats["deferred_unloaded"] == 1,
          str(stats["deferred_unloaded"]))
    check("zero corrections", stats["corrections"] == 0,
          str(stats["corrections"]))
    check("the ghost stayed in chunk 0", near(rig2.ghost.x, 15.5),
          repr(rig2.ghost.x))

    # ... but the speed gate needs no world, and an unloaded chunk must not
    # become a place where an impossible step is free.
    over_gate = rig2.move(16.5, FLOOR_TOP, 8.5)
    check("a step over the gate is still corrected without the geometry",
          over_gate is not None and over_gate.reason == "speed", str(over_gate))
    check("and it is put back", over_gate is not None
          and near(over_gate.x, 15.5), str(over_gate))
    check("the deferral was counted as well",
          rig2.stats()["deferred_unloaded"] == 2,
          str(rig2.stats()["deferred_unloaded"]))


def check_spectator_and_leave() -> None:
    print("\n12. A spectator is passed through unjudged")
    world = wall_world()
    rig = Rig(world, spectator=True)
    rig.move(0.5, FLOOR_TOP, 5.5)
    check("flying through the wall is not corrected",
          rig.move(50.5, FLOOR_TOP, 5.5) is None)
    stats = rig.stats()
    check("it is counted", stats["skipped_spectator"] == 1,
          str(stats["skipped_spectator"]))
    check("zero corrections", stats["corrections"] == 0)
    check("the ghost follows them, so the next mode change is judged from the "
          "right place", near(rig.ghost.x, 50.5), repr(rig.ghost.x))

    print("\n13. A disconnect drops the ghost")
    rig.hide()
    check("there is no ghost left", rig.ghost is None)
    check("it is counted", rig.stats()["removed"] == 1,
          str(rig.stats()["removed"]))
    check("a rejoin is seeded, not judged",
          rig.move(0.5, FLOOR_TOP, 5.5) is None)
    check("and counted as a second seed", rig.stats()["seeded"] == 2,
          str(rig.stats()["seeded"]))
    check("reset() forgets a ghost too", rig.validator.reset("p1"))
    check("a reset is counted", rig.stats()["resets"] == 1,
          str(rig.stats()["resets"]))
    check("resetting an unknown player is False",
          rig.validator.reset("nobody") is False)


def check_counters() -> None:
    print("\n14. The counters add up")
    world = wall_world()
    rig = Rig(world)
    rig.move(0.5, FLOOR_TOP, 5.5)
    rig.move(0.5, FLOOR_TOP, 5.5)                    # still
    rig.move(3.5, FLOOR_TOP, 5.5)                    # corrected
    rig.move(5.5, FLOOR_TOP, 5.5)                    # suppressed
    for _ in range(10):
        rig.validator.tick()
    rig.move(3.5, FLOOR_TOP, 5.5)                    # corrected again
    stats = rig.stats()
    check("every view lands in exactly one bucket",
          stats["views"] == (stats["still"] + stats["judged"]
                             + stats["too_far"] + stats["deferred_unloaded"]
                             + stats["overflow"]),
          f"views={stats['views']} buckets="
          f"{stats['still']}+{stats['judged']}+{stats['too_far']}"
          f"+{stats['deferred_unloaded']}+{stats['overflow']}")
    check("every violation is sent, suppressed or undeliverable",
          stats["violations"] == (stats["corrections"]
                                  + stats["corrections_suppressed"]
                                  + stats["undeliverable"]),
          f"violations={stats['violations']} corrections={stats['corrections']} "
          f"suppressed={stats['corrections_suppressed']}")
    check("violations split into path and speed",
          stats["violations"] == stats["violations_path"]
          + stats["violations_speed"],
          f"{stats['violations_path']} + {stats['violations_speed']}")
    check("players is reported from the ghost table",
          stats["players"] == 1, str(stats["players"]))
    check("stats() is a copy", rig.validator.stats() is not stats)

    print("\n15. An undeliverable correction is counted, not claimed")
    rig2 = Rig(wall_world())
    rig2.move(0.5, FLOOR_TOP, 5.5)
    rig2.sink.close()
    check("a closed sink produces no packet", rig2.move(3.5, FLOOR_TOP, 5.5) is None)
    stats = rig2.stats()
    check("it is counted as undeliverable", stats["undeliverable"] == 1,
          str(stats["undeliverable"]))
    check("and not as a correction", stats["corrections"] == 0,
          str(stats["corrections"]))
    check("the violation was still recorded", stats["violations"] == 1,
          str(stats["violations"]))


def main() -> int:
    from registry import init_registries

    init_registries(log_summary=False)

    global STONE, SLAB
    STONE = block_states.default_state_id("minecraft:stone")
    SLAB = block_states.state_id("minecraft:oak_slab", {"type": "bottom"})

    print("Player movement authority check (physics/player.py)")
    check_legal_walk()
    check_path_around_wall()
    check_wall_crossing()
    check_threshold()
    check_speed_gate()
    check_cooldown()
    check_handshake()
    check_rotation()
    check_step_up()
    check_still_and_unloaded()
    check_spectator_and_leave()
    check_counters()

    print()
    if failures:
        print(f"{len(failures)} of {checks} check(s) FAILED:")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print(f"All {checks} player authority checks passed.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception:
        import traceback

        traceback.print_exc()
        raise SystemExit(1)
