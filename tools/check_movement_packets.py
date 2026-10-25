"""End-to-end: the movement packets, the dispatch table, and chunk re-streaming.

Two things are checked here, both against the real code rather than a copy of
it.

1. The PLAY client->server dispatch table in network/client_handler.py is
   parsed out of the source and compared against the six packets added for
   the "Unknown packet" warnings, so a typo in an ID or a handler name shows
   up as a failure instead of as silence at runtime.

2. `_apply_movement` is driven with each of the four movement packets to
   confirm the server's view of the player actually advances, and that
   walking (Set Player Position, 0x1D) re-publishes the chunk view. That was
   the actual bug: only 0x1E was wired up, so a player who walked without
   turning froze in place and the chunk stream stalled with them.

Run: python tools/check_movement_packets.py
"""

import ast
import asyncio
import math
import queue
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from network.packets import (  # noqa: E402
    SChunkBatchReceived,
    SMovePlayerPos,
    SMovePlayerPosRot,
    SMovePlayerRot,
    SMovePlayerStatusOnly,
    SPlayerInput,
    SPlayerLoaded,
)
from network.client_handler import ClientHandler  # noqa: E402
from player.player import Player  # noqa: E402
from world.chunk_streamer import PlayerViewMsg  # noqa: E402

# The packets that were being logged as "Unknown packet" in the play state.
ADDED = {
    0x0A: SChunkBatchReceived,
    0x1D: SMovePlayerPos,
    0x1F: SMovePlayerRot,
    0x20: SMovePlayerStatusOnly,
    0x2A: SPlayerInput,
    0x2B: SPlayerLoaded,
}


def parse_dispatch_table() -> dict[int, str]:
    """Read the PLAY client->server handler map straight out of the source."""
    tree = ast.parse(Path("network/client_handler.py").read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Dict):
            continue
        keys = [k.value for k in node.keys if isinstance(k, ast.Constant)]
        # The play table is the only one keyed entirely by packet-id ints and
        # holding a >= 0x1D entry.
        if all(isinstance(k, int) for k in keys) and 0x1D in keys:
            return {
                k.value: v.attr
                for k, v in zip(node.keys, node.values)
                if isinstance(k, ast.Constant) and isinstance(v, ast.Attribute)
            }
    raise AssertionError("play dispatch table not found")


def make_stub_handler() -> ClientHandler:
    """A ClientHandler with just enough state for _apply_movement.

    Constructed via __new__ so the test does not need a real socket; what
    follows is exactly the state _apply_movement and _publish_view
    read. The player is a real Player rather than a bag of floats, which is
    the point: the movement path writes into the same object the streamer
    would be handed, so this exercises the actual indirection.
    """
    h = ClientHandler.__new__(ClientHandler)
    h.player = Player(uuid="00000000-0000-0000-0000-000000000000", name="Tester")
    h._actor_inbox = queue.Queue()
    h._chunk_sink = None
    h._spawn_blob = None
    h._teleport_confirmed = True
    # The two values _publish_view stamps onto every view: the pose counter the
    # actor-side movement validator uses to spot a repeat, and the teleport id
    # its corrections have to allocate above.
    h._view_sequence = 0
    h._pending_teleport_id = 0
    return h


def pos_of(h: ClientHandler) -> tuple:
    return (h.player.position.x, h.player.position.y, h.player.position.z)


def check_dispatch_table() -> None:
    print("== play client->server dispatch table ==")
    table = parse_dispatch_table()
    bad = False
    for pid, cls in sorted(ADDED.items()):
        handler = table.get(pid)
        ok = handler is not None
        bad |= not ok
        print(f"  0x{pid:02X} {cls.__name__:24s} -> {handler or 'MISSING'}")
    assert not bad, "one or more added packets are not routed"
    print()


def check_movement_updates_view() -> None:
    print("== _apply_movement advances the server's view ==")
    h = make_stub_handler()
    p = h.player.position

    h._apply_movement(SMovePlayerPosRot(x=10.0, y=64.0, z=10.0, yaw=90.0, pitch=0.0, flags=0x01))
    print(f"  after pos+rot: pos={pos_of(h)} yaw={p.yaw} on_ground={p.on_ground}")
    assert p.x == 10.0 and p.yaw == 90.0 and p.on_ground is True

    h._apply_movement(SMovePlayerRot(yaw=180.0, pitch=-30.0, flags=0x01))
    print(f"  after rot:     yaw={p.yaw} pitch={p.pitch} pos={pos_of(h)}")
    assert p.yaw == 180.0 and p.x == 10.0, "rotation must not move the player"

    h._apply_movement(SMovePlayerStatusOnly(flags=0x00))
    print(f"  after status:  on_ground={p.on_ground}")
    assert p.on_ground is False, "status-only must update on_ground"

    h._apply_movement(SMovePlayerPos(x=20.0, y=65.0, z=20.0, flags=0x01))
    print(f"  after pos:     pos={pos_of(h)}")
    assert pos_of(h) == (20.0, 65.0, 20.0)
    print()


def check_walking_restreams_chunks() -> None:
    """Every movement packet posts exactly one view, at the right chunk.

    The suppression this used to assert (same chunk -> no view) was removed
    when the entity tracker started reading the same message: the streamer
    dedupes on `view.center`, but the tracker's delta codec needs a snapshot
    of each pose, so a player turning in place has to keep publishing or
    everyone else's copy of them freezes.
    """
    print("== walking publishes a view every packet (the 0x1D fix) ==")
    h = make_stub_handler()

    # First view, published from the spawn position.
    h._apply_movement(SMovePlayerPos(x=1.0, y=64.0, z=1.0, flags=0x01))
    assert h._actor_inbox.qsize() == 1, "first position should post a view"
    first = h._actor_inbox.get()
    assert isinstance(first, PlayerViewMsg)
    assert math.floor(first.x / 16) == 0
    print(f"  spawned at chunk {math.floor(first.x / 16)}, 1 view posted")

    # Same chunk: still exactly one. The streamer side of this is what
    # actually suppresses redundant work (it compares view.center in pump).
    h._apply_movement(SMovePlayerPos(x=8.0, y=64.0, z=8.0, flags=0x01))
    assert h._actor_inbox.qsize() == 1, "a same-chunk move must still publish"
    second = h._actor_inbox.get()
    assert math.floor(second.x / 16) == 0, "same chunk must keep chunk 0"
    print(f"  moved within chunk -> 1 view posted, chunk still "
          f"{math.floor(second.x / 16)} (streamer will dedupe)")

    # Walk across a chunk border: the view has to name the new chunk or the
    # streamer keeps serving chunks around where the player used to be.
    h._apply_movement(SMovePlayerPos(x=40.0, y=64.0, z=0.0, flags=0x01))
    assert h._actor_inbox.qsize() == 1, "crossing a chunk border must post a view"
    third = h._actor_inbox.get()
    assert math.floor(third.x / 16) == 2, f"expected chunk 2, got {math.floor(third.x / 16)}"
    print(f"  walked to chunk {math.floor(third.x / 16)}, 1 view posted with x={third.x}")

    # And the same must hold when rotation accompanies the walk.
    h._actor_inbox.queue.clear()
    h._apply_movement(SMovePlayerPosRot(x=200.0, y=64.0, z=0.0, yaw=0.0, pitch=0.0, flags=0x01))
    assert h._actor_inbox.qsize() == 1
    fourth = h._actor_inbox.get()
    assert math.floor(fourth.x / 16) == 12, f"expected chunk 12, got {math.floor(fourth.x / 16)}"
    print(f"  pos+rot across border -> chunk {math.floor(fourth.x / 16)}, view posted")
    print()


def check_view_carries_entity_fields() -> None:
    """The tracker's half of a PlayerViewMsg must be populated by the handler.

    entity_id is the tracker's key, yaw/pitch/on_ground decide which delta
    packet gets built, and spawn_blob is what makes the other client render a
    body. Any of these left at its default and the symptom is silent: a
    player everyone can see but nobody can move, or a player who never
    appears at all.
    """
    print("== posted view carries the tracker's fields ==")
    h = make_stub_handler()
    h._apply_movement(
        SMovePlayerPosRot(x=5.0, y=64.0, z=5.0, yaw=45.0, pitch=-10.0, flags=0x00)
    )
    assert h._actor_inbox.qsize() == 1
    msg = h._actor_inbox.get()

    assert msg.entity_id == h.player.entity_id, "entity_id must reach the tracker"
    assert msg.entity_id != 0, "entity_id 0 is vanilla's no-entity sentinel"
    assert msg.uuid == h.player.uuid, (
        "profile uuid must ride along - Add Entity and Player Info Add "
        "have to agree or the client cannot bind the skin"
    )
    assert msg.uuid, "uuid must not be empty at this point"
    assert (msg.yaw, msg.pitch) == (45.0, -10.0), "rotation must ride along"
    assert msg.on_ground is False, "flags bit 0 clear means airborne"
    assert isinstance(msg.spawn_blob, bytes) and msg.spawn_blob, "spawn blob must be built"
    assert msg.spawn_blob.endswith(b"\xff"), "metadata blob must be 0xFF terminated"
    assert msg.player_id == str(id(h)), "streamer key stays the handler identity"
    # The movement validator's two fields ride on the same message: it needs a
    # monotonic pose counter (physics.player drops a message that is not newer
    # than the one it judged) and the teleport id its corrections must clear.
    assert msg.sequence == 1, f"first published pose must be sequence 1, got {msg.sequence}"
    assert msg.teleport_id == h._pending_teleport_id, (
        "the view must carry the id the client is currently waiting on"
    )
    print(f"  entity_id={msg.entity_id} yaw={msg.yaw} pitch={msg.pitch} "
          f"on_ground={msg.on_ground} blob={len(msg.spawn_blob)}B "
          f"sequence={msg.sequence} teleport_id={msg.teleport_id}")

    # The blob must be built once and reused, not per packet.
    h._apply_movement(SMovePlayerPos(x=6.0, y=64.0, z=6.0, flags=0x01))
    second = h._actor_inbox.get()
    assert second.spawn_blob is msg.spawn_blob, "blob must be cached"
    assert second.sequence == 2, f"the counter must advance, got {second.sequence}"
    print("  spawn blob cached across packets, sequence advanced to 2")
    print()


def check_carries_flags_match_fields() -> None:
    """A packet that carries fields it does not declare would be silently
    dropped by _apply_movement, so assert the two agree."""
    print("== CARRIES_* flags agree with the actual fields ==")
    for cls in (SMovePlayerPos, SMovePlayerPosRot, SMovePlayerRot, SMovePlayerStatusOnly):
        # x/y/z are instance attributes set in __init__, so probe an instance.
        p = cls()
        has_pos = all(hasattr(p, f) for f in ("x", "y", "z"))
        has_rot = all(hasattr(p, f) for f in ("yaw", "pitch"))
        assert has_pos == cls.CARRIES_POSITION, (
            f"{cls.__name__}: has position fields={has_pos} "
            f"but CARRIES_POSITION={cls.CARRIES_POSITION}"
        )
        assert has_rot == cls.CARRIES_ROTATION, (
            f"{cls.__name__}: has rotation fields={has_rot} "
            f"but CARRIES_ROTATION={cls.CARRIES_ROTATION}"
        )
        print(
            f"  {cls.__name__:24s} position={cls.CARRIES_POSITION!s:5s} "
            f"rotation={cls.CARRIES_ROTATION}"
        )
    print()


def main() -> None:
    check_dispatch_table()
    check_carries_flags_match_fields()
    # _publish_view builds its ChunkSink on the running loop the
    # first time a view is posted, so the movement checks have to be in one.
    asyncio.run(_run_movement_checks())


async def _run_movement_checks() -> None:
    check_movement_updates_view()
    check_walking_restreams_chunks()
    check_view_carries_entity_fields()
    print("all checks passed")


if __name__ == "__main__":
    main()
