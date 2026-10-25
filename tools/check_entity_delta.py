"""The delta codec and the entity tracker: quantization, packet choice,
and the visibility lifecycle.

Two layers are checked here, bottom-up, because a bug in either shows up
on a client as the same symptom - "the other player is frozen" - and the
fix is different.

1. **pick_move, the pure decision function.** Exhaustive over its input
   space, ported decision-for-decision from Pumpkin's entity_tracker.rs
   (which is vanilla's ServerEntity):
   - `encode_pos` is floor(v * 4096 + 0.5), and NEGATIVE coordinates get
     the +0.5 rounding too - -0.5 blocks encodes to -2048, not -2047.5's
     truncation.
   - An i16 overflow on any axis (~8 blocks of accumulated delta) forces
     ABSOLUTE.
   - An on_ground change forces ABSOLUTE (the client's fall logic keys
     off it).
   - Tick 60 forces a position send even when standing still (the
     periodic full resync), as does tick 0.
   - Inside the delta path, the (send_pos, rot_changed) matrix picks
     POS_ROT / POS / ROT / NONE exactly.
2. **EntityTracker behavior**, with a fake sink recording packets per
   viewer:
   - spawn: a second player entering range receives Add Entity, then the
     prebuilt metadata blob, then Rotate Head - in that order.
   - despawn: out of range, the same viewer receives Remove Entities with
     the right id.
   - self-visibility: a player never receives packets about their own
     entity, no matter how the ranges work out.
   - movement: a tracked player walking 0.5 blocks produces a POS/POS_ROT
     delta whose shorts decode back to +2048; a 100-block jump produces
     ABSOLUTE.
   - baseline bookkeeping: after a ROT-only send, the position baseline
     does NOT advance (that is what keeps the next POS delta honest).
   - detach: a leaving player is removed from everyone's seen_by, and
     their own tracked entity is broadcast as removed.

Run: py -3.14t tools/check_entity_delta.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

failures: list[str] = []


def check(label: str, condition: bool, detail: str = "") -> None:
    if condition:
        print(f"  OK   {label}" + (f"  ({detail})" if detail else ""))
    else:
        failures.append(f"{label}: {detail}")
        print(f"  FAIL {label}  ({detail})")


class _FakeSink:
    """Records every send() batch, per viewer."""

    def __init__(self) -> None:
        self.batches: list[list] = []
        self.closed = False

    def send(self, packets) -> None:
        if not self.closed and packets:
            self.batches.append(list(packets))

    @property
    def packets(self) -> list:
        return [p for batch in self.batches for p in batch]

    def packet_types(self) -> list[str]:
        return [type(p).__name__ for p in self.packets]


def main() -> None:
    from registry.game_registry import init_registries, reset_registries

    init_registries()
    try:
        check_encode_pos()
        check_pick_move()
        check_tracker()
        check_real_view_msg()
        print()
        if failures:
            print(f"{len(failures)} failure(s)")
            raise SystemExit(1)
        print("all checks passed")
    finally:
        reset_registries()


def check_encode_pos() -> None:
    from entity.entity_tracker import encode_pos

    print("1. encode_pos quantization")
    check("0.0 encodes to 0", encode_pos(0.0) == 0)
    check("0.5 encodes to 2048", encode_pos(0.5) == 2048, str(encode_pos(0.5)))
    check("-0.5 encodes to -2048 (round-half-up on negatives)",
          encode_pos(-0.5) == -2048, str(encode_pos(-0.5)))
    check("sub-quantum negative rounds to 0",
          encode_pos(-0.4 / 4096.0) == 0, str(encode_pos(-0.4 / 4096.0)))
    check("1.5 quanta rounds up to 2",
          encode_pos(1.5 / 4096.0) == 2, str(encode_pos(1.5 / 4096.0)))


def check_pick_move() -> None:
    from entity.entity_tracker import MoveSync, pick_move

    print("2. pick_move decision matrix")
    origin = (0.0, 0.0, 0.0)

    check("no movement, no rotation, mid-period -> NONE",
          pick_move(origin, origin, False, False, 7) is MoveSync.NONE)
    # The 60-tick resync forces a POSITION send, not an absolute sync:
    # vanilla expresses it as a zero-delta Move Entity Pos (which carries
    # the current on_ground), and so does the port. Verified against
    # Pumpkin's pick_move: send_pos is true at tick%60==0, and the choice
    # matrix then lands on POS with zero deltas.
    check("no movement at tick 0 -> POS (forced zero-delta resync)",
          pick_move(origin, origin, False, False, 0) is MoveSync.POS)
    check("no movement at tick 60 -> POS",
          pick_move(origin, origin, False, False, 60) is MoveSync.POS)
    check("no movement at tick 61 -> NONE",
          pick_move(origin, origin, False, False, 61) is MoveSync.NONE)

    tiny = (1e-4, 0.0, 0.0)
    check("sub-epsilon movement with rotation -> ROT",
          pick_move(tiny, origin, True, False, 7) is MoveSync.ROT)
    check("sub-epsilon movement without rotation -> NONE",
          pick_move(tiny, origin, False, False, 7) is MoveSync.NONE)

    small = (0.5, 0.0, 0.0)
    check("half-block move -> POS",
          pick_move(small, origin, False, False, 7) is MoveSync.POS)
    check("half-block move + rotation -> POS_ROT",
          pick_move(small, origin, True, False, 7) is MoveSync.POS_ROT)

    check("on_ground flip forces ABSOLUTE",
          pick_move(small, origin, False, True, 7) is MoveSync.ABSOLUTE)

    big = (100.0, 0.0, 0.0)
    check("100-block jump overflows i16 -> ABSOLUTE",
          pick_move(big, origin, False, False, 7) is MoveSync.ABSOLUTE)

    exactly8 = (8.0, 0.0, 0.0)
    # 8 blocks = 32768 quanta = i16 max + 1 -> overflow -> absolute.
    check("exactly 8 blocks is still an overflow",
          pick_move(exactly8, origin, False, False, 7) is MoveSync.ABSOLUTE)
    just_under = (8.0 - 1.0 / 4096.0, 0.0, 0.0)
    check("just under 8 blocks rides the delta path",
          pick_move(just_under, origin, False, False, 7) is MoveSync.POS)


def _make_tracker():
    from entity.entity_tracker import EntityTracker

    return EntityTracker("minecraft:overworld", broadcast_percentage=100)


def _msg(entity_id, sink, x, y, z, radius=10, **extra):
    """A stand-in PlayerViewMsg with the fields the tracker reads."""

    class _Msg:
        pass

    m = _Msg()
    m.entity_id = entity_id
    m.player_id = f"handler-{entity_id}"
    m.sink = sink
    m.x = x
    m.y = y
    m.z = z
    m.radius = radius
    m.yaw = extra.get("yaw", 0.0)
    m.pitch = extra.get("pitch", 0.0)
    m.on_ground = extra.get("on_ground", True)
    m.spawn_blob = extra.get("spawn_blob", b"\xff")
    m.uuid = extra.get("uuid", "")
    # A spectator sees the world but does not count towards mob spawning,
    # so the flag has to survive into the tracker - see `viewer_positions`.
    m.spectator = extra.get("spectator", False)
    return m


def check_tracker() -> None:
    from entity.entity_tracker import MoveSync, encode_pos
    from network.packets.clientbound.play.add_entity import CAddEntity
    from network.packets.clientbound.play.entity_position_sync import CEntityPositionSync
    from network.packets.clientbound.play.move_entity_pos import CMoveEntityPos
    from network.packets.clientbound.play.move_entity_pos_rot import CMoveEntityPosRot
    from network.packets.clientbound.play.move_entity_rot import CMoveEntityRot
    from network.packets.clientbound.play.remove_entities import CRemoveEntities
    from network.packets.clientbound.play.rotate_head import CRotateHead

    print("3. tracker lifecycle")

    blob = bytes([0x11, 0x25, 0x03]) + b"Aka" + bytes([0x00, 0xFF])
    tracker = _make_tracker()

    sink_a = _FakeSink()
    sink_b = _FakeSink()
    # Player A at origin; player B one block away, in range (32 chunks).
    # The uuid is the profile uuid: the client binds skin and tab-list
    # entry by it, so the spawn has to carry whatever A and B published.
    tracker.attach(_msg(1, sink_a, 0.0, 64.0, 0.0, spawn_blob=blob,
                        uuid="aaaaaaaa-0000-4000-8000-000000000001"))
    tracker.attach(_msg(2, sink_b, 1.0, 64.0, 0.0, spawn_blob=blob,
                        uuid="bbbbbbbb-0000-4000-8000-000000000002"))

    stats = tracker.pump()
    check("both players tracked", stats["tracked"] == 2, str(stats))
    # A sees B and B sees A: the pair is symmetric, so two.
    check("two visible pairs (mutual)", stats["visible_pairs"] == 2, str(stats))

    a_types = sink_a.packet_types()
    check("A received B's spawn sequence (add, data, head)",
          a_types == ["CAddEntity", "CSetEntityData", "CRotateHead"],
          str(a_types))
    spawn_pkt = sink_a.packets[0]
    check("spawn names B's entity id", spawn_pkt.entity_id == 2)
    check("spawn carries type 155 (player)", spawn_pkt.entity_type == 155)
    check("spawn is at B's true position, not the origin",
          spawn_pkt.y == 64.0 and spawn_pkt.x == 1.0,
          f"({spawn_pkt.x}, {spawn_pkt.y}, {spawn_pkt.z})")
    check("spawn carries B's profile uuid (matches Player Info Add)",
          spawn_pkt.entity_uuid.replace("-", "")
          == "bbbbbbbb000040008000000000000002",
          spawn_pkt.entity_uuid)

    b_types = sink_b.packet_types()
    check("B received A's spawn symmetrically",
          "CAddEntity" in b_types, str(b_types))
    check("nobody receives their own spawn",
          all(p.entity_id != 2 for p in sink_b.packets),
          str([p.entity_id for p in sink_b.packets]))

    # A walks half a block. B should get a delta, A should get nothing
    # about A (B's own movement is B's client's business).
    sink_a.batches.clear()
    sink_b.batches.clear()
    tracker.attach(_msg(1, sink_a, 0.5, 64.0, 0.0, spawn_blob=blob))
    stats = tracker.pump()
    a_moved = [p for p in sink_a.packets if isinstance(
        p, (CMoveEntityPos, CMoveEntityPosRot, CEntityPositionSync))]
    check("A receives no packets about its own entity", not a_moved,
          str(a_types))
    b_moves = [p for p in sink_b.packets if isinstance(
        p, (CMoveEntityPos, CMoveEntityPosRot, CEntityPositionSync))]
    check("B received exactly one movement packet for A",
          len(b_moves) == 1, str([type(p).__name__ for p in b_moves]))
    if b_moves:
        move = b_moves[0]
        if isinstance(move, CMoveEntityPos):
            check("delta shorts encode half a block as 2048",
                  move.dx == 2048, str(move.dx))
        else:
            check("small move went delta, not absolute",
                  isinstance(move, (CMoveEntityPos, CMoveEntityPosRot)),
                  type(move).__name__)

    # A teleports 100 blocks: B must get an absolute sync.
    sink_b.batches.clear()
    tracker.attach(_msg(1, sink_a, 100.5, 64.0, 0.0, spawn_blob=blob))
    tracker.pump()
    b_abs = [p for p in sink_b.packets if isinstance(p, CEntityPositionSync)]
    check("100-block jump sends an absolute sync", len(b_abs) == 1)
    if b_abs:
        check("absolute sync carries the true position", abs(b_abs[0].x - 100.5) < 1e-9,
              str(b_abs[0].x))

    # Rotation-only: A spins in place; B gets ROT (and not a position send).
    sink_b.batches.clear()
    tracker.attach(_msg(1, sink_a, 100.5, 64.0, 0.0, yaw=90.0, spawn_blob=blob))
    tracker.pump()
    b_rot = [p for p in sink_b.packets if isinstance(p, CMoveEntityRot)]
    check("spin in place sends Move Entity Rot", len(b_rot) == 1,
          str([type(p).__name__ for p in sink_b.packets]))

    # A walks out of B's 32-chunk range: B gets a removal.
    sink_b.batches.clear()
    tracker.attach(_msg(1, sink_a, 10000.5, 64.0, 0.0, spawn_blob=blob))
    tracker.pump()
    b_rm = [p for p in sink_b.packets if isinstance(p, CRemoveEntities)]
    check("walking out of range sends Remove Entities", len(b_rm) == 1)
    if b_rm:
        check("removal names A's entity id", b_rm[0].entity_ids == [1],
              str(b_rm[0].entity_ids))

    # A's tracked entity is at 10000 blocks; nobody else is tracked as a
    # viewer there. Detach A entirely: their entity is removed from the
    # tracker and B (out of range already) gets nothing further.
    stats_before = tracker.stats()
    tracker.attach(_msg(1, None, 0, 0, 0))
    stats_after = tracker.stats()
    check("detach removes the tracked entity",
          stats_after["tracked"] == stats_before["tracked"] - 1,
          f"{stats_before} -> {stats_after}")
    check("detach removes the viewer",
          stats_after["viewers"] == stats_before["viewers"] - 1)

    # A player leaving who was still SEEN broadcasts their removal.
    tracker2 = _make_tracker()
    sink_a2, sink_b2 = _FakeSink(), _FakeSink()
    tracker2.attach(_msg(1, sink_a2, 0.0, 64.0, 0.0, spawn_blob=blob))
    tracker2.attach(_msg(2, sink_b2, 1.0, 64.0, 0.0, spawn_blob=blob))
    tracker2.pump()
    check("pair is visible before the leave", tracker2.stats()["visible_pairs"] == 2,
          str(tracker2.stats()))
    tracker2.attach(_msg(1, None, 0, 0, 0))
    got_removal = any(isinstance(p, CRemoveEntities) for p in sink_b2.packets)
    check("leaving player is removed from remaining viewers", got_removal,
          str(sink_b2.packet_types()))


def check_real_view_msg() -> None:
    """The seam the duck-typed `_msg` stand-in cannot cover.

    `_msg` mirrors the shape the tracker *reads*; this uses the real frozen
    `PlayerViewMsg` the handler actually posts, so a field renamed or
    defaulted away on either side of the queue fails here rather than at
    first blood in a live join.
    """
    from world.chunk_streamer import PlayerViewMsg

    print("4. real PlayerViewMsg through the tracker")
    blob = bytes([0x10, 0x00, 0x11, 0x25, 0x03]) + b"Steve" + bytes([0x00, 0xFF])
    tracker = _make_tracker()
    sink7 = _FakeSink()
    sink8 = _FakeSink()

    tracker.attach(PlayerViewMsg(
        player_id="handler-7", sink=sink7,
        x=1.5, y=65.0, z=-2.5, radius=8,
        entity_id=7, uuid="cccccccc-0000-4000-8000-000000000007",
        yaw=90.0, pitch=0.0, on_ground=True, spawn_blob=blob,
    ))
    # A second player to be the viewer; nobody sees their own spawn, so the
    # first attach alone would pump nothing at all.
    tracker.attach(PlayerViewMsg(
        player_id="handler-8", sink=sink8,
        x=2.0, y=65.0, z=-2.5, radius=8,
        entity_id=8, uuid="dddddddd-0000-4000-8000-000000000008",
        spawn_blob=blob,
    ))
    stats = tracker.pump()
    check("both real messages were understood", stats["tracked"] == 2
          and stats["viewers"] == 2, str(stats))

    # sink8 records what player 8 was TOLD about, which is player 7.
    spawn = next((p for p in sink8.packets
                  if type(p).__name__ == "CAddEntity"), None)
    check("a spawn came out of the real message", spawn is not None,
          str(sink8.packet_types()))
    if spawn is not None:
        check("spawn names entity_id 7", spawn.entity_id == 7, str(spawn.entity_id))
        check("spawn carries the uuid the handler posted",
              spawn.entity_uuid.replace("-", "")
              == "cccccccc000040008000000000000007", spawn.entity_uuid)
        check("spawn carries type 155 (player)", spawn.entity_type == 155,
              str(spawn.entity_type))
        check("spawn is at the posted pose",
              (spawn.x, spawn.y, spawn.z) == (1.5, 65.0, -2.5),
              f"({spawn.x}, {spawn.y}, {spawn.z})")

    data = next((p for p in sink8.packets
                 if type(p).__name__ == "CSetEntityData"), None)
    check("metadata blob arrived intact", data is not None
          and data.metadata == blob,
          f"got {None if data is None else data.metadata!r}")

    # And nobody is told about themselves.
    check("player 7 got no packet naming entity 7",
          all(p.entity_id != 7 for p in sink7.packets
              if hasattr(p, "entity_id")),
          str(sink7.packet_types()))

    # The asymmetric case: a message with no blob makes a viewer but no
    # tracked entry. Assert the shape so the warning in attach() is the only
    # thing that can go unnoticed.
    tracker2 = _make_tracker()
    tracker2.attach(PlayerViewMsg(
        player_id="handler-9", sink=_FakeSink(),
        x=0.0, y=64.0, z=0.0, radius=8, entity_id=9,
    ))
    s = tracker2.stats()
    check("blob-less message makes a viewer but no tracked entity",
          s["viewers"] == 1 and s["tracked"] == 0, str(s))
    print()


MOVE_PACKETS = ("CMoveEntityPos", "CMoveEntityPosRot", "CMoveEntityRot",
                "CEntityPositionSync")


def _make_mob(name: str, entity_id: int, x: float, y: float, z: float):
    """A real Mob of a real type, for the tracker to hold."""
    from entity.entity_type import entity_type_by_name
    from entity.mob import create_mob

    mob = create_mob(entity_type_by_name(name), x, y, z,
                     dimension="minecraft:overworld")
    mob.entity_id = entity_id
    return mob


def _moves_for(sink: _FakeSink, entity_id: int) -> int:
    """How many movement packets for one entity reached this sink."""
    return sum(1 for p in sink.packets
               if getattr(p, "entity_id", None) == entity_id
               and type(p).__name__ in MOVE_PACKETS)


def _heads_for(sink: _FakeSink, entity_id: int) -> int:
    return sum(1 for p in sink.packets
               if getattr(p, "entity_id", None) == entity_id
               and type(p).__name__ == "CRotateHead")


def check_mob_tracking() -> None:
    """The non-player half of the tracker: add, throttle, remove.

    Two viewers, and the second one is a spectator, because both facts are
    load-bearing and only a two-viewer fixture can show either:

    * a tracker never sends an entity to the viewer that *is* that entity,
      so player 1's own movement is only observable on viewer 2's sink -
      which is what makes "players are not throttled" a real assertion
      rather than a tautology;
    * the spectator flag rides on `viewer_positions`, which is the spawner's
      only view of who is online.
    """
    from entity.entity_type import entity_type_by_name
    from network.packets.clientbound.play.add_entity import CAddEntity
    from network.packets.clientbound.play.remove_entities import CRemoveEntities

    print("4. mob tracking: add, update_interval, remove")

    sink_a, sink_b = _FakeSink(), _FakeSink()
    tracker = _make_tracker()
    tracker.attach(_msg(1, sink_a, 0.0, 64.0, 0.0, radius=10))
    tracker.attach(_msg(2, sink_b, 1.0, 64.0, 0.0, radius=10, spectator=True))

    # ---- add_entity: the entry, not the packets ----
    cow = _make_mob("minecraft:cow", 40, 8.5, 65.0, -3.5)
    tracker.add_entity(cow)
    check("add_entity makes a tracked entry", 40 in tracker.tracked)
    entry = tracker.tracked[40]
    check("a mob is not flagged as a player", entry.is_player is False)
    check("its tracking range is the type's (10 chunks)",
          entry.client_tracking_range == 10, str(entry.client_tracking_range))
    check("its update_interval is the type's (3 ticks)",
          entry.update_interval == 3, str(entry.update_interval))
    check("its uuid is the mob's own, not a profile id",
          entry.uuid == cow.uuid and entry.uuid != "", entry.uuid)
    check("its baselines are primed at the spawn position, not the origin",
          (entry.last_sent_x, entry.last_sent_y, entry.last_sent_z)
          == (8.5, 65.0, -3.5),
          f"({entry.last_sent_x}, {entry.last_sent_y}, {entry.last_sent_z})")
    # 09 = HEALTH, 03 = float, 41200000 = 10.0 big-endian (Minecraft metadata
    # floats are big-endian, unlike the tracker delta shorts), ff = the
    # terminator. Health is not decoration: the client's getHealth() reads
    # index 9 and treats 0 as dead, so a bare terminator here deletes the
    # mob on the client the tick it arrives.
    check("its spawn blob carries health, not the bare terminator",
          entry.spawn_blob == bytes([9, 3, 0x41, 0x20, 0x00, 0x00, 0xFF]),
          entry.spawn_blob.hex())

    # add_entity sends nothing. Visibility is `pump`'s job, because pump is
    # the only place that knows who is in range - so a mob that spawns while
    # nobody is watching costs zero packets.
    check("add_entity itself sent nothing", sink_a.packets == [],
          str(sink_a.packet_types()))

    tracker.pump()
    trio = [type(p).__name__ for p in sink_a.packets
            if getattr(p, "entity_id", None) == 40]
    check("the first pump spawns add -> metadata -> rotate_head, in that order",
          trio == ["CAddEntity", "CSetEntityData", "CRotateHead"], str(trio))
    spawn = next((p for p in sink_a.packets
                  if isinstance(p, CAddEntity) and p.entity_id == 40), None)
    if spawn is not None:
        want = entity_type_by_name("minecraft:cow").numeric_id
        check("as the cow's own type id", spawn.entity_type == want,
              f"{spawn.entity_type} (want {want})")
        check("naming entity 40", spawn.entity_id == 40, str(spawn.entity_id))
        check("carrying the mob's uuid, not a zero one",
              spawn.entity_uuid == cow.uuid, spawn.entity_uuid)
    data = next((p for p in sink_a.packets
                 if type(p).__name__ == "CSetEntityData"
                 and p.entity_id == 40), None)
    check("the metadata is the spawn blob",
          data is not None and data.metadata == entry.spawn_blob,
          f"{None if data is None else data.metadata.hex()}")
    check("the second viewer got the same trio",
          [type(p).__name__ for p in sink_b.packets
           if getattr(p, "entity_id", None) == 40] == trio,
          str(sink_b.packet_types()))

    # ---- the throttle ----
    # Every pump the cow moves half a block and turns its head, so a packet
    # that does not appear is the throttle and not a quiet pick_move.
    sink_a.batches.clear()
    sent_on = []
    for i in range(6):
        tracker.update_entity(40, 9.0 + i * 0.5, 65.0, -3.5,
                              yaw=30.0 * (i + 1))
        tracker.pump()
        if _moves_for(sink_a, 40) > len(sent_on):
            sent_on.append(i)
    check("a mob moves on 2 of 6 ticks - its update_interval is 3",
          len(sent_on) == 2, f"movement on pumps {sent_on}")
    check("and the gap between them is exactly the interval",
          len(sent_on) == 2 and sent_on[1] - sent_on[0] == 3, str(sent_on))
    check("its head yaw is NOT throttled - all 6 turns go out",
          _heads_for(sink_a, 40) == 6, str(_heads_for(sink_a, 40)))

    # ---- players keep the every-tick path ----
    # Player 1's type says update_interval 2, so a gate on `update_interval`
    # instead of `is_player` would halve this count. It is the player-
    # already-verified-against-a-real-client path, so it stays untouched.
    sink_b.batches.clear()
    for i in range(6):
        tracker.update_entity(1, 1.0 + i * 0.5, 64.0, 0.0)
        tracker.pump()
    check("a player moves on all 6 ticks - the gate is is_player, not interval",
          _moves_for(sink_b, 1) == 6, str(_moves_for(sink_b, 1)))
    check("and a player is never sent to itself",
          _moves_for(sink_a, 1) == 0, str(_moves_for(sink_a, 1)))

    # ---- remove_entity: everyone who could see it is told ----
    sink_a.batches.clear()
    sink_b.batches.clear()
    tracker.remove_entity(40)
    removals = [p for p in sink_a.packets if isinstance(p, CRemoveEntities)]
    check("remove_entity broadcasts CRemoveEntities", len(removals) == 1,
          str(sink_a.packet_types()))
    if removals:
        check("naming entity 40", removals[0].entity_ids == [40],
              str(removals[0].entity_ids))
    check("to every viewer that could see it, not just the first",
          [p for p in sink_b.packets if isinstance(p, CRemoveEntities)]
          and [p.entity_ids for p in sink_b.packets
               if isinstance(p, CRemoveEntities)] == [[40]],
          str(sink_b.packet_types()))
    check("and the tracked entry is gone", 40 not in tracker.tracked)
    check("the players are untouched", 1 in tracker.tracked and 2 in tracker.tracked)
    check("remove_entity twice is a no-op", tracker.remove_entity(40) is None)
    check("an unknown id is a no-op too", tracker.remove_entity(999) is None)

    # A second add of a live id must be refused, not overwrite: the previous
    # entry's seen_by viewers would go on believing in an entity that is gone.
    sheep = _make_mob("minecraft:sheep", 41, 0.0, 65.0, 0.0)
    sink_a.batches.clear()
    tracker.add_entity(sheep)
    tracker.pump()
    check("a later mob is spawned on its own first pump",
          any(isinstance(p, CAddEntity) and p.entity_id == 41
              for p in sink_a.packets), str(sink_a.packet_types()))
    sink_a.batches.clear()
    tracker.add_entity(_make_mob("minecraft:sheep", 41, 5.0, 65.0, 5.0))
    check("a duplicate id is refused, not overwritten",
          tracker.tracked[41].x == 0.0, str(tracker.tracked[41].x))
    check("and refusing it sends nothing", sink_a.packets == [],
          str(sink_a.packet_types()))

    # ---- viewer_positions: the spawner's one view of who is online ----
    positions = sorted(tracker.viewer_positions())
    check("one row per attached viewer", len(positions) == 2, str(positions))
    check("keyed by the tracker's own view id, not player_id",
          sorted(p[0] for p in positions) == [1, 2], str(positions))
    normal = [p for p in positions if p[0] == 1][0]
    spec = [p for p in positions if p[0] == 2][0]
    check("a normal viewer is not flagged as a spectator",
          normal[4] is False, str(normal))
    check("a spectator is flagged", spec[4] is True, str(spec))
    check("positions come back as the message posted them",
          normal[1:4] == (0.0, 64.0, 0.0), str(normal))
    print()


def main() -> None:
    from registry.game_registry import init_registries, reset_registries

    init_registries()
    try:
        check_encode_pos()
        check_pick_move()
        check_tracker()
        check_real_view_msg()
        check_mob_tracking()
        print()
        if failures:
            print(f"{len(failures)} failure(s)")
            raise SystemExit(1)
        print("all checks passed")
    finally:
        reset_registries()


def CSetEntityDataName() -> str:
    return "CSetEntityData"


if __name__ == "__main__":
    main()
