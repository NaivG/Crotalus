"""
Entity tracker - who can see whom, and the delta codec that moves them.

Owned by the dimension actor thread, alongside the chunk streamer it was
modelled on (`world/chunk_streamer.py`). Both exist because the same
question has two answers: the chunk streamer sends *terrain* to a player
based on where the player is; the tracker sends *entities* to a player
based on where both are.

Thread ownership is the design, stated once: this object is constructed on
and only ever touched by its dimension actor thread. The event loop's only
involvement is `PlayerViewMsg` snapshots arriving through the actor inbox
and `ChunkSink.send` going back out. No locks anywhere.

**Baselines live here, not on the Entity.** Pumpkin stores
`last_sent_pos/yaw/pitch` on the Entity itself, which works because its
entities are single-threaded per world. Here a player's `Entity` belongs to
the event loop while its observer bookkeeping belongs to this thread, so
the delta baselines are observer state and live on `TrackedEntity`. They
are still shared across all viewers (one delta computation per entity per
tick, broadcast to everyone), which is the part worth keeping from Pumpkin.

**Players are tracked, not special-cased.** Pumpkin skips `send_changes`
for player entities because it broadcasts player movement inline in the
movement-packet handler instead. Crotalus has no such inline path - the
handler's contract ends at updating the Player and posting a view message
- so the tracker moves players like any other entity. This is vanilla's
`ServerEntity` shape and it changes nothing on the wire.

The visibility rule is vanilla's, plus the one config knob this server
already shipped:

    range_blocks = min(type.client_tracking_range, player view distance)
                   * 16 * entity_broadcast_range_percentage / 100

A player is never sent packets about their own entity: their client
authors that entity's movement itself, and a spurious spawn/move would
fight the client's prediction.
"""

from __future__ import annotations

import enum
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from loguru import logger

# Vanilla's movement threshold: a movement below this squared length is
# "standing still" (2^-17, the constant vanilla's ServerEntity compares
# against). Not f32.EPSILON despite resembling it.
_MOVE_EPSILON_SQ = 7.6293945e-6

# The delta codec's fixed point: positions are quantized to 1/4096 blocks
# and the per-axis difference must fit a signed short (~8 blocks). Anything
# larger, and the codec falls back to an absolute sync.
_POSITION_SCALE = 4096.0
_I16_MIN = -32768
_I16_MAX = 32767

# Vanilla forces a full absolute resync this often, so a long series of
# accumulated small deltas can never drift the client's belief far from
# the server's truth.
_FORCE_ABSOLUTE_INTERVAL = 60


def _pack_angle(degrees: float) -> int:
    """A yaw/pitch as the one byte the wire carries.

    Indirected so that "did this angle change" is decided by exactly one
    function. It is now called from three places - the move decision, the
    throttled path and the head-yaw path - and two of them must agree, or a
    mob's head would resend a rotation that its body has already stopped
    tracking, forever.
    """
    from network.buffer import ByteBuffer

    return ByteBuffer.pack_angle(degrees)


def encode_pos(value: float) -> int:
    """Vanilla `VecDeltaCodec.encode`: floor(value * 4096 + 0.5).

    The +0.5 makes it round-half-up, matching Java's Math.round on the
    scaled value; floor() on its own would bias every negative coordinate
    down by half a quantum.
    """
    return math.floor(value * _POSITION_SCALE + 0.5)


class MoveSync(enum.Enum):
    """Which packet shape describes this entity's movement this tick."""

    POS = "pos"            # 0x33, position delta only
    ROT = "rot"            # 0x36, rotation only
    POS_ROT = "pos_rot"    # 0x34, both deltas
    ABSOLUTE = "absolute"  # 0x23, full resync
    NONE = "none"          # nothing worth sending

    @property
    def sends_position(self) -> bool:
        return self is not MoveSync.ROT and self is not MoveSync.NONE

    @property
    def sends_rotation(self) -> bool:
        return self is not MoveSync.POS and self is not MoveSync.NONE


def pick_move(pos: Tuple[float, float, float],
              last_sent: Tuple[float, float, float],
              rot_changed: bool,
              ground_changed: bool,
              tick: int) -> MoveSync:
    """Vanilla `ServerEntity`'s packet choice, as a pure function.

    Order of decisions, ported from Pumpkin's `pick_move`
    (entity_tracker.rs): the delta path requires the per-axis 1/4096
    difference to fit an i16 and `on_ground` to be unchanged - either
    failing forces an absolute sync. Within the delta path, send position
    when it moved beyond the epsilon (or the 60-tick forced resync is due)
    and rotation when the packed angles differ.

    Two of Pumpkin's preconditions are absent by design this round:
    `teleport_delay` (nothing teleports entities yet - the join teleport
    happens before any viewer exists) and `was_riding` (no vehicles).
    Both belong in the `ground_changed` slot of the precondition chain
    when they arrive.
    """
    dx = pos[0] - last_sent[0]
    dy = pos[1] - last_sent[1]
    dz = pos[2] - last_sent[2]
    moved_sq = dx * dx + dy * dy + dz * dz
    send_pos = moved_sq >= _MOVE_EPSILON_SQ or tick % _FORCE_ABSOLUTE_INTERVAL == 0

    delta_fits = all(
        _I16_MIN <= encode_pos(n) - encode_pos(b) <= _I16_MAX
        for n, b in zip(pos, last_sent)
    )

    if delta_fits and not ground_changed:
        if send_pos and rot_changed:
            return MoveSync.POS_ROT
        if send_pos:
            return MoveSync.POS
        if rot_changed:
            return MoveSync.ROT
        return MoveSync.NONE
    return MoveSync.ABSOLUTE


@dataclass
class _TrackedEntity:
    """One entity as the tracker sees it: a wire-side snapshot plus the
    delta baselines and who is watching.

    This is a projection, not the Entity object - for players it is fed by
    PlayerViewMsg snapshots, precisely so the actor thread never touches
    the event loop's Player.
    """

    entity_id: int
    entity_type_name: str
    client_tracking_range: int
    is_player: bool

    # Prebuilt Set Entity Data payload sent right after the spawn packet.
    spawn_blob: bytes = b"\xff"

    # The persistent id, as opposed to `entity_id`. For a player this is the
    # profile UUID, and it MUST match the one sent in Player Info Add: the
    # client looks a remote player's skin and name up by UUID, so a spawn
    # carrying a different value than the tab list silently renders a
    # nobody. Empty means "unassigned" and falls back to the zero UUID.
    uuid: str = ""

    # Live snapshot.
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    yaw: float = 0.0
    pitch: float = 0.0
    on_ground: bool = True

    # Delta baselines, as of the last packet each viewer family received.
    # Angles are stored packed (the byte the wire sees) because that is the
    # resolution "did it change" is decided at - a 0.1-degree turn the wire
    # cannot express must not wake the packet choice.
    last_sent_x: float = 0.0
    last_sent_y: float = 0.0
    last_sent_z: float = 0.0
    last_sent_yaw: int = 0
    last_sent_pitch: int = 0
    last_sent_head_yaw: int = 0

    # on_ground as of the last ABSOLUTE sync: the delta path is only legal
    # while this equals the current value.
    was_on_ground: bool = True

    # Viewer entity ids currently seeing this entity.
    seen_by: Set[int] = field(default_factory=set)

    tick_count: int = 0

    # Ticks between automatic position syncs, from the type's
    # `update_interval`. Applied to non-players only - see `pump`.
    update_interval: int = 1

    def note_snapshot(self, x: float, y: float, z: float,
                      yaw: float, pitch: float, on_ground: bool) -> None:
        self.x = x
        self.y = y
        self.z = z
        self.yaw = yaw
        self.pitch = pitch
        self.on_ground = on_ground

    def prime_baselines(self) -> None:
        """Start the delta story from the current snapshot, so the first
        pump does not replay the whole spawn position as a "move"."""
        from network.buffer import ByteBuffer

        self.last_sent_x = self.x
        self.last_sent_y = self.y
        self.last_sent_z = self.z
        self.last_sent_yaw = ByteBuffer.pack_angle(self.yaw)
        self.last_sent_pitch = ByteBuffer.pack_angle(self.pitch)
        self.last_sent_head_yaw = ByteBuffer.pack_angle(self.yaw)
        self.was_on_ground = self.on_ground


@dataclass
class _Viewer:
    """One player's socket and position, as seen from this thread."""

    entity_id: int
    sink: object  # world.chunk_streamer.ChunkSink, duck-typed to avoid a cycle
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    radius: int = 10  # view distance in chunks
    spectator: bool = False
    """Whether this client is in spectator mode.

    A spectator is a viewer - they see everything in range - but they are
    not a spawn trigger and they hold no local mob cap, which is why the
    spawner filters on this rather than on `radius`. Carried as a plain
    field rather than re-read from the `Player`, because the tracker is
    deliberately never handed a mutable cross-thread object.
    """

    def note(self, x: float, y: float, z: float, radius: int,
             spectator: bool = False) -> None:
        self.x = x
        self.y = y
        self.z = z
        self.radius = radius
        self.spectator = spectator


class EntityTracker:
    """The per-dimension entity tracking table."""

    def __init__(self, dimension_name: str,
                 broadcast_percentage: int = 100) -> None:
        self.dimension_name = dimension_name
        self.broadcast_percentage = max(0, min(10000, broadcast_percentage))
        self.tracked: Dict[int, _TrackedEntity] = {}
        self.views: Dict[int, _Viewer] = {}
        # entity ids we have already complained about for a missing blob, so
        # the complaint is once per join rather than twenty times a second.
        self._blob_warned: Set[int] = set()

    # ----- inbound (from the event loop, via the actor inbox) -----

    def attach(self, msg) -> None:
        """A PlayerViewMsg: a player attached, moved, or detached.

        `msg.sink is None` means detach. Detaching a player both stops them
        seeing things and removes *them* from everyone else's view - the
        two halves of leaving, one message.
        """
        if msg.sink is None:
            self._detach(msg.entity_id)
            return

        viewer = self.views.get(msg.entity_id)
        if viewer is None:
            viewer = _Viewer(entity_id=msg.entity_id, sink=msg.sink)
            self.views[msg.entity_id] = viewer
        viewer.note(msg.x, msg.y, msg.z, msg.radius,
                    getattr(msg, "spectator", False))

        tracked = self.tracked.get(msg.entity_id)
        if tracked is None:
            blob = getattr(msg, "spawn_blob", None)
            if blob is not None:
                self._add_tracked_player(
                    msg.entity_id, blob,
                    getattr(msg, "uuid", ""),
                    msg.x, msg.y, msg.z,
                    getattr(msg, "yaw", 0.0), getattr(msg, "pitch", 0.0),
                    getattr(msg, "on_ground", True),
                )
            else:
                # Asymmetric state, and a confusing one: the player gets a
                # viewer (they can see everyone) but no tracked entry (nobody
                # can see them). The only producer of this message fills the
                # blob in, so reaching here means a second producer was added
                # and forgot. Say so rather than let "why can nobody see
                # me?" become a bug hunt.
                if msg.entity_id not in self._blob_warned:
                    self._blob_warned.add(msg.entity_id)
                    logger.warning(
                        f"[{self.dimension_name}] PlayerViewMsg for entity "
                        f"{msg.entity_id} has no spawn_blob; that player will "
                        f"be visible to nobody until it is supplied"
                    )
        else:
            tracked.note_snapshot(
                msg.x, msg.y, msg.z,
                getattr(msg, "yaw", 0.0), getattr(msg, "pitch", 0.0),
                getattr(msg, "on_ground", True),
            )

    def _add_tracked_player(self, entity_id: int, spawn_blob: bytes,
                            uuid: str,
                            x: float, y: float, z: float,
                            yaw: float, pitch: float, on_ground: bool) -> None:
        from entity import entity_type_by_name

        player_type = entity_type_by_name("minecraft:player")
        tracked = _TrackedEntity(
            entity_id=entity_id,
            entity_type_name=player_type.name,
            client_tracking_range=player_type.client_tracking_range,
            is_player=True,
            spawn_blob=spawn_blob,
            uuid=uuid,
        )
        # Snapshot AND baselines in the same breath: a tracked entity whose
        # baselines start at the origin would broadcast its whole spawn
        # position as a "move" on the first pump, and worse, spawn packets
        # would go out at (0, 0, 0).
        tracked.note_snapshot(x, y, z, yaw, pitch, on_ground)
        tracked.prime_baselines()
        self.tracked[entity_id] = tracked
        logger.debug(f"[{self.dimension_name}] now tracking player entity {entity_id}")

    # ----- direct updates (same thread; used for future non-players) -----

    def update_entity(self, entity_id: int, x: float, y: float, z: float,
                      yaw: float = 0.0, pitch: float = 0.0,
                      on_ground: bool = True) -> None:
        """Refresh a tracked entity's snapshot in place."""
        tracked = self.tracked.get(entity_id)
        if tracked is not None:
            tracked.note_snapshot(x, y, z, yaw, pitch, on_ground)

    # ----- non-player entities (the natural spawner's path) -----

    def add_entity(self, entity) -> None:
        """Put a mob into the tracking table.

        The mirror of `_add_tracked_player`, and the reason it does not
        simply call that: a mob is not a player, and the two differ in three
        places that all matter downstream.

        * `is_player=False` gates the `update_interval` throttle in `pump`.
          Players keep the every-tick path this server has already run
          against a real client; mobs are on the type's interval.
        * `uuid` is the mob's own, not a profile id. It goes into the
          remove/replace bookkeeping and must be distinct per mob, or two
          cows would be one entity as far as anything keyed by uuid knows.
        * The snapshot comes from the mob's `EntityPosition`, not from a
          message - the spawner runs on the actor thread and hands over the
          object directly, so there is no copy to make and no race to have.

        The baselines are primed in the same breath as the snapshot, for the
        reason `_add_tracked_player` gives: a tracked entity whose baselines
        start at the origin broadcasts its whole spawn position as a move on
        the first pump, and spawns at (0, 0, 0).

        Refuses an id that is already tracked, because `tracked` is a dict
        keyed by the id the packets name: overwriting silently would leave
        the previous entry's `seen_by` viewers believing they can still see
        an entity that no longer exists. The spawner never does this (ids
        come from the global allocator), so it is a guard, not a path.
        """
        entity_id = entity.entity_id
        if entity_id in self.tracked:
            logger.warning(
                f"[{self.dimension_name}] entity {entity_id} is already "
                f"tracked; add_entity refused"
            )
            return
        entity_type = entity.entity_type
        if entity_type is None:
            logger.warning(
                f"[{self.dimension_name}] entity {entity_id} has no type; "
                f"nothing the client could render"
            )
            return

        pos = entity.position
        tracked = _TrackedEntity(
            entity_id=entity_id,
            entity_type_name=entity_type.name,
            client_tracking_range=entity_type.client_tracking_range,
            is_player=False,
            spawn_blob=entity.spawn_blob() if hasattr(entity, "spawn_blob")
            else b"\xff",
            uuid=entity.uuid,
            update_interval=max(1, entity_type.update_interval),
        )
        tracked.note_snapshot(pos.x, pos.y, pos.z, pos.yaw, pos.pitch,
                              pos.on_ground)
        tracked.prime_baselines()
        self.tracked[entity_id] = tracked
        # No viewer is told anything here. `pump` decides who can see it and
        # sends the add/metadata/rotate trio then, which is the only place
        # that knows the range rules - and it means a mob spawned while no
        # one is watching costs nothing.

    def remove_entity(self, entity_id: int) -> None:
        """Take a mob out of the tracking table and tell its viewers.

        The counterpart to `add_entity`, and the two halves of leaving that
        `EntityTracker.detach` does for a player. Called by the spawner when
        a mob's chunk leaves memory: the world has forgotten the block, so
        the entity has to go, and every client still holding it has to be
        told or it keeps rendering a ghost at a position the world no longer
        agrees exists.

        A no-op for an id that was never tracked, because the natural order
        is "broadcast first, then drop from the world" and a spawner that
        crashes between the two must not crash again here.
        """
        tracked = self.tracked.pop(entity_id, None)
        if tracked is None:
            return
        self._broadcast_removal(tracked)

    def viewer_positions(self):
        """Every attached viewer's id, position and spectator flag.

        `[(entity_id, x, y, z, spectator), ...]`, as a plain snapshot. The
        natural spawner reads it once per round to learn who is online and
        where; going through a snapshot rather than a live reference is the
        tracker's whole contract - a mutable `_Viewer` handed across threads
        would be read while the event loop wrote it.

        The key is the tracker's own `views` key, so it is the same value
        `PlayerViewMsg.entity_id` carries, not `PlayerViewMsg.player_id`.
        The two are different things (a runtime entity id versus an
        integration-side handle) and mixing them would give the spawner two
        unrelated "who is this" namespaces for one player.
        """
        return [
            (viewer.entity_id, viewer.x, viewer.y, viewer.z, viewer.spectator)
            for viewer in self.views.values()
        ]

    # ----- detaching -----

    def _detach(self, entity_id: int) -> None:
        """A player left: they stop seeing, and they stop being seen."""
        self.views.pop(entity_id, None)
        self._blob_warned.discard(entity_id)
        tracked = self.tracked.pop(entity_id, None)
        if tracked is not None:
            self._broadcast_removal(tracked)
        for other in self.tracked.values():
            other.seen_by.discard(entity_id)

    def _broadcast_removal(self, tracked: _TrackedEntity) -> None:
        """Tell everyone who could see this entity that it is gone."""
        from network.packets.clientbound.play.remove_entities import CRemoveEntities

        if not tracked.seen_by:
            return
        packet = CRemoveEntities([tracked.entity_id])
        for viewer_id in list(tracked.seen_by):
            viewer = self.views.get(viewer_id)
            if viewer is not None:
                viewer.sink.send([packet])
        tracked.seen_by.clear()

    def detach_all(self) -> None:
        """Drop every view and tracked entity. Actor-shutdown path: the
        streamer is closing its sinks, so no removal packets are sent -
        the clients are about to lose the server anyway."""
        self.views.clear()
        self.tracked.clear()
        self._blob_warned.clear()

    # ----- the tick -----

    def pump(self) -> Dict[str, int]:
        """One tick's worth of visibility updates and movement deltas.

        Runs after the dimension tick and before the chunk streamer's pump,
        inside the same measured tick window. Order matters: visibility
        first, so a viewer who just came into range receives a spawn
        *before* the movement deltas that follow.
        """
        from network.buffer import ByteBuffer

        # 1. Visibility: every (entity, viewer) pair, spawn or despawn.
        for tracked in self.tracked.values():
            for viewer_id, viewer in self.views.items():
                if viewer_id == tracked.entity_id:
                    continue  # a client never sees its own entity
                if self._in_range(tracked, viewer):
                    if viewer_id not in tracked.seen_by:
                        tracked.seen_by.add(viewer_id)
                        self._send_spawn(tracked, viewer)
                else:
                    if viewer_id in tracked.seen_by:
                        tracked.seen_by.discard(viewer_id)
                        from network.packets.clientbound.play.remove_entities import (
                            CRemoveEntities,
                        )
                        viewer.sink.send([CRemoveEntities([tracked.entity_id])])

        # 2. Movement deltas: one packet set per entity, fanned out to
        #    everyone currently seeing it.
        for tracked in self.tracked.values():
            if not tracked.seen_by:
                continue
            tracked.tick_count += 1

            # Non-players are throttled to their type's `update_interval`
            # (3 ticks for every animal in `entity_tracking.json`, 20 for an
            # item frame). Players are NOT: vanilla's tracker never sends a
            # player's own position through this path at all - it is the one
            # type with `track_deltas: false`, whose movement is driven by
            # the client's own input - and this server deliberately chose
            # to track players as ordinary entities, a path already verified
            # against a real 1.21.11 client. Throttling them to the
            # player's `update_interval: 2` would change verified behaviour
            # for a gain nobody asked for, so the gate is `is_player`.
            #
            # Head yaw is outside the gate in vanilla and here too: it is one
            # byte and it is what makes a mob's head turn, and a mob whose
            # body syncs every 3 ticks but whose head syncs every tick is
            # what the client expects.
            if (not tracked.is_player and tracked.update_interval > 0
                    and tracked.tick_count % tracked.update_interval != 0):
                self._send_head_yaw(tracked)
                continue

            packed_yaw = ByteBuffer.pack_angle(tracked.yaw)
            packed_pitch = ByteBuffer.pack_angle(tracked.pitch)
            rot_changed = (
                packed_yaw != tracked.last_sent_yaw
                or packed_pitch != tracked.last_sent_pitch
            )

            move = pick_move(
                (tracked.x, tracked.y, tracked.z),
                (tracked.last_sent_x, tracked.last_sent_y, tracked.last_sent_z),
                rot_changed,
                tracked.was_on_ground != tracked.on_ground,
                tracked.tick_count,
            )

            if move is MoveSync.NONE:
                packets: List = []
            else:
                packets = self._build_move_packets(tracked, move, packed_yaw,
                                                   packed_pitch)

            self._append_head_yaw(tracked, packets)

            if packets:
                for viewer_id in tracked.seen_by:
                    viewer = self.views.get(viewer_id)
                    if viewer is not None:
                        viewer.sink.send(packets)

            self._note_sent(tracked, move)

        return self.stats()

    def _append_head_yaw(self, tracked: _TrackedEntity,
                         packets: List) -> None:
        """Add a rotate-head packet if the head has turned. Into `packets`."""
        from network.packets.clientbound.play.rotate_head import CRotateHead

        packed_head = _pack_angle(tracked.yaw)
        if packed_head != tracked.last_sent_head_yaw:
            packets.append(CRotateHead(tracked.entity_id, tracked.yaw))
            tracked.last_sent_head_yaw = packed_head

    def _send_head_yaw(self, tracked: _TrackedEntity) -> None:
        """The same head-yaw check, for a tick where the position is throttled."""
        from network.packets.clientbound.play.rotate_head import CRotateHead

        packed_head = _pack_angle(tracked.yaw)
        if packed_head == tracked.last_sent_head_yaw:
            return
        packet = CRotateHead(tracked.entity_id, tracked.yaw)
        tracked.last_sent_head_yaw = packed_head
        for viewer_id in tracked.seen_by:
            viewer = self.views.get(viewer_id)
            if viewer is not None:
                viewer.sink.send([packet])

    def _in_range(self, tracked: _TrackedEntity, viewer: _Viewer) -> bool:
        """The visibility rule, all squared distances."""
        range_blocks = (
            min(tracked.client_tracking_range, viewer.radius)
            * 16
            * self.broadcast_percentage
            / 100
        )
        dx = tracked.x - viewer.x
        dy = tracked.y - viewer.y
        dz = tracked.z - viewer.z
        return dx * dx + dy * dy + dz * dz <= range_blocks * range_blocks

    def _send_spawn(self, tracked: _TrackedEntity, viewer: _Viewer) -> None:
        """Pumpkin's add_pairing sequence, minus the packets this round
        does not have: spawn, then metadata, then head rotation."""
        from network.packets.clientbound.play.add_entity import CAddEntity
        from network.packets.clientbound.play.rotate_head import CRotateHead
        from network.packets.clientbound.play.set_entity_data import CSetEntityData

        from entity import entity_type_by_name

        entity_type = entity_type_by_name(tracked.entity_type_name)
        spawn = CAddEntity(
            entity_id=tracked.entity_id,
            # The profile uuid, the same one Player Info Add published for
            # this player. The client joins the two by this value; a
            # mismatch leaves the body unskinned and unlisted.
            entity_uuid=tracked.uuid or "00000000-0000-0000-0000-000000000000",
            entity_type=entity_type.numeric_id if entity_type else 0,
            x=tracked.x,
            y=tracked.y,
            z=tracked.z,
            pitch=tracked.pitch,
            yaw=tracked.yaw,
            head_yaw=tracked.yaw,  # Java: server-side head yaw starts at yaw
            data=0,
        )
        metadata = CSetEntityData(tracked.entity_id, tracked.spawn_blob)
        head = CRotateHead(tracked.entity_id, tracked.yaw)
        viewer.sink.send([spawn, metadata, head])

    def _build_move_packets(self, tracked: _TrackedEntity, move: MoveSync,
                            packed_yaw: int, packed_pitch: int) -> List:
        """The MoveSync choice turned into packets, with the baselines
        advanced by exactly what was sent."""
        from network.packets.clientbound.play.entity_position_sync import (
            CEntityPositionSync,
        )
        from network.packets.clientbound.play.move_entity_pos import CMoveEntityPos
        from network.packets.clientbound.play.move_entity_pos_rot import (
            CMoveEntityPosRot,
        )
        from network.packets.clientbound.play.move_entity_rot import CMoveEntityRot

        if move is MoveSync.ABSOLUTE:
            tracked.was_on_ground = tracked.on_ground
            return [CEntityPositionSync(
                entity_id=tracked.entity_id,
                x=tracked.x, y=tracked.y, z=tracked.z,
                yaw=tracked.yaw, pitch=tracked.pitch,
                on_ground=tracked.on_ground,
            )]
        if move is MoveSync.POS_ROT:
            return [CMoveEntityPosRot(
                entity_id=tracked.entity_id,
                dx=self._delta(tracked.last_sent_x, tracked.x),
                dy=self._delta(tracked.last_sent_y, tracked.y),
                dz=self._delta(tracked.last_sent_z, tracked.z),
                yaw=tracked.yaw, pitch=tracked.pitch,
                on_ground=tracked.on_ground,
            )]
        if move is MoveSync.POS:
            return [CMoveEntityPos(
                entity_id=tracked.entity_id,
                dx=self._delta(tracked.last_sent_x, tracked.x),
                dy=self._delta(tracked.last_sent_y, tracked.y),
                dz=self._delta(tracked.last_sent_z, tracked.z),
                on_ground=tracked.on_ground,
            )]
        # ROT
        return [CMoveEntityRot(
            entity_id=tracked.entity_id,
            yaw=tracked.yaw, pitch=tracked.pitch,
            on_ground=tracked.on_ground,
        )]

    @staticmethod
    def _delta(last_sent: float, current: float) -> int:
        """The i16 fixed-point delta the move packets carry."""
        delta = encode_pos(current) - encode_pos(last_sent)
        return max(_I16_MIN, min(_I16_MAX, delta))

    def _note_sent(self, tracked: _TrackedEntity, move: MoveSync) -> None:
        """Advance the baselines to match what just went out. Skipping the
        position half for ROT (and vice versa) is the whole reason the
        deltas stay small."""
        from network.buffer import ByteBuffer

        if move.sends_position:
            tracked.last_sent_x = tracked.x
            tracked.last_sent_y = tracked.y
            tracked.last_sent_z = tracked.z
        if move.sends_rotation:
            tracked.last_sent_yaw = ByteBuffer.pack_angle(tracked.yaw)
            tracked.last_sent_pitch = ByteBuffer.pack_angle(tracked.pitch)

    def stats(self) -> Dict[str, int]:
        return {
            "tracked": len(self.tracked),
            "viewers": len(self.views),
            "visible_pairs": sum(len(t.seen_by) for t in self.tracked.values()),
        }
