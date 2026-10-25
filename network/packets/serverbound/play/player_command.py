"""
Player Command (Serverbound, Play) - packet id 0x29

Discrete state changes the client pushes: start/stop sneaking,
start/stop sprinting, leave bed, horse jump begin/end, open vehicle
inventory, start flying elytra. The classic "sprint detection" packet
on a half-sprint player: a single packet when they double-tap W and
another when they let go.

The `jump_boost` field is the horse-jump bar value (0..100) at the
moment of the action; for actions other than horse-jump it is 0.

The 1.21.6 renumbering dropped start/stop sneaking (the client now
sends them as Set Entity Data), but in 1.21.11 they are still on the
wire and need to be parseable. This implementation keeps both name and
id for all of them.

Packet ID 0x29 per Pumpkin @ 42c903493 assets/packets.json (protocol 774).
"""

from network.buffer import ByteBuffer
from network.packet import ServerBoundPacket, register_packet
from network.protocol import ProtocolState, Direction


@register_packet(ProtocolState.PLAY, Direction.CLIENT_TO_SERVER, 0x29)
class SPlayerCommand(ServerBoundPacket):
    """Player Command - discrete player state change.

    Packet ID: 0x29
    """

    name = "Player Command"
    packet_id = 0x29

    # Action ids, in 1.21.11 ordering.
    START_SNEAKING = 0
    STOP_SNEAKING = 1
    LEAVE_BED = 2
    START_SPRINTING = 3
    STOP_SPRINTING = 4
    START_HORSE_JUMP = 5
    STOP_HORSE_JUMP = 6
    OPEN_VEHICLE_INVENTORY = 7
    START_FLYING_ELYTRA = 8

    def __init__(
        self,
        entity_id: int = 0,
        action_id: int = 0,
        jump_boost: int = 0,
    ):
        super().__init__()
        self.entity_id = entity_id
        self.action_id = action_id
        self.jump_boost = jump_boost

    def write(self, buf: ByteBuffer) -> None:
        buf.write_varint(self.entity_id)
        buf.write_varint(self.action_id)
        buf.write_varint(self.jump_boost)

    def read(self, buf: ByteBuffer) -> None:
        self.entity_id = buf.read_varint()
        self.action_id = buf.read_varint()
        self.jump_boost = buf.read_varint()