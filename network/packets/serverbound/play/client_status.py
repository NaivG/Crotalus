"""
Client Command (Serverbound, Play)

Two unrelated intents in one packet, distinguished by action_id:
0 = perform respawn, 1 = request stats.

Renamed from "Client Status" in 1.21.5; the old name is kept as an alias in
network.packets so older imports keep working.
"""

from network.buffer import ByteBuffer
from network.packet import ServerBoundPacket, register_packet
from network.protocol import ProtocolState, Direction


@register_packet(ProtocolState.PLAY, Direction.CLIENT_TO_SERVER, 0x0B)
class SClientCommand(ServerBoundPacket):
    """Client command.

    Packet ID: 0x0B

    Actions:
    0: Perform respawn
    1: Request stats
    """

    name = "Client Command"
    packet_id = 0x0B

    ACTION_RESPAWN = 0
    ACTION_REQUEST_STATS = 1

    def __init__(self, action_id: int = 0):
        super().__init__()
        self.action_id = action_id

    def write(self, buf: ByteBuffer) -> None:
        buf.write_varint(self.action_id)

    def read(self, buf: ByteBuffer) -> None:
        self.action_id = buf.read_varint()
