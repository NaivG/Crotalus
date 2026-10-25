"""
Disconnect (Play, Clientbound)

Forces the client to disconnect while in the play state, showing the reason
on the dedicated disconnection screen. This is the kick path: server
shutdown, bans, and any post-join rejection all end here. Distinct from the
login/config disconnects only by state — same payload, different packet ID,
and the client resolves IDs against the state it is currently in.

Packet ID 0x20 per Pumpkin @ 42c903493 assets/packets.json (protocol 774).
"""

from network.buffer import ByteBuffer
from network.packet import ClientBoundPacket, register_packet
from network.protocol import ProtocolState, Direction


@register_packet(ProtocolState.PLAY, Direction.SERVER_TO_CLIENT, 0x20)
class CDisconnectPlay(ClientBoundPacket):
    """Server kicks a client that is in the world.

    Fields:
    - json_reason: JSON text component, NBT-encoded on the wire
    """

    name = "Disconnect (Play)"
    packet_id = 0x20

    def __init__(self, json_reason: str = '{"text": "Disconnected"}'):
        super().__init__()
        self.json_reason = json_reason

    def write(self, buf: ByteBuffer) -> None:
        buf.write_component(self.json_reason)

    def read(self, buf: ByteBuffer) -> None:
        self.json_reason = buf.read_component()
