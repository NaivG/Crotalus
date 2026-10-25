"""
Set Subtitle Text (Clientbound, Play) - packet id 0x6E

The smaller line that sits under a Set Title Text (0x70). Animation timings
are not set here -- those come from Set Titles Animation (0x71) -- but the
text shows with whatever animation the most-recent timing packet gave.

Packet ID 0x6E per Pumpkin @ 42c903493 assets/packets.json (protocol 774).
"""

from network.buffer import ByteBuffer
from network.packet import ClientBoundPacket, register_packet
from network.protocol import ProtocolState, Direction
from datatypes import write_nbt


@register_packet(ProtocolState.PLAY, Direction.SERVER_TO_CLIENT, 0x6E)
class CSetSubtitleText(ClientBoundPacket):
    """Set Subtitle Text - the line under a title.

    Packet ID: 0x6E
    """

    name = "Set Subtitle Text"
    packet_id = 0x6E

    def __init__(self, text: str | dict = ""):
        super().__init__()
        self.text = text

    def write(self, buf: ByteBuffer) -> None:
        component = self.text if isinstance(self.text, dict) else {"text": self.text}
        buf.write(write_nbt(component))

    def read(self, buf: ByteBuffer) -> None:
        # Not implementing read for server-side
        pass