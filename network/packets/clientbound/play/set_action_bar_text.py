"""
Set Action Bar Text (Clientbound, Play) - packet id 0x55

Renders a transient line of text above the hotbar but below the title
slot. The text is a vanilla text component (NBT on the wire, JSON in this
module's API).

The vanilla server uses this for the "Saving world..." and "Scoreboard"
overlays; Crotalus uses it as the action-bar overlay channel. The text
fades with the title-stack settings if a title is up; with no title, the
line still appears for a few seconds then disappears.

Packet ID 0x55 per Pumpkin @ 42c903493 assets/packets.json (protocol 774).
"""

from network.buffer import ByteBuffer
from network.packet import ClientBoundPacket, register_packet
from network.protocol import ProtocolState, Direction
from datatypes import write_nbt


@register_packet(ProtocolState.PLAY, Direction.SERVER_TO_CLIENT, 0x55)
class CSetActionBarText(ClientBoundPacket):
    """Set Action Bar Text - one transient overlay line.

    Packet ID: 0x55
    """

    name = "Set Action Bar Text"
    packet_id = 0x55

    def __init__(self, text: str | dict = ""):
        super().__init__()
        self.text = text

    def write(self, buf: ByteBuffer) -> None:
        component = self.text if isinstance(self.text, dict) else {"text": self.text}
        buf.write(write_nbt(component))

    def read(self, buf: ByteBuffer) -> None:
        # Not implementing read for server-side
        pass