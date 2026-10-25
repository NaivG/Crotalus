"""
Chat Command (Serverbound, Play)

A player typed something starting with "/" in the chat box. The client strips
the slash and sends the rest as this packet rather than a Chat Message, which
is why an unimplemented command path shows up as an "Unknown packet" warning
with the raw bytes spelling out the command text.

Which commands the client even knows about is decided by the Declare Commands
packet; without it the client treats every argument as unsigned and always
uses this packet. The signed variant (0x07) only appears once the server
declares arguments that carry signatures.
"""

from network.buffer import ByteBuffer
from network.packet import ServerBoundPacket, register_packet
from network.protocol import ProtocolState, Direction


@register_packet(ProtocolState.PLAY, Direction.CLIENT_TO_SERVER, 0x06)
class SChatCommand(ServerBoundPacket):
    """A command typed by the client, excluding the leading '/'.

    Packet ID: 0x06

    Fields:
    - command: the command line as typed, e.g. "gamemode creative"
    """

    name = "Chat Command"
    packet_id = 0x06

    def __init__(self, command: str = ""):
        super().__init__()
        self.command = command

    def write(self, buf: ByteBuffer) -> None:
        buf.write_string(self.command)

    def read(self, buf: ByteBuffer) -> None:
        self.command = buf.read_string()
