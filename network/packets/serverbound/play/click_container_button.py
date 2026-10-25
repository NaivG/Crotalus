"""
Click Container Button (Serverbound, Play) - packet id 0x10

The player clicked a numbered button on an open container (the kind
drawn by the merchant, enchanting table, villager trade, lectern page
flip, etc.). The server looks up the button id against the open
container's button table and decides what to do.

Packet ID 0x10 per Pumpkin @ 42c903493 assets/packets.json (protocol 774).
"""

from network.buffer import ByteBuffer
from network.packet import ServerBoundPacket, register_packet
from network.protocol import ProtocolState, Direction


@register_packet(ProtocolState.PLAY, Direction.CLIENT_TO_SERVER, 0x10)
class SClickContainerButton(ServerBoundPacket):
    """Click Container Button - click a numbered container button.

    Packet ID: 0x10
    """

    name = "Click Container Button"
    packet_id = 0x10

    def __init__(self, window_id: int = 0, button_id: int = 0):
        super().__init__()
        self.window_id = window_id
        self.button_id = button_id

    def write(self, buf: ByteBuffer) -> None:
        buf.write_varint(self.window_id)
        buf.write_varint(self.button_id)

    def read(self, buf: ByteBuffer) -> None:
        self.window_id = buf.read_varint()
        self.button_id = buf.read_varint()