"""
Chunk Batch Received (Serverbound, Play)

The client's answer to a chunk batch: it reports how many chunks per tick it
can absorb, which is how the server learns to stop sending faster than the
client can apply. A value of 0.0 means "send nothing more until asked", which
the vanilla client sends when it is overloaded.
"""

from network.buffer import ByteBuffer
from network.packet import ServerBoundPacket, register_packet
from network.protocol import ProtocolState, Direction


@register_packet(ProtocolState.PLAY, Direction.CLIENT_TO_SERVER, 0x0A)
class SChunkBatchReceived(ServerBoundPacket):
    """Chunk batch received.

    Packet ID: 0x0A
    """

    name = "Chunk Batch Received"
    packet_id = 0x0A

    def __init__(self, desired_chunks_per_tick: float = 0.0):
        super().__init__()
        self.desired_chunks_per_tick = desired_chunks_per_tick

    def write(self, buf: ByteBuffer) -> None:
        buf.write_float(self.desired_chunks_per_tick)

    def read(self, buf: ByteBuffer) -> None:
        self.desired_chunks_per_tick = buf.read_float()
