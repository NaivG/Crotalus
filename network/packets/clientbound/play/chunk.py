"""
Chunk streaming packets (Clientbound, Play)

The seven packets a server needs to keep a client's view of the world in sync.
IDs are for protocol 774 (Minecraft 1.21.11) and were read out of Pumpkin's
1.21.11 commit rather than its working tree, which tracks a later version:

    git show 42c903493:assets/packets.json

    0x0B (11)  chunk_batch_finished
    0x0C (12)  chunk_batch_start
    0x25 (37)  forget_level_chunk
    0x2C (44)  level_chunk_with_light
    0x2F (47)  light_update
    0x5C (92)  set_chunk_cache_center
    0x5D (93)  set_chunk_cache_radius

`CLevelChunkWithLight` carries bytes straight from `world.chunk_codec.py`
rather than re-deriving them here -- the codec owns the layout, this owns the
framing.

`CChunkBatchStart` / `CChunkBatchFinished` bracket a run of chunks so the
client knows when a batch is complete. Since 1.20.2 the batch carries no
"desired chunks per tick" field; the client acknowledges with a float
(`chunk_batch_received`) and the server does its own throttling.
"""

from network.buffer import ByteBuffer
from network.packet import ClientBoundPacket, register_packet
from network.protocol import ProtocolState, Direction


# Chunk coordinates are signed but travel as plain (non-zigzag) unsigned
# VarInts, so a negative value is its 32-bit two's complement. ByteBuffer
# refuses negative VarInts outright, and write_signed_varint would zigzag it,
# which is the wrong encoding for a position -- hence the explicit pair here.

def _write_coord(buf: ByteBuffer, value: int) -> None:
    """Write a signed chunk coordinate as an unsigned VarInt."""
    buf.write_varint(value & 0xFFFFFFFF)


def _read_coord(buf: ByteBuffer) -> int:
    """Read an unsigned VarInt back into a signed chunk coordinate."""
    value = buf.read_varint() & 0xFFFFFFFF
    return value - 0x100000000 if value >= 0x80000000 else value


@register_packet(ProtocolState.PLAY, Direction.SERVER_TO_CLIENT, 0x5C)
class CSetChunkCacheCenter(ClientBoundPacket):
    """Update Center Chunk - moves the client's chunk cache origin.

    Packet ID: 0x5C

    Must be sent before the chunks around the new position, and before any
    `CLevelChunkWithLight` for them, or the client will build its render
    distance around the wrong centre.
    """

    name = "Set Chunk Cache Center"
    packet_id = 0x5C

    def __init__(self, chunk_x: int = 0, chunk_z: int = 0):
        super().__init__()
        self.chunk_x = chunk_x
        self.chunk_z = chunk_z

    def write(self, buf: ByteBuffer) -> None:
        _write_coord(buf, self.chunk_x)
        _write_coord(buf, self.chunk_z)

    def read(self, buf: ByteBuffer) -> None:
        self.chunk_x = _read_coord(buf)
        self.chunk_z = _read_coord(buf)


@register_packet(ProtocolState.PLAY, Direction.SERVER_TO_CLIENT, 0x5D)
class CSetChunkCacheRadius(ClientBoundPacket):
    """Update View Distance - sets how far the client should keep chunks.

    Packet ID: 0x5D
    """

    name = "Set Chunk Cache Radius"
    packet_id = 0x5D

    def __init__(self, radius: int = 10):
        super().__init__()
        self.radius = radius

    def write(self, buf: ByteBuffer) -> None:
        buf.write_varint(self.radius)

    def read(self, buf: ByteBuffer) -> None:
        self.radius = buf.read_varint()


@register_packet(ProtocolState.PLAY, Direction.SERVER_TO_CLIENT, 0x2C)
class CLevelChunkWithLight(ClientBoundPacket):
    """Chunk Data and Update Light - the main chunk packet.

    Packet ID: 0x2C

    `payload` is a fully encoded body from `world.chunk_codec.py`. This class
    deliberately knows nothing about the body layout; see that module for the
    heightmap / section / light structure.
    """

    name = "Level Chunk With Light"
    packet_id = 0x2C

    def __init__(self, payload: bytes = b""):
        super().__init__()
        self.payload = payload

    def write(self, buf: ByteBuffer) -> None:
        buf.write_bytes(self.payload)

    def read(self, buf: ByteBuffer) -> None:
        self.payload = buf.read_bytes(buf.readable_bytes())


@register_packet(ProtocolState.PLAY, Direction.SERVER_TO_CLIENT, 0x2F)
class CLightUpdate(ClientBoundPacket):
    """Light Update - light data for a chunk already sent.

    Packet ID: 0x2F

    `light_payload` is the light block produced by `ChunkCodec.encode_light`,
    shared with `CLevelChunkWithLight` so the two can never drift apart.
    """

    name = "Light Update"
    packet_id = 0x2F

    def __init__(self, chunk_x: int = 0, chunk_z: int = 0, light_payload: bytes = b""):
        super().__init__()
        self.chunk_x = chunk_x
        self.chunk_z = chunk_z
        self.light_payload = light_payload

    def write(self, buf: ByteBuffer) -> None:
        _write_coord(buf, self.chunk_x)
        _write_coord(buf, self.chunk_z)
        buf.write_bytes(self.light_payload)

    def read(self, buf: ByteBuffer) -> None:
        self.chunk_x = _read_coord(buf)
        self.chunk_z = _read_coord(buf)
        self.light_payload = buf.read_bytes(buf.readable_bytes())


@register_packet(ProtocolState.PLAY, Direction.SERVER_TO_CLIENT, 0x25)
class CUnloadChunk(ClientBoundPacket):
    """Forget Level Chunk - drops a chunk from the client's cache.

    Packet ID: 0x25

    Since 1.20.2 the position is a *single* packed i64, not two integers:
    the low 32 bits are X (unsigned) and the high 32 bits are Z (signed).
    Writing two VarInts here desyncs the stream permanently.
    """

    name = "Forget Level Chunk"
    packet_id = 0x25

    def __init__(self, chunk_x: int = 0, chunk_z: int = 0):
        super().__init__()
        self.chunk_x = chunk_x
        self.chunk_z = chunk_z

    def write(self, buf: ByteBuffer) -> None:
        key = ((self.chunk_z & 0xFFFFFFFF) << 32) | (self.chunk_x & 0xFFFFFFFF)
        # write_long packs a *signed* int64, so an unsigned pattern with the top
        # bit set has to be folded into range or struct raises.
        if key >= 0x8000000000000000:
            key -= 0x10000000000000000
        buf.write_long(key)

    def read(self, buf: ByteBuffer) -> None:
        key = buf.read_long()
        self.chunk_x = key & 0xFFFFFFFF
        if self.chunk_x >= 0x80000000:
            self.chunk_x -= 0x100000000
        self.chunk_z = (key >> 32) & 0xFFFFFFFF
        if self.chunk_z >= 0x80000000:
            self.chunk_z -= 0x100000000


@register_packet(ProtocolState.PLAY, Direction.SERVER_TO_CLIENT, 0x0C)
class CChunkBatchStart(ClientBoundPacket):
    """Chunk Batch Start - opens a run of chunk packets.

    Packet ID: 0x0C

    The body is empty; the packet ID alone is the signal.
    """

    name = "Chunk Batch Start"
    packet_id = 0x0C

    def write(self, buf: ByteBuffer) -> None:
        pass

    def read(self, buf: ByteBuffer) -> None:
        pass


@register_packet(ProtocolState.PLAY, Direction.SERVER_TO_CLIENT, 0x0B)
class CChunkBatchFinished(ClientBoundPacket):
    """Chunk Batch Finished - closes a run, reporting how many chunks it held.

    Packet ID: 0x0B
    """

    name = "Chunk Batch Finished"
    packet_id = 0x0B

    def __init__(self, chunk_count: int = 0):
        super().__init__()
        self.chunk_count = chunk_count

    def write(self, buf: ByteBuffer) -> None:
        buf.write_varint(self.chunk_count)

    def read(self, buf: ByteBuffer) -> None:
        self.chunk_count = buf.read_varint()
