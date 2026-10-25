"""
Particle (Clientbound, Play) - packet id 0x2E

Spawns a particle at a position. The particle *kind* is a VarInt registry
index into minecraft:particle_type; `count` is the number of particles in
the spawn batch (long range particles spawn one packet with many count, short
range particles spawn many packets with one count each). The data layout
after position depends on the particle kind, so the simplest approach is
to encode the per-kind bytes from the encoder that knows the format and
ship them as the trailing blob.

Packet ID 0x2E per Pumpkin @ 42c903493 assets/packets.json (protocol 774).
"""

from network.buffer import ByteBuffer
from network.packet import ClientBoundPacket, register_packet
from network.protocol import ProtocolState, Direction


@register_packet(ProtocolState.PLAY, Direction.SERVER_TO_CLIENT, 0x2E)
class CParticle(ClientBoundPacket):
    """Particle - spawn a particle batch.

    Packet ID: 0x2E
    """

    name = "Particle"
    packet_id = 0x2E

    # Minecart / client code reads these as unsigned VarInts, so a few
    # well-known particle ids are exposed for callers who do not want to
    # dig through the registry.
    PARTICLE_AMBIENT_ENTITY_EFFECT = 0
    PARTICLE_EXPLOSION = 8
    PARTICLE_BLOCK = 14
    PARTICLE_DUST = 15
    PARTICLE_FLAME = 27
    PARTICLE_HEART = 34
    PARTICLE_TOTEM_OF_UNDYING = 53

    def __init__(
        self,
        particle_id: int = 0,
        long_distance: bool = False,
        x: float = 0.0,
        y: float = 0.0,
        z: float = 0.0,
        dx: float = 0.0,
        dy: float = 0.0,
        dz: float = 0.0,
        max_speed: float = 0.0,
        count: int = 1,
        data: bytes = b"",
    ):
        super().__init__()
        self.particle_id = particle_id
        # Long-distance: the client renders out to the simulation distance
        # instead of the chunk viewer distance, used by e.g. dragon breath.
        self.long_distance = long_distance
        self.x = x
        self.y = y
        self.z = z
        self.dx = dx
        self.dy = dy
        self.dz = dz
        self.max_speed = max_speed
        self.count = count
        # Per-kind payload. Empty for particles that have no extra fields.
        self.data = bytes(data)

    def write(self, buf: ByteBuffer) -> None:
        buf.write_varint(self.particle_id)
        buf.write_bool(self.long_distance)
        buf.write_double(self.x)
        buf.write_double(self.y)
        buf.write_double(self.z)
        buf.write_float(self.dx)
        buf.write_float(self.dy)
        buf.write_float(self.dz)
        buf.write_float(self.max_speed)
        buf.write_varint(self.count)
        if self.data:
            buf.write_bytes(self.data)

    def read(self, buf: ByteBuffer) -> None:
        self.particle_id = buf.read_varint()
        self.long_distance = buf.read_bool()
        self.x = buf.read_double()
        self.y = buf.read_double()
        self.z = buf.read_double()
        self.dx = buf.read_float()
        self.dy = buf.read_float()
        self.dz = buf.read_float()
        self.max_speed = buf.read_float()
        self.count = buf.read_varint()
        if buf.readable_bytes() > 0:
            self.data = buf.read_bytes(buf.readable_bytes())
        else:
            self.data = b""