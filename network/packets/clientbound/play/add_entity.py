"""
Add Entity (Clientbound, Play) - packet id 0x01

The one spawn packet. 1.21.11 has no separate "Spawn Player": every entity
- mobs, items, arrows, and players - arrives through this packet, and a
player is distinguished only by carrying type id 155 plus a Set Entity Data
blob whose profile entry (index 17) names their model.

The velocity field is an `LpVec3` ("linearly packed Vec3", 1.21.2+): NOT the
three i16 shorts of older protocols. A zero velocity is the single byte
0x00; a non-zero one is a 6-byte packed value of three 15-bit components
sharing one scale factor, plus a VarInt tail when that factor exceeds 3.
Standing-spawned entities are the zero case; the packing below implements
the full format anyway because "it is always zero" is exactly the kind of
assumption that survives until the first knockback.

Angle encoding: degrees scaled to a byte, floor(d * 256 / 360). Yaw wraps
(`rem_euclid(360)` first); pitch does not - vanilla's cast makes pitch = -90
encode as 192, and the client undoes the wrap by the same arithmetic. The
wire order is pitch, THEN yaw, THEN head_yaw, which is not the order the
constructor arguments read and has been verified against Pumpkin's struct
at the 1.21.11 commit (`pumpkin-protocol/src/java/client/play/spawn_entity.rs`).

The `data` VarInt is entity-type-specific (painting variants, item ids, ...);
a player carries 0.
"""

from network.buffer import ByteBuffer
from network.packet import ClientBoundPacket, register_packet
from network.protocol import ProtocolState, Direction

# LpVec3 thresholds, from vanilla's codec (Pumpkin's velocity.rs is a
# line-for-line port; the constants are not guessable from the wire).
_MAX_VELOCITY_CLAMP = 1.7179869183e10
_MIN_VELOCITY_MAGNITUDE = 3.051944088384301e-5
_MAX_15_BIT_VALUE = 32766.0


@register_packet(ProtocolState.PLAY, Direction.SERVER_TO_CLIENT, 0x01)
class CAddEntity(ClientBoundPacket):
    """Add Entity - spawn an entity the client does not know yet.

    Packet ID: 0x01
    """

    name = "Add Entity"
    packet_id = 0x01

    def __init__(
        self,
        entity_id: int = 0,
        entity_uuid: str = "",
        entity_type: int = 0,
        x: float = 0.0,
        y: float = 0.0,
        z: float = 0.0,
        velocity_x: float = 0.0,
        velocity_y: float = 0.0,
        velocity_z: float = 0.0,
        pitch: float = 0.0,
        yaw: float = 0.0,
        head_yaw: float = 0.0,
        data: int = 0,
    ):
        super().__init__()
        self.entity_id = entity_id
        self.entity_uuid = entity_uuid
        self.entity_type = entity_type
        self.x = x
        self.y = y
        self.z = z
        self.velocity_x = velocity_x
        self.velocity_y = velocity_y
        self.velocity_z = velocity_z
        self.pitch = pitch
        self.yaw = yaw
        self.head_yaw = head_yaw
        self.data = data

    def write(self, buf: ByteBuffer) -> None:
        buf.write_varint(self.entity_id)
        buf.write_uuid(self.entity_uuid)
        buf.write_varint(self.entity_type)
        buf.write_double(self.x)
        buf.write_double(self.y)
        buf.write_double(self.z)
        write_lp_vec3(
            buf, self.velocity_x, self.velocity_y, self.velocity_z
        )
        # Wire order is pitch, yaw, head_yaw. Pitch is not wrapped; yaw is.
        buf.write_byte(int(self.pitch * 256.0 / 360.0) & 0xFF)
        buf.write_byte(int((self.yaw % 360.0) * 256.0 / 360.0) & 0xFF)
        buf.write_byte(int((self.head_yaw % 360.0) * 256.0 / 360.0) & 0xFF)
        buf.write_varint(self.data)

    def read(self, buf: ByteBuffer) -> None:
        self.entity_id = buf.read_varint()
        self.entity_uuid = buf.read_uuid()
        self.entity_type = buf.read_varint()
        self.x = buf.read_double()
        self.y = buf.read_double()
        self.z = buf.read_double()
        velocity = read_lp_vec3(buf)
        self.velocity_x, self.velocity_y, self.velocity_z = velocity
        self.pitch = buf.read_unsigned_byte() * 360.0 / 256.0
        self.yaw = buf.read_unsigned_byte() * 360.0 / 256.0
        self.head_yaw = buf.read_unsigned_byte() * 360.0 / 256.0
        self.data = buf.read_varint()


def write_lp_vec3(buf: ByteBuffer, vx: float, vy: float, vz: float) -> None:
    """The LpVec3 velocity encoding: zero is one byte, else 6 bytes + scale.

    Three components share one scale factor `l = ceil(max |component|)`.
    Each is normalized into [-1, 1], quantized to 15 bits, and the three
    quantized values are bit-packed alongside a 3-bit scale header into a
    48-bit integer written as 2 raw bytes + 1 big-endian i32. A scale above
    3 does not fit the header, so its top bits ride a trailing VarInt.
    """
    def clamp(value: float) -> float:
        if value != value:  # NaN
            return 0.0
        return max(-_MAX_VELOCITY_CLAMP, min(_MAX_VELOCITY_CLAMP, value))

    d, e, f = clamp(vx), clamp(vy), clamp(vz)
    g = max(abs(d), abs(e), abs(f))

    if g < _MIN_VELOCITY_MAGNITUDE:
        buf.write_byte(0)
        return

    scale = int(g) if g == int(g) else int(g) + 1  # ceil, without math.ceil
    # ... but scale must be at least 1 so the division is sane.
    scale = max(1, scale)
    needs_varint = scale > 3

    header = (scale & 3) | 4 if needs_varint else scale

    def to_long(value: float) -> int:
        # round() is banker's rounding in Python; vanilla wants half-away-
        # from-zero. On the values that reach here (|v| <= 1) the difference
        # is one ulp of a 15-bit quantity - irrelevant - but be explicit.
        return int((value * 0.5 + 0.5) * _MAX_15_BIT_VALUE + 0.5)

    q = (
        header
        | (to_long(d / scale) << 3)
        | (to_long(e / scale) << 18)
        | (to_long(f / scale) << 33)
    )

    buf.write_byte(q & 0xFF)
    buf.write_byte((q >> 8) & 0xFF)
    # The remaining 32 bits go out as a big-endian signed int, exactly like
    # Java's writeInt - the value's 48 bits only reach 47, so the sign bit
    # of the i32 is never set.
    signed = q >> 16
    if signed >= 0x80000000:
        signed -= 0x100000000
    buf.write_int(signed)
    if needs_varint:
        buf.write_varint(scale >> 2)


def read_lp_vec3(buf: ByteBuffer) -> tuple[float, float, float]:
    """Inverse of `write_lp_vec3`; used by tests, not by the server."""
    first = buf.read_unsigned_byte()
    if first == 0:
        return (0.0, 0.0, 0.0)
    second = buf.read_unsigned_byte()
    rest = buf.read_int() & 0xFFFFFFFF  # the i32 is unsigned once widened
    q = first | (second << 8) | (rest << 16)

    header = q & 7
    if header & 4:
        scale_low = header & 3
        scale_high = buf.read_varint()
        scale = (scale_high << 2) | scale_low
    else:
        scale = header

    def from_long(raw: int) -> float:
        value = raw / _MAX_15_BIT_VALUE * 2.0 - 1.0
        return value * scale

    x = from_long((q >> 3) & 0x7FFF)
    y = from_long((q >> 18) & 0x7FFF)
    z = from_long((q >> 33) & 0x7FFF)
    return (x, y, z)
