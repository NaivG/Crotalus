"""
Update Time (Clientbound, Play)

The client keeps its own clock: it advances `dayTime` every tick and uses it for
the sky colour, the lightmap's sky brightness and the sun/moon position. Without
this packet it starts from its own default and the world's day/night cycle is
whatever that default plus the login time says -- which is a different time of
day than the server's, and the reason a server at noon can be watched from a
client that renders dusk (everything sky-lit dims, block light does not).

Vanilla (`ServerLevel.tickTime`) broadcasts this every 20 ticks, so the client
is never more than a second away from the server's clock, and it says whether
the client should keep advancing on its own in between:

    World Age        i64 BE   ticks since the world was created; `/time set`
                              does not move it
    Time of Day      i64 BE   the world (or region) time, modulo 24000 is the
                              time of day: 0 sunrise, 6000 noon, 12000 sunset,
                              18000 midnight
    Time Increasing  bool     true when the day/night cycle runs

Packet ID: 0x6F (protocol: set_time), protocol 774
"""

from network.buffer import ByteBuffer
from network.packet import ClientBoundPacket, register_packet
from network.protocol import ProtocolState, Direction


@register_packet(ProtocolState.PLAY, Direction.SERVER_TO_CLIENT, 0x6F)
class CUpdateTime(ClientBoundPacket):
    """The server's clock, for a client that is keeping its own."""

    name = "Update Time"
    packet_id = 0x6F

    def __init__(self, world_age: int = 0, time_of_day: int = 6000,
                 increasing: bool = True):
        super().__init__()
        self.world_age = world_age
        self.time_of_day = time_of_day
        #: False for a dimension whose time is pinned (`DimensionSettings
        #: .fixed_time`: the End's twilight), so the client stops advancing and
        #: stays where it was told to be.
        self.increasing = increasing

    def write(self, buf: ByteBuffer) -> None:
        buf.write_long(self.world_age)
        buf.write_long(self.time_of_day)
        buf.write_bool(self.increasing)

    def read(self, buf: ByteBuffer) -> None:
        self.world_age = buf.read_long()
        self.time_of_day = buf.read_long()
        self.increasing = buf.read_bool()

    def __repr__(self) -> str:  # pragma: no cover - a log line
        return (f"<CUpdateTime age={self.world_age} "
                f"time={self.time_of_day} increasing={self.increasing}>")
