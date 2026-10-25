"""
Player Input (Serverbound, Play)

Which movement keys the client is holding right now. Sent whenever the set
changes, not every tick, so a player standing still sends none at all.

Mojang collapsed this to a single bitfield byte in 1.21.2. It used to be a
VarInt of the same seven flags followed by two floats (strafe and forward
analogue axes); the floats are gone, so a server no longer sees partial
keypresses — the packet is now purely "these keys are down".
"""

from network.buffer import ByteBuffer
from network.packet import ServerBoundPacket, register_packet
from network.protocol import ProtocolState, Direction


@register_packet(ProtocolState.PLAY, Direction.CLIENT_TO_SERVER, 0x2A)
class SPlayerInput(ServerBoundPacket):
    """Player input flags.

    Packet ID: 0x2A
    """

    name = "Player Input"
    packet_id = 0x2A

    FORWARD = 0x01
    BACKWARD = 0x02
    LEFT = 0x04
    RIGHT = 0x08
    JUMP = 0x10
    SNEAK = 0x20
    SPRINT = 0x40

    def __init__(self, input_flags: int = 0):
        super().__init__()
        self.input_flags = input_flags

    def _is_pressed(self, flag: int) -> bool:
        return bool(self.input_flags & flag)

    @property
    def forward(self) -> bool:
        return self._is_pressed(self.FORWARD)

    @property
    def backward(self) -> bool:
        return self._is_pressed(self.BACKWARD)

    @property
    def left(self) -> bool:
        return self._is_pressed(self.LEFT)

    @property
    def right(self) -> bool:
        return self._is_pressed(self.RIGHT)

    @property
    def jump(self) -> bool:
        return self._is_pressed(self.JUMP)

    @property
    def sneak(self) -> bool:
        return self._is_pressed(self.SNEAK)

    @property
    def sprint(self) -> bool:
        return self._is_pressed(self.SPRINT)

    def active_keys(self) -> list[str]:
        """Names of the held keys, for logging a readable form."""
        names = (
            (self.FORWARD, "forward"),
            (self.BACKWARD, "backward"),
            (self.LEFT, "left"),
            (self.RIGHT, "right"),
            (self.JUMP, "jump"),
            (self.SNEAK, "sneak"),
            (self.SPRINT, "sprint"),
        )
        return [name for flag, name in names if self._is_pressed(flag)]

    def write(self, buf: ByteBuffer) -> None:
        buf.write_byte(self.input_flags)

    def read(self, buf: ByteBuffer) -> None:
        self.input_flags = buf.read_unsigned_byte()
