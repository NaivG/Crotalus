"""
Signed Chat Command (Serverbound, Play)

The signed-chat counterpart of Chat Command (0x06). The client uses this
instead once the server declares (via Declare Commands) that the command's
arguments need message signatures, so e.g. /msg content can be verified
end-to-end.

Crotalus runs without message signing, so the signature block is dead weight
here -- but the fields still have to be consumed in order, because the buffer
position is shared state and a half-read packet would misalign anything that
reads after it. The signature bytes are read and dropped.
"""

from network.buffer import ByteBuffer
from network.packet import ServerBoundPacket, register_packet
from network.protocol import ProtocolState, Direction

# Signature length is fixed by the protocol (RSA-2048), not length-prefixed.
SIGNATURE_SIZE = 256

# Fixed BitSet(20): 20 bits round up to 3 bytes on the wire.
ACKNOWLEDGED_SIZE = 3


@register_packet(ProtocolState.PLAY, Direction.CLIENT_TO_SERVER, 0x07)
class SChatCommandSigned(ServerBoundPacket):
    """A command with signed arguments.

    Packet ID: 0x07

    Fields:
    - command: the command line as typed, excluding the '/'
    - timestamp: millis since epoch, when the command was executed
    - salt: salt for the argument signatures
    - argument_signatures: [(argument name, 256-byte signature), ...]
    - message_count: running chat counter, as in the chat acknowledge path
    - acknowledged: Fixed BitSet(20), 3 raw bytes
    - checksum: present since 1.21.5
    """

    name = "Signed Chat Command"
    packet_id = 0x07

    def __init__(self, command: str = ""):
        super().__init__()
        self.command = command
        self.timestamp: int = 0
        self.salt: int = 0
        self.argument_signatures: list = []
        self.message_count: int = 0
        self.acknowledged: bytes = b""
        self.checksum: int = 0

    def write(self, buf: ByteBuffer) -> None:
        buf.write_string(self.command)
        buf.write_long(self.timestamp)
        buf.write_long(self.salt)
        buf.write_varint(len(self.argument_signatures))
        for arg_name, signature in self.argument_signatures:
            buf.write_string(arg_name)
            buf.write_bytes(signature)
        buf.write_varint(self.message_count)
        buf.write_bytes(self.acknowledged)
        buf.write_byte(self.checksum)

    def read(self, buf: ByteBuffer) -> None:
        self.command = buf.read_string()
        self.timestamp = buf.read_long()
        self.salt = buf.read_long()

        arg_count = buf.read_varint()
        self.argument_signatures = []
        for _ in range(arg_count):
            arg_name = buf.read_string()
            signature = buf.read_bytes(SIGNATURE_SIZE)
            self.argument_signatures.append((arg_name, signature))

        self.message_count = buf.read_varint()
        self.acknowledged = buf.read_bytes(ACKNOWLEDGED_SIZE)
        self.checksum = buf.read_unsigned_byte()
