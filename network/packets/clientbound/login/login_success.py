"""
Login Success (Clientbound)

Confirms authentication and hands the client the UUID and username it will use
for the rest of the session. Note the name is server-assigned: a server may
rewrite the username here, and the client adopts whatever arrives.
"""

from network.buffer import ByteBuffer
from network.packet import ClientBoundPacket, register_packet
from network.protocol import ProtocolState, Direction


@register_packet(ProtocolState.LOGIN, Direction.SERVER_TO_CLIENT, 0x02)
class CLoginSuccess(ClientBoundPacket):
    """Login successful, contains player info.

    Fields:
    - uuid: String (player UUID)
    - username: String (player name)
    - number_of_properties: VarInt
    - properties: List of {name, value, is_signed, signature}
    """

    name = "Login Success"
    packet_id = 0x02

    def __init__(
        self,
        uuid: str = "",
        username: str = "",
        properties: list = None
    ):
        super().__init__()
        self.uuid = uuid
        self.username = username
        self.properties = properties or []

    def write(self, buf: ByteBuffer) -> None:
        buf.write_uuid(self.uuid)
        buf.write_string(self.username)
        buf.write_varint(len(self.properties))
        for prop in self.properties:
            buf.write_string(prop["name"])
            buf.write_string(prop["value"])
            buf.write_bool(prop.get("signed", False))
            if prop.get("signed"):
                buf.write_string(prop["signature"])

    def read(self, buf: ByteBuffer) -> None:
        self.uuid = buf.read_uuid()
        self.username = buf.read_string()
        num_props = buf.read_varint()
        self.properties = []
        for _ in range(num_props):
            prop = {
                "name": buf.read_string(),
                "value": buf.read_string(),
                "signed": buf.read_bool()
            }
            if prop["signed"]:
                prop["signature"] = buf.read_string()
            self.properties.append(prop)
