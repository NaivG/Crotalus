"""
Known Packs (Serverbound, Config)

The client's answer to CKnownPacks: the subset of packs it also recognises.
The intersection is what governs which registry entries may be sent without
inline NBT - a client missing a pack must receive the full data.

Wire format is three strings per pack (namespace, id, version), not one
combined string. The client keeps only the entries matching its own packs on
all three fields, so a version it does not have is dropped silently and the
reply comes back empty.
"""

from network.buffer import ByteBuffer
from network.packet import ServerBoundPacket, register_packet
from network.protocol import ProtocolState, Direction


@register_packet(ProtocolState.CONFIG, Direction.CLIENT_TO_SERVER, 0x07)
class SKnownPacks(ServerBoundPacket):
    """Serverbound Known Packs - Client responds with subset of packs it also knows.

    Packet ID: 0x07

    Format:
    - packs: Array of (namespace, id, version) triples, e.g.
             {"namespace": "minecraft", "id": "core", "version": "1.21.11"}
    """

    name = "Known Packs (Response)"
    packet_id = 0x07

    # A vanilla client has one built-in pack, so one in practice. Bounded
    # because the count is a client-supplied varint and the loop below runs
    # before anything can fail: an absurd count would otherwise spin
    # read_varint on an exhausted buffer instead of raising once.
    MAX_PACKS = 64

    def __init__(self, packs: list = None):
        """Initialize with the packs the client also knows.

        Args:
            packs: List of dicts with 'namespace', 'id' and 'version' keys,
                  each a string. Same shape CKnownPacks carries, so the two
                  sides of the negotiation can be intersected directly.
        """
        super().__init__()
        self.packs = packs or []

    def write(self, buf: ByteBuffer) -> None:
        buf.write_varint(len(self.packs))
        for pack in self.packs:
            buf.write_string(pack["namespace"])
            buf.write_string(pack["id"])
            buf.write_string(pack["version"])

    def read(self, buf: ByteBuffer) -> None:
        num_packs = buf.read_varint()
        if not 0 <= num_packs <= self.MAX_PACKS:
            raise ValueError(f"Known Packs count out of bounds: {num_packs}")

        self.packs = [
            {
                "namespace": buf.read_string(),
                "id": buf.read_string(),
                "version": buf.read_string(),
            }
            for _ in range(num_packs)
        ]

    def knows(self, namespace: str, identifier: str, version: str) -> bool:
        """Whether the client echoed this exact pack back.

        All three fields have to match. The version is the one that actually
        decides it: same namespace and id with a different version means the
        client has a different revision of that pack, and its contents cannot
        stand in for ours.
        """
        return any(
            pack["namespace"] == namespace
            and pack["id"] == identifier
            and pack["version"] == version
            for pack in self.packs
        )
