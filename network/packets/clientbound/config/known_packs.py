"""
Known Packs (Clientbound, Config)

Opens the data-pack negotiation: the server lists the packs it knows. The
client answers with SKnownPacks naming the subset it also has, and only that
intersection may have its NBT omitted from CRegistryData.
"""

from network.buffer import ByteBuffer
from network.packet import ClientBoundPacket, register_packet
from network.protocol import ProtocolState, Direction


@register_packet(ProtocolState.CONFIG, Direction.SERVER_TO_CLIENT, 0x0E)
class CKnownPacks(ClientBoundPacket):
    """Clientbound Known Packs - Server sends list of data packs known to server.

    This is sent during configuration to negotiate which data packs both sides
    recognize, allowing NBT data for those packs to be omitted from Registry Data.

    Packet ID: 0x0E

    Format:
    - packs: Array of KnownPack
      - namespace: string (e.g., "minecraft")
      - id: string (e.g., "core")
      - version: string (e.g., "1.21.10" for minecraft:core)
    """

    name = "Known Packs"
    packet_id = 0x0E

    def __init__(self, packs: list = None):
        """Initialize with list of known packs.

        Args:
            packs: List of dicts with 'namespace', 'id', 'version' keys
                  e.g., [{"namespace": "minecraft", "id": "core", "version": "1"}]
        """
        super().__init__()
        self.packs = packs or []

    def write(self, buf: ByteBuffer) -> None:
        buf.write_varint(len(self.packs))
        for pack in self.packs:
            buf.write_string(pack.get("namespace", ""))
            buf.write_string(pack.get("id", ""))
            buf.write_string(pack.get("version", ""))

    def read(self, buf: ByteBuffer) -> None:
        num_packs = buf.read_varint()
        self.packs = []
        for _ in range(num_packs):
            self.packs.append({
                "namespace": buf.read_string(),
                "id": buf.read_string(),
                "version": buf.read_string()
            })

    @staticmethod
    def create_minecraft_core_pack(version: str = "1") -> 'CKnownPacks':
        """Create Known Packs with just minecraft:core.

        Args:
            version: Version string, e.g., "1" for vanilla compatibility

        Returns:
            CKnownPacks packet with minecraft:core
        """
        return CKnownPacks([
            {
                "namespace": "minecraft",
                "id": "core",
                "version": version
            }
        ])
