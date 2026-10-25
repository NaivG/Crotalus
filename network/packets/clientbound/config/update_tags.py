"""
Update Tags (Clientbound)

Sends every tag definition the client needs - which blocks are planks, which
items are fuels, and so on. Unlike registry data, tag data is never covered by
the known-packs optimisation and must always be sent in full.

The subtle part: tag entries are numeric registry IDs (VarInt), not string
identifiers. A wrong ID does not error, it silently tags the wrong object.
"""

from network.buffer import ByteBuffer
from network.packet import ClientBoundPacket, register_packet
from network.protocol import ProtocolState, Direction


@register_packet(ProtocolState.CONFIG, Direction.SERVER_TO_CLIENT, 0x0D)
class CUpdateTags(ClientBoundPacket):
    """Update Tags - Server sends all tag data.

    Tag data is NEVER sourced from known packs and must always be sent in full.

    Packet ID: 0x0D

    Format:
    - tags: Array of RegistryTags
      - registry_id: string (e.g., "minecraft:point_of_interest_type")
      - tags: Array of Tag
        - tag_name: Identifier (e.g., "minecraft:acquirable_job_site")
        - entries: Prefixed Array of VarInt (numeric IDs, not strings!)

    Note: Tag entries are NUMERIC IDs (VarInt), not string identifiers.
    The IDs correspond to the registry's internal numeric IDs.
    """

    name = "Update Tags"
    packet_id = 0x0D

    def __init__(self, tags: list = None):
        super().__init__()
        self.tags = tags or []

    def write(self, buf: ByteBuffer) -> None:
        buf.write_varint(len(self.tags))

        for registry_tag in self.tags:
            buf.write_string(registry_tag.get("registry_id", ""))
            buf.write_varint(len(registry_tag.get("tags", [])))

            for tag in registry_tag.get("tags", []):
                buf.write_string(tag.get("name", ""))
                # Protocol: entries are Array of VarInt (numeric IDs, not strings)
                entries = tag.get("entries", [])
                buf.write_varint(len(entries))
                for entry in entries:
                    # Entry can be int (numeric ID) or dict with 'id' key for compat
                    if isinstance(entry, dict):
                        buf.write_varint(entry.get("id", 0))
                    else:
                        buf.write_varint(entry)

    def read(self, buf: ByteBuffer) -> None:
        num_registries = buf.read_varint()
        self.tags = []

        for _ in range(num_registries):
            registry_tag = {
                "registry_id": buf.read_string(),
                "tags": []
            }

            num_tags = buf.read_varint()
            for _ in range(num_tags):
                tag = {
                    "name": buf.read_string(),
                    "entries": []
                }

                # Protocol: entries are Array of VarInt (numeric IDs, not strings)
                num_entries = buf.read_varint()
                for _ in range(num_entries):
                    # Each entry is a VarInt (numeric ID), not a string
                    entry_id = buf.read_varint()
                    tag["entries"].append(entry_id)

                registry_tag["tags"].append(tag)

            self.tags.append(registry_tag)
