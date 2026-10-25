"""Ad-hoc check: decode realistic client->server bytes for the newly added packets.

Feeds hand-built wire bytes through the real PacketCodec + GLOBAL_REGISTRY path
that Connection._handle_packet uses, so this covers framing, lookup and decode
together rather than testing the classes in isolation.

Run: python tools/check_new_play_packets.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from network.packet import GLOBAL_REGISTRY, PacketCodec
from network.protocol import ProtocolState, Direction
import network.packets  # noqa: F401  - import for @register_packet side effects

# Realistic client->server bytes: packet id followed by payload.
WIRE = {
    0x0A: bytes.fromhex("0a" + "40c00000"),  # Chunk Batch Received, 6.0/tick
    0x1D: bytes.fromhex(
        "1d" + "4059200000000000" + "4050000000000000" + "c069080000000000" + "01"
    ),  # Set Player Position, on ground
    0x1F: bytes.fromhex("1f" + "42b40000" + "c2340000" + "01"),  # Rotation, on ground
    0x20: bytes.fromhex("20" + "00"),  # Movement Flags, airborne
    0x2A: bytes.fromhex("2a" + "35"),  # Player Input, forward|left|jump|sneak
    0x2B: bytes.fromhex("2b"),  # Player Loaded, no payload
    0x1E: bytes.fromhex(
        "1e"
        + "4059200000000000"
        + "4050000000000000"
        + "c069080000000000"
        + "42b40000"
        + "c2340000"
        + "01"
    ),  # Position and Rotation
}


def main() -> None:
    codec = PacketCodec(ProtocolState.PLAY, compression_threshold=-1)

    for pid, data in sorted(WIRE.items()):
        decoded_id, payload = codec.decode_packet(data)
        assert decoded_id == pid, f"id mismatch: {decoded_id:#04x} != {pid:#04x}"

        cls = GLOBAL_REGISTRY.get(ProtocolState.PLAY, Direction.CLIENT_TO_SERVER, decoded_id)
        assert cls is not None, f"0x{pid:02X} not registered"

        packet = GLOBAL_REGISTRY.create_packet(
            ProtocolState.PLAY, Direction.CLIENT_TO_SERVER, decoded_id, payload
        )
        assert packet is not None

        print(f"0x{pid:02X} {data.hex():52s} -> {packet}")
        if hasattr(packet, "active_keys"):
            print(" " * 24 + f"keys: {packet.active_keys()}")
        if hasattr(packet, "on_ground"):
            print(" " * 24 + f"on_ground={packet.on_ground}")


if __name__ == "__main__":
    main()
