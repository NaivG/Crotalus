"""
End-to-end check for the chunk streaming packets.

Exercises the packets through the real encoder (PacketCodec, so the packet ID
and length framing are included) and verifies each body against the expected
bytes for protocol 774.

Run: python tools/check_chunk_packets.py
"""

from __future__ import annotations

import struct
import sys
import zlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from network.buffer import ByteBuffer
from network.packet import PacketCodec
from network.packets import (
    CChunkBatchFinished,
    CChunkBatchStart,
    CLevelChunkWithLight,
    CSetChunkCacheCenter,
    CSetChunkCacheRadius,
    CLightUpdate,
    CUnloadChunk,
)
from network.protocol import Direction, ProtocolState
from world.chunk_codec import ChunkCodec
from tools.check_chunk_codec import make_chunk

# Expected IDs for protocol 774, clientbound PLAY.
EXPECTED_IDS = {
    "CChunkBatchFinished": 0x0B,
    "CChunkBatchStart": 0x0C,
    "CUnloadChunk": 0x25,
    "CLevelChunkWithLight": 0x2C,
    "CLightUpdate": 0x2F,
    "CSetChunkCacheCenter": 0x5C,
    "CSetChunkCacheRadius": 0x5D,
}


def body_of(packet, compression: int = -1) -> tuple[int, bytes]:
    """Encode a packet and split it into (packet_id, body).

    Mirrors what `Connection` does on the way out, including the compression
    frame: [len VarInt][data_len VarInt][payload].
    """
    codec = PacketCodec(state=ProtocolState.PLAY, compression_threshold=compression)
    raw = codec.encode_packet(packet, ByteBuffer())
    reader = ByteBuffer()
    reader.write(raw)
    reader.reset_read()
    frame_len = reader.read_varint()
    assert reader.readable_bytes() == frame_len, "frame length mismatch"

    data = reader.read_bytes(reader.readable_bytes())
    if compression >= 0:
        data_len = ByteBuffer(); data_len.write(data); data_len.reset_read()
        # Note the prefix is the *uncompressed* size, so the compressed bytes
        # run to the end of the frame rather than for `size` bytes.
        size = data_len.read_varint()
        if size:
            data = zlib.decompress(data_len.read_bytes(data_len.readable_bytes()))
            assert len(data) == size, f"uncompressed {len(data)} != declared {size}"
        else:
            data = data_len.read_bytes(data_len.readable_bytes())

    body_reader = ByteBuffer()
    body_reader.write(data)
    body_reader.reset_read()
    return body_reader.read_varint(), body_reader.read_bytes(body_reader.readable_bytes())


def test_ids() -> None:
    for cls, expected in EXPECTED_IDS.items():
        actual = globals()[cls].packet_id
        assert actual == expected, f"{cls}: id 0x{actual:02X} != 0x{expected:02X}"
    print(f"  all {len(EXPECTED_IDS)} packet IDs match protocol 774")


def test_registry() -> None:
    """Every chunk packet must be reachable by ID from a live connection."""
    from network.packet import GLOBAL_REGISTRY

    for cls, expected in EXPECTED_IDS.items():
        found = GLOBAL_REGISTRY.get(ProtocolState.PLAY, Direction.SERVER_TO_CLIENT, expected)
        assert found is globals()[cls], f"registry[{expected:#04x}] -> {found}"
    print("  all registered in GLOBAL_REGISTRY for (PLAY, SERVER_TO_CLIENT)")


def test_small_packets() -> None:
    # Negative coordinates go out as the 32-bit two's complement in an
    # unsigned VarInt; the sign is only restored on read.
    pid, body = body_of(CSetChunkCacheCenter(-3, 7))
    assert pid == 0x5C
    r = ByteBuffer(); r.write(body); r.reset_read()
    assert r.read_varint() == 0xFFFFFFFD, "x=-3 must be two's complement on the wire"
    assert r.read_varint() == 7
    assert r.readable_bytes() == 0

    r2 = ByteBuffer(); r2.write(body); r2.reset_read()
    pkt = CSetChunkCacheCenter()
    pkt.read(r2)
    assert (pkt.chunk_x, pkt.chunk_z) == (-3, 7), f"read back {pkt.chunk_x},{pkt.chunk_z}"
    print("  CSetChunkCacheCenter: signed coords survive the unsigned VarInt (x=-3, z=7)")

    pid, body = body_of(CSetChunkCacheRadius(12))
    assert pid == 0x5D
    r = ByteBuffer(); r.write(body); r.reset_read()
    assert r.read_varint() == 12 and r.readable_bytes() == 0
    print("  CSetChunkCacheRadius: single VarInt round trip")

    pid, body = body_of(CChunkBatchStart())
    assert pid == 0x0C and body == b"", f"batch start body should be empty, got {body!r}"
    print("  CChunkBatchStart: empty body")

    pid, body = body_of(CChunkBatchFinished(37))
    assert pid == 0x0B
    r = ByteBuffer(); r.write(body); r.reset_read()
    assert r.read_varint() == 37 and r.readable_bytes() == 0
    print("  CChunkBatchFinished: count round trip (37)")


def test_unload_chunk_packing() -> None:
    """UnloadChunk is one packed i64, not two integers -- the easy thing to
    get wrong, and getting it wrong desyncs the connection permanently."""
    cases = [(0, 0), (1, 2), (-1, -1), (2147483647, -2147483648), (-3, 7), (5, -9)]
    for x, z in cases:
        pid, body = body_of(CUnloadChunk(x, z))
        assert pid == 0x25
        assert len(body) == 8, f"expected 8 bytes, got {len(body)}"
        key = struct.unpack(">q", body)[0] & 0xFFFFFFFFFFFFFFFF  # raw 64-bit pattern

        expect_x = x & 0xFFFFFFFF
        expect_z = z & 0xFFFFFFFF
        assert key == ((expect_z << 32) | expect_x), f"({x},{z}) key {key:#018x}"
        assert (key & 0xFFFFFFFF) == expect_x
        assert ((key >> 32) & 0xFFFFFFFF) == expect_z

        # And the packet's own read() must invert it.
        pkt = CUnloadChunk()
        r = ByteBuffer(); r.write(body); r.reset_read()
        pkt.read(r)
        assert (pkt.chunk_x, pkt.chunk_z) == (x, z), f"round trip {pkt.chunk_x},{pkt.chunk_z} != {x},{z}"
    print(f"  CUnloadChunk: {len(cases)} positions pack/unpack as a single i64")


def test_chunk_packet() -> None:
    codec = ChunkCodec()
    chunk = make_chunk(3, -7)
    payload = codec.encode(chunk)

    pid, body = body_of(CLevelChunkWithLight(payload))
    assert pid == 0x2C
    assert body == payload, "packet body must be the codec payload verbatim"
    print(f"  CLevelChunkWithLight: 0x2C, body is the {len(body)} B codec payload")

    # With compression on (what the server does in PLAY), the same chunk should
    # collapse hard -- that is the number that decides join time.
    pid_c, body_c = body_of(CLevelChunkWithLight(payload), compression=256)
    assert pid_c == 0x2C
    assert body_c == payload, "compressed frame does not round trip"

    codec_obj = PacketCodec(state=ProtocolState.PLAY, compression_threshold=256)
    framed = codec_obj.encode_packet(CLevelChunkWithLight(payload), ByteBuffer())
    r = ByteBuffer(); r.write(framed); r.reset_read()
    r.read_varint()
    data_len = r.read_varint()
    assert data_len > 0, "expected a compressed frame"
    on_wire = len(framed)
    print(f"  compressed: {len(payload)} B -> {on_wire} B framed ({len(payload)/on_wire:.0f}x)")


def test_light_packet() -> None:
    codec = ChunkCodec()
    chunk = make_chunk(3, -7)
    light = codec.encode_light(chunk)

    pid, body = body_of(CLightUpdate(3, -7, light))
    assert pid == 0x2F
    r = ByteBuffer(); r.write(body); r.reset_read()
    assert r.read_varint() == 3
    assert r.read_varint() == 0xFFFFFFF9, "z=-7 must be two's complement on the wire"
    assert r.read_bytes(r.readable_bytes()) == light
    print(f"  CLightUpdate: 0x2F, {len(light)} B light body round trips")

    # encode_light must agree with the tail of a full chunk encode.
    full = codec.encode(chunk)
    assert full.endswith(light), "light block differs between encode() and encode_light()"
    print("  encode_light() matches the light block inside encode()")


def main() -> int:
    test_ids()
    test_registry()
    test_small_packets()
    test_unload_chunk_packing()
    test_chunk_packet()
    test_light_packet()
    print("\nAll chunk packet checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
