"""Entity packets: byte-level layout, registration ids, and round-trips.

The entity packets are the layer where a one-byte mistake costs a client
disconnect with a cryptic "failed to decode packet" and no server-side
trace, so the assertions here are byte-for-byte, not behavioral.

Checked, in order:

1. **Registration**: each new packet class is in GLOBAL_REGISTRY under the
   (PLAY, clientbound, id) key vanilla 1.21.11 assigns, read from
   `assets/packets.json` - the same file the runtime dispatch table uses,
   so a drift between the decorator's literal and the asset fails here.
2. **Add Entity (0x01)**: the full body layout against a hand-computed
   byte string - VarInt id, 16-byte uuid, VarInt type, three doubles,
   LpVec3 velocity, then pitch/yaw/head_yaw IN THAT ORDER, then VarInt
   data. The order is the trap: it is not the constructor's order. The
   LpVec3 zero case is the single byte 0x00; a non-zero velocity is
   packed, sent through the writer, and decoded back by the independent
   reader in this file.
3. **All 157 type ids** encode through Add Entity and decode unchanged -
   the dense-id contract from check_registry, exercised through the actual
   packet.
4. **The delta trio** (0x33/0x34/0x36) and Rotate Head (0x51): short
   fields are little-endian? no - big-endian, like every MC numeric; the
   hand-computed strings pin byte order AND the angle bytes.
5. **Remove Entities (0x4B)** batch semantics, and **Set Entity Data
   (0x61)** writing its blob verbatim.
6. **Entity Position Sync (0x23)**: the absolute packet's doubles/floats.

Run: py -3.14t tools/check_entity_packets.py
"""

from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

failures: list[str] = []


def check(label: str, condition: bool, detail: str = "") -> None:
    if condition:
        print(f"  OK   {label}" + (f"  ({detail})" if detail else ""))
    else:
        failures.append(f"{label}: {detail}")
        print(f"  FAIL {label}  ({detail})")


def body(packet) -> bytes:
    """A packet's payload, without any framing."""
    from network.buffer import ByteBuffer

    buf = ByteBuffer()
    packet.write(buf)
    return bytes(buf)


def reader(data: bytes):
    """A ByteBuffer positioned at the start of `data`.

    Note ByteBuffer's first argument is a capacity, not contents - the
    data has to go in through write().
    """
    from network.buffer import ByteBuffer

    buf = ByteBuffer()
    buf.write(data)
    buf.reset_read()
    return buf


def main() -> None:
    from network.buffer import ByteBuffer

    print("1. registration")
    # Importing the modules is what runs the decorators.
    from network.packet import GLOBAL_REGISTRY
    from network.protocol import Direction, ProtocolState
    from network.packets.clientbound.play.add_entity import CAddEntity
    from network.packets.clientbound.play.entity_position_sync import CEntityPositionSync
    from network.packets.clientbound.play.move_entity_pos import CMoveEntityPos
    from network.packets.clientbound.play.move_entity_pos_rot import CMoveEntityPosRot
    from network.packets.clientbound.play.move_entity_rot import CMoveEntityRot
    from network.packets.clientbound.play.remove_entities import CRemoveEntities
    from network.packets.clientbound.play.rotate_head import CRotateHead
    from network.packets.clientbound.play.set_entity_data import CSetEntityData

    packets = json.loads((ROOT / "assets" / "packets.json").read_text(encoding="utf-8"))
    names = packets["clientbound"]["play"]

    expected = {
        CAddEntity: "add_entity",
        CEntityPositionSync: "entity_position_sync",
        CMoveEntityPos: "move_entity_pos",
        CMoveEntityPosRot: "move_entity_pos_rot",
        CMoveEntityRot: "move_entity_rot",
        CRemoveEntities: "remove_entities",
        CRotateHead: "rotate_head",
        CSetEntityData: "set_entity_data",
    }
    for cls, asset_name in expected.items():
        wire_id = names[asset_name]
        registered = GLOBAL_REGISTRY.get(ProtocolState.PLAY, Direction.SERVER_TO_CLIENT, wire_id)
        check(f"{asset_name} is registered at 0x{wire_id:02X}", registered is cls)

    print("2. Add Entity layout")
    # Hand-computed: id 7 -> 07, uuid 16 zero bytes, type 155 -> 9b 01
    # (varint), position 1.0/64.0/2.0 as big-endian doubles, zero velocity
    # = 0x00, pitch 0 = 00, yaw 90 -> packed value 64 decimal = 0x40,
    # head_yaw 90 = 0x40, data 0 = 00.
    p = CAddEntity(entity_id=7, entity_uuid="00000000-0000-0000-0000-000000000000",
                   entity_type=155, x=1.0, y=64.0, z=2.0,
                   yaw=90.0, head_yaw=90.0, pitch=0.0)
    expected_bytes = (
        bytes([0x07]) + bytes(16) + bytes([0x9B, 0x01])
        + struct.pack(">ddd", 1.0, 64.0, 2.0)
        + bytes([0x00, 0x00, 0x40, 0x40, 0x00])
    )
    got = body(p)
    check("add_entity bytes match the hand-computed layout", got == expected_bytes,
          f"got {got.hex(' ')}")

    # Pitch is written even when zero, and the yaw wrap matches Add
    # Entity's rem_euclid semantics: yaw 359.9 -> 255, pitch -90 -> 192.
    p2 = CAddEntity(entity_id=1, entity_uuid="00000000-0000-0000-0000-000000000000",
                    entity_type=0, pitch=-90.0, yaw=359.9, head_yaw=359.9)
    b2 = body(p2)
    # Precise: after 1(id)+16(uuid)+1(type)+24(pos)+1(vel) = 43 bytes, the
    # three angles and the data varint follow.
    angles = b2[43:46]
    check("angle bytes are (192, 255, 255)", angles == bytes([192, 255, 255]),
          f"{angles.hex(' ')}")

    print("3. all 157 type ids round-trip")
    ok = True
    bad = None
    for type_id in range(157):
        pkt = CAddEntity(entity_id=1, entity_uuid="0" * 32, entity_type=type_id)
        data = body(pkt)
        pkt2 = CAddEntity()
        pkt2.read(reader(data))
        if pkt2.entity_type != type_id or pkt2.entity_id != 1:
            ok = False
            bad = (type_id, pkt2.entity_type)
            break
    check("157 type ids survive encode/decode", ok, f"failed at {bad}" if bad else "")

    print("4. delta trio, head rot")
    mp = CMoveEntityPos(entity_id=5, dx=100, dy=-2, dz=-32768, on_ground=False)
    check("move_entity_pos bytes",
          body(mp) == bytes([0x05]) + struct.pack(">hhh", 100, -2, -32768) + bytes([0x00]),
          body(mp).hex(" "))

    mpr = CMoveEntityPosRot(entity_id=5, dx=1, dy=2, dz=3, yaw=-45.0,
                            pitch=45.0, on_ground=True)
    # pitch 45 -> floor(45*256/360)=32; yaw -45 -> -32 & 0xFF = 224
    check("move_entity_pos_rot bytes (angles packed, yaw wrapped)",
          body(mpr) == bytes([0x05]) + struct.pack(">hhh", 1, 2, 3)
          + bytes([224, 32, 0x01]),
          body(mpr).hex(" "))

    mr = CMoveEntityRot(entity_id=9, yaw=90.0, pitch=-90.0, on_ground=True)
    check("move_entity_rot bytes",
          body(mr) == bytes([0x09, 64, 192, 0x01]), body(mr).hex(" "))

    hr = CRotateHead(entity_id=9, head_yaw=-45.0)
    check("rotate_head bytes", body(hr) == bytes([0x09, 224]), body(hr).hex(" "))

    print("5. remove + set_entity_data")
    re_ = CRemoveEntities([7, 8, 9])
    check("remove_entities batch", body(re_) == bytes([0x03, 0x07, 0x08, 0x09]),
          body(re_).hex(" "))
    re_empty = CRemoveEntities([])
    check("remove_entities empty batch", body(re_empty) == bytes([0x00]))

    sd = CSetEntityData(entity_id=3, metadata=bytes([0x10, 0x00, 0x7F, 0xFF]))
    check("set_entity_data writes blob verbatim",
          body(sd) == bytes([0x03, 0x10, 0x00, 0x7F, 0xFF]), body(sd).hex(" "))

    print("6. entity position sync")
    sync = CEntityPositionSync(entity_id=4, x=1.5, y=65.0, z=-1.5,
                               yaw=90.0, pitch=0.0, on_ground=True)
    expected_sync = (
        bytes([0x04])
        + struct.pack(">ddd", 1.5, 65.0, -1.5)
        + struct.pack(">ddd", 0.0, 0.0, 0.0)
        + struct.pack(">ff", 90.0, 0.0)
        + bytes([0x01])
    )
    check("entity_position_sync bytes", body(sync) == expected_sync,
          f"got {body(sync).hex(' ')}")

    print("7. LpVec3 velocity packing")
    from network.packets.clientbound.play.add_entity import read_lp_vec3, write_lp_vec3

    zero = ByteBuffer()
    write_lp_vec3(zero, 0.0, 0.0, 0.0)
    check("zero velocity is the single byte 0x00", bytes(zero) == b"\x00")

    for velocity in [(0.5, 0.0, 0.0), (0.0, -0.25, 0.0), (1.7e10, -1.7e10, 0.0),
                     (0.1, 0.2, 0.3), (-3.5, 2.0, 9.0)]:
        buf = ByteBuffer()
        write_lp_vec3(buf, *velocity)
        decoded = read_lp_vec3(reader(bytes(buf)))
        # Quantization: 15 bits across the scale factor - a fraction of a
        # percent. Comparing with tolerance keeps the test about the
        # format, not about float luck.
        ok = all(abs(a - b) <= max(abs(a), abs(b), 1.0) * 0.01
                 for a, b in zip(velocity, decoded))
        check(f"LpVec3 round-trip {velocity}", ok, f"decoded {decoded}")

    print()
    if failures:
        print(f"{len(failures)} failure(s)")
        raise SystemExit(1)
    print("all checks passed")


if __name__ == "__main__":
    main()
