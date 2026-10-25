"""Ad-hoc check: byte layout for the tab list packets (0x43 / 0x44).

Player Info Update is the packet the client consults *before* it will render a
Player Chat Message, so a layout mistake here does not fail loudly -- it
surfaces as the player typing a message and getting a "chat validation error"
toast instead. That makes it worth walking the wire in the client's order
rather than trusting the encoder round-trips.

The field order below is the client's action-enum ordinal order, which fixes
both the bit values and the sequence an entry writes its fields in. Reading it
in any other order passes a length check and still desyncs.

Run: python tools/check_player_info_packets.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from network.buffer import ByteBuffer
from network.packets import (
    ACTION_ADD_PLAYER,
    ACTION_UPDATE_GAME_MODE,
    ACTION_UPDATE_HAT,
    ACTION_UPDATE_LATENCY,
    ACTION_UPDATE_LISTED,
    CPlayerInfoRemove,
    CPlayerInfoUpdate,
    JOIN_ACTIONS,
    PlayerInfoEntry,
)

PLAYER_UUID = "d75ce882-5f9c-3ea1-9cbd-b32484beccd4"
PLAYER_NAME = "Atoc501"


class _Reader:
    """Walks a wire buffer the way the client would, asserting as it goes."""

    def __init__(self, raw: bytes):
        self.raw = raw
        self.off = 0

    def byte(self) -> int:
        v = self.raw[self.off]
        self.off += 1
        return v

    def boolean(self) -> bool:
        return bool(self.byte())

    def varint(self) -> int:
        val = shift = 0
        while True:
            byte = self.raw[self.off]
            self.off += 1
            val |= (byte & 0x7F) << shift
            if not byte & 0x80:
                return val
            shift += 7

    def string(self) -> str:
        return self.take(self.varint()).decode()

    def uuid(self) -> str:
        import uuid as uuid_mod
        return str(uuid_mod.UUID(bytes=self.take(16)))

    def take(self, n: int) -> bytes:
        v = self.raw[self.off:self.off + n]
        self.off += n
        return v

    def done(self) -> bool:
        return self.off == len(self.raw)


def check_update_layout() -> None:
    packet = CPlayerInfoUpdate.add_player(
        PlayerInfoEntry(PLAYER_UUID, PLAYER_NAME, latency=42)
    )
    buf = ByteBuffer()
    packet.write(buf)
    raw = bytes(buf._data)

    r = _Reader(raw)

    # A byte, not a VarInt: the client reads the mask as an EnumSet.
    mask = r.byte()
    assert mask == JOIN_ACTIONS, f"action mask: got {mask:#04x}"

    assert r.varint() == 1, "entry count"
    assert r.uuid() == PLAYER_UUID, "entry uuid"

    # Fields follow in action-ordinal order, and only if the bit is set.
    assert r.string() == PLAYER_NAME, "ADD_PLAYER name"
    assert r.varint() == 0, "ADD_PLAYER property count"
    assert r.varint() == 0, "UPDATE_GAME_MODE"
    assert r.boolean() is True, "UPDATE_LISTED"
    assert r.varint() == 42, "UPDATE_LATENCY"
    assert r.boolean() is True, "UPDATE_HAT"

    assert r.done(), f"trailing bytes: {len(raw) - r.off}"
    print(f"update layout ok -- {len(raw)} bytes consumed exactly, in client order")


def check_optional_bits_are_absent() -> None:
    """A mask that omits a bit must not write that field at all.

    This is the property the mask exists for: entries are shared across
    packets with different masks, and a field written unconditionally would
    desync every entry after it.
    """
    entry = PlayerInfoEntry(PLAYER_UUID, PLAYER_NAME)
    buf = ByteBuffer()
    CPlayerInfoUpdate([entry], actions=ACTION_ADD_PLAYER).write(buf)
    raw = bytes(buf._data)

    r = _Reader(raw)
    assert r.byte() == ACTION_ADD_PLAYER, "mask"
    assert r.varint() == 1, "entry count"
    assert r.uuid() == PLAYER_UUID, "entry uuid"
    assert r.string() == PLAYER_NAME, "name"
    assert r.varint() == 0, "property count"
    assert r.done(), f"trailing bytes: {len(raw) - r.off}"
    print("mask scoping ok -- ADD_PLAYER alone writes uuid+name and stops")


def check_multi_entry() -> None:
    """Two players in one packet, each with a different name."""
    other = "11111111-2222-3333-4444-555555555555"
    buf = ByteBuffer()
    CPlayerInfoUpdate(
        [
            PlayerInfoEntry(PLAYER_UUID, PLAYER_NAME),
            PlayerInfoEntry(other, "Steve", listed=False),
        ]
    ).write(buf)
    raw = bytes(buf._data)

    r = _Reader(raw)
    assert r.byte() == JOIN_ACTIONS, "mask"
    assert r.varint() == 2, "entry count"

    assert r.uuid() == PLAYER_UUID
    assert r.string() == PLAYER_NAME
    assert r.varint() == 0, "property count"
    r.varint()
    assert r.boolean() is True, "first listed"
    r.varint()
    r.boolean()

    assert r.uuid() == other
    assert r.string() == "Steve"
    assert r.varint() == 0, "property count"
    r.varint()
    assert r.boolean() is False, "second listed"
    r.varint()
    r.boolean()

    assert r.done(), f"trailing bytes: {len(raw) - r.off}"
    print("multi-entry ok -- 2 entries, per-entry listed flag independent")


def check_remove_layout() -> None:
    other = "11111111-2222-3333-4444-555555555555"
    buf = ByteBuffer()
    CPlayerInfoRemove([PLAYER_UUID, other]).write(buf)
    raw = bytes(buf._data)

    r = _Reader(raw)
    assert r.varint() == 2, "uuid count"
    assert r.uuid() == PLAYER_UUID, "first uuid"
    assert r.uuid() == other, "second uuid"
    assert r.done(), f"trailing bytes: {len(raw) - r.off}"
    print(f"remove layout ok -- {len(raw)} bytes consumed exactly")


def check_chat_session_is_not_optional() -> None:
    """Setting INITIALIZE_CHAT without a session is a bug, not an empty state.

    The session is a plain field once the bit is chosen -- unlike the
    unsigned-content field in Player Chat, there is no boolean in front of
    it, so "absent" is spelled by not choosing the bit at all.
    """
    from network.packets.clientbound.play.player_info_update import (
        ACTION_INITIALIZE_CHAT,
    )

    entry = PlayerInfoEntry(PLAYER_UUID, PLAYER_NAME)
    buf = ByteBuffer()
    try:
        CPlayerInfoUpdate([entry], actions=ACTION_INITIALIZE_CHAT).write(buf)
    except ValueError as e:
        assert "no chat session" in str(e), str(e)
        print("chat session ok -- INITIALIZE_CHAT without a session is rejected")
        return
    raise AssertionError("expected ValueError for INITIALIZE_CHAT with no session")


def main() -> None:
    check_update_layout()
    check_optional_bits_are_absent()
    check_multi_entry()
    check_remove_layout()
    check_chat_session_is_not_optional()
    print("\nall player info checks passed")


if __name__ == "__main__":
    main()
