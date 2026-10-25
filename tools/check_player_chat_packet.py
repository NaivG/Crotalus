"""Ad-hoc check: byte layout and index bookkeeping for Player Chat (0x3F).

Player Chat is the one packet whose fields cannot be read off the wiki: the
record has eight components where the older documented shape has thirteen, the
Message is a component rather than a string, and Chat Type is an "ID or inline"
field whose zero is a reserved sentinel rather than registry entry 0. A packet
built to the documented shape decodes without complaint for most of its length
and then runs off the end of the buffer, so this walks the wire exactly the way
the 1.21.11 client does -- in the order transcribed from its own
ClientboundPlayerChatPacket -- rather than reading the fields back off the
object.

Run: python tools/check_player_chat_packet.py
"""

import asyncio
import struct
import sys
import time
import uuid as uuid_mod
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from network.buffer import ByteBuffer
from network.packets import CPlayerChatMessage, CSystemChatMessage
from player.player import Player

from core.config import config
from core.server import CrotalusServer

SENDER_UUID = "11111111-2222-3333-4444-555555555555"

TAG_END = 0x00
TAG_STRING = 0x08
TAG_COMPOUND = 0x0A


class _Reader:
    """Walks a wire buffer the way the client would, asserting as it goes."""

    def __init__(self, raw: bytes):
        self.raw = raw
        self.off = 0

    def varint(self) -> int:
        val = shift = 0
        while True:
            byte = self.raw[self.off]
            self.off += 1
            val |= (byte & 0x7F) << shift
            if not byte & 0x80:
                return val
            shift += 7

    def ushort(self) -> int:
        val = struct.unpack(">H", self.raw[self.off:self.off + 2])[0]
        self.off += 2
        return val

    def long(self) -> int:
        val = struct.unpack(">q", self.raw[self.off:self.off + 8])[0]
        self.off += 8
        return val

    def take(self, n: int) -> bytes:
        val = self.raw[self.off:self.off + n]
        self.off += n
        return val

    def boolean(self) -> bool:
        val = bool(self.raw[self.off])
        self.off += 1
        return val

    def expect(self, byte: int, what: str) -> None:
        assert self.raw[self.off] == byte, f"expected {byte:#04x} for {what}"
        self.off += 1

    def component(self) -> str:
        """Read a network-NBT component holding a single "text" string.

        The root is an unnamed compound, so it carries neither a length nor a
        name -- just entries terminated by TAG_End.
        """
        self.expect(TAG_COMPOUND, "root TAG_Compound")
        text = None
        while True:
            tag = self.raw[self.off]
            self.off += 1
            if tag == TAG_END:
                return text
            assert tag == TAG_STRING, f"expected TAG_String, got {tag:#04x}"
            key = self.take(self.ushort()).decode()
            value = self.take(self.ushort()).decode()
            if key == "text":
                text = value


def check_layout() -> None:
    packet = CPlayerChatMessage(
        SENDER_UUID, "Atoc501", "hello world", index=3, global_index=7
    )
    buf = ByteBuffer()
    packet.write(buf)
    raw = bytes(buf._data)

    r = _Reader(raw)
    assert r.varint() == 7, "Global Index"
    assert str(uuid_mod.UUID(bytes=r.take(16))) == SENDER_UUID, "Sender"
    assert r.varint() == 3, "Index"
    assert r.boolean() is False, "Message Signature must be absent"

    # Signed body: the plaintext a signature would have covered.
    n = r.varint()
    assert r.take(n).decode() == "hello world", "Signed body content"
    assert abs(r.long() - time.time() * 1000) < 60_000, "Timestamp"
    r.long()  # salt, unused while unsigned
    assert r.varint() == 0, "last-seen signature array must be empty"

    # Unsigned Content is a *prefixed optional*. That boolean is a real field;
    # leaving it out shifts every byte after it, and the client does not fail
    # where the shift starts -- it fails several fields later, on a byte that
    # happens to look like an NBT tag id.
    assert r.boolean() is True, "Unsigned Content must be present"
    assert r.component() == "hello world", "Unsigned Content"

    assert r.varint() == 0, "Message Filter must be pass-through"

    # Chat Type is ID-or-inline: zero means an inline definition follows and
    # everything else is a registry index offset by one. minecraft:chat sits at
    # registry index 0, so the wire value is 1. Sending 0 here is what killed
    # the first attempt -- the client went looking for the inline definition,
    # ate the rest of the packet as a registry-id array, and ran out of bytes.
    wire_chat_type = r.varint()
    assert wire_chat_type != 0, "Chat Type 0 is the inline sentinel, not entry 0"
    assert wire_chat_type - 1 == 0, "Chat Type must resolve to minecraft:chat"

    assert r.component() == "Atoc501", "Sender Name"
    assert r.boolean() is False, "Target Name must be absent"

    assert r.off == len(raw), f"trailing bytes: {len(raw) - r.off}"
    print(f"layout ok -- {len(raw)} bytes consumed exactly, in client order")


def check_index_bookkeeping() -> None:
    """Two recipients of the same line must not share a counter or a packet."""

    class Conn:
        # A real Player, so the counter under test is the real one: the
        # server now claims the Global Index off the connection's player
        # rather than off the connection.
        def __init__(self):
            self.in_game = True
            self.remote_address = "test"
            self.player = Player(name="test")
            self.sent: list[CPlayerChatMessage] = []

        async def send_packet(self, packet):
            self.sent.append(packet)

    class Server:
        broadcast_player_chat = CrotalusServer.broadcast_player_chat

        def __init__(self):
            self._lock = asyncio.Lock()
            self._connections = {"a": Conn(), "b": Conn()}

    async def run() -> Server:
        server = Server()
        await server.broadcast_player_chat("Atoc501", SENDER_UUID, 0, "hi")
        await server.broadcast_player_chat("Steve", SENDER_UUID, 0, "yo")
        return server

    server = asyncio.run(run())
    a, b = (server._connections[k].sent for k in ("a", "b"))

    assert [p.global_index for p in a] == [0, 1], "client a Global Index"
    assert [p.global_index for p in b] == [0, 1], "client b Global Index"
    assert all(pa is not pb for pa in a for pb in b), "packets must not be shared"
    assert all(p.message == m for p, m in zip(a, ("hi", "yo"))), "Message"
    assert all(p.sender_name == n for p, n in zip(a, ("Atoc501", "Steve"))), "Sender Name"
    print("index bookkeeping ok -- per-recipient counters 0,1; per-sender Index 0,0")


def check_config_fallback() -> None:
    """player-chat-packet = false must reproduce the old System Chat line."""

    class Conn:
        def __init__(self):
            self.in_game = True
            self.remote_address = "test"
            self.player = Player(name="test")
            self.sent: list = []

        async def send_packet(self, packet):
            self.sent.append(packet)

    class Server:
        broadcast_player_chat = CrotalusServer.broadcast_player_chat
        broadcast_system_message = CrotalusServer.broadcast_system_message

        def __init__(self):
            self._lock = asyncio.Lock()
            self._connections = {"a": Conn()}

    async def run(flag: bool):
        config.set("server.player-chat-packet", flag)
        server = Server()
        await server.broadcast_player_chat("Atoc501", SENDER_UUID, 0, "hi")
        return server._connections["a"].sent[0]

    for flag, expected in ((True, CPlayerChatMessage), (False, CSystemChatMessage)):
        packet = asyncio.run(run(flag))
        assert isinstance(packet, expected), f"player-chat-packet={flag}"
    print("config fallback ok -- False still relays 'Atoc501: hi' as System Chat")


def main() -> None:
    check_layout()
    check_index_bookkeeping()
    check_config_fallback()
    print("\nall player chat checks passed")


if __name__ == "__main__":
    main()
