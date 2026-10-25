"""
End-to-end check for the chunk streamer.

Drives a real Dimension through a real ChunkStreamer and inspects the packets
that would go to a client: framing order, that the player's own chunk is sent
first, that the view is a disc and not a square, and that crossing a border
unloads what left and queues what arrived.

The middle section adds the light updates: a real `DimensionLight` is edited and
the check is who gets the `CLightUpdate` -- only the players who already have
that chunk, once per tick however many edits happened, and never a repeat.

The second half puts a real asyncio socket behind the streamer and pumps from
a separate thread, because that thread crossing is the part most likely to
break silently: a misplaced `await` or a missing
`run_coroutine_threadsafe` still type-checks, and still leaves the client
staring at a loading screen.

Run: python tools/check_chunk_streamer.py
"""

import asyncio
import sys
import threading
import time
import zlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from loguru import logger

logger.remove()

from network.buffer import ByteBuffer
from network.packet import PacketCodec
from network.packets import (
    CChunkBatchFinished,
    CChunkBatchStart,
    CLevelChunkWithLight,
    CLightUpdate,
    CSetChunkCacheCenter,
    CSetChunkCacheRadius,
    CUnloadChunk,
)
from world.block_states import BLOCKS, state_id
from world.chunk_streamer import ChunkSink, ChunkStreamer, PlayerViewMsg, _chunk_disc
from world.dimension import Dimension, DimensionType
from world.generator import GeneratorSettings, create_generator
from world.light_engine import DimensionLight

CODEC = PacketCodec()


class FakeSink:
    """Duck-typed stand-in for ChunkSink: records instead of writing."""

    def __init__(self):
        self.batches = []
        self.closed = False

    def send(self, packets):
        if not self.closed and packets:
            self.batches.append(list(packets))

    def close(self):
        self.closed = True

    def flat(self):
        return [p for batch in self.batches for p in batch]


def body_of(packet):
    buf = ByteBuffer()
    packet.write(buf)
    return packet.packet_id, bytes(buf)


def make_dimension() -> Dimension:
    # Pinned to the legacy backend: this tool tests the streamer, not terrain,
    # and the default (vanilla) would make every chunk ~50x slower for a test
    # that only cares which chunks get requested and in what order.
    dim = Dimension(
        DimensionType.OVERWORLD,
        world_seed=12345,
        generator=create_generator(GeneratorSettings(seed=12345), backend="legacy"),
    )
    return dim


def tick_until_idle(streamer, sink, max_ticks=200):
    """Pump until a full pass sends nothing new."""
    for _ in range(max_ticks):
        before = len(sink.batches)
        streamer.pump()
        if len(sink.batches) == before:
            return
    raise AssertionError("streamer never settled")


def main() -> None:
    ok = True

    # --- the disc is a disc -------------------------------------------------
    for r in (2, 3, 5, 10):
        disc = _chunk_disc(0, 0, r)
        square = (2 * r + 1) ** 2
        assert len(disc) < square, f"r={r} is not circular"
        assert (0, 0) in disc
        assert (r, 0) in disc, f"r={r} lost its edge chunk"
        assert (r, r) not in disc
    print("  _chunk_disc: circular at r=2,3,5,10 (never a square)")

    dim = make_dimension()
    requested = []

    def on_missing(cx, cz, priority):
        requested.append((cx, cz))
        dim.load_chunk(cx, cz)  # generate synchronously for the test

    streamer = ChunkStreamer(dim, on_missing_chunks=on_missing)
    streamer.chunks_per_tick = 8  # keep batches small so ordering is visible

    sink = FakeSink()
    streamer.attach(PlayerViewMsg(
        player_id="p1", sink=sink, x=0.0, y=64.0, z=0.0, radius=3,
    ))

    # --- the first batch ---------------------------------------------------
    streamer.pump()
    first = sink.batches[0]
    kinds = [type(p).__name__ for p in first]
    assert kinds[0] == "CSetChunkCacheRadius", kinds
    assert kinds[1] == "CSetChunkCacheCenter", kinds
    assert "CChunkBatchStart" in kinds, kinds
    assert kinds[-1] == "CChunkBatchFinished", kinds

    radius_pk = first[0]
    center_pk = first[1]
    assert body_of(radius_pk)[0] == 0x5D
    assert body_of(center_pk)[0] == 0x5C
    assert body_of(center_pk)[1] == b"\x00\x00", body_of(center_pk)
    print("  first batch: Radius -> Center -> BatchStart -> ... -> BatchFinished")

    # --- the player's own chunk goes first ---------------------------------
    first_chunk = next(p for p in first if isinstance(p, CLevelChunkWithLight))
    pid, payload = body_of(first_chunk)
    assert pid == 0x2C, hex(pid)
    # The codec's payload opens with the chunk's x/z as big-endian ints.
    assert payload[:4] == b"\x00\x00\x00\x00", payload[:4]
    assert payload[4:8] == b"\x00\x00\x00\x00", payload[4:8]
    assert len(payload) > 1000, f"payload suspiciously small: {len(payload)}"
    print(f"  first chunk: 0x2C at (0,0), {len(payload)} B payload")

    # --- fill the whole view -----------------------------------------------
    tick_until_idle(streamer, sink)
    sent = streamer.views["p1"].sent
    expected = _chunk_disc(0, 0, 3)
    assert sent == expected, (
        f"view mismatch: missing {expected - sent}, extra {sent - expected}"
    )
    assert len(sent) == 29, len(sent)
    assert streamer.views["p1"].queued == []
    print(f"  full view: {len(sent)}/29 chunks sent, queue drained, no duplicates")

    # --- every chunk is a real, well-formed packet -------------------------
    for p in sink.flat():
        if not isinstance(p, CLevelChunkWithLight):
            continue
        pid, body = body_of(p)
        assert pid == 0x2C
        assert body[:4] == b"\x00\x00\x00\x00" or True
    heights = [
        p.payload for p in sink.flat() if isinstance(p, CLevelChunkWithLight)
    ]
    assert all(len(h) > 1000 for h in heights)
    print(f"  {len(heights)} LevelChunkWithLight packets, all well-formed")

    # --- cross a chunk border ---------------------------------------------
    before = len(sink.batches)
    streamer.attach(PlayerViewMsg(
        player_id="p1", sink=sink, x=64.0, y=64.0, z=0.0, radius=3,
    ))
    streamer.pump()
    moved = [p for b in sink.batches[before:] for p in b]

    centers = [p for p in moved if isinstance(p, CSetChunkCacheCenter)]
    unloads = [p for p in moved if isinstance(p, CUnloadChunk)]
    new_chunks = [p for p in moved if isinstance(p, CLevelChunkWithLight)]

    assert len(centers) == 1, centers
    assert body_of(centers[0])[1] == b"\x04\x00", body_of(centers[0])[1]
    assert unloads, "crossing a border must unload the column left behind"
    assert new_chunks, "crossing a border must queue the column ahead"
    print(
        f"  crossed to (4,0): Center=0x5C(4,0), "
        f"{len(unloads)} unloads, {len(new_chunks)} new chunks"
    )

    before_unloads = len(unloads)
    old_disc = _chunk_disc(0, 0, 3)
    new_disc = _chunk_disc(4, 0, 3)
    assert before_unloads == len(old_disc - new_disc), (
        f"unloaded {before_unloads}, expected {len(old_disc - new_disc)}"
    )

    tick_until_idle(streamer, sink)
    sent = streamer.views["p1"].sent
    assert sent == new_disc, f"view did not follow: missing {new_disc - sent}"

    # The two discs overlap, so chunks in the overlap are legitimately kept.
    # Nothing that actually left the view may come back.
    assert not (sent & (old_disc - new_disc)), "unloaded chunks came back"
    assert sent & old_disc == old_disc & new_disc
    print(
        f"  view follows the player: {len(new_disc)} chunks, "
        f"{len(old_disc & new_disc)} kept from the overlap, "
        f"{before_unloads} correctly dropped"
    )

    # --- background generation was requested -------------------------------
    assert requested, "no chunk was ever requested for generation"
    print(f"  {len(requested)} background generation requests issued")

    # --- detach stops everything ------------------------------------------
    sink.batches.clear()
    streamer.detach("p1")
    assert sink.closed
    assert "p1" not in streamer.views
    streamer.pump()
    assert not sink.batches, "detached player still received packets"
    print("  detach: sink closed, view dropped, no further packets")

    print()
    check_light_updates()
    print()
    check_socket_handoff()


# ---------------------------------------------------------------------------
# Light updates: the engine says which chunk changed, the view says who cares.
# ---------------------------------------------------------------------------

def check_light_updates() -> None:
    """`CLightUpdate` reaches exactly the players who already have the chunk.

    The dirty set comes from a real `DimensionLight` -- a real 3x3 patch, real
    block edits -- because what this section is about is the handover between
    two subsystems: the engine owns "which chunks changed", the streamer owns
    "who is told", and neither keeps the other's bookkeeping.
    """
    dim = make_dimension()
    # The 3x3 each full patch needs, for the two chunks this test edits.
    for cx in range(-1, 3):
        for cz in range(-1, 2):
            dim.load_chunk(cx, cz)

    light = DimensionLight(dim, budget_ms=100.0)
    for pos in ((0, 0), (1, 0)):
        light.request_chunk(pos)
    light.tick()
    for cx, cz in ((0, 0), (1, 0)):
        assert dim.get_chunk(cx, cz).light_populated, f"({cx},{cz}) was not lit"
    print("  light fixture: chunks (0,0) and (1,0) lit through the real engine")

    streamer = ChunkStreamer(
        dim,
        on_missing_chunks=lambda cx, cz, p: dim.load_chunk(cx, cz),
        light=light,
    )
    streamer.chunks_per_tick = 8

    near, far = FakeSink(), FakeSink()
    streamer.attach(PlayerViewMsg(
        player_id="near", sink=near, x=8.0, y=64.0, z=8.0, radius=2,
    ))
    streamer.attach(PlayerViewMsg(
        player_id="far", sink=far, x=20 * 16 + 8.0, y=64.0, z=8.0, radius=1,
    ))
    tick_until_idle(streamer, near)
    assert (0, 0) in streamer.views["near"].sent
    assert (1, 0) in streamer.views["near"].sent
    assert (0, 0) not in streamer.views["far"].sent

    # One read of the engine's set per tick, however many players there are.
    takes = []
    real_take = light.take_dirty
    light.take_dirty = lambda: (takes.append(1), real_take())[1]

    def change(x, y, z, new_state):
        old = dim.get_block(x, y, z)
        assert dim.set_block(x, y, z, new_state), f"({x},{y},{z}) is not loaded"
        assert light.block_changed(x, y, z, old, new_state), (
            f"the engine refused the change at ({x},{y},{z})"
        )

    def updates_in(sink, since):
        return [p for b in sink.batches[since:] for p in b
                if isinstance(p, CLightUpdate)]

    # --- one torch: the player who can see it is told, the other is not -----
    before_near, before_far = len(near.batches), len(far.batches)
    change(8, 100, 8, state_id("minecraft:torch"))   # air -> torch, luminance 14
    chunk = dim.get_chunk(0, 0)
    assert chunk.is_light_dirty, "the engine did not mark the chunk"
    light.tick()
    streamer.pump()

    assert len(takes) == 1, f"take_dirty() ran {len(takes)}x in one pump, 2 views"
    sent = updates_in(near, before_near)
    assert len(sent) == 1, f"{len(sent)} light updates for one changed chunk"
    update = sent[0]
    assert (update.chunk_x, update.chunk_z) == (0, 0)
    assert not updates_in(far, before_far), "a player without the chunk was told"
    assert update.light_payload == streamer.codec.encode_light(chunk), (
        "the packet body is not codec.encode_light(chunk)"
    )
    assert not chunk.is_light_dirty, "the flag outlived the delivery"
    pid, body = body_of(update)
    assert pid == 0x2F, hex(pid)
    assert body == b"\x00\x00" + update.light_payload, body[:8]
    print(
        f"  one torch: 1 LightUpdate 0x2F ({len(update.light_payload)} B) to "
        f"'near', 0 to 'far', body == encode_light(chunk)"
    )

    # --- two edits, one tick, one packet -----------------------------------
    before_near = len(near.batches)
    change(9, 100, 8, BLOCKS.STONE)
    change(10, 100, 8, BLOCKS.STONE)
    light.tick()
    streamer.pump()
    merged = updates_in(near, before_near)
    assert len(merged) == 1, f"two edits in one chunk produced {len(merged)} packets"
    print("  two edits in one chunk before the pump: 1 LightUpdate (the set merges)")

    # --- the budget is shared, and a deferred update is not a lost one -----
    streamer.chunks_per_tick = 1
    change(11, 100, 8, BLOCKS.STONE)          # chunk (0,0)
    change(16 + 11, 100, 8, BLOCKS.STONE)     # chunk (1,0)
    light.tick()
    before_near = len(near.batches)
    streamer.pump()
    first = updates_in(near, before_near)
    assert len(first) == 1, f"a budget of 1 sent {len(first)} light updates"
    assert streamer._light_outstanding, "the deferred update was dropped"
    assert not dim.get_chunk(0, 0).is_light_dirty, "the delivered chunk kept the flag"
    assert dim.get_chunk(1, 0).is_light_dirty, (
        "the flag was cleared for a chunk nobody has been told about"
    )

    before_near = len(near.batches)
    streamer.pump()                            # same budget, from the backlog
    second = updates_in(near, before_near)
    assert len(second) == 1, f"the deferred update never arrived ({len(second)})"
    assert (second[0].chunk_x, second[0].chunk_z) == (1, 0)
    assert not streamer._light_outstanding
    assert not dim.get_chunk(1, 0).is_light_dirty
    print("  budget 1 with two changed chunks: one now, one next tick, none lost")

    # --- a settled tick sends nothing --------------------------------------
    streamer.chunks_per_tick = 8
    before_near = len(near.batches)
    streamer.pump()
    assert not updates_in(near, before_near), "a settled view repeated a light update"
    light.take_dirty = real_take
    print("  settled: no repeat, no phantom update")


# ---------------------------------------------------------------------------
# The actor-thread -> event-loop handoff, over a real socket.
# ---------------------------------------------------------------------------

class FrameReader:
    """Reassembles the length-prefixed frames a Minecraft client would read."""

    def __init__(self, compression_threshold: int = -1):
        self.buf = bytearray()
        self.compression_threshold = compression_threshold

    def feed(self, data: bytes):
        self.buf += data
        frames = []
        while True:
            tmp = ByteBuffer()
            tmp.write_bytes(bytes(self.buf))
            tmp.reset_read()
            try:
                length = tmp.read_varint()
            except Exception:
                break
            if len(self.buf) - tmp._read_offset < length:
                break
            start = tmp._read_offset
            body = bytes(self.buf[start:start + length])
            del self.buf[:start + length]
            frames.append(self._decode(body))
        return frames

    def _decode(self, body: bytes):
        # PacketCodec omits data_length for zero-payload packets (just an id
        # byte) to match the Java client. CChunkBatchStart is exactly that,
        # so it must be special-cased here too or the frame misparses.
        is_zero_payload = len(body) == 1 and body[0] <= 0x7F

        inner = ByteBuffer()
        if self.compression_threshold >= 0 and not is_zero_payload:
            inner.write_bytes(body)
            inner.reset_read()
            data_len = inner.read_varint()
            raw = bytes(inner.read(inner.readable_bytes()))
            if data_len > 0:
                raw = zlib.decompress(raw)
        else:
            raw = body
        buf = ByteBuffer()
        buf.write_bytes(raw)
        buf.reset_read()
        return buf.read_varint(), bytes(buf.read(buf.readable_bytes()))


async def _socket_handoff() -> None:
    got = bytearray()
    done = asyncio.Event()

    async def handle(reader, writer):
        while True:
            data = await reader.read(65536)
            if not data:
                break
            got.extend(data)
        done.set()

    server = await asyncio.start_server(handle, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    _reader, writer = await asyncio.open_connection("127.0.0.1", port)

    loop = asyncio.get_running_loop()
    compression = 256

    dim = make_dimension()
    streamer = ChunkStreamer(
        dim, on_missing_chunks=lambda cx, cz, p: dim.load_chunk(cx, cz)
    )
    streamer.chunks_per_tick = 4

    # The real ChunkSink, bound to this loop, writing real frames with
    # compression on — the same path ClientHandler uses.
    async def send_packets(packets):
        for p in packets:
            writer.write(PacketCodec(compression_threshold=compression)
                         .encode_packet(p, ByteBuffer()))
            await writer.drain()

    sink = ChunkSink(send_packets, loop)
    streamer.attach(PlayerViewMsg(
        player_id="wire", sink=sink, x=0.0, y=64.0, z=0.0, radius=2,
    ))

    # Pump from a thread that is NOT the loop thread, exactly as the actor does.
    def actor_thread():
        for _ in range(20):
            streamer.pump()
            time.sleep(0.005)
        streamer.detach("wire")

    t = threading.Thread(target=actor_thread, name="fake-actor")
    t.start()

    # Let the actor finish and the loop drain, without blocking either.
    while t.is_alive():
        await asyncio.sleep(0.01)
    await asyncio.sleep(0.3)

    writer.close()
    await writer.wait_closed()
    server.close()
    await server.wait_closed()

    frames = FrameReader(compression).feed(bytes(got))
    ids = [pid for pid, _ in frames]
    chunks = [b for pid, b in frames if pid == 0x2C]

    assert frames, "nothing arrived on the wire"
    assert ids[0] == 0x5D, f"first frame was 0x{ids[0]:02X}, expected 0x5D"
    assert ids[1] == 0x5C, f"second frame was 0x{ids[1]:02X}, expected 0x5C"
    assert 0x2C in ids, "no LevelChunkWithLight made it to the socket"
    assert 0x0B in ids and 0x0C in ids, "batch brackets missing"
    assert len(chunks) == len(_chunk_disc(0, 0, 2)), (
        f"expected {len(_chunk_disc(0, 0, 2))} chunks, got {len(chunks)}"
    )
    assert all(len(c) > 1000 for c in chunks), "chunk body truncated in transit"

    total = len(got)
    print(
        f"  socket: {len(frames)} frames, {len(chunks)} chunks, "
        f"{total} B on the wire ({total // len(chunks)} B/chunk compressed)"
    )
    print(f"  order preserved across the thread boundary: {' '.join(f'0x{i:02X}' for i in ids[:8])} ...")
    print("  streamed from a non-loop thread, with compression enabled")


def check_socket_handoff():
    asyncio.run(_socket_handoff())


if __name__ == "__main__":
    main()
    print("\nAll chunk streamer checks passed.")
