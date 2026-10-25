"""The world clock reaching the client: one packet, one cadence, one value.

The client keeps its own clock (`dayTime` advances every tick, and the sky, the
lightmap's sky brightness and the sun all read from it). `CUpdateTime` is the
only thing that ties it to the server's, and the symptom of not sending it is
not an error but a *look*: a server at noon watched from a client that renders
dusk, where sky-lit terrain goes dim and block light does not, so torches and
glowstone look like the only correct lighting in the world.

Checked, in order:

1. **Registration and layout**: `CUpdateTime` is at 0x6F, registered under
   (PLAY, clientbound), and its body is exactly `i64 world_age`, `i64
   time_of_day`, one byte `increasing` -- hand-computed against the bytes
   protocol 774 (the 1.21.11 snapshot in `JavaEditon-Packets.mdx`) describes.
2. **The clock advances**: `Dimension.tick` moves `world_time` and `day_time`,
   and `advance_clock(gap - 1)` catches up the ticks a busy actor discarded, so
   the clock tracks the wall clock rather than the number of ticks that ran.
3. **A joining player is told the time immediately** -- once for the attach, not
   once per view update, because a moving player posts the same message every
   tick and the sky is not worth 20 packets a second.
4. **Everyone watching is told again every 20 ticks** and not in between, with
   the value the dimension's clock has at that moment.
5. **A pinned dimension says so**: `increasing=False` where
   `DimensionSettings.fixed_time` is set (the End's twilight), which is the same
   rule `Dimension.advance_clock` uses to freeze its clock.

Run: py -3.14t tools/check_world_time.py
"""

from __future__ import annotations

import queue as queue_module
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.ticker import TickMsg  # noqa: E402
from network.packet import GLOBAL_REGISTRY  # noqa: E402
from network.protocol import ProtocolState, Direction  # noqa: E402
import network.packets  # noqa: F401,E402  - @register_packet side effects
from network.packets.clientbound.play.update_time import CUpdateTime  # noqa: E402
from world.chunk_streamer import PlayerViewMsg  # noqa: E402
from world.dimension import Dimension, DimensionType  # noqa: E402
from world.dimension_actor import TIME_SYNC_TICKS, DimensionActor  # noqa: E402

failures: list[str] = []


def check(label: str, condition: bool, detail: str = "") -> None:
    if condition:
        print(f"  OK   {label}" + (f"  ({detail})" if detail else ""))
    else:
        failures.append(f"{label}: {detail}")
        print(f"  FAIL {label}  ({detail})")


class FakeSink:
    """A player's packet sink: records batches, and nothing else."""

    def __init__(self):
        self.batches: list[list] = []
        self.closed = False

    def send(self, packets) -> None:
        if not self.closed:
            self.batches.append(list(packets))

    def close(self) -> None:
        self.closed = True

    def times(self) -> list[CUpdateTime]:
        return [p for batch in self.batches for p in batch
                if isinstance(p, CUpdateTime)]


def body(packet) -> bytes:
    from network.buffer import ByteBuffer

    buf = ByteBuffer()
    packet.write(buf)
    return bytes(buf)


def make_dimension(dim_type: DimensionType = DimensionType.OVERWORLD) -> Dimension:
    """A dimension with no blocks behind it.

    The clock and the packet do not touch terrain, and the vanilla generator
    costs about a second per chunk (`check_chunk_lifecycle` says the same), so
    the generator here is the cheap legacy one and nothing ever asks it for a
    chunk.
    """
    from world.generator import GeneratorSettings, create_generator

    return Dimension(
        dim_type,
        world_seed=12345,
        generator=create_generator(GeneratorSettings(seed=12345), backend="legacy"),
    )


def make_actor(dimension, sink):
    actor = DimensionActor(dimension, queue_module.Queue())
    actor._handle(PlayerViewMsg(player_id="p1", sink=sink, x=8.0, y=64.0, z=8.0,
                                radius=1, simulation_radius=1))
    return actor


def tick(actor, n: int, start: int = 0) -> None:
    for i in range(start + 1, start + n + 1):
        actor._handle(TickMsg(tick_no=i, sent_at=time.monotonic()))


def main() -> int:
    print("1. the packet: id, registration, bytes")
    assert GLOBAL_REGISTRY.get(ProtocolState.PLAY, Direction.SERVER_TO_CLIENT, 0x6F) \
        is CUpdateTime, "CUpdateTime is not registered at 0x6F"
    raw = body(CUpdateTime(world_age=1234, time_of_day=6000, increasing=True))
    want = (1234).to_bytes(8, "big", signed=True) + \
           (6000).to_bytes(8, "big", signed=True) + b"\x01"
    check("registered at 0x6F for (PLAY, clientbound)", True)
    check("body is i64 + i64 + bool, big-endian", raw == want,
          f"{raw.hex()} vs {want.hex()}")
    check("no framing of its own (the codec adds it)", len(raw) == 17, f"{len(raw)} B")
    stopped = body(CUpdateTime(0, 6000, increasing=False))
    check("a frozen clock writes increasing=False", stopped.endswith(b"\x00"))

    print("\n2. the dimension's clock advances")
    dimension = make_dimension()
    start_age, start_day = dimension.world_time, dimension.day_time
    actor = make_actor(dimension, FakeSink())
    tick(actor, 5)
    check("five ticks move both counters by five",
          (dimension.world_time, dimension.day_time)
          == (start_age + 5, start_day + 5),
          f"{dimension.world_time - start_age} ticks")

    print("\n3. a joining player is told the time at once")
    sink = FakeSink()
    actor2 = make_actor(make_dimension(), sink)
    first = sink.times()
    check("one packet on attach", len(first) == 1, f"{len(first)} packets")
    check("carrying the dimension's clock", first and first[0].time_of_day == 6000,
          f"{first[0].time_of_day if first else None}")
    check("and the cycle running", first and first[0].increasing is True)

    print("\n4. everyone watching is corrected every 20 ticks")
    dimension3 = make_dimension()
    sink3 = FakeSink()
    actor3 = make_actor(dimension3, sink3)
    baseline = len(sink3.times())
    tick(actor3, TIME_SYNC_TICKS - 1)
    check("nothing before the 20th tick", len(sink3.times()) == baseline,
          f"{len(sink3.times()) - baseline} extra")
    tick(actor3, 1, start=TIME_SYNC_TICKS - 1)
    check("one at the 20th tick", len(sink3.times()) == baseline + 1,
          f"{len(sink3.times()) - baseline} extra")
    tick(actor3, TIME_SYNC_TICKS)
    check("and one at the 40th", len(sink3.times()) == baseline + 2,
          f"{len(sink3.times()) - baseline} extra")
    latest = sink3.times()[-1]
    check("reporting the clock as of that tick",
          latest.time_of_day == dimension3.day_time,
          f"packet {latest.time_of_day} vs clock {dimension3.day_time}")
    check("the world age is not the time of day",
          latest.world_age == dimension3.world_time)

    print("\n5. a moving player does not re-trigger the join packet")
    moved = len(sink3.times())
    actor3._handle(PlayerViewMsg(player_id="p1", sink=sink3, x=40.0, y=64.0, z=8.0,
                                 radius=1, simulation_radius=1))
    check("moving the view sends no time packet", len(sink3.times()) == moved,
          f"{len(sink3.times()) - moved} extra")

    print("\n6. a pinned dimension tells the client not to advance")
    end = make_dimension(DimensionType.END)
    end_sink = FakeSink()
    end_actor = make_actor(end, end_sink)
    tick(end_actor, 25)
    pinned = end_sink.times()
    check("the End's clock is pinned by its settings",
          end.settings.fixed_time is not None, f"fixed_time={end.settings.fixed_time}")
    check("the client is told the cycle does not run",
          pinned and all(not packet.increasing for packet in pinned),
          f"{len(pinned)} packets")
    check("and the clock itself did not move", end.day_time == 6000,
          f"{end.day_time}")

    print()
    if failures:
        print(f"RESULT: FAIL ({len(failures)})")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print("RESULT: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
