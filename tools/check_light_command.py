"""`/light`: signed coordinates, the three relight shapes, and lazy queueing.

Two things this pins that a fixture cannot show by looking at light values:

1. **`-13` is a coordinate, not a flag.** Coordinates are signed and most of the
   map is west or north of the origin, so treating a leading `-` as "this is an
   option" makes `/light relight -13 0` -- and every query there -- answer
   "Unknown option: -13". Only `--` makes a token an option.
2. **A relight request is a queue, not a loop.** `/light relight all` on a
   loaded dimension is a few hundred deque appends and an immediate answer; the
   patches run inside the light tick's budget afterwards. A version that patched
   inline would time the console out (and `/light` already waits at most
   `LIGHT_QUERY_TIMEOUT` for the actor).

Checked, in order:

1. **Argument parsing**: negative world and chunk coordinates survive, `--dim`,
   `--dim=` and a trailing dimension name still work, and a real flag is still
   refused.
2. **The three relight shapes** (`<cx> <cz>`, `<cx> <cz> <radius>`, `all`),
   including the refusals: negative radius, a radius that would queue a
   city-sized square, and a missing coordinate.
3. **Lazy queueing**: `relight_area` counts targets, skips what has no generated
   chunk behind it, and grows `pending_chunks` without computing anything.
4. **`all` covers the dimension**: every generated chunk in the manager, blank
   claims excluded.
5. **The command end to end, with no actor** (the tools/tests path): the inline
   drain actually lights the chunks it queued, and the output says so.

Run: py -3.14t tools/check_light_command.py
"""

from __future__ import annotations

import asyncio
import contextlib
import io
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from console.command import CommandContext  # noqa: E402
from console.commands import LightCommand, _parse_light_args  # noqa: E402
from world.chunk import Chunk, ChunkPos  # noqa: E402
from world.dimension import DimensionType  # noqa: E402
from world.light_engine import DimensionLight  # noqa: E402
from world.light_query import relight_all, relight_area  # noqa: E402

failures: list[str] = []


def check(label: str, condition: bool, detail: str = "") -> None:
    if condition:
        print(f"  OK   {label}" + (f"  ({detail})" if detail else ""))
    else:
        failures.append(f"{label}: {detail}")
        print(f"  FAIL {label}  ({detail})")


class FakeDimension:
    """Just enough of a `Dimension` for the light query layer.

    Real `Chunk` objects, because the engine reads their section arrays; a stub
    dimension around them, because the clock, the tickets and the generator have
    nothing to do with `/light`.
    """

    def __init__(self):
        self.chunks: dict = {}
        self.chunk_manager = SimpleNamespace(chunks=self.chunks)
        self.settings = SimpleNamespace(min_y=-64, height=384, has_skylight=True,
                                        fixed_time=None)
        self.light = None

    def get_chunk(self, chunk_x: int, chunk_z: int):
        chunk = self.chunks.get(ChunkPos(chunk_x, chunk_z))
        return chunk

    def add(self, chunk_x: int, chunk_z: int, *, generated: bool = True):
        chunk = Chunk(position=ChunkPos(chunk_x, chunk_z))
        chunk.is_generated = generated
        self.chunks[chunk.position] = chunk
        return chunk


class FakeWorld:
    def __init__(self, dimension):
        self.dimensions = {DimensionType.OVERWORLD: dimension}
        # No `chunk_actor_inbox`: nobody owns the dimension, so the command does
        # the work on this thread. That is the branch a tool exercises.
        self.chunk_actor_inbox = None


def run(cmd, *args) -> str:
    """Execute a console command and capture what it printed."""
    stream = io.StringIO()
    with contextlib.redirect_stdout(stream):
        asyncio.run(cmd.execute(CommandContext(args=list(args))))
    return stream.getvalue()


def main() -> int:
    print("1. argument parsing: signed coordinates are coordinates")
    positional, dim, error = _parse_light_args(["-13", "70", "-2"])
    check("a negative x/y/z parse as coordinates",
          (positional, dim, error) == (["-13", "70", "-2"], None, None),
          f"{positional} {error}")
    positional, dim, error = _parse_light_args(["relight", "-13", "-2"])
    check("a negative chunk coordinate parses too",
          (positional, dim, error) == (["relight", "-13", "-2"], None, None),
          f"{positional} {error}")
    positional, dim, error = _parse_light_args(["relight", "-1", "0", "--dim", "nether"])
    check("--dim still comes out", (positional, dim, error) == (["relight", "-1", "0"], "nether", None),
          f"{positional} {dim}")
    positional, dim, error = _parse_light_args(["relight", "4", "5", "--dim=end"])
    check("--dim=end still comes out", (positional, dim, error) == (["relight", "4", "5"], "end", None),
          f"{positional} {dim}")
    positional, dim, error = _parse_light_args(["relight", "4", "5", "nether"])
    check("a bare trailing dimension name still comes out",
          (positional, dim, error) == (["relight", "4", "5"], "nether", None),
          f"{positional} {dim}")
    _positional, _dim, error = _parse_light_args(["relight", "4", "5", "--time"])
    check("an unknown flag is still refused", error == "Unknown option: --time", str(error))

    print("\n2. the three relight shapes")
    dimension = FakeDimension()
    command = LightCommand(world_getter=lambda: FakeWorld(dimension))
    target = command._relight_target(["12", "-7"])
    check("one chunk", target == (False, 12, -7, 0), str(target))
    target = command._relight_target(["12", "-7", "2"])
    check("a square", target == (False, 12, -7, 2), str(target))
    target = command._relight_target(["all"])
    check("all", target == (True, 0, 0, 0), str(target))
    check("a negative radius is refused", command._relight_target(["0", "0", "-1"]) is None)
    check("a city-sized radius is refused", command._relight_target(["0", "0", "999"]) is None)
    check("a missing coordinate is refused", command._relight_target(["7"]) is None)
    check("garbage is refused", command._relight_target(["x", "0"]) is None)

    print("\n3. queueing is lazy: counts now, patches later")
    dimension = FakeDimension()
    dimension.add(0, 0)
    dimension.add(1, 0)
    dimension.add(-1, 0)
    dimension.add(0, 1, generated=False)          # a blank claim
    engine = DimensionLight(dimension, chunks_per_tick=4)
    dimension.light = engine
    answer = relight_area(dimension, engine, 0, 0, radius=1)
    check("a 3x3 request asks for nine chunks", answer["chunks"] == 9, str(answer))
    check("three have a generated chunk behind them", answer["queued"] == 3, str(answer))
    check("the blank claim is skipped, the rest counted",
          answer["skipped"] == 6, str(answer))
    check("nothing was computed yet", engine.chunks_lit == 0, str(engine.chunks_lit))
    check("the queue holds exactly what was accepted",
          engine.pending_chunks == 3, str(engine.pending_chunks))

    print("\n4. 'all' means every generated chunk")
    answer = relight_all(dimension, engine)
    check("three chunks, one skipped claim",
          (answer["chunks"], answer["queued"], answer["skipped"]) == (4, 3, 1), str(answer))
    check("with no anchor there is no 3x3 line", "neighbours" not in answer)

    print("\n5. the command end to end, with no actor")
    # A 5x5 grid centred on (-1,-1): every chunk in the 3x3 request has its own
    # 3x3 inside this, which is what a patch needs (`region.ready`) -- with a
    # thinner grid the engine correctly *defers* the patches whose neighbours
    # are missing, and the check would be measuring the fixture.
    dimension = FakeDimension()
    for cx in range(-3, 2):
        for cz in range(-3, 2):
            dimension.add(cx, cz)
    output = run(command, "relight", "-1", "-1", "1")
    check("negative chunk coordinates reach the dimension",
          "Unknown option" not in output, output.strip()[:120])
    lit = [chunk for chunk in dimension.chunks.values() if chunk.light_populated]
    check("the 3x3 around (-1,-1) was queued and lit", len(lit) == 9, f"{len(lit)} chunks")
    check("the other sixteen chunks were left alone", len(dimension.chunks) - len(lit) == 16,
          f"{len(dimension.chunks)} chunks in the fixture")

    print("\n6. 'all' is a queue, not a loop")
    # An engine already attached to the dimension owns the draining (a tool's
    # engine, or the actor's), so the command must queue and return -- never
    # tick 700 patches on the console thread while the console waits.
    queued_dimension = FakeDimension()
    for cx in range(-3, 2):
        for cz in range(-3, 2):
            queued_dimension.add(cx, cz)
    queued_engine = DimensionLight(queued_dimension, chunks_per_tick=1)
    queued_dimension.light = queued_engine
    queued_command = LightCommand(world_getter=lambda: FakeWorld(queued_dimension))
    output = run(queued_command, "relight", "all")
    check("one queued patch per generated chunk", queued_engine.pending_chunks == 25,
          f"{queued_engine.pending_chunks} queued")
    check("and nothing was computed in the call", queued_engine.chunks_lit == 0,
          f"{queued_engine.chunks_lit} patches")
    check("the command reports it as queued, not as an error",
          "区块未加载" not in output and "失败" not in output, output.strip()[:80])

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
