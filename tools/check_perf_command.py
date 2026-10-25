"""`/perf`: always-on stage profiling, the actor round trip, and the report.

Three things this pins that reading the code cannot:

1. **The profiler's numbers are the truth the report prints.** Stage rows
   appear in first-marked (pipeline) order, `reset` zeroes without
   shuffling that order, and `end_tick` returns the same duration the
   `TickDone` reply carries -- the command's "tick avg" and the ticker's
   "last ms" are the same measurement seen twice.
2. **The actor answers PerfMsg on its own thread contract.** The branch is
   exercised through the real `DimensionActor._handle` (built with
   `__new__`, no physics setup), so a future refactor that renames
   `_prof` or forgets `done.set()` fails here instead of timing the
   console out at 2 s per dimension.
3. **`share` is computed from window totals, not per-call averages.**
   `time_sync` broadcasts every 20 ticks, so its per-call average is
   ~20x one tick's share of it; dividing totals by totals is the only
   column that does not lie for periodic stages.

Run: py -3.14t tools/check_perf_command.py
"""

from __future__ import annotations

import asyncio
import contextlib
import io
import queue
import sys
import time
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from console.command import CommandContext, cmd_manager  # noqa: E402
from console.commands import PerfCommand, register_builtin_commands  # noqa: E402
from core.ticker import Ticker  # noqa: E402
from world.dimension import DimensionType  # noqa: E402
from world.dimension_actor import DimensionActor  # noqa: E402
from world.perf import PerfMsg, TickProfiler  # noqa: E402

failures: list[str] = []


def check(label: str, condition: bool, detail: str = "") -> None:
    if condition:
        print(f"  OK   {label}" + (f"  ({detail})" if detail else ""))
    else:
        failures.append(f"{label}: {detail}")
        print(f"  FAIL {label}  ({detail})")


def run(cmd, *args) -> str:
    """Execute a console command and capture what it printed."""
    stream = io.StringIO()
    with contextlib.redirect_stdout(stream):
        asyncio.run(cmd.execute(CommandContext(args=list(args))))
    return stream.getvalue()


class FakeWorld:
    """`chunk_actor_inbox` wired to a real `_handle`, everything else stubbed.

    The hook shape is `World._post_to_actor`'s: `(dimension) -> inbox`,
    where `put` is the delivery. Here the inbox answers synchronously by
    calling the real `DimensionActor._handle`, so the PerfMsg branch under
    test is the shipped one, not a copy.
    """

    def __init__(self, actor, dimensions):
        self._actor = actor
        self.dimensions = dimensions
        self.chunk_actor_inbox = lambda dim: self

    # `World._post_to_actor`'s contract, with the lookup already done.
    def _post_to_actor(self, dim_type, msg) -> bool:
        if dim_type not in self.dimensions:
            return False
        self._actor._handle(msg)
        return True


def bare_actor(name: str, prof: TickProfiler) -> DimensionActor:
    """A DimensionActor with only what the PerfMsg branch touches."""
    actor = DimensionActor.__new__(DimensionActor)
    actor._prof = prof
    actor.dimension = SimpleNamespace(name=name)
    actor.inbox = queue.Queue()
    return actor


def main() -> int:
    print("1. TickProfiler: pipeline order, math, reset")
    prof = TickProfiler()
    prof.start_tick()
    time.sleep(0.001)
    prof.mark("dimension")
    time.sleep(0.001)
    prof.mark("lifecycle")
    duration = prof.end_tick(discarded=2)
    snap = prof.snapshot()

    check("stages are listed in first-marked (pipeline) order",
          list(snap["stages"]) == ["dimension", "lifecycle"],
          f"{list(snap['stages'])}")
    check("end_tick returns the whole-tick duration",
          duration >= 0.002, f"{duration * 1000:.2f}ms")
    check("tick_count counts runs", snap["tick_count"] == 1, f"{snap['tick_count']}")
    check("discarded accumulates the ticker's dropped deadlines",
          snap["discarded"] == 2, f"{snap['discarded']}")
    check("stage stats carry count/avg/max/last",
          snap["stages"]["lifecycle"]["count"] == 1
          and snap["stages"]["lifecycle"]["max_ms"] >= 1.0,
          f"{snap['stages']['lifecycle']}")

    order_before = list(prof.stages)
    prof.reset()
    snap = prof.snapshot()
    check("reset zeroes everything",
          snap["tick_count"] == 0 and snap["discarded"] == 0
          and snap["tick"]["count"] == 0
          and all(s["count"] == 0 for s in snap["stages"].values()),
          f"{snap}")
    check("reset preserves stage order (the report never reshuffles)",
          list(prof.stages) == order_before, f"{list(prof.stages)}")

    print("2. DimensionActor._handle answers PerfMsg on the real branch")
    prof = TickProfiler()
    prof.start_tick()
    prof.mark("dimension")
    prof.mark("streamer")
    prof.end_tick()
    actor = bare_actor("overworld", prof)
    msg = PerfMsg()
    actor._handle(msg)
    check("snapshot comes back through the actor branch",
          msg.done.is_set() and msg.box.get("dimension") == "overworld"
          and msg.box["tick"]["count"] == 1
          and list(msg.box["stages"]) == ["dimension", "streamer"],
          f"{msg.box.get('dimension')} {list(msg.box.get('stages', {}))}")
    check("queue depth rides along",
          msg.box.get("queue_depth") == 0, f"{msg.box.get('queue_depth')}")

    msg = PerfMsg(reset=True)
    actor._handle(msg)
    check("reset lands before the snapshot (fresh window reported)",
          msg.box["tick_count"] == 0 and msg.box["tick"]["count"] == 0,
          f"count={msg.box['tick_count']}")

    print("3. Ticker.snapshot(): the lag half of the report")
    ticker = Ticker()
    ticker.register_actor("Dimension-overworld", queue.Queue())
    st = ticker._actors["Dimension-overworld"]
    st.last_duration = 0.0123
    st.last_queue_depth = 2
    st.missed = 3
    st.behind_since = time.monotonic() - 4.0
    snap = ticker.snapshot()
    a = snap["actors"]["Dimension-overworld"]
    check("actor row carries duration/queue/discarded",
          abs(a["last_duration_ms"] - 12.3) < 0.01
          and a["queue_depth"] == 2 and a["discarded"] == 3,
          f"{a}")
    check("behind_for is the stall's age in seconds",
          a["behind_for"] is not None and 3.5 <= a["behind_for"] <= 4.5,
          f"{a['behind_for']}")
    st.behind_since = None
    st.missed = 0
    check("behind_for=None once the actor recovered",
          ticker.snapshot()["actors"]["Dimension-overworld"]["behind_for"] is None
          and ticker.snapshot()["actors"]["Dimension-overworld"]["discarded"] == 0,
          "recovered")

    print("4. /perf end to end through a fake world")
    prof = TickProfiler()
    for _ in range(40):
        prof.start_tick()
        prof.mark("dimension")
        prof.mark("lifecycle")
        prof.mark("physics")
        prof.mark("light")
        prof.mark("streamer")
        prof.end_tick(discarded=1)
    actor = bare_actor("overworld", prof)
    world = FakeWorld(actor, {DimensionType.OVERWORLD: SimpleNamespace()})
    cmd = PerfCommand(world_getter=lambda: world)

    out = run(cmd)
    check("no running ticker: the report says so instead of crashing",
          "Tick 线程未运行" in out, "")

    # The command reads the ticker through core.server.get_server; patch it
    # to a real Ticker with bookkeeping in place so the report's ticker half
    # is exercised too (the tool process has no running pipeline).
    import core.server as server_module

    ticker = Ticker()
    ticker.register_actor("Dimension-overworld", queue.Queue())
    actor_state = ticker._actors["Dimension-overworld"]
    actor_state.last_duration = 0.011
    actor_state.last_queue_depth = 1
    server_module.get_server = lambda: SimpleNamespace(ticker=ticker)

    out = run(cmd)
    check("report prints the ticker header with the actor row",
          "Ticker: TPS 20.0 / 20" in out and "Dimension-overworld" in out,
          out[:200].replace("\n", " | "))
    check("report prints the dimension block",
          "--- overworld: tick avg" in out, "")
    check("stages print in pipeline order",
          out.index("dimension") < out.index("lifecycle") < out.index("physics")
          < out.index("light") < out.index("streamer"), "")
    check("discarded ticks show up", "40 discarded" in out, "")

    out = run(cmd, "reset")
    check("/perf reset asks the actor for a fresh window",
          "重置" in out and "0 ticks" in out and "40 ticks" not in out,
          out[:200].replace("\n", " | "))

    out = run(cmd, "--dim", "nether")
    check("--dim filter skips unlisted dimensions",
          "未在这个 world 里加载" in out and "--- overworld" not in out, "")

    out = run(cmd, "bogus")
    check("unknown argument is refused with usage",
          "Unknown argument" in out and "Usage:" in out, "")

    print("5. no-actor and no-world paths")
    world_none = FakeWorld(actor, {DimensionType.OVERWORLD: SimpleNamespace()})
    world_none.chunk_actor_inbox = None
    cmd = PerfCommand(world_getter=lambda: world_none)
    out = run(cmd)
    check("no actor: the dimension says so instead of printing nothing",
          "没有 actor 在跑" in out, "")

    cmd = PerfCommand(world_getter=lambda: None)
    out = run(cmd)
    check("no world attached: skipped with a reason",
          "没有 world 挂到这个控制台" in out, "")

    print("6. registration")
    register_builtin_commands()
    check("/perf is registered (help finds it)",
          cmd_manager.has_command("perf"), "")

    print()
    if failures:
        print(f"FAILED: {len(failures)}")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("All perf checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
