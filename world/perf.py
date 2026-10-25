"""
Per-stage tick profiling.

`TickProfiler` lives on the dimension actor and is fed by `_on_tick`: one
`perf_counter()` call per pipeline stage, ~1 µs per tick at 20 TPS -- cheap
enough to be always on, so `/perf` answers from history instead of having
to be switched on before the lag happens (the profiler you enable after
noticing a stall is the one that never catches it).

`PerfMsg` is the console's way in: posted through `World._post_to_actor`
and answered on the actor thread, the same round trip `StatsMsg` uses --
the profiler is actor-owned state like everything else on that thread.
The message lives here rather than in `console/commands.py` because the
actor must not import `console` (the dependency direction the tree keeps),
and in its own module rather than `world/dimension_actor.py` so the
console does not pull the physics/entity import tree just to read a
dataclass.
"""

import threading
import time
from dataclasses import dataclass, field


class StageStat:
    """Rolling numbers for one pipeline stage, since start (or last reset)."""

    __slots__ = ("count", "total", "max", "last")

    def __init__(self) -> None:
        self.count = 0
        self.total = 0.0
        self.max = 0.0
        self.last = 0.0

    def add(self, seconds: float) -> None:
        self.count += 1
        self.total += seconds
        self.last = seconds
        if seconds > self.max:
            self.max = seconds

    def reset(self) -> None:
        self.count = 0
        self.total = 0.0
        self.max = 0.0
        self.last = 0.0

    def as_dict(self) -> dict:
        avg = self.total / self.count if self.count else 0.0
        return {
            "count": self.count,
            "avg_ms": avg * 1000.0,
            "max_ms": self.max * 1000.0,
            "last_ms": self.last * 1000.0,
        }


class TickProfiler:
    """Accumulates per-stage timings across ticks.

    Stage entries are created in first-seen order, which for an actor that
    has run at least one tick is pipeline order -- `/perf` prints them in
    that order without this class carrying a name list. `reset` preserves
    the order on purpose (it zeroes in place rather than clearing the
    dict), so a reset never shuffles the report's rows.
    """

    def __init__(self) -> None:
        self.stages: dict[str, StageStat] = {}
        self.tick = StageStat()  # whole-tick durations, one entry per tick run
        self.tick_count = 0
        self.discarded = 0  # ticks the Ticker dropped while this actor was busy
        self._stage_start = 0.0
        self._tick_start = 0.0

    def start_tick(self) -> None:
        now = time.perf_counter()
        self._tick_start = now
        self._stage_start = now

    def mark(self, stage: str) -> None:
        """Close the stage that just ran and open the next one."""
        now = time.perf_counter()
        self.stages.setdefault(stage, StageStat()).add(now - self._stage_start)
        self._stage_start = now

    def end_tick(self, discarded: int = 0) -> float:
        """Close the tick; returns its duration for the TickDone reply.

        `discarded` is how many deadlines the Ticker dropped for this actor
        while it was busy (the `gap - 1` fast-forward `_on_tick` already
        computes). Counted here so the actor's own report says "I was
        behind N times" without the console reading Ticker internals.
        """
        duration = time.perf_counter() - self._tick_start
        self.tick.add(duration)
        self.tick_count += 1
        self.discarded += discarded
        return duration

    def snapshot(self) -> dict:
        return {
            "tick_count": self.tick_count,
            "discarded": self.discarded,
            "tick": self.tick.as_dict(),
            "stages": {name: stat.as_dict() for name, stat in self.stages.items()},
        }

    def reset(self) -> None:
        for stat in self.stages.values():
            stat.reset()
        self.tick.reset()
        self.tick_count = 0
        self.discarded = 0


@dataclass
class PerfMsg:
    """Ask the owning actor for a snapshot of its per-stage tick timing.

    `reset=True` zeroes the accumulators before the snapshot, so `/perf
    reset` reports a (near-)empty window and the next `/perf` reads a fresh
    one -- e.g. right after reproducing a stall.
    """

    reset: bool = False
    box: dict = field(default_factory=dict)
    done: threading.Event = field(default_factory=threading.Event)
