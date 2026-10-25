"""
Game tick pacing.

The Ticker owns the 20 Hz clock and nothing else. Every deadline it posts
a TickMsg into each dimension actor's inbound queue; the actor replies
with TickDone once its tick has run. Game logic never executes here —
dimension threads own that.

Lag policy: ticks are discarded, never queued. If an actor is still busy
when a deadline arrives, that tick is dropped for it (recorded as lag);
if the Ticker itself falls behind schedule, the missed deadlines are
collapsed into the next one. Either way the next TickMsg always carries
the latest tick number, so a recovered dimension fast-forwards its clock
instead of replaying missed ticks.
"""

import queue
import time
from dataclasses import dataclass
from typing import Optional

from loguru import logger

from core.threads import ServerThread


@dataclass(frozen=True)
class TickMsg:
    """One tick of work, dispatched by the Ticker."""

    tick_no: int    # global tick number; skips ahead when ticks were discarded
    sent_at: float  # time.monotonic() at dispatch


@dataclass(frozen=True)
class TickDone:
    """An actor's answer to a TickMsg."""

    actor: str        # actor name, as registered on the Ticker
    tick_no: int      # which TickMsg this answers
    duration: float   # seconds spent inside the tick
    queue_depth: int  # messages still waiting in the actor's inbox


class _ActorTickState:
    """Per-actor tick accounting — one of these per registered dimension."""

    def __init__(self, name: str, inbox: "queue.Queue"):
        self.name = name
        self.inbox = inbox

        # Tick bookkeeping ("每个tick单独追踪自己的队列")
        self.pending_tick: Optional[int] = None  # tick_no in flight, None = idle
        self.missed = 0                          # deadlines discarded since last done
        self.behind_since: Optional[float] = None
        self.last_report = 0.0

        # Per-tick observations from TickDone
        self.last_duration = 0.0
        self.last_queue_depth = 0


class Ticker(ServerThread):
    """Drives every dimension actor at tick_rate and tracks who keeps up."""

    def __init__(self, tick_rate: int = 20):
        super().__init__("Ticker")
        self.tick_rate = tick_rate
        self.tick_interval = 1.0 / tick_rate

        self._actors: dict[str, _ActorTickState] = {}
        self.done_inbox: queue.Queue = queue.Queue()

        self._tick_count = 0
        self._tps = float(tick_rate)

    def register_actor(self, name: str, inbox: "queue.Queue") -> None:
        """Register a dimension actor's inbound queue for tick dispatch."""
        self._actors[name] = _ActorTickState(name, inbox)

    def _run(self) -> None:
        logger.info(f"Ticker started (target: {self.tick_rate} TPS)")
        next_deadline = time.monotonic()
        window_count = 0
        window_start = next_deadline

        while self._running:
            now = time.monotonic()
            if now >= next_deadline:
                # Discard-to-latest: if even the Ticker missed deadlines
                # (GC pause, OS stall), collapse them into one tick whose
                # number jumped to the newest deadline.
                skipped = int((now - next_deadline) // self.tick_interval)
                self._tick_count += 1 + skipped
                next_deadline += (skipped + 1) * self.tick_interval
                if skipped:
                    logger.warning(
                        f"Ticker running {skipped} tick(s) behind schedule, "
                        f"discarding to tick {self._tick_count}"
                    )

                self._dispatch_tick(self._tick_count, now)

                window_count += 1 + skipped
                if window_count >= self.tick_rate:
                    span = now - window_start
                    self._tps = window_count / span if span > 0 else float(self.tick_rate)
                    window_count = 0
                    window_start = now

            # Sleep until the next deadline, waking early for TickDone replies.
            timeout = min(max(next_deadline - time.monotonic(), 0.0), 0.05)
            try:
                done = self.done_inbox.get(timeout=timeout)
            except queue.Empty:
                continue
            self._on_done(done)

        logger.info("Ticker stopped")

    def _dispatch_tick(self, tick_no: int, now: float) -> None:
        """Send this tick to every idle actor; discard it for busy ones."""
        for st in self._actors.values():
            if st.pending_tick is not None:
                # Actor still busy: discard this tick for it. Its next
                # TickMsg will carry a later tick_no — no catch-up run.
                st.missed += 1
                if st.behind_since is None:
                    st.behind_since = now
                if self._should_report_lag(st, now, "behind"):
                    st.last_report = now
                    logger.warning(
                        f"{st.name} is behind: tick {st.pending_tick} still running, "
                        f"discarded tick {tick_no} ({st.missed} missed so far)"
                    )
            else:
                st.pending_tick = tick_no
                st.inbox.put(TickMsg(tick_no=tick_no, sent_at=now))

    def _on_done(self, done: TickDone) -> None:
        """Fold a TickDone reply into the actor's accounting."""
        st = self._actors.get(done.actor)
        if st is None:
            logger.warning(f"TickDone from unknown actor {done.actor}")
            return

        st.pending_tick = None
        st.last_duration = done.duration
        st.last_queue_depth = done.queue_depth

        if st.missed:
            now = time.monotonic()
            if self._should_report_lag(st, now, "late_done"):
                st.last_report = now
                logger.warning(
                    f"{st.name} recovered late: tick {done.tick_no} took "
                    f"{done.duration * 1000:.1f}ms, {st.missed} tick(s) discarded, "
                    f"inbox backlog={done.queue_depth}"
                )
            st.missed = 0
            st.behind_since = None

    def _should_report_lag(self, st: _ActorTickState, now: float, event: str) -> bool:
        """Decide whether a lag event for this actor deserves a log line.

        Called with event="behind" when a deadline was discarded (the actor
        is not keeping up) and event="late_done" when the actor finished a
        tick that had already missed one or more deadlines.

        The discard policy itself (never queue ticks, resume at the latest
        deadline) is fixed in _dispatch_tick / _on_done — this function
        only shapes *reporting*. Things to weigh when writing this policy:

        - Logging every discarded tick at 20 Hz drowns the console; never
          logging hides a dimension that died quietly.
        - "behind" and "late_done" both firing for one stall doubles the
          noise — maybe only one of them, or only the summary, matters.
        - A single missed tick is a hiccup (GC, page fault); sustained lag
          is qualitatively different and may deserve escalation, not just
          a log line.

        TODO: replace the default below with the reporting policy you want.
        Default: always report the first discarded tick of a stall, then at
        most one line per second per dimension; one-tick stalls report only
        via "behind", not via "late_done".
        """
        if event == "late_done" and st.missed <= 1:
            return False  # hiccup already reported by the "behind" line
        if event == "behind" and st.missed == 1:
            return True  # first discarded tick of this stall
        return (now - st.last_report) >= 1.0

    def snapshot(self) -> dict:
        """Per-actor lag bookkeeping for observers (the console /perf command).

        Read from another thread by design: the fields are plain ints and
        floats the ticker thread only rewrites, so a reader sees either the
        old or the new value, never a torn one. `behind_for` is how long
        the actor has been missing deadlines (None = currently keeping up);
        `discarded` is ticks dropped since the actor last finished one.
        """
        now = time.monotonic()
        actors = {}
        for st in self._actors.values():
            actors[st.name] = {
                "last_duration_ms": st.last_duration * 1000.0,
                "queue_depth": st.last_queue_depth,
                "discarded": st.missed,
                "behind_for": (now - st.behind_since
                               if st.behind_since is not None else None),
            }
        return {
            "tps": self._tps,
            "tick_rate": self.tick_rate,
            "tick_count": self._tick_count,
            "actors": actors,
        }

    @property
    def tps(self) -> float:
        """Ticks per second actually dispatched recently."""
        return self._tps

    @property
    def tick_count(self) -> int:
        """Global tick number of the most recent dispatch."""
        return self._tick_count
