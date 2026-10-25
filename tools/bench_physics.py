"""What the physics engine costs per tick, and where the cost goes.

    python tools/bench_physics.py [--entities 500] [--radius 12]

The numbers this prints are the ones `PHYSICS_PLAN.md` §5.3 asks for:

* **us per entity per tick** for the three states a body is ever in -- resting
  (asleep, support probe only), falling (free sweep), and pressed against a
  wall (sweep plus the step-up retry);
* **the full tick at `--entities` bodies**, against the 50 ms tick budget the
  server shares with chunk streaming;
* **the sleeping ratio**: the same bodies with all of them awake versus the
  steady state where most of them are standing still. This is the measurement
  that decides whether sleeping earns its complexity, and the plan's answer was
  that it is the real lever rather than the kernel language;
* **the load-level gate**: the same bodies with and without the ticket model, at
  four ticking ratios. The first two rows are what the gate costs when nothing
  is paused (it is one `np.unique` over the body positions plus one query per
  distinct chunk); the last two are what a lazy world saves;
* **the cold-start cost** of the first `move_batch` call, which must never land
  inside a tick (`warm_up()` exists for exactly that, and the number is printed
  so the claim is checkable rather than asserted).

The world is synthetic: a `--radius` square of chunks with a stone floor, built
straight into `ChunkSection.blocks` through `raw_sections()`. No generator, no
registry-driven terrain, so the comparison is about the solver and not about
how long a chunk takes to make.

Run: python tools/bench_physics.py

Exit code 0 when it completes. It is a benchmark, not a check -- the target is
printed next to the measurement instead of being enforced, because a benchmark
that fails on a busy machine teaches nobody anything.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Callable, List, Tuple

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from entity.position import EntityPosition  # noqa: E402
from physics.block_shapes import block_shapes  # noqa: E402
from physics.bodies import BodyStore  # noqa: E402
from physics.kernel import MAX_CANDIDATES, move_batch, warm_up  # noqa: E402
from physics.solver import GRAVITY, PhysicsSolver  # noqa: E402
from physics.world_view import WorldView  # noqa: E402
from world.chunk import Chunk, ChunkPos  # noqa: E402

#: The tick budget the plan measures against: 20 Hz, shared with chunk
#: streaming, entity tracking and everything else the actor does.
TICK_BUDGET_MS = 50.0

#: The target from `PHYSICS_PLAN.md` §5.3: 500 bodies, 20% of them moving.
TARGET_ENTITIES = 500
TARGET_BUSY_FRACTION = 0.2
TARGET_MS = 5.0

MIN_Y, HEIGHT = -64, 384
FLOOR_SECTION = (0 - MIN_Y) // 16


class Stub:
    """A body's owner as far as the solver is concerned: position + box."""

    __slots__ = ("entity_id", "entity_type", "position")

    def __init__(self, entity_id: int, width: float, height: float,
                 x: float, y: float, z: float) -> None:
        self.entity_id = entity_id
        self.entity_type = type("T", (), {"width": width, "height": height})()
        self.position = EntityPosition(x=x, y=y, z=z)


def build_world(radius: int) -> WorldView:
    """A `(2r+1)²` square of chunks with one stone layer at y = 0."""
    from world import block_states

    stone = block_states.default_state_id("minecraft:stone")
    view = WorldView(MIN_Y, HEIGHT)
    chunks = []
    for cx in range(-radius, radius + 1):
        for cz in range(-radius, radius + 1):
            chunk = Chunk(position=ChunkPos(cx, cz), min_y=MIN_Y, height=HEIGHT)
            section = chunk.sections[FLOOR_SECTION]
            section.blocks.reshape(16, 16, 16)[0, :, :] = stone
            section.recompute_non_air_count()
            chunks.append(chunk)
    view.rebuild(chunks)
    return view


def best_of(fn: Callable[[], None], repeat: int = 7,
            prepare: Callable[[], None] | None = None) -> float:
    """Seconds for the fastest of `repeat` runs, after two discarded warm runs.

    `prepare` runs before each measured call, which is how a launch is re-applied
    between ticks: without it the launched bodies would land, fall asleep, and
    every later run would measure a cheaper world than the one being claimed.
    """
    def once() -> float:
        if prepare is not None:
            prepare()
        return _timed(fn)

    once()
    once()
    return min(once() for _ in range(repeat))


def _timed(fn: Callable[[], None]) -> float:
    started = time.perf_counter()
    fn()
    return time.perf_counter() - started


def row(label: str, seconds: float, detail: str = "") -> None:
    print(f"  {label:34s} {seconds * 1e3:9.3f} ms {seconds * 1e6:10.2f} us"
          + (f"   {detail}" if detail else ""))


def make_population(view: WorldView, count: int, radius: int,
                    busy_fraction: float, gate=None) -> Tuple[PhysicsSolver, BodyStore, int, Callable[[], None]]:
    """A settled world with a fraction of its bodies genuinely in motion.

    Every body is dropped from the air first and allowed to land, so the steady
    state being measured is "mostly standing still" rather than "everything
    falling". The returned `relaunch` re-applies the launch before each measured
    tick, because a body that lands sleeps and would otherwise make every later
    measurement cheaper than the one being claimed.
    """
    solver = PhysicsSolver(view, block_shapes(), MIN_Y, HEIGHT,
                           ticks_entities=gate)
    solver.warm_up()
    store = solver.bodies
    rng = np.random.default_rng(11)
    span = radius * 16 - 2
    for index in range(count):
        x = float(rng.uniform(-span, span))
        z = float(rng.uniform(-span, span))
        store.add(entity_id=index + 1, x=x, y=float(rng.uniform(20.0, 40.0)),
                  z=z, width=0.6, height=1.8)
    for _ in range(160):
        solver.tick()
    busy = int(count * busy_fraction)

    def relaunch() -> None:
        for index in range(busy):
            store.vy[index] = 3.0
            store.wake(index)

    return solver, store, busy, relaunch


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--entities", type=int, default=TARGET_ENTITIES)
    parser.add_argument("--radius", type=int, default=12,
                        help="chunks per side of the synthetic world")
    args = parser.parse_args()

    from registry import init_registries

    init_registries(log_summary=False)

    print(f"world: {(2 * args.radius + 1) ** 2} chunks "
          f"({args.radius * 32}x{args.radius * 32} blocks), "
          f"y {MIN_Y}..{MIN_Y + HEIGHT - 1}")

    print("\n1. Cold start (must not happen inside a tick)")
    started = time.perf_counter()
    warm_up()
    print(f"  first move_batch call (JIT compile)  {(time.perf_counter() - started) * 1e3:.1f} ms")
    started = time.perf_counter()
    warm_up()
    print(f"  a later call (cache=True hit)        {(time.perf_counter() - started) * 1e3:.4f} ms")

    view = build_world(args.radius)
    shapes = block_shapes()

    print("\n2. One body, one tick, by state")
    one = build_world(2)
    solver = PhysicsSolver(one, shapes, MIN_Y, HEIGHT)
    solver.warm_up()
    store = solver.bodies
    slot = store.add(entity_id=1, x=8.5, y=1.0, z=8.5, width=0.6, height=1.8)
    candidates = np.zeros((MAX_CANDIDATES, 6), dtype=np.float64)
    keys, slots = one.index_arrays()
    sections = one.section_arrays()
    motion = np.zeros((1, 3))
    probe = np.zeros(1, dtype=np.uint8)
    status = np.zeros(1, dtype=np.uint8)
    active = np.array([slot], dtype=np.int32)

    # A wall two blocks ahead, so the "blocked" case really is blocked.
    from world import block_states
    stone = block_states.default_state_id("minecraft:stone")
    wall = Chunk(position=ChunkPos(0, 0), min_y=MIN_Y, height=HEIGHT)
    section = wall.sections[FLOOR_SECTION]
    # `blocks` is `[y][z][x]` flattened, so this is the x = 9 column, y = 1..2.
    section.blocks.reshape(16, 16, 16)[0, :, :] = stone
    section.blocks.reshape(16, 16, 16)[1:3, :, 9] = stone
    section.recompute_non_air_count()
    one.add(wall)
    keys, slots = one.index_arrays()
    sections = one.section_arrays()

    def one_call(vx: float, vy: float, vz: float) -> None:
        motion[0, 0] = vx
        motion[0, 1] = vy
        motion[0, 2] = vz
        move_batch(store.x, store.y, store.z, store.vx, store.vy, store.vz,
                   store.width, store.height, active, motion, probe, status,
                   sections, one.sections_per_chunk, one.min_section_y,
                   keys, slots, int(keys.shape[0]),
                   shapes.offsets, shapes.ids, shapes.min_xyz, shapes.max_xyz,
                   candidates, 0.6)

    row("zero motion (fast path, no reads)", best_of(lambda: one_call(0.0, 0.0, 0.0)))
    row("falling at terminal velocity", best_of(lambda: one_call(0.0, -3.92, 0.0)))
    row("walking into a wall (step-up retry)", best_of(lambda: one_call(0.3, -0.08, 0.0)))
    row("walking on open ground", best_of(lambda: one_call(0.3, -0.08, 0.3)))

    print("\n3. A full tick, and the sleeping ratio")
    total = args.entities
    for label, fraction in (("all awake (worst case)", 1.0),
                            (f"{int(TARGET_BUSY_FRACTION * 100)}% moving (steady state)",
                             TARGET_BUSY_FRACTION),
                            ("none moving (all asleep)", 0.0)):
        population, ports, busy, relaunch = make_population(
            view, total, args.radius, fraction)
        seconds = best_of(population.tick, repeat=5,
                          prepare=relaunch if busy else None)
        stats = population.stats()
        row(f"{total} bodies, {label}", seconds,
            f"{seconds / total * 1e6:.2f} us/body  "
            f"awake={stats['awake']} asleep={stats['sleeping']}")
        del population, ports

    print("\n4. Against the budget")
    population, _, busy, relaunch = make_population(
        view, total, args.radius, TARGET_BUSY_FRACTION)
    seconds = best_of(population.tick, repeat=5, prepare=relaunch)
    ms = seconds * 1e3
    print(f"  {total} bodies, {int(TARGET_BUSY_FRACTION * 100)}% moving: "
          f"{ms:.3f} ms/tick")
    print(f"  plan target: < {TARGET_MS} ms/tick  -> "
          f"{'MET' if ms < TARGET_MS else 'MISSED'}")
    print(f"  share of the {TICK_BUDGET_MS:.0f} ms tick budget: "
          f"{ms / TICK_BUDGET_MS * 100:.1f}%")
    print(f"  bodies this tick could afford at {TICK_BUDGET_MS:.0f} ms: "
          f"{int(TICK_BUDGET_MS / max(ms, 1e-9) * total)}")
    stats = population.stats()
    print(f"  solver counters: {stats}")
    print(f"  worst single tick seen: {population.max_step:.3f} ms")

    print("\n5. The load-level gate (Chunk.mdx): what it costs, what it saves")
    # The gate is asked once per distinct chunk per tick (`np.unique` over the
    # packed body chunks), so the rows separate the two halves of its cost: the
    # vector work plus the per-chunk Python loop (the lambda rows), and the
    # ticket query itself (the real `TicketManager` row). The last two rows are
    # the payoff -- a body in a chunk that is not entity-ticking is skipped
    # before gravity, the sweep and the probe.
    from world.chunk_ticket import PLAYER_SIMULATION, TicketManager

    tickets = TicketManager("bench")
    tickets.add(PLAYER_SIMULATION, ChunkPos(0, 0), radius=args.radius, key="bench")

    def real_gate(chunk_x: int, chunk_z: int) -> bool:
        return tickets.ticks_entities(ChunkPos(chunk_x, chunk_z))

    # A second, four-player table: the query scans the ticket positions, so its
    # per-call cost grows with how many players are loading chunks, and that is
    # the number a live server actually pays. Four tickets still cover the whole
    # world (radius == the world's), so every body ticks in both rows.
    crowd = TicketManager("bench-crowd")
    for index, (px, pz) in enumerate(((0, 0), (6, 6), (-6, 6), (6, -6))):
        crowd.add(PLAYER_SIMULATION, ChunkPos(px, pz), radius=args.radius,
                  key=f"p{index}")

    def crowd_gate(chunk_x: int, chunk_z: int) -> bool:
        return crowd.ticks_entities(ChunkPos(chunk_x, chunk_z))

    for label, gate in (("no gate (the reference)", None),
                        ("gate, every body ticking (lambda)",
                         lambda cx, cz: True),
                        ("gate, a real ticket table (1 player)", real_gate),
                        ("gate, a real ticket table (4 players)", crowd_gate),
                        ("gate, half the world ticking", lambda cx, cz: cx >= 0),
                        ("gate, nothing ticking", lambda cx, cz: False)):
        population, _, busy, relaunch = make_population(
            view, total, args.radius, TARGET_BUSY_FRACTION, gate=gate)
        seconds = best_of(population.tick, repeat=5, prepare=relaunch)
        stats = population.stats()
        row(f"{total} bodies, {label}", seconds,
            f"awake={stats['awake']} paused={stats['paused_not_ticking']}")
        del population

    print("\n6. What the shapes cost (context, not a physics path)")
    started = time.perf_counter()
    block_shapes(refresh=True)
    print(f"  rebuilding the whole state -> AABB table: "
          f"{(time.perf_counter() - started) * 1e3:.1f} ms (once per process)")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception:
        import traceback

        traceback.print_exc()
        raise SystemExit(1)
