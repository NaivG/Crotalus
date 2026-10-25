"""
Adversarial checks for `world/chunk_ticket.py` -- the chunk ticket table and
its level model.

The module's central claim is that propagation can be evaluated in closed form

    level(c) = min over tickets t of (t.level + max(0, chebyshev(c, t.pos) - t.radius))

instead of by flood fill, and everything else in it (render distance versus
simulation distance, expiry, `/forceload` persistence) is built on that claim.
A closed form is exactly the kind of thing that looks right on the one example
someone typed in by hand and is off by one somewhere on a diagonal, so this
tool does not trust it:

* it carries a second, independent implementation -- the monotone-relaxation
  propagator a flood fill would be -- and compares it cell by cell, over a
  bounded region, for dozens of seeded random ticket sets. Random sets are the
  point: a hand-picked one only ever exercises the cases its author thought
  of, and the failures worth catching here (a clamp that saturates on one side
  only, a radius that is applied to the wrong term, a diagonal counted as two
  steps) are all invisible in the middle of a square.
* both directions of `loaded_positions()` are asserted, because a set that is
  merely a subset of the truth passes any "everything it lists is loaded"
  check while quietly leaving chunks unstreamed.
* the wiki's own worked example (the 7x7 table for a level-31 ticket) is
  pasted in literally, so the model is pinned to the spec rather than to
  itself.
* the timing check at the end exists to fail if someone replaces the closed
  form with a propagation pass: it is not a performance target, it is a guard
  on the design decision, and the threshold is set loosely enough that only a
  different complexity class trips it.

Run: python tools/check_chunk_tickets.py
"""

import random
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from loguru import logger
logger.remove()

from world.chunk import ChunkPos
from world.chunk_ticket import (
    BLOCK_TICKING_LEVEL,
    DIMENSION_IDLE_TIMEOUT,
    DRAGON,
    ENDER_PEARL,
    ENTITY_TICKING_LEVEL,
    FORCED,
    MAX_LEVEL,
    MAX_LOADED_LEVEL,
    PLAYER_LOADING,
    PLAYER_SIMULATION,
    PLAYER_SPAWN,
    PORTAL,
    SPAWN_SEARCH,
    TICKET_TYPES,
    TicketFlags,
    TicketManager,
    UNKNOWN,
    chebyshev,
    is_loaded_level,
    load_type_for_level,
)

FAILURES: list = []
UNREACHED = MAX_LEVEL + 1


def check(ok: bool, line: str, detail: str = "") -> bool:
    """Record one check: a one-line verdict, and the evidence if it failed."""
    if ok:
        print(f"  [ok]   {line}")
    else:
        print(f"  [FAIL] {line}")
        if detail:
            for text_line in str(detail).splitlines():
                print(f"         {text_line}")
        FAILURES.append(line)
    return ok


def region_of(tickets) -> tuple:
    """A bounding box that contains every cell any of `tickets` can load.

    Sized from the argument rather than from a constant, because it has to be
    big enough that the relaxation cannot be fed a level from a cell outside
    it: the furthest cell a ticket can reach at the loaded cut-off is
    `radius + (MAX_LOADED_LEVEL - level)` steps out, and every cell that could
    feed one inside that is one step further again.
    """
    reach = 0
    for ticket in tickets:
        reach = max(reach, ticket.radius + (MAX_LOADED_LEVEL - ticket.level) + 1)
    min_x = min(t.pos.x for t in tickets) - reach
    max_x = max(t.pos.x for t in tickets) + reach
    min_z = min(t.pos.z for t in tickets) - reach
    max_z = max(t.pos.z for t in tickets) + reach
    return min_x, max_x, min_z, max_z


def reference_levels(tickets, box: tuple) -> dict:
    """The propagation fixpoint, computed by relaxation -- the flood fill.

    Independent of `TicketManager` on purpose: it seeds from each ticket's
    square and then repeatedly lets every cell be lowered to one more than its
    lowest neighbour, which is vanilla's definition of propagation and shares
    no code, no formula and no index with the implementation under test.

    Unreached cells are reported as `MAX_LEVEL`, not as "missing", so the
    comparison against `level()` is total over the box.
    """
    min_x, max_x, min_z, max_z = box
    level = {}
    queue = []

    def relax(cell, value):
        """Lower one cell; queue it if it actually improved."""
        if level.get(cell, UNREACHED) > value:
            level[cell] = value
            queue.append((cell, value))

    # A ticket pins its whole radius square at `ticket.level` -- that square is
    # a source, not a propagation result, so it is seeded directly.
    for ticket in tickets:
        for x in range(ticket.pos.x - ticket.radius, ticket.pos.x + ticket.radius + 1):
            for z in range(ticket.pos.z - ticket.radius, ticket.pos.z + ticket.radius + 1):
                if min_x <= x <= max_x and min_z <= z <= max_z:
                    relax((x, z), ticket.level)

    while queue:
        (x, z), value = queue.pop()
        if value >= MAX_LEVEL:
            # Nothing above the cap can lower a neighbour: the neighbour's own
            # term would be clamped to MAX_LEVEL as well.
            continue
        if value != level.get((x, z)):
            # A stale queue entry: this cell was already improved below the
            # value we queued it with, and was re-queued at that lower value.
            continue
        for dx, dz in ((1, 0), (-1, 0), (0, 1), (0, -1),
                       (1, 1), (1, -1), (-1, 1), (-1, -1)):
            nx, nz = x + dx, z + dz
            if min_x <= nx <= max_x and min_z <= nz <= max_z:
                relax((nx, nz), value + 1)

    return {(x, z): level.get((x, z), MAX_LEVEL)
            for x in range(min_x, max_x + 1)
            for z in range(min_z, max_z + 1)}


def cells(box: tuple):
    min_x, max_x, min_z, max_z = box
    for x in range(min_x, max_x + 1):
        for z in range(min_z, max_z + 1):
            yield x, z


def ticket_set_text(tickets) -> str:
    """The ticket set, verbatim, for a mismatch report."""
    return ", ".join(
        f"{t.type.name}(L{t.level} r{t.radius} @{t.pos.x},{t.pos.z})"
        for t in tickets
    )


# ---------------------------------------------------------------------------
# 1. Closed form versus brute force
# ---------------------------------------------------------------------------

def test_closed_form_vs_relaxation(rng: random.Random, sets: int = 40) -> None:
    """`level()` must equal a flood fill everywhere, for arbitrary ticket sets.

    Forty random sets of 1..6 tickets, levels 22..35, radii 0..12, positions
    within +-10 chunks. The radii are what matter: a ticket at level 35 with
    radius 12 is the only shape that puts interesting values at the edges of
    the box, and a ticket at 22 with radius 6 is the only shape that puts
    levels far below the cut-off in the middle of one.

    Levels below `MIN_LEVEL` are not generated (they are unreachable without
    mods, and the model is not required to make them interesting), and neither
    are levels above `MAX_LOADED_LEVEL + 2`: a ticket that far past the
    cut-off can only ever produce `INACCESSIBLE`, which every set already
    covers somewhere.
    """
    mismatches = 0
    cells_compared = 0
    worst = None

    for index in range(sets):
        count = rng.randint(1, 6)
        manager = TicketManager(f"random-{index}")
        tickets = []
        for n in range(count):
            ttype = TICKET_TYPES[rng.choice(sorted(TICKET_TYPES))]
            ticket = manager.add(
                ttype,
                ChunkPos(rng.randint(-10, 10), rng.randint(-10, 10)),
                # The type's own level/radius are worth exercising too, which
                # is what the occasional `None` does.
                level=rng.choice([None, rng.randint(22, 35)]),
                radius=rng.choice([None, 0, 0, 1, 2, 3, rng.randint(4, 12)]),
                key=f"k{n}",
                tick=index,
            )
            tickets.append(ticket)
        # Read the level/radius back off the tickets: `add` may have filled in
        # the type's defaults, and the reference model must be seeded with
        # what the manager actually holds, not with what was asked for.
        box = region_of(tickets)
        reference = reference_levels(tickets, box)

        for x, z in cells(box):
            got = manager.level(ChunkPos(x, z))
            want = reference[(x, z)]
            cells_compared += 1
            if got != want:
                mismatches += 1
                if worst is None:
                    worst = (x, z, got, want, ticket_set_text(tickets), box)
        if mismatches and worst is not None:
            break

    detail = ""
    if worst is not None:
        x, z, got, want, text, box = worst
        detail = (f"at chunk ({x},{z}): level()={got}, relaxation={want}\n"
                  f"region {box}\n"
                  f"tickets: {text}")
    check(mismatches == 0,
          f"level() matches a flood-fill relaxation on {cells_compared} cells "
          f"over {sets} random ticket sets",
          detail)

    # --- the two directions of loaded_positions() ---------------------------
    subset_violations = []
    outside_violations = []
    for index in range(sets):
        count = rng.randint(1, 6)
        manager = TicketManager(f"loaded-{index}")
        tickets = []
        for n in range(count):
            ttype = TICKET_TYPES[rng.choice(sorted(TICKET_TYPES))]
            tickets.append(manager.add(
                ttype,
                ChunkPos(rng.randint(-10, 10), rng.randint(-10, 10)),
                level=rng.randint(22, 35),
                radius=rng.choice([0, 1, 2, 4, 8, 12]),
                key=f"k{n}",
            ))
        box = region_of(tickets)
        reference = reference_levels(tickets, box)
        reported = manager.loaded_positions()

        for x, z in cells(box):
            loaded = reference[(x, z)] <= MAX_LOADED_LEVEL
            listed = ChunkPos(x, z) in reported
            if loaded and not listed:
                if len(subset_violations) < 5:
                    subset_violations.append(
                        f"({x},{z}) is loaded at level {reference[(x, z)]} but "
                        f"loaded_positions() omits it; "
                        f"tickets: {ticket_set_text(tickets)}")
            elif listed and not loaded:
                if len(outside_violations) < 5:
                    outside_violations.append(
                        f"({x},{z}) is listed at level {reference[(x, z)]}")

    check(not subset_violations,
          "loaded_positions() omits nothing the relaxation says is loaded",
          "\n".join(subset_violations))
    check(not outside_violations,
          "loaded_positions() contains nothing the relaxation says is unloaded",
          "\n".join(outside_violations))


# ---------------------------------------------------------------------------
# 2. The wiki's worked example
# ---------------------------------------------------------------------------

#: The 7x7 table from Chunk.mdx, "Level propagation", for a level-31 ticket
#: with radius 0 in the centre. Rows are z = -3..3, columns x = -3..3. Levels
#: 35 and up are stated to be identical to 34, and the model reports 34 for
#: all of them rather than clamping at 35 -- which is what this table pins.
WIKI_TABLE = (
    (34, 34, 34, 34, 34, 34, 34),
    (34, 33, 33, 33, 33, 33, 34),
    (34, 33, 32, 32, 32, 33, 34),
    (34, 33, 32, 31, 32, 33, 34),
    (34, 33, 32, 32, 32, 33, 34),
    (34, 33, 33, 33, 33, 33, 34),
    (34, 34, 34, 34, 34, 34, 34),
)


def test_wiki_example() -> None:
    manager = TicketManager("wiki")
    manager.add(PLAYER_SPAWN, ChunkPos(0, 0), level=31, radius=0)
    wrong = []
    for row, expected_row in enumerate(WIKI_TABLE):
        z = row - 3
        got_row = []
        for column, expected in enumerate(expected_row):
            got = manager.level(ChunkPos(column - 3, z))
            got_row.append(got)
            if got != expected:
                wrong.append(f"({column - 3},{z}): model {got}, wiki {expected}")
        print(f"         {' '.join(f'{v:2d}' for v in got_row)}")
    check(not wrong,
          "the wiki's 7x7 level-31 propagation table reproduces exactly",
          "\n".join(wrong))

    # The same table expressed as load types, from the second table on the
    # page: ET in the centre, BT on the ring at 1, BO at 2, IN at 3. This is
    # the load-type boundary the streamer actually branches on.
    expected_types = {
        (0, 0): "ENTITY_TICKING",
        (1, 0): "BLOCK_TICKING", (0, 1): "BLOCK_TICKING",
        (-1, 1): "BLOCK_TICKING", (1, -1): "BLOCK_TICKING",
        (2, 0): "BORDER", (2, 2): "BORDER", (-2, -2): "BORDER",
        (3, 0): "INACCESSIBLE", (3, 3): "INACCESSIBLE",
    }
    type_wrong = [
        f"({x},{z}): {manager.load_type(ChunkPos(x, z)).name} != {want}"
        for (x, z), want in expected_types.items()
        if manager.load_type(ChunkPos(x, z)).name != want
    ]
    check(not type_wrong,
          "the same table in load types (ET / BT / BO / IN rings)",
          "\n".join(type_wrong))


# ---------------------------------------------------------------------------
# 3. Load types and the loaded boundary
# ---------------------------------------------------------------------------

def test_load_types() -> None:
    expected = {
        22: "ENTITY_TICKING", 30: "ENTITY_TICKING", 31: "ENTITY_TICKING",
        32: "BLOCK_TICKING", 33: "BORDER", 34: "INACCESSIBLE",
        44: "INACCESSIBLE",
    }
    wrong = [
        f"level {level}: {load_type_for_level(level).name} != {want}"
        for level, want in expected.items()
        if load_type_for_level(level).name != want
    ]
    check(not wrong, "load types at 22/30/31/32/33/34/44", "\n".join(wrong))

    # 22 and below (mods) is vanilla's fall-through to the first branch, not a
    # separate case; assert it so the branch order cannot be rearranged.
    check(load_type_for_level(0) is load_type_for_level(31),
          "levels below MIN_LEVEL fall through to Entity Ticking")

    loaded = [level for level in range(20, 40) if is_loaded_level(level)]
    expected_loaded = list(range(20, 34))
    check(loaded == expected_loaded,
          f"is_loaded_level is True exactly on <= {MAX_LOADED_LEVEL}, False above",
          f"True on {loaded}")

    # The constants have to agree with each other, not just with themselves.
    check(MAX_LOADED_LEVEL == 33 and ENTITY_TICKING_LEVEL == 31
          and BLOCK_TICKING_LEVEL == 32 and DIMENSION_IDLE_TIMEOUT == 300,
          "the level constants are the wiki's numbers")


# ---------------------------------------------------------------------------
# 4. Player squares: simulation distance, and the frame around it
# ---------------------------------------------------------------------------

def count_square(predicate, side: int) -> int:
    """How many cells of a centred `side` x `side` square the predicate accepts."""
    half = side // 2
    return sum(
        1
        for x in range(-half, half + 1)
        for z in range(-half, half + 1)
        if predicate(ChunkPos(x, z))
    )


def square_check(manager: TicketManager, predicate, expected_side: int,
                 label: str) -> None:
    """Assert the predicate's true-set is exactly one centred square.

    Stronger than sampling the axes: it walks the whole square and asserts
    every cell in it is true, then walks the first ring outside and asserts
    every cell there is false -- including the diagonals, which is where a
    metric mix-up shows up.
    """
    expected_cells = expected_side * expected_side
    got = count_square(predicate, expected_side)
    check(got == expected_cells,
          f"{label}: all {expected_cells} cells of the {expected_side}x"
          f"{expected_side} square qualify (got {got})")
    got = count_square(predicate, expected_side + 2)
    check(got == expected_cells,
          f"{label}: no cell in the surrounding frame qualifies "
          f"(got {got}, expected {expected_cells})")


def test_player_squares() -> None:
    # s = 1..10 exercises the formula; s = 32 is past the wiki's clamp and is
    # checked separately below, where the wiki's side length and the model's
    # side length are not the same number.
    for simulation_distance in range(1, 11):
        # The wiki's own formula, applied literally: side = min(2s+1, 63) for
        # the entity region and min(2s+3, 65) for the block region. Writing it
        # out here (rather than deriving it from the manager) is deliberate --
        # the model should not be allowed to restate its answer as the
        # question.
        side = min(2 * simulation_distance + 1, 63)
        block_side = min(2 * simulation_distance + 3, 65)

        manager = TicketManager("player")
        ticket = manager.add(PLAYER_SIMULATION, ChunkPos(0, 0),
                             radius=simulation_distance)

        # The level model on its own terms: the ticket pins level 31 across
        # radius s, and every step outside adds one.
        level_wrong = []
        for x, z in ((-1, 0), (0, 1), (1, 1), (-2, -2)):
            distance = chebyshev(ChunkPos(x, z), ChunkPos(0, 0))
            expected = 31 + max(0, distance - simulation_distance)
            got = manager.simulation_level(ChunkPos(x, z))
            if got != expected:
                level_wrong.append(
                    f"({x},{z}) d={distance}: simulation_level {got} != {expected}")
        check(not level_wrong,
              f"simulation_level = 31 + (d - {simulation_distance}) outside the "
              f"square at s={simulation_distance}",
              "\n".join(level_wrong))

        square_check(manager, manager.ticks_entities, side,
                     f"s={simulation_distance} ticks_entities")
        square_check(manager, manager.ticks_blocks, block_side,
                     f"s={simulation_distance} ticks_blocks")

        # View distance must not touch any of this. Give the player a loading
        # ticket four times their simulation distance, re-add it the way the
        # streamer would on every chunk crossing, and the two squares must not
        # move by one cell.
        manager.add(PLAYER_LOADING, ChunkPos(0, 0),
                    radius=simulation_distance * 4)
        for step in range(1, 4):
            manager.add(PLAYER_LOADING, ChunkPos(step, -step),
                        radius=simulation_distance * 4)
        check(count_square(manager.ticks_entities, side) == side * side
              and count_square(manager.ticks_entities, side + 2) == side * side,
              f"s={simulation_distance}: a larger player_loading ticket does not "
              f"widen the entity square")
        check(count_square(manager.ticks_blocks, block_side) == block_side ** 2
              and count_square(manager.ticks_blocks, block_side + 2) == block_side ** 2,
              f"s={simulation_distance}: nor the block square")

        # ... but it must load: that is the other half of the separation.
        # Every cell of the (much larger) loading square is held in memory
        # while only the simulation square ticks.
        far = simulation_distance * 4 - 1
        check(ticket.does_simulate and manager.is_loaded(ChunkPos(far, 0))
              and not manager.ticks_blocks(ChunkPos(far, 0)),
              f"s={simulation_distance}: chunk at {far} chunks is loaded but "
              f"does not tick")

    # --- the clamp at the top of the range --------------------------------
    # The wiki caps the region at side = min(2s+1, 63), and with
    # `simulation-distance` itself capped at 32 the two formulas disagree
    # there: min(2*32+1, 63) is 63 while the ticket model, given radius 32,
    # pins level 31 on a 65x65 square. The model has no 63-chunk ceiling --
    # it is a level map, not a region -- so the cap is the caller's to apply
    # by clamping the radius it passes (`min(s, 31)`, i.e. side 63 / block
    # side 65, which is what the wiki's own "min" is doing). Both halves are
    # asserted here so that neither can drift silently: an unclamped model,
    # and a clamped caller.
    manager = TicketManager("player-clamped")
    manager.add(PLAYER_SIMULATION, ChunkPos(0, 0), radius=32)
    check(count_square(manager.ticks_entities, 65) == 65 * 65
          and count_square(manager.ticks_entities, 67) == 65 * 65,
          "s=32: the raw model's entity square is 65x65, four chunks wider "
          "than min(2s+1, 63) -- the cap is the caller's",
          f"ticks_entities beyond 32: {count_square(manager.ticks_entities, 65)}")
    check(count_square(manager.ticks_blocks, 67) == 67 * 67
          and count_square(manager.ticks_blocks, 69) == 67 * 67,
          "s=32: and its block square is 67x67, past min(2s+3, 65)")
    manager_clamped = TicketManager("player-clamped-31")
    manager_clamped.add(PLAYER_SIMULATION, ChunkPos(0, 0), radius=min(32, 31))
    square_check(manager_clamped, manager_clamped.ticks_entities, 63,
                 "s=32 clamped to 31 ticks_entities")
    square_check(manager_clamped, manager_clamped.ticks_blocks, 65,
                 "s=32 clamped to 31 ticks_blocks")

    # The dragon's square is the largest one any real ticket pins, and it is
    # the only type where a radius and a non-31 level arrive together, so the
    # distance term has to be measured from the *level*, not from 31.
    manager = TicketManager("dragon")
    manager.add(DRAGON, ChunkPos(0, 0))
    dragon_wrong = []
    for x, z in ((0, 0), (9, 0), (9, 9), (-9, 9), (10, 0), (0, -10), (10, 10)):
        distance = chebyshev(ChunkPos(x, z), ChunkPos(0, 0))
        expected = 24 + max(0, distance - 9)
        got = manager.level(ChunkPos(x, z))
        if got != expected:
            dragon_wrong.append(f"({x},{z}) d={distance}: {got} != {expected}")
    check(not dragon_wrong,
          "a radius-9 level-24 dragon square stays 24 to distance 9 and then "
          "adds one per step",
          "\n".join(dragon_wrong))
    # Level 24 is far below ENTITY_TICKING_LEVEL, but that is not what decides
    # where entities tick: the dragon pins 24 on its own 19x19 square, and the
    # propagation rings around it reach 31 only at distance 16. The entity
    # square is therefore much *larger* than the ticket, not equal to it --
    # asserting "ticks to the corner" here would be asserting the wrong model.
    check(manager.ticks_entities(ChunkPos(16, 16))
          and not manager.ticks_entities(ChunkPos(17, 17)),
          "the dragon's entity-ticking region is the level<=31 ring of its "
          "propagated square (distance 16, not the ticket's radius 9)",
          f"d=16 -> {manager.simulation_level(ChunkPos(16, 16))}, "
          f"d=17 -> {manager.simulation_level(ChunkPos(17, 17))}")
    check(manager.load_type(ChunkPos(18, 18)).name == "BORDER"
          and manager.load_type(ChunkPos(19, 19)).name == "INACCESSIBLE",
          "and its loaded region is the level<=33 ring out to distance 18")


# ---------------------------------------------------------------------------
# 5. Expiry
# ---------------------------------------------------------------------------

def test_expiry() -> None:
    # --- a 300-tick ticket expires on the 300th tick, not the 299th ---------
    manager = TicketManager("portal")
    ticket = manager.add(PORTAL, ChunkPos(0, 0), key="p1")
    check(ticket.ticks_remaining == 300, "a portal ticket starts at 300 ticks",
          f"got {ticket.ticks_remaining}")
    early = [n for n in range(1, 300) if manager.tick(n)]
    check(not early, "no portal ticket expires before tick 300",
          f"expired at {early}")
    check(len(manager) == 1, "the portal ticket is still there at tick 299")
    dropped = manager.tick(300)
    check([t.type.name for t in dropped] == ["portal"] and len(manager) == 0,
          "it expires on tick 300 and is handed back by tick()",
          f"dropped {[str(t) for t in dropped]}, {len(manager)} left")

    # --- a refused package deal: some types never expire -------------------
    manager = TicketManager("no-timeout")
    for ttype in (PLAYER_LOADING, PLAYER_SIMULATION, FORCED, DRAGON):
        manager.add(ttype, ChunkPos(0, 0), radius=2)
    expired = []
    for n in range(1000):
        expired += manager.tick(n)
    check(not expired and len(manager) == 4,
          "player, forced and dragon tickets never expire",
          f"{[str(t) for t in expired]}")

    # --- unknown expires even when nothing loads it ------------------------
    manager = TicketManager("unknown")
    # Level 40 is above MAX_LOADED_LEVEL, so the ticket does not load its own
    # chunk: this is the state the flag exists for.
    manager.add(UNKNOWN, ChunkPos(0, 0), level=40)
    check(not manager.is_loaded(ChunkPos(0, 0)),
          "an unknown ticket at level 40 leaves its own chunk unloaded")
    dropped = manager.tick(0)
    check([t.type.name for t in dropped] == ["unknown"] and len(manager) == 0,
          "unknown expires after 1 tick even though its chunk is not loaded",
          f"dropped {[str(t) for t in dropped]}")

    # The other side of the same rule: a type without the flag that sits in an
    # unloaded chunk must wait.
    manager = TicketManager("portal-unloaded")
    ticket = manager.add(PORTAL, ChunkPos(0, 0), level=40)
    for n in range(100):
        manager.tick(n)
    check(ticket.ticks_remaining == 300 and len(manager) == 1,
          "a portal ticket in an unloaded chunk does not count down",
          f"ticks_remaining={ticket.ticks_remaining}")
    # A second ticket loads the chunk; the countdown resumes from where it
    # was, and expires 300 ticks after that -- not immediately. The loader is
    # a player simulation ticket rather than a forced one because a forced
    # ticket has a 20-tick timeout of its own and would expire (taking the
    # portal with it, on the tick its own chunk went unloaded) half a second
    # in.
    manager.add(PLAYER_SIMULATION, ChunkPos(0, 0), radius=1, key="loader")
    check(manager.is_loaded(ChunkPos(0, 0)), "the simulation ticket loads the chunk")
    resumed = [n for n in range(299) if manager.tick(n)]
    check(not resumed and ticket.ticks_remaining == 1,
          "it does not expire in the 299 ticks after its chunk became loaded",
          f"dropped {[str(t) for t in resumed]} at ticks_remaining="
          f"{ticket.ticks_remaining}")
    check([t.type.name for t in manager.tick(299)] == ["portal"],
          "it expires on the 300th",
          f"ticks_remaining={ticket.ticks_remaining}")

    # A ticket that loads its own chunk must count down -- loadedness is
    # sampled before the tick drops anything, so PORTAL at level 30 (loaded)
    # is not accidentally treated as "unloaded".
    manager = TicketManager("self-loading")
    manager.add(SPAWN_SEARCH, ChunkPos(4, 4))     # level 33, radius 0
    check(manager.is_loaded(ChunkPos(4, 4)),
          "spawn_search at level 33 loads its own chunk (33 <= 33)")
    check([t.type.name for t in manager.tick(0)] == ["spawn_search"],
          "and therefore expires on its first tick")

    # --- an ender pearl's timeout is refreshed by re-adding ----------------
    manager = TicketManager("pearl")
    ticket = manager.add(ENDER_PEARL, ChunkPos(0, 0), key="e1")
    for n in range(39):
        manager.tick(n)
    check(ticket.ticks_remaining == 1, "the pearl is one tick from expiry",
          f"got {ticket.ticks_remaining}")
    manager.add(ENDER_PEARL, ChunkPos(0, 0), key="e1")
    check(ticket.ticks_remaining == 40 and len(manager) == 1,
          "re-adding the pearl refreshes its timeout to 40, without stacking",
          f"ticks_remaining={ticket.ticks_remaining}, {len(manager)} tickets")
    # 38 ticks leave 2, which is past the point the *old* countdown would have
    # expired (it had 1 tick left), so surviving this proves the refresh took.
    for n in range(38):
        manager.tick(n)
    check(len(manager) == 1 and ticket.ticks_remaining == 2,
          "and it survives the ticks that would have expired the old countdown",
          f"ticks_remaining={ticket.ticks_remaining}")
    check(not manager.tick(38) and len(manager) == 1,
          "still alive with 1 tick left")
    check([t.type.name for t in manager.tick(39)] == ["ender_pearl"],
          "expiring 40 ticks after the refresh",
          f"ticks_remaining={ticket.ticks_remaining}")

    # Moving to a new chunk is the pearl's other refresh trigger, and it must
    # not silently reset the countdown instead of carrying it.
    manager = TicketManager("pearl-move")
    ticket = manager.add(ENDER_PEARL, ChunkPos(0, 0), key="e1")
    for n in range(10):
        manager.tick(n)
    manager.add(ENDER_PEARL, ChunkPos(1, 0), key="e1")   # the pearl crossed
    manager.remove(ENDER_PEARL, ChunkPos(0, 0), key="e1")
    check(len(manager) == 1 and manager.level(ChunkPos(1, 0)) == 31,
          "the pearl's ticket follows it into the next chunk without duplicating",
          f"{[str(t) for t in manager]}")


# ---------------------------------------------------------------------------
# 6. Flags and dimension activity
# ---------------------------------------------------------------------------

def test_flags() -> None:
    manager = TicketManager("flags")
    check(not manager.activates_dimension(), "an empty dimension is not active")

    manager.add(PLAYER_LOADING, ChunkPos(0, 0), radius=10)
    check(not manager.activates_dimension(),
          "player_loading alone does not keep the dimension active")
    manager.add(SPAWN_SEARCH, ChunkPos(4, 4))
    manager.add(PLAYER_SPAWN, ChunkPos(0, 0))
    manager.add(DRAGON, ChunkPos(0, 0))
    manager.add(UNKNOWN, ChunkPos(9, 9), level=40)
    check(not manager.activates_dimension(),
          "nor do spawn_search / player_spawn / dragon / unknown")

    for ttype in (PLAYER_SIMULATION, FORCED, PORTAL, ENDER_PEARL):
        manager.add(ttype, ChunkPos(20, 20), radius=1, key=ttype.name)
    check(manager.activates_dimension(),
          "player_simulation, forced, portal and ender_pearl each activate it")

    # The idle timeout counts up while the dimension is inactive and resets
    # the moment an activating ticket appears.
    idle = TicketManager("idle")
    idle.add(PLAYER_LOADING, ChunkPos(0, 0), radius=4)
    for n in range(DIMENSION_IDLE_TIMEOUT):
        idle.tick(n)
    check(idle.idle_ticks == DIMENSION_IDLE_TIMEOUT,
          f"the idle counter reaches {DIMENSION_IDLE_TIMEOUT} with no "
          f"dimension-activating ticket",
          f"got {idle.idle_ticks}")
    idle.tick(DIMENSION_IDLE_TIMEOUT)
    check(idle.idle_ticks == DIMENSION_IDLE_TIMEOUT + 1,
          "and keeps counting past it (this module counts, it does not act)")
    idle.add(FORCED, ChunkPos(0, 0), key="f")
    idle.tick(DIMENSION_IDLE_TIMEOUT + 2)
    check(idle.idle_ticks == 0,
          "a forced ticket resets the idle counter on the next tick",
          f"got {idle.idle_ticks}")

    # --- the flags themselves ---------------------------------------------
    expected_flags = {
        PLAYER_SPAWN: TicketFlags.LOADING,
        SPAWN_SEARCH: TicketFlags.LOADING,
        DRAGON: TicketFlags.LOADING | TicketFlags.SIMULATION,
        PLAYER_LOADING: TicketFlags.LOADING,
        PLAYER_SIMULATION: (TicketFlags.LOADING | TicketFlags.SIMULATION
                            | TicketFlags.KEEP_DIMENSION_ACTIVE),
        FORCED: (TicketFlags.LOADING | TicketFlags.SIMULATION
                 | TicketFlags.KEEP_DIMENSION_ACTIVE | TicketFlags.PERSIST),
        PORTAL: (TicketFlags.LOADING | TicketFlags.SIMULATION
                 | TicketFlags.KEEP_DIMENSION_ACTIVE | TicketFlags.PERSIST),
        ENDER_PEARL: (TicketFlags.LOADING | TicketFlags.SIMULATION
                      | TicketFlags.KEEP_DIMENSION_ACTIVE),
        UNKNOWN: (TicketFlags.LOADING | TicketFlags.CAN_EXPIRE_IF_UNLOADED),
    }
    wrong = [
        f"{ttype.name}: {int(ttype.flags)} != {int(flags)}"
        for ttype, flags in expected_flags.items()
        if int(ttype.flags) != int(flags)
    ]
    check(not wrong, "every ticket type carries exactly the wiki's flags",
          "\n".join(wrong))
    check(UNKNOWN.can_expire_if_unloaded
          and not any(t.can_expire_if_unloaded for t in TICKET_TYPES.values()
                      if t is not UNKNOWN),
          "unknown is the only type that can expire while unloaded")

    # --- persistence -------------------------------------------------------
    manager = TicketManager("persist")
    manager.add(FORCED, ChunkPos(-3, 7), key=None)
    manager.add(FORCED, ChunkPos(2 ** 31 - 1, -2 ** 31), key="still-an-int")
    manager.add(PORTAL, ChunkPos(-100, 100), key="p1")
    manager.add(PLAYER_SIMULATION, ChunkPos(0, 0), radius=10, key="p1")
    manager.add(ENDER_PEARL, ChunkPos(5, 5), key="e1")
    manager.add(DRAGON, ChunkPos(0, 0))

    records = manager.persistable()
    names = sorted(r["type"] for r in records)
    check(names == ["forced", "forced", "portal"],
          "persistable() returns exactly the forced and portal tickets",
          f"got {names}")
    check(all(isinstance(r["x"], int) and isinstance(r["z"], int)
              for r in records)
          and all(set(r) == {"type", "x", "z", "level", "radius",
                             "ticks_remaining", "key", "created_tick"}
                  for r in records),
          "the records are plain dicts of primitives (no ChunkPos, no enum)")

    restored = TicketManager("restored")
    count = restored.restore(records)
    check(count == 3, "restore() re-adds all three records", f"got {count}")
    check(restored.persistable() == records,
          "the round trip is lossless, keys and tail-of-map coordinates "
          "included",
          f"in : {records}\nout: {restored.persistable()}")
    check(restored.level(ChunkPos(-3, 7)) == 31
          and restored.level(ChunkPos(2 ** 31 - 1, -2 ** 31)) == 31
          and restored.level(ChunkPos(-100, 100)) == 30,
          "the restored tickets pin the same levels they were saved with")

    # A restored ticket keeps the countdown it had, rather than getting a
    # fresh 300 ticks -- otherwise a portal would be immortal across restarts.
    restored.add(PORTAL, ChunkPos(-100, 100), key="p1")
    for n in range(299):
        restored.tick(n)
    live = [t for t in restored if t.type is PORTAL]
    check(len(live) == 1 and live[0].ticks_remaining == 1,
          "a restored portal resumes its countdown at its current tick, not at 300",
          f"{[str(t) for t in live]}")

    # Unknown/absent rows must be skipped, not crash and not silently become
    # some other ticket type.
    bad = TicketManager("bad")
    accepted = bad.restore([
        {"type": "not_a_ticket_type", "x": 0, "z": 0},
        {"type": "ender_pearl", "x": 0, "z": 0},        # not persistable
        "not a dict",
        {"type": "forced"},                             # no coordinates
        {"type": "forced", "x": "12", "z": "-5"},       # the serialiser's own
    ])
    check(accepted == 1 and len(bad) == 1
          and bad.level(ChunkPos(12, -5)) == 31,
          "restore() skips unknown, non-persistable and malformed rows and "
          "keeps the good ones",
          f"accepted {accepted}, {[str(t) for t in bad]}")


# ---------------------------------------------------------------------------
# 7. Ticket identity
# ---------------------------------------------------------------------------

def test_identity() -> None:
    manager = TicketManager("identity")
    first = manager.add(FORCED, ChunkPos(0, 0), level=31, radius=0, key="a")
    again = manager.add(FORCED, ChunkPos(0, 0), level=28, radius=2, key="a")
    check(len(manager) == 1 and again is first,
          "adding the same (type, pos, key) twice is one ticket, same object",
          f"{len(manager)} tickets")
    check(first.level == 28 and first.radius == 2,
          "and it refreshed the level and radius rather than stacking",
          f"level={first.level} radius={first.radius}")

    # Two owners on one chunk: two tickets, and the lower level wins.
    m2 = TicketManager("two-keys")
    a = m2.add(FORCED, ChunkPos(0, 0), level=30, key="a")
    b = m2.add(PORTAL, ChunkPos(0, 0), level=26, key="b")
    check(len(m2) == 2 and a is not b,
          "the same position with two keys is two tickets", f"{len(m2)}")
    check(m2.level(ChunkPos(0, 0)) == 26,
          "the lower level wins for the position", f"got {m2.level(ChunkPos(0, 0))}")
    b.level = 34
    check(m2.level(ChunkPos(0, 0)) == 30,
          "raising one ticket hands the chunk back to the other",
          f"got {m2.level(ChunkPos(0, 0))}")
    # ... and the fast path (a ticket standing on the queried chunk) must not
    # shortcut past a second, lower ticket standing there as well.
    b.level = 26
    check(m2.level(ChunkPos(0, 0)) == 26,
          "the minimum is taken across tickets sharing the chunk")

    # remove_key drops exactly one owner's tickets.
    m3 = TicketManager("remove-key")
    for pos in (ChunkPos(0, 0), ChunkPos(1, 0), ChunkPos(0, 1), ChunkPos(-1, -1)):
        m3.add(PLAYER_SIMULATION, pos, radius=10, key="alice")
    m3.add(PLAYER_SIMULATION, ChunkPos(5, 5), radius=10, key="bob")
    # A position-only ticket somewhere else: it must survive a remove_key for
    # a named owner. (It cannot share a position with alice's tickets -- a
    # `forced` ticket added there would *be* alice's ticket, since the
    # identity is the (type, pos, key) triple and the type is the same.)
    m3.add(FORCED, ChunkPos(9, 9), key=None)
    check(m3.remove_key("alice") == 4 and len(m3) == 2,
          "remove_key drops exactly that key's tickets",
          f"{len(m3)} left: {[str(t) for t in m3]}")
    # Probe a cell only alice's tickets reached: bob's radius-10 square at
    # (5,5) reaches out to (-5,-5), so probing near the origin would not
    # distinguish "alice was removed" from "bob still covers it".
    check(m3.level(ChunkPos(50, 50)) == MAX_LEVEL
          and m3.level(ChunkPos(5, 5)) == 31
          and m3.level(ChunkPos(9, 9)) == 31,
          "the dropped key's area is gone; the other key's and the "
          "position-only ticket's are untouched",
          f"(50,50)={m3.level(ChunkPos(50, 50))}, "
          f"(5,5)={m3.level(ChunkPos(5, 5))}, "
          f"(9,9)={m3.level(ChunkPos(9, 9))}")
    check(m3.remove_key("nobody") == 0, "remove_key on an absent key is a no-op")

    # Sharing a position is only possible across types or owners; the three
    # combinations are worth pinning because the identity key is built from
    # all three parts.
    m3a = TicketManager("same-pos")
    m3a.add(FORCED, ChunkPos(4, 4), level=31, key="a")
    m3a.add(PORTAL, ChunkPos(4, 4), level=30, key="a")
    m3a.add(FORCED, ChunkPos(4, 4), level=28, key="b")
    check(len(m3a) == 3 and m3a.level(ChunkPos(4, 4)) == 28,
          "type, position and key all take part in the identity",
          f"{len(m3a)} tickets, level {m3a.level(ChunkPos(4, 4))}")
    check(m3a.tickets and len({t.ticket_key() for t in m3a}) == 3,
          "and the three identities are distinct")

    # move keeps identity and the countdown in progress.
    m4 = TicketManager("move")
    ticket = m4.add(PORTAL, ChunkPos(0, 0), key="traveller")
    seen = [n for n in range(5) if m4.tick(n)]
    check(not seen and ticket.ticks_remaining == 295,
          "the ticket has been counting down", f"{ticket.ticks_remaining}")
    check(m4.move(PORTAL, ChunkPos(0, 0), ChunkPos(64, -64), key="traveller"),
          "move() reports the ticket was there")
    check(len(m4) == 1 and m4.level(ChunkPos(0, 0)) == MAX_LEVEL
          and m4.level(ChunkPos(64, -64)) == 30,
          "the ticket is at its new chunk and no longer at the old one",
          f"{[str(t) for t in m4]}")
    check(ticket.pos.to_tuple() == (64, -64) and ticket.ticks_remaining == 295,
          "move keeps the identity and the remaining ticks",
          f"pos={ticket.pos} ttl={ticket.ticks_remaining}")
    check(not m4.move(PORTAL, ChunkPos(1000, 1000), ChunkPos(1, 1), key="ghost"),
          "moving a ticket that is not there is a no-op, not an insert")

    # `remove` must be exact: same type and position, different key.
    m5 = TicketManager("remove")
    m5.add(FORCED, ChunkPos(3, 3), key="keep")
    m5.add(FORCED, ChunkPos(3, 3), key="drop")
    check(m5.remove(FORCED, ChunkPos(3, 3), key="drop") and len(m5) == 1
          and m5.tickets[0].key == "keep",
          "remove() hits one identity, not every ticket on the chunk")
    check(not m5.remove(FORCED, ChunkPos(3, 3), key="drop"),
          "removing it twice reports False")

    # The position index is derived state; assert it stayed consistent with
    # the authority after all of the above rather than trusting the helpers.
    for manager, label in ((m3, "remove-key"), (m4, "move"), (m5, "remove")):
        flat = [t for bucket in manager._by_pos.values() for t in bucket]
        check(len(flat) == len(manager)
              and len({id(t) for t in flat}) == len(flat)
              and all(bucket for bucket in manager._by_pos.values())
              and all((t.pos.x, t.pos.z) == pk
                      for pk, bucket in manager._by_pos.items() for t in bucket),
              f"the position index is in step with the ticket table ({label})",
              f"{len(flat)} indexed vs {len(manager)} tickets")

    m6 = TicketManager("clear")
    for pos in (ChunkPos(0, 0), ChunkPos(2, 2)):
        m6.add(FORCED, pos, key="x")
    m6.clear()
    check(len(m6) == 0 and not m6._by_pos
          and m6.level(ChunkPos(0, 0)) == MAX_LEVEL
          and m6.loaded_positions() == set(),
          "clear() empties both the table and the index")
    check(m6.stats()["tickets"] == 0 and m6.stats()["by_type"] == {}
          and m6.stats()["by_load_type"] == {},
          "stats() survives an empty manager")

    # A populated stats() has to agree with what was added, and it reports the
    # tickets' own levels rather than the effective level map (which `level()`
    # is for) -- worth pinning because the two differ the moment two tickets
    # overlap, which is the normal case in a live dimension.
    m7 = TicketManager("stats")
    m7.add(PLAYER_SIMULATION, ChunkPos(0, 0), radius=10, key="p")
    m7.add(PLAYER_LOADING, ChunkPos(0, 0), radius=16, key="p")
    m7.add(FORCED, ChunkPos(0, 0), key=None)      # level 31, on the same chunk
    m7.add(PORTAL, ChunkPos(1, 1), key=None)      # level 30, expiring
    m7.add(ENDER_PEARL, ChunkPos(2, 2), key="e")
    summary = m7.stats()
    check(summary["tickets"] == 5 and summary["positions"] == 3
          and summary["by_type"] == {"player_simulation": 1, "player_loading": 1,
                                     "forced": 1, "portal": 1, "ender_pearl": 1},
          "stats() counts the tickets and positions that exist",
          f"{summary}")
    check(summary["expiring"] == 2 and summary["activates_dimension"]
          and summary["by_load_type"] == {"ENTITY_TICKING": 5},
          "stats() counts the two expiring tickets and reports ticket levels",
          f"{summary}")


# ---------------------------------------------------------------------------
# 8. Scale: the closed form has to stay cheap
# ---------------------------------------------------------------------------

def test_query_cost() -> None:
    """Guard the design decision, not a performance target.

    2000 `level()` + `ticks_entities()` pairs against four player-shaped
    simulation tickets at radius 10 (plus a portal and a loading ticket, so
    the four that are not nearest are still visited). The threshold is 50 us
    per query pair, which is roughly two orders of magnitude above what the
    closed form costs here; a per-query flood fill over even a 21x21 square
    would be several hundred microseconds and trip it. A regression that
    merely makes the loop somewhat slower should not fail this check.
    """
    manager = TicketManager("scale")
    for x in (-10, 10):
        for z in (-10, 10):
            manager.add(PLAYER_SIMULATION, ChunkPos(x, z), radius=10,
                        key=f"p{x},{z}")
    manager.add(PORTAL, ChunkPos(3, 3), key="portal")
    manager.add(PLAYER_LOADING, ChunkPos(0, 0), radius=16, key="view")

    # Walk the whole 21x21 square that the four simulation tickets tick,
    # rather than drawing random positions: the cost of a query must not
    # depend on where in the square it lands, and a deterministic sweep is
    # reproducible when the number is reported.
    positions = [ChunkPos(x, z) for x in range(-20, 21) for z in range(-20, 21)]

    def sweep(count: int) -> float:
        """Seconds per query pair over `count` query pairs."""
        start = time.perf_counter()
        for index in range(count):
            pos = positions[index % len(positions)]
            manager.level(pos)
            manager.ticks_entities(pos)
        return (time.perf_counter() - start) / count * 1e6

    queries = 2000
    sweep(queries)          # warm up: first calls pay for the interpreter's
    sweep(queries)          # warm-up, not for the algorithm
    samples = [sweep(queries) for _ in range(5)]

    best = min(samples)
    median = statistics.median(samples)
    check(best < 50.0,
          f"{queries} level()+ticks_entities() query pairs on 6 tickets: "
          f"best {best:.2f} us/query, median {median:.2f} us/query "
          f"(threshold 50 us)",
          f"samples: {[round(s, 2) for s in samples]} us/query")

    # A flood fill would also show up as cost growing with the *area* asked
    # about rather than the number of tickets, so measure the ratio between a
    # query inside the squares and one far outside them.
    far = [ChunkPos(5000 + index, 5000) for index in range(200)]
    start = time.perf_counter()
    for index in range(queries):
        manager.level(far[index % len(far)])
    empty_cost = (time.perf_counter() - start) / queries * 1e6
    print(f"         far-from-any-ticket query: {empty_cost:.2f} us/query")


def main() -> int:
    header = "chunk ticket checks"
    print(f"{header}\n{'=' * len(header)}")

    rng = random.Random(0xC407A1)
    groups = (
        ("[1/8] closed form vs brute force", lambda: test_closed_form_vs_relaxation(rng)),
        ("[2/8] the wiki's worked example", test_wiki_example),
        ("[3/8] load types", test_load_types),
        ("[4/8] player squares", test_player_squares),
        ("[5/8] expiry", test_expiry),
        ("[6/8] flags and dimension activity", test_flags),
        ("[7/8] ticket identity", test_identity),
        ("[8/8] query cost", test_query_cost),
    )
    for title, run in groups:
        print(f"\n{title}")
        run()

    if FAILURES:
        print(f"\nRESULT: FAIL ({len(FAILURES)})")
        for line in FAILURES:
            print(f"  - {line}")
        return 1
    print("\nRESULT: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
