"""
Cross-thread light queries: the actor side and the console side of `/light`.

The light arrays belong to the dimension actor's thread, so a console command
cannot read them directly -- that is a real race on a free-threaded build, not
a theoretical one. The established way across is `StatsMsg`
(`world/chunk_lifecycle.py`): the caller posts a message carrying a `box` dict
and a `threading.Event`, the owning thread fills the box and *then* sets the
event, and the caller waits with a timeout.

This module holds the two messages and the one function that produces their
answers, so that neither side has to import the other:

* the console (`console/commands.py`) posts the message and renders the box;
* the actor (`world/dimension_actor.py`) calls `read_light` and sets `done`;
* with no actor -- a tool or a test -- the console calls `read_light` itself,
  on the thread that owns the dimension, because nobody else is touching it.

`read_light` is deliberately the *only* producer of the answer. A second
implementation on the actor side would be a second answer to "how bright is it
here", and the console's job is to print it.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Dict

#: Seconds `/light` waits for the dimension actor's reply. `World.get_chunk_stats`
#: waits 2.0 for the same reason: long enough for a busy actor to finish the tick
#: it is in, short enough that the console does not look hung.
LIGHT_QUERY_TIMEOUT = 2.0


@dataclass
class LightQueryMsg:
    """Ask the owning thread for the light at one world position.

    `box` keys `read_light` fills:

        loaded        bool         the chunk is in memory
        in_range      bool         y is inside the dimension's build height
        min_y/max_y   int          that height range, for the message
        populated     bool         chunk.light_populated: real light or not
        sky           int | None   sky light (0 in a dimension without a sky)
        block         int | None   block light
        level         int          max(sky, block), or the fallback value
        fallback      bool         `level` came from the heightmap approximation
        top           int | None   that approximation's reference surface
        has_skylight  bool         dimension.settings.has_skylight
        error         str | None   the read failed; the other keys are absent

    `loaded` is always present, so "the box is empty but `done` was set" means
    the handler is incomplete rather than that the chunk is missing.
    """

    x: int = 0
    y: int = 0
    z: int = 0
    box: Dict[str, object] = field(default_factory=dict)
    done: threading.Event = field(default_factory=threading.Event)


@dataclass
class LightRelightMsg:
    """Ask the owning thread to queue full 3x3 patches.

    Three shapes, one message:

        one chunk     `chunk_x, chunk_z`, `radius` 0, `all_loaded` False
        a square      `radius` > 0: the (2r+1)^2 chunks around the anchor
        everything    `all_loaded` True: every generated chunk the dimension has

    The batch shapes exist because a single-chunk patch is the *repair* path for
    a whole save: light written before the engine existed is the all-15 default,
    and the only way to correct it is to queue every chunk (`world/README.md`
    calls that out as an explicit command rather than something automatic).

    `box` keys the handler fills:

        chunks      int          how many were asked for
        queued      int          how many `DimensionLight.queue_chunk` accepted
        skipped     int          targets with no generated chunk behind them
        loaded      bool         `queued` > 0
        neighbours  int          the anchor's 3x3, one-chunk shape only
        pending     int          engine backlog after queueing
        lit_now     bool         optional: the handler also ran patches
        error       str | None

    The engine's entry point is `DimensionLight.queue_chunk(chunk_x, chunk_z)`
    (`world/light_engine.py`); the recompute itself happens on the actor's next
    `DimensionLight.tick()`, inside `light-budget-ms`. "Queued" is what
    "accepted" means here, which is why the command reports the queue and not a
    light value -- and why queueing 700 chunks costs one deque append each and
    answers immediately.
    """

    chunk_x: int = 0
    chunk_z: int = 0
    radius: int = 0
    all_loaded: bool = False
    box: Dict[str, object] = field(default_factory=dict)
    done: threading.Event = field(default_factory=threading.Event)


def read_light(dimension, x: int, y: int, z: int) -> Dict[str, object]:
    """One position's light. Call on the thread that owns the dimension.

    It asks the same three questions the spawner asks
    (`entity/spawn_placement.py`, the light engine's one consumer), so `/light`
    cannot disagree with a spawn decision:

    * no chunk -> `loaded` False. An unloaded chunk is a *refusal*, never a
      level: reporting 0 would make a player's unloaded neighbourhood read as a
      cave, and reporting 15 as daylight;
    * `light_populated` False -> the stored arrays are untouched defaults, so
      the answer is the heightmap approximation the spawner falls back to,
      flagged with `fallback` True. It is deliberately *not* presented as
      engine output;
    * otherwise -> the real nibbles, and `level` is `max(sky, block)`.

    An unavailable `entity.spawn_placement` is reported in `error`; nothing
    here raises, because the caller is a console loop or a tick handler and
    both would rather print a line than die.
    """
    answer: Dict[str, object] = {}
    chunk = dimension.get_chunk(x >> 4, z >> 4)
    answer["loaded"] = chunk is not None
    if chunk is None:
        return answer

    min_y = chunk.min_y
    max_y = chunk.min_y + chunk.height - 1
    answer["min_y"] = min_y
    answer["max_y"] = max_y
    answer["in_range"] = min_y <= y <= max_y
    if not answer["in_range"]:
        return answer

    answer["populated"] = bool(chunk.light_populated)
    answer["has_skylight"] = bool(
        getattr(getattr(dimension, "settings", None), "has_skylight", True)
    )
    try:
        from entity.spawn_placement import (
            block_light_at,
            sky_light_at,
            top_motion_blocking_no_leaves,
        )
    except ImportError as error:  # pragma: no cover - only without entity/
        answer["error"] = f"light query unavailable: {error}"
        return answer

    if not answer["populated"]:
        # No real light in this chunk: answer the way the spawner would rather
        # than reading a default array as if the engine had written it.
        top = top_motion_blocking_no_leaves(dimension, x, z)
        answer["fallback"] = True
        answer["top"] = top
        answer["sky"] = None
        answer["block"] = None
        answer["level"] = 15 if (top is not None and y > top) else 0
        return answer

    sky = sky_light_at(dimension, x, y, z)
    block = block_light_at(dimension, x, y, z)
    answer["fallback"] = False
    answer["sky"] = sky
    answer["block"] = block
    answer["level"] = max(sky, block)
    return answer


def relight_area(dimension, light, chunk_x: int, chunk_z: int,
                 radius: int = 0) -> Dict[str, object]:
    """Queue a full patch for a square of chunks around the anchor.

    `radius` is in chunks: 0 is the anchor alone (the "this one chunk is wrong"
    request), 1 is the 3x3 around it. The square is a *target list*, not a loop
    over patches -- `_queue_relight` only appends to the engine's queue, and the
    patches run later inside the light tick's budget. A radius that reaches past
    the loaded region is normal, not an error: those targets come back counted in
    `skipped`.
    """
    if radius <= 0:
        targets = [(chunk_x, chunk_z)]
    else:
        targets = [
            (chunk_x + dx, chunk_z + dz)
            for dx in range(-radius, radius + 1)
            for dz in range(-radius, radius + 1)
        ]
    return _queue_relight(dimension, light, targets, anchor=(chunk_x, chunk_z))


def relight_all(dimension, light) -> Dict[str, object]:
    """Queue a full patch for every generated chunk this dimension holds.

    The explicit full-map recompute: the repair for a save whose light was
    written before the engine existed (`ChunkSection.sky_light` still the all-15
    default, which nothing else would ever correct -- `world/README.md` lists
    that as a job for a command exactly like this one). It reads the chunk
    manager's key set once, on the thread that owns it, and queues the chunks
    whose terrain is real: patching a blank claim would compute the field of an
    all-air chunk that the generator is about to replace, and the lifecycle
    queues the real chunk for light anyway when it arrives.
    """
    chunks = getattr(getattr(dimension, "chunk_manager", None), "chunks", None) or {}
    targets = [(pos.x, pos.z) for pos in list(chunks.keys())]
    return _queue_relight(dimension, light, targets, anchor=None)


def _generated(dimension, chunk_x: int, chunk_z: int):
    """The chunk at that coord when its terrain is real, else None."""
    chunk = dimension.get_chunk(chunk_x, chunk_z)
    if chunk is None:
        return None
    # `getattr` default True: a chunk that says nothing about generation is one
    # a fixture built by hand, terrain included.
    return chunk if getattr(chunk, "is_generated", True) else None


def _queue_relight(dimension, light, targets, anchor) -> Dict[str, object]:
    """Queue one full patch per target and count what happened.

    `light` is the dimension's `DimensionLight`, or None when no engine is
    running for this dimension. Queueing is the whole operation: the patches run
    on the next `tick()`, inside `light-budget-ms`, so a command that waited for
    the result would be a command that can time out because the server is busy --
    exactly the failure this message shape exists to avoid. That is also what
    makes a 700-chunk request cheap: 700 deque appends and an immediate answer,
    with the work showing up in `pending` and in `chunks_lit` as it drains.
    """
    answer: Dict[str, object] = {"chunks": 0, "queued": 0, "skipped": 0}
    if light is None:
        answer["error"] = "no light engine is running for this dimension"
        return answer

    for chunk_x, chunk_z in targets:
        answer["chunks"] += 1
        if _generated(dimension, chunk_x, chunk_z) is None:
            answer["skipped"] += 1
            continue
        light.queue_chunk(chunk_x, chunk_z)
        answer["queued"] += 1

    answer["loaded"] = answer["queued"] > 0
    if anchor is not None and answer["chunks"] == 1:
        # Only meaningful for the single-chunk shape: a 3x3 patch needs all nine
        # in memory, so this is the number that explains a patch which does not
        # seem to run. For a batch it would be one arbitrary chunk's nine.
        present = 0
        for dz in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dimension.get_chunk(anchor[0] + dx, anchor[1] + dz) is not None:
                    present += 1
        answer["neighbours"] = present
    answer["pending"] = light.pending_chunks
    return answer
