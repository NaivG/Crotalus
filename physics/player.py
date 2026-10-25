"""Server-authoritative validation of a player's movement path.

    validator = PlayerMoveValidator(view, shapes)   # once, on the actor thread
    validator.warm_up()                             # compile before the first tick

    # world/dimension_actor.py, the PlayerViewMsg branch of `_handle`:
    self.player_validator.on_view_message(msg)
    # world/dimension_actor.py, `_on_tick`, next to the physics tick:
    self.player_validator.tick()

That is the whole wiring: one call per movement packet, one per tick. The
validator reads the same `WorldView` and `BlockShapes` the mob solver sweeps
against, so a player is judged against exactly the geometry the mobs are and
neither of them owns a copy of the world.

What one movement packet does
-----------------------------

A player is **client-predicted**. `network/client_handler.py:_apply_movement`
still writes the reported position into `Player.position` -- that path is
verified and relied on by the streamer, the tracker and the saver -- and the
same packet is published to the actor thread as a `PlayerViewMsg`
(`client_handler.py:_publish_view`). Here, per message:

1. the actor-side **ghost** for that player remembers where the *server* says
   they are;
2. the displacement from the ghost to the reported position is swept through
   the same `physics.kernel.move_batch` the mobs use, with the player's
   0.6 x 1.8 box, giving the furthest point along the claimed path the world
   allows;
3. **legal** (nothing was clipped, and the horizontal displacement is within
   the speed gate): the ghost moves to the swept position and no packet is
   sent -- the client's prediction stands;
4. **illegal**: a `CSynchronizePlayerPosition` is built and handed to
   `msg.sink.send()`, the existing "safe from any thread" bridge
   (`world/chunk_streamer.py:79`). The handshake is the join one -- a teleport
   id plus `STeleportConfirm` back into
   `client_handler.py:_handle_teleport_confirm` -- because a second teleport
   path is a second set of bugs.

The four rules that decide whether this feels right or rubber-bands
------------------------------------------------------------------

1. **Threshold and hysteresis.** A deviation at or below
   `correction_threshold` (default 0.5 blocks) sends **nothing**. Client
   prediction and our sweep never agree to the last bit, and correcting every
   tick is what makes a player feel they are being shoved around.
2. **Cooldown.** After a correction no further correction goes out for
   `cooldown_ticks` (default 10 = half a second). Suppressed corrections are
   counted (`corrections_suppressed`), so "the cooldown is eating real
   violations" is visible instead of silent.
3. **Corrections carry a teleport id** and use the existing handshake. The id
   sequence is seeded from the one the handler publishes on every view
   (`PlayerViewMsg.teleport_id`), so the two never disagree about an id the
   client is actually waiting on.
4. **Rotation is never corrected.** yaw/pitch are copied from the client's
   message into the correction verbatim. Only position is judged, because only
   position is something this server has an opinion about.

The ghost is always the *physical* position
-------------------------------------------

Whether or not a packet goes out, the ghost takes the swept position for a
path violation. That is what stops a sub-threshold penetration from
accumulating: a client that claims to be 0.2 blocks inside a wall is not
corrected, but the server's copy of them stays at the wall face, so creeping
through the wall a fifth of a block at a time never gets anywhere.

The "before" position of a step is the ghost's, not the position the client
reported last packet. `_apply_movement` hands over the *after* position (it
publishes the pose it just folded in); the before side is the server's own
record, because a sweep that started wherever the client says it was would let
a cheat move the origin of its own check on every packet.

Deliberate deviation from vanilla: no input replay
--------------------------------------------------

Vanilla replays the client's input, simulates it, and answers a mismatch with
`moved too quickly` / `moved wrongly` plus a teleport back. This server has no
input system -- `SPlayerInput` is logged and dropped
(`client_handler.py:_handle_player_input`), there is no jump, no sprint and no
friction model -- so a replay would be *wrong*, not merely expensive: it would
simulate a player that does not match the game the client is playing, and the
result would be a correction on every tick.

So the client's prediction stays the source of truth and the server only
rejects what the collision engine can prove impossible:

* the **path** from the last server-side position to the reported one is swept
  against real block shapes. If the sweep has to clip it, the swept point is
  where the player really can be, and that is the correction.
* a crude **speed gate** (`speed_cap * SPEED_MARGIN`, horizontal only) catches
  the displacement no legitimate tick produces, which is the teleport-hack
  shape rather than a speed-hack shape. A step over the gate is rejected
  wholesale -- there is no honest way to guess how much of it was real -- so
  the correction is the last position the server believed, which is vanilla's
  own answer to `moved too quickly`.

Known gaps, deliberately not faked
----------------------------------

* **The speed gate is horizontal only.** A terminal-velocity fall is 3.92
  blocks/tick and is perfectly legal, so gating the total displacement (as
  `PHYSICS_PLAN.md` §4.5 words it) would rubber-band everyone who walks off a
  cliff. Vertical speed is therefore not gated at all: a client that reports
  flying upwards 100 blocks per tick is not caught by this round.
* **Spectators are not judged.** They fly through blocks by design; clipping
  their path and correcting it would be wrong. The message already carries the
  flag, so this is a one-line exemption, and it is counted.
* **The first pose from a player is taken on trust** (`seeded`). There is no
  previous position to sweep from, and vanilla likewise trusts the client's
  first report after a teleport.
* **A server-side teleport must call `reset()`.** The client's next report is a
  legal jump of arbitrary length; judging it against the pre-teleport position
  would correct a player the *server* moved. The only teleport today is the
  join one, whose first report seeds the ghost, so nothing needs to call it
  yet -- `reset()` exists so that `/tp` and respawn have somewhere to go.
* **A chunk in the swept span that is not resident means "cannot judge"**: the
  step is skipped, the ghost does not move, and `deferred_unloaded` counts it.
  Unloaded is neither air nor solid (`PHYSICS_PLAN.md` §§4.7, 八.2). The speed
  gate is the exception, because it needs no world: a step over it is corrected
  even when the geometry could not answer, or an unloaded chunk would be a
  place where an impossible displacement is free.
* **A displacement too long to sweep** (`max_sweep`, 16 blocks per axis) is
  answered by the speed gate alone. Sweeping a 10^9-block claim would walk the
  kernel's block range until the actor thread gave up, so the bound is there
  for the same reason the gate is crude. Counted as `too_far`.
* **Creative flight is only just inside the gate** (0.545 blocks/tick against a
  0.625 threshold); sprint-flying, elytra and vehicles are outside it and would
  be corrected. There are no vehicles and no elytra in this server yet, and the
  cap is a constructor argument so the round that adds them can raise it.

Thread ownership
----------------

Everything here belongs to the dimension actor thread, like the rest of
`physics/`. The one thing that crosses a thread boundary is the correction
packet, and it crosses through `ChunkSink.send` -- the sanctioned bridge, no
locks. The `Player` object is never touched: the ghost is a copy, exactly as
`ChunkStreamer` keeps a copy of the position and `EntityTracker` keeps a
snapshot.

Config keys that will feed the arguments
----------------------------------------

`core/config.py` and `config.toml` belong to the config round, so this module
takes keyword arguments with documented defaults instead of reading config:

* `correction_threshold` <- `movement-correction-threshold` (planned, 0.5)
* `cooldown_ticks`       <- `movement-correction-cooldown-ticks` (planned, 10)
* `speed_cap`            <- `movement-speed-cap` (planned, see `DEFAULT_SPEED_CAP`)

and the whole validator is behind the planned `server-authoritative-movement`
switch, which the actor reads when it decides whether to call
`on_view_message` at all.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

from network.packets import CSynchronizePlayerPosition
from physics.block_shapes import BlockShapes
from physics.kernel import (MAX_CANDIDATES, STATUS_DEFERRED, STATUS_OVERFLOW,
                            move_batch, warm_up)
from physics.solver import STEP_HEIGHT
from physics.world_view import WorldView

__all__ = [
    "PlayerMoveValidator", "PlayerGhost", "Correction",
    "PLAYER_WIDTH", "PLAYER_HEIGHT", "SPEED_MARGIN", "SPRINT_SPEED",
    "DEFAULT_SPEED_CAP", "DEFAULT_CORRECTION_THRESHOLD",
    "DEFAULT_COOLDOWN_TICKS", "MAX_SWEEP_BLOCKS",
]

#: The vanilla player collision box. Not read from an entity type: a player is
#: not an `Entity` here (`Player` lives in `player/` and owns no physics state),
#: and the box is fixed until poses land.
PLAYER_WIDTH = 0.6
PLAYER_HEIGHT = 1.8

#: Vanilla's sprinting speed on the surface: 5.612 m/s over 20 ticks.
SPRINT_SPEED = 0.2806

#: The speed gate's cap, in blocks per tick, before `SPEED_MARGIN`.
#:
#: It is deliberately not walking speed: the gate's job is to catch the
#: teleport-hack shape (a whole-block-scale or larger jump inside one tick),
#: while every legitimate tick has to pass. `speed_cap * SPEED_MARGIN` = 0.625
#: blocks/tick clears `SPRINT_SPEED` with room for a sprint-jump's extra,
#: knockback, a packet that arrives in a burst, and creative flight (0.545) --
#: the last one only just, which is the known gap in the module docstring.
#: Sprint-flying (1.09) and elytra speeds do not exist in this server yet; the
#: round that adds them raises this.
DEFAULT_SPEED_CAP = 0.5

#: Headroom over `speed_cap`, from `PHYSICS_PLAN.md` §4.5 ("speed_cap x 1.25").
SPEED_MARGIN = 1.25

#: A deviation below this many blocks is not worth a packet: the client's
#: prediction and our sweep are allowed to disagree by this much forever.
DEFAULT_CORRECTION_THRESHOLD = 0.5

#: Ticks a player is left alone after a correction: 10 ticks is half a second,
#: which comfortably covers a round trip to a client on a bad connection. A
#: shorter cooldown risks a second correction landing on a client that is still
#: applying the first, which is the rubber-band the cooldown exists to prevent.
DEFAULT_COOLDOWN_TICKS = 10

#: The longest displacement per axis that is worth sweeping. Past it the claim
#: is not geometry any more -- a hostile 10^9-block jump would walk the kernel's
#: block range for minutes on the actor thread -- so the speed gate decides
#: alone. 16 blocks is four terminal-velocity falls, or 57 ticks of sprinting.
MAX_SWEEP_BLOCKS = 16.0


@dataclass(frozen=True)
class Correction:
    """One position the server pulled a player back to, and why.

    Returned by `on_view_message` for the packet it just sent, so a caller can
    log or assert on the decision without decoding the packet. `reason` is
    `"path"` (the collision engine clipped the claimed path) or `"speed"` (the
    displacement was beyond the gate).
    """

    player_id: str
    teleport_id: int
    x: float
    y: float
    z: float
    #: Copied from the client's message, never invented (rule 4).
    yaw: float
    pitch: float
    #: How far the client's claim was from the swept position, in blocks.
    deviation: float
    #: The claimed horizontal displacement, in blocks.
    horizontal: float
    reason: str


class PlayerGhost:
    """The actor thread's copy of one player: where the server says they are.

    A plain object rather than a `BodyStore` slot: the store keys bodies by an
    integer entity id, and a player's identity here is the handler-identity
    string the streamer and the tracker already key by (`player_id`), which is
    what a `PlayerViewMsg` carries on every packet. There is no per-tick
    integration to do -- the client supplies the displacement -- so the ghost is
    a position, a box, and the two counters that implement the cooldown and the
    teleport handshake.
    """

    __slots__ = ("player_id", "x", "y", "z", "width", "height",
                 "cooldown", "next_teleport_id", "last_sequence", "steps")

    def __init__(self, player_id: str, x: float, y: float, z: float,
                 width: float, height: float, teleport_id: int = 0) -> None:
        self.player_id = player_id
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)
        self.width = float(width)
        self.height = float(height)
        #: Ticks left before another correction may go out (rule 2).
        self.cooldown = 0
        #: The next teleport id this validator may use. Seeded above the last
        #: id the handler published, so it never collides with a teleport the
        #: client is already waiting to confirm.
        self.next_teleport_id = int(teleport_id) + 1
        #: Highest `PlayerViewMsg.sequence` seen; 0 means "unsequenced".
        self.last_sequence = 0
        #: Judged steps, for diagnostics.
        self.steps = 0

    @property
    def position(self) -> Tuple[float, float, float]:
        return (self.x, self.y, self.z)

    def __repr__(self) -> str:  # pragma: no cover - diagnostics only
        return (f"<PlayerGhost {self.player_id} "
                f"({self.x:.3f}, {self.y:.3f}, {self.z:.3f}) "
                f"cooldown={self.cooldown}>")


class PlayerMoveValidator:
    """Checks a player's reported path and produces corrections.

    Constructed and used on the owning dimension actor thread. One instance per
    dimension, not per player: the per-player state is the `ghosts` dict, and a
    player who disconnects is dropped from it by the same message that tells the
    streamer and the tracker (`sink=None`).
    """

    __slots__ = ("view", "shapes", "correction_threshold", "cooldown_ticks",
                 "speed_cap", "speed_threshold", "width", "height",
                 "step_height", "max_sweep", "ghosts",
                 "_x", "_y", "_z", "_vx", "_vy", "_vz", "_width", "_height",
                 "_active", "_motion", "_probe", "_status", "_candidates",
                 "_stats")

    def __init__(self, view: WorldView, shapes: BlockShapes, *,
                 correction_threshold: float = DEFAULT_CORRECTION_THRESHOLD,
                 cooldown_ticks: int = DEFAULT_COOLDOWN_TICKS,
                 speed_cap: float = DEFAULT_SPEED_CAP,
                 width: float = PLAYER_WIDTH, height: float = PLAYER_HEIGHT,
                 step_height: float = STEP_HEIGHT,
                 max_sweep: float = MAX_SWEEP_BLOCKS) -> None:
        #: The same view the solver sweeps against, so "the world" means one
        #: thing in a tick. Not copied, not wrapped.
        self.view = view
        self.shapes = shapes
        self.correction_threshold = float(correction_threshold)
        self.cooldown_ticks = int(cooldown_ticks)
        self.speed_cap = float(speed_cap)
        #: The actual gate: `speed_cap` with its margin already applied.
        self.speed_threshold = float(speed_cap) * SPEED_MARGIN
        self.width = float(width)
        self.height = float(height)
        self.step_height = float(step_height)
        self.max_sweep = float(max_sweep)
        self.ghosts: Dict[str, PlayerGhost] = {}

        # One body's worth of scratch: the kernel takes arrays, and a
        # per-message validation is inherently one body. `_x`/`_y`/`_z` come
        # back holding the swept position, which is the answer.
        self._x = np.zeros(1, dtype=np.float64)
        self._y = np.zeros(1, dtype=np.float64)
        self._z = np.zeros(1, dtype=np.float64)
        self._vx = np.zeros(1, dtype=np.float64)
        self._vy = np.zeros(1, dtype=np.float64)
        self._vz = np.zeros(1, dtype=np.float64)
        self._width = np.zeros(1, dtype=np.float64)
        self._height = np.zeros(1, dtype=np.float64)
        self._active = np.zeros(1, dtype=np.int32)
        self._motion = np.zeros((1, 3), dtype=np.float64)
        self._probe = np.zeros(1, dtype=np.uint8)
        self._status = np.zeros(1, dtype=np.uint8)
        self._candidates = np.zeros((MAX_CANDIDATES, 6), dtype=np.float64)

        self._stats: Dict[str, int] = {
            "players": 0, "views": 0, "seeded": 0, "judged": 0, "still": 0,
            "violations": 0, "violations_path": 0, "violations_speed": 0,
            "corrections": 0, "corrections_suppressed": 0, "undeliverable": 0,
            "too_far": 0, "deferred_unloaded": 0, "overflow": 0,
            "skipped_spectator": 0, "stale_views": 0, "resets": 0, "removed": 0,
        }

    def warm_up(self) -> None:
        """Compile the kernel now, never inside a tick.

        Idempotent, and shared with `PhysicsSolver.warm_up` -- whichever runs
        first pays the LLVM compile, and a restart pays only the cache check
        (`PHYSICS_PLAN.md` §九.2: the compile must not land in a tick).
        """
        warm_up()

    # ----- the per-packet entry point -----

    def on_view_message(self, msg) -> Optional[Correction]:
        """Judge one published player pose. Returns the correction, if any.

        Called for every `PlayerViewMsg`, which `_apply_movement` publishes once
        per movement packet. The message carries the position the client just
        claimed; the position it is judged *from* is the ghost's, not the
        client's own previous report -- trusting the client's "I was here" would
        let a cheat reset the sweep origin on every packet.

        The fields read: `player_id`, `sink` (`None` = the player left), `x`,
        `y`, `z`, `yaw`, `pitch`, `spectator`, `sequence`, `teleport_id`. A
        stand-in message needs those and nothing else.
        """
        player_id = msg.player_id
        if msg.sink is None:
            # The same message that detaches the streamer and the tracker: a
            # player who left must not leave a ghost behind, or the next join
            # under the same id would be judged against a stale position.
            if self._drop(player_id):
                self._stats["removed"] += 1
            return None

        ghost = self.ghosts.get(player_id)
        if ghost is None:
            self._seed(msg)
            return None

        if msg.sequence and msg.sequence <= ghost.last_sequence:
            # Out of order or a repeat. The inbox is one ordered stream per
            # dimension, so this should not happen; counted rather than
            # assumed, because judging a replayed packet twice is exactly the
            # kind of thing that shows up as an unexplained correction.
            self._stats["stale_views"] += 1
            return None
        if msg.sequence:
            ghost.last_sequence = msg.sequence
        if msg.teleport_id >= ghost.next_teleport_id:
            # The handler has told the client about a newer teleport than we
            # have used; our next correction must not reuse that id.
            ghost.next_teleport_id = msg.teleport_id + 1

        if msg.spectator:
            # Flying through blocks is what a spectator does. Following the
            # report keeps the ghost usable if they switch back to survival.
            ghost.x, ghost.y, ghost.z = float(msg.x), float(msg.y), float(msg.z)
            self._stats["skipped_spectator"] += 1
            return None

        self._stats["views"] += 1
        return self._judge(ghost, msg)

    def _judge(self, ghost: PlayerGhost, msg) -> Optional[Correction]:
        """Sweep the claimed displacement and decide what to do about it."""
        reported_x = float(msg.x)
        reported_y = float(msg.y)
        reported_z = float(msg.z)
        dx = reported_x - ghost.x
        dy = reported_y - ghost.y
        dz = reported_z - ghost.z
        horizontal = math.hypot(dx, dz)
        speed_ok = horizontal <= self.speed_threshold
        span = max(abs(dx), abs(dy), abs(dz))

        if dx == 0.0 and dy == 0.0 and dz == 0.0:
            # A rotation-only or status-only packet republishes the same
            # position. Nothing to sweep and nothing to judge: the world is not
            # read at all, which is what keeps a standing player free.
            self._stats["still"] += 1
            ghost.steps += 1
            return None

        swept: Optional[Tuple[float, float, float]] = None
        if span > self.max_sweep:
            # Not geometry any more -- see `MAX_SWEEP_BLOCKS`. The speed gate
            # answers alone below.
            self._stats["too_far"] += 1
        else:
            status = self._sweep(ghost, dx, dy, dz)
            if status & STATUS_DEFERRED:
                self._stats["deferred_unloaded"] += 1
            elif status & STATUS_OVERFLOW:
                self._stats["overflow"] += 1
            else:
                swept = (float(self._x[0]), float(self._y[0]), float(self._z[0]))
                self._stats["judged"] += 1

        if swept is None:
            # The world could not answer. A step that is fine by speed is
            # skipped outright -- never guessed at -- and the ghost stays put.
            if speed_ok:
                return None
            deviation = horizontal
            reason = "speed"
            target = ghost.position
        else:
            deviation = math.dist(swept, (reported_x, reported_y, reported_z))
            if deviation > self.correction_threshold:
                reason = "path"
                target = swept
            elif not speed_ok:
                reason = "speed"
                target = ghost.position
            else:
                # Legal: the client's prediction stands, no packet.
                ghost.x, ghost.y, ghost.z = swept
                ghost.steps += 1
                return None

        self._stats["violations"] += 1
        self._stats["violations_path" if reason == "path" else "violations_speed"] += 1
        ghost.steps += 1
        if reason == "path" and swept is not None:
            # The ghost is the physical position whether or not a packet goes
            # out: a sub-threshold penetration stays at the wall face instead of
            # accumulating, and a player twenty blocks into a wall is judged from
            # the wall, not from where they claim to be. It is also exactly where
            # the correction below sends them, so the server believes what it
            # just told the client.
            #
            # A *speed* violation does NOT move the ghost, even when the path
            # itself was clean: there the correction is "you did not move", and a
            # ghost that had already followed the claim would make the next
            # packet disagree with the teleport in flight -- a rubber-band
            # oscillation between the two positions rather than one correction.
            ghost.x, ghost.y, ghost.z = swept

        if ghost.cooldown > 0:
            self._stats["corrections_suppressed"] += 1
            return None

        correction = Correction(
            player_id=ghost.player_id,
            teleport_id=ghost.next_teleport_id,
            x=target[0], y=target[1], z=target[2],
            # Rule 4: the client's own rotation, untouched.
            yaw=float(msg.yaw), pitch=float(msg.pitch),
            deviation=deviation, horizontal=horizontal, reason=reason,
        )
        ghost.next_teleport_id += 1
        ghost.cooldown = self.cooldown_ticks

        sink = msg.sink
        if getattr(sink, "closed", False):
            # A disconnect in progress. The ghost still moved, but claiming a
            # packet went out when the sink is closed is how "the client never
            # got the correction" becomes invisible.
            self._stats["undeliverable"] += 1
            return None
        sink.send([CSynchronizePlayerPosition(
            teleport_id=correction.teleport_id,
            x=correction.x, y=correction.y, z=correction.z,
            # The movement packets carry no velocity and a correction is not an
            # impulse: the client zeroes its own delta when it accepts one it
            # did not ask for. Vanilla sends zero here too.
            velocity_x=0.0, velocity_y=0.0, velocity_z=0.0,
            yaw=correction.yaw, pitch=correction.pitch,
            flags=0,  # absolute position
        )])
        self._stats["corrections"] += 1
        return correction

    def _sweep(self, ghost: PlayerGhost, dx: float, dy: float, dz: float) -> int:
        """One `move_batch` call for one player. Returns its status byte.

        The body is the ghost's, so `_x`/`_y`/`_z` come back holding the swept
        position. `probe` is 0 on purpose: a probe would leave the scratch
        arrays alone, and the swept point *is* the answer here.
        """
        self._x[0] = ghost.x
        self._y[0] = ghost.y
        self._z[0] = ghost.z
        self._width[0] = ghost.width
        self._height[0] = ghost.height
        self._motion[0, 0] = dx
        self._motion[0, 1] = dy
        self._motion[0, 2] = dz
        self._probe[0] = 0

        sections = self.view.section_arrays()
        keys, slots = self.view.index_arrays()
        move_batch(self._x, self._y, self._z, self._vx, self._vy, self._vz,
                   self._width, self._height, self._active, self._motion,
                   self._probe, self._status,
                   sections, self.view.sections_per_chunk,
                   self.view.min_section_y, keys, slots, int(keys.shape[0]),
                   self.shapes.offsets, self.shapes.ids,
                   self.shapes.min_xyz, self.shapes.max_xyz,
                   self._candidates, self.step_height)
        return int(self._status[0])

    # ----- ghosts -----

    def _seed(self, msg) -> PlayerGhost:
        """Take a player's first pose on trust and start their ghost there."""
        ghost = PlayerGhost(msg.player_id, msg.x, msg.y, msg.z,
                            self.width, self.height, msg.teleport_id)
        self.ghosts[msg.player_id] = ghost
        self._stats["seeded"] += 1
        return ghost

    def _drop(self, player_id: str) -> bool:
        return self.ghosts.pop(player_id, None) is not None

    def for_player(self, player_id: str) -> Optional[PlayerGhost]:
        """The actor-side ghost for a player, or None when there is none."""
        return self.ghosts.get(player_id)

    def reset(self, player_id: str) -> bool:
        """Forget a player's ghost so the next message re-seeds it.

        For a **server-side** teleport (`/tp`, respawn, a dimension change): the
        client's next report is a legal jump of arbitrary length, and judging it
        against the pre-teleport position would correct a player the server
        itself just moved. Returns False when there was no ghost.
        """
        dropped = self._drop(player_id)
        if dropped:
            self._stats["resets"] += 1
        return dropped

    def tick(self) -> None:
        """One actor tick: decay every cooldown (rule 2).

        Driven from `DimensionActor._on_tick`. Note that the ticker discards
        ticks when an actor is behind (`core/ticker.py`), so this counts
        *processed* ticks: an overloaded dimension decays cooldowns more slowly,
        which is the safe direction for a mechanism whose purpose is to avoid
        flooding a client.
        """
        for ghost in self.ghosts.values():
            if ghost.cooldown > 0:
                ghost.cooldown -= 1

    # ----- reporting -----

    def stats(self) -> Dict[str, int]:
        """The counters a `StatsMsg` box reports, as a copy."""
        self._stats["players"] = len(self.ghosts)
        return dict(self._stats)

    def __len__(self) -> int:
        return len(self.ghosts)

    def __repr__(self) -> str:  # pragma: no cover - diagnostics only
        return (f"<PlayerMoveValidator players={len(self.ghosts)} "
                f"threshold={self.correction_threshold} "
                f"gate={self.speed_threshold:.3f}>")
