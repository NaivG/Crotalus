"""
LivingEntity - the level between `Entity` and the things that bleed.

Pumpkin's hierarchy is `Mob -> LivingEntity -> Entity`, with players inserted
on the same shelf as mobs by way of `LivingEntity`. The class carries the
features common to anything that can be hurt and healed: a current and maximum
health, a per-tick invulnerability window after taking damage (`hurt_cooldown`),
the absorption layer (yellow hearts) that sits in front of `health`, the
status-effect map, the last-damage bookkeeping, and the small set of
interaction entry points (`damage`, `heal`, `attack`, `interact`, `knockback`,
fall damage). The mob class above this only owns the AI - the goal selector,
the controls it writes and the per-class behaviour - and the player class
above this only owns the parts that are not common: name, food, experience,
game mode.

**Why a regular class, not a `@dataclass`.** The fields here are not a flat
record; `damage` mutates several of them, `tick` decrements one and may flip
another, and `Entity` is the dataclass for the wire-stable identity block.
Pulling `LivingEntity` under `@dataclass` would re-introduce the constraints
the existing layer is deliberately avoiding.

**Single-thread invariant.** A living entity lives on one thread, the same
as `Entity`: the event loop for players, the dimension actor for mobs.
Mutation is plain attribute access. Crossing threads still happens only via
snapshots (`PlayerViewMsg` and friends), never by sharing this object.

**Equipment and effects are stubs of the right shape, not implementations.**
A full equipment system needs items, and items are not in this round. What
this class carries is the slots the wire and the client predict (`equipment`
dict keyed by slot name) and an `active_effects` dict keyed by effect id, so
`Mob.tick` and `Player.tick` can already iterate them and the next round's
potion and item writers have a defined place to write into.
"""

from __future__ import annotations

from typing import Dict, NamedTuple, Optional

from entity.entity import Entity
from entity.entity_type import EntityType
from entity.position import EntityPosition


#: Default maximum health for a living entity whose type does not say otherwise.
#: Vanilla's player attribute is 20, and most mobs sit somewhere similar; this
#: is the value used when `EntityType.max_health` is None (which happens for
#: most entity types in the asset table - vanilla sets it per class, not per
#: type, and the registry mirrors that gap).
DEFAULT_MAX_HEALTH: float = 20.0

#: Vanilla's `LivingEntity.invulnerableTime` default after taking damage:
#: ticks (10 = half a second) where further damage is ignored. Short enough
#: that mashing attack still hits, long enough that one swing does not eat
#: two ticks of health.
DEFAULT_HURT_COOLDOWN: int = 10

#: Vanilla's `LivingEntity.NO_KNOCKBACK` modifier source; we carry the name
#: rather than the value because nothing compares against it yet, but the
#: string is what `damage_type` callers pass.
DEFAULT_KNOCKBACK_RESISTANCE: float = 0.0


class _ActiveEffect(NamedTuple):
    """A status-effect record.

    Vanilla's `StatusEffectInstance` is richer (ambient, hideParticles,
    showIcon, factor data). This is the half the real implementation will
    keep: remaining duration and amplifier. Anything else is read off
    the effect id through the registry.

    A `NamedTuple` rather than a small class because we never mutate the
    record in place - `add_effect` builds a new one and reassigns the
    dict entry - which keeps mutation sites explicit.
    """

    remaining: int
    amplifier: int


class LivingEntity(Entity):
    """Anything that has health, can be hurt, and can carry status effects.

    `Mob` and `Player` both extend this; `Entity` itself is left open for the
    things that do not bleed (items, projectiles, vehicles - the things
    Pumpkin puts on the same `Entity` base but not on `LivingEntity`).
    """

    def __init__(self,
                 entity_id: int = 0,
                 uuid: str = "",
                 entity_type: Optional[EntityType] = None,
                 position: Optional[EntityPosition] = None,
                 dimension: str = "minecraft:overworld",
                 removed: bool = False,
                 max_health: Optional[float] = None,
                 health: Optional[float] = None) -> None:
        """Build a living entity.

        `max_health` falls back to the entity type's row, then to the
        `DEFAULT_MAX_HEALTH`. `health` defaults to `max_health` so a freshly
        spawned mob is at full hearts - the only way to come back lower is
        `damage`.
        """
        super().__init__(
            entity_id=entity_id,
            uuid=uuid,
            entity_type=entity_type,
            position=position if position is not None else EntityPosition(),
            dimension=dimension,
            removed=removed,
        )

        # ----- health -----
        # Two fields, deliberately, not one. The next round's combat writes
        # `health`; a single field would make "respawn at full" mean "revive
        # the corpse". Same shape as Pumpkin, same shape as the mob/player
        # classes had before this refactor.
        self.max_health: float = self._resolve_max_health(max_health)
        self.health: float = self.max_health if health is None else self._clamp_health(float(health))

        # ----- damage bookkeeping -----
        # The amount actually applied last time `damage` ran, post-armor and
        # post-absorption. Used by `last_hurt_by_player` and friends - those
        # are not consulted yet, but the field is here so the death message
        # has somewhere to come from when it lands.
        self.last_damage_taken: float = 0.0

        # Vanilla's `livingEntity.hurtTime` would also be ticked here; we
        # collapse hurt time and hurt cooldown onto one counter, because the
        # wire sends only one (the red-flash state is derived from it on the
        # client).
        self.hurt_cooldown: int = 0
        """Ticks of invulnerability left. Decremented every `tick` and reset
        to `DEFAULT_HURT_COOLDOWN` by `damage`."""

        # The damage source of the last hit, as the type's resource name
        # (`minecraft:mob_attack`, `minecraft:player_attack`, ...). `None`
        # means "no damage taken this lifetime". Pumpkin keeps a richer
        # `CombatTracker`; we keep the string for panic attribution.
        self.last_damage_type: Optional[str] = None

        # Entity id of whoever last hit us, and game-time of the hit. The
        # second is used by vanilla's `getKillCredit` (a 100-tick / 5-second
        # window), so it is the world tick count, not millis.
        self.last_attacker_id: int = 0
        self.last_attacked_time: int = -10**9

        # Vanilla's yellow hearts. Absorption is subtracted from incoming
        # damage before `health` is touched. Cleared on respawn.
        self.absorption: float = 0.0

        # ----- status effects -----
        # `effect_id -> _ActiveEffect`. Pumpkin stores the full
        # `StatusEffect` struct; this round has no effects implemented, so
        # the value side of the dict is just duration+amplifier. The
        # iteration in `_tick_potion_effects` is the integration point for
        # the real implementation when it lands.
        self.active_effects: Dict[str, _ActiveEffect] = {}

        # ----- death state -----
        # `dead` means `on_death` has run. `death_time` is the vanilla
        # 0..20 animation counter the client plays; reset on respawn.
        self.dead: bool = False
        self.death_time: int = 0

        # ----- fall damage + combat staging -----
        # Vanilla's `fallDistance` accumulator. Written by physics (when it
        # arrives) every tick; `damage` calls `cause_fall_damage` once after
        # physics has run, which is why this lives here rather than on
        # `EntityPosition` (it is a *combat* input, not a movement field).
        self.fall_distance: float = 0.0

        # Vanilla's `knockback` event accepts a `strength` and writes to the
        # entity's velocity. We expose it as a method; the callsite (when it
        # lands, presumably in the combat handlers) reads
        # `entity_type.fire_immune` to gate flames-style no-knockback.
        self.knockback_resistance: float = DEFAULT_KNOCKBACK_RESISTANCE

        # ----- interaction surface -----
        # Stub equipment map. The wire-side writes (helmet, chest, legs,
        # boots, main hand, off hand) will land here; this round does not
        # implement them, but the slot constants and the dict are the
        # shape the next layer codes against.
        self.equipment: Dict[str, Optional[str]] = {
            "head": None,
            "chest": None,
            "legs": None,
            "feet": None,
            "mainhand": None,
            "offhand": None,
        }

        # Knocks into the per-tick mob controls. Lives here because players
        # also jump (a future sprint/climb system would share this) - the
        # alternative (a flag on `Mob` only) is the same boundary drift
        # Pumpkin's `LivingEntity.jumping` exists to prevent.
        self.jumping: bool = False

        # Ticks since the entity existed. Bumped by `tick`, read by combat
        # bookkeeping (`getKillCredit` uses a 100-tick window). Not bumped
        # elsewhere.
        self._age: int = 0

    # ------------------------------------------------------------------ helpers

    def _resolve_max_health(self, override: Optional[float]) -> float:
        """Pick the max-health this entity is born with.

        Order: caller override, then the type's row, then the default. The
        type is usually set by the time we get here (the spawner does it,
        and `Player.__init__` pins it to `minecraft:player`); for the rare
        caller that omits both, fall back to `DEFAULT_MAX_HEALTH` rather
        than raising - an untyped living entity is "for testing", not "for
        production", and tests want a number, not an exception.
        """
        if override is not None:
            return float(override)
        if self.entity_type is not None and self.entity_type.max_health is not None:
            return float(self.entity_type.max_health)
        return DEFAULT_MAX_HEALTH

    def _clamp_health(self, value: float) -> float:
        """Bring a health value into the `[0, max_health]` range.

        Negative `health` would underflow `tick_regeneration` comparisons;
        >max would let a freshly loaded player from a modded save start
        above their cap and confuse the client's HUD. Both are clamped
        silently - this is internal state normalisation, not a public
        check.
        """
        if value < 0.0:
            return 0.0
        if value > self.max_health:
            return self.max_health
        return value

    # ------------------------------------------------------------------ health

    @property
    def is_alive(self) -> bool:
        """True while the entity has a positive health bar and is not `dead`.

        Both halves are needed. A living entity whose `damage` ran the
        death path has `dead=True` even if `health` was not yet clamped to
        zero; conversely, a freshly spawned mob with `health=0` is "alive"
        for one tick (`dead` is False until the tick that flips it).
        """
        return not self.dead and self.health > 0.0

    @property
    def is_dead(self) -> bool:
        return self.dead or self.health <= 0.0

    def damage(self, amount: float, source: Optional[str] = None,
               attacker_id: int = 0) -> bool:
        """Apply damage, honoring `hurt_cooldown` and absorption.

        Returns True if any damage was actually applied; the return is
        load-bearing for the combat system the next round will build
        (the packet-level Damage Event expects a yes/no).

        Order matches `LivingEntity.hurt` in the decompiled vanilla source:

        1. Skip if already dead.
        2. Skip if `hurt_cooldown > 0`.
        3. Subtract from `absorption` first; anything left over hits
           `health`.
        4. Reset `hurt_cooldown` and the last-damage bookkeeping.
        5. If `health` reached zero, run the death path.
        """
        if self.dead:
            return False
        if self.hurt_cooldown > 0:
            return False
        if amount <= 0.0:
            return False

        # Absorption eats damage before health. Vanilla subtracts only
        # `min(amount, absorption)` and keeps the rest for health; we
        # mirror that line-for-line.
        if self.absorption > 0.0:
            absorbed = min(self.absorption, amount)
            self.absorption -= absorbed
            amount -= absorbed

        self.last_damage_taken = amount
        self.health = self._clamp_health(self.health - amount)
        self.hurt_cooldown = DEFAULT_HURT_COOLDOWN
        self.last_damage_type = source
        if attacker_id != 0:
            self.last_attacker_id = attacker_id
            self.last_attacked_time = self.entity_age

        if self.health <= 0.0 and not self.dead:
            self.die()

        return True

    def heal(self, amount: float) -> bool:
        """Restore health up to `max_health`.

        Negative or zero amounts are no-ops; the return is the same shape
        as `damage`'s so a heal-effect handler can ignore the call without
        an extra branch.
        """
        if self.dead:
            return False
        if amount <= 0.0:
            return False
        self.health = self._clamp_health(self.health + amount)
        return True

    def set_health(self, value: float) -> None:
        """Force `health` to a value, clamped and dead-respecting.

        Used by `/effect` give/clear, by scoreboard-driven health bars,
        and by anything else that owns the value rather than computes it.
        Setting below zero on an alive entity does NOT call `die` - that
        is `damage`'s job, and a setter is "replace the value", not "apply
        damage".
        """
        self.health = self._clamp_health(float(value))

    def set_max_health(self, value: float, refill: bool = True) -> None:
        """Change `max_health`.

        `refill=True` (default) rescales `health` proportionally so the bar
        doesn't drop when the cap moves up. This matches vanilla's behavior
        for attributes - you don't suddenly lose hearts because a piece of
        gear raised the cap.
        """
        if value <= 0.0:
            return
        if refill and self.max_health > 0.0:
            ratio = self.health / self.max_health
            self.health = min(value, value * ratio)
        else:
            self.health = min(self.health, value)
        self.max_health = value

    # -------------------------------------------------------------- death path

    def die(self) -> None:
        """Run the death path.

        Pumpkin does a long chain here (drops, score, broadcast event,
        particles). The shape we expose is the minimum: set `dead`,
        start the animation timer, then call `on_death` so subclasses can
        extend without re-implementing the bookkeeping.
        """
        if self.dead:
            return
        self.dead = True
        self.health = 0.0
        self.death_time = 0
        self.absorption = 0.0
        self.on_death()

    def on_death(self) -> None:
        """Override to add type-specific death behavior.

        Called exactly once, by `die`. Subclasses override (drops, score,
        boss kill music, ...); the base implementation is a no-op because
        there is no global death side effect on this server yet.
        """

    def respawn(self) -> None:
        """Bring a dead entity back to full health.

        The mirror of `die`. Clears `dead`, refunds `health`, drops the
        effect list and the absorption overlay. The vanilla attribute
        reroll (random 0..max on respawn) is NOT implemented; a Crotalus
        respawn always returns the nominal max.
        """
        self.dead = False
        self.death_time = 0
        self.health = self.max_health
        self.absorption = 0.0
        self.active_effects.clear()
        self.hurt_cooldown = 0
        self.last_damage_taken = 0.0
        self.last_damage_type = None

    # ----------------------------------------------------- status effects (stub)

    def add_effect(self, effect_id: str, duration_ticks: int,
                   amplifier: int = 0) -> bool:
        """Add or refresh a status effect.

        Stubs out a sensible shape until the real effect system lands.
        The boolean return is "did anything change" - used by the entity
        events layer to decide whether to send an Add Mob Effect packet.
        """
        if duration_ticks <= 0:
            self.remove_effect(effect_id)
            return False
        existing = self.active_effects.get(effect_id)
        if existing is not None:
            # Refresh duration to the longer of the two; only upgrade the
            # amplifier (vanilla: a smaller-amplifier potion does not
            # downgrade an existing one).
            self.active_effects[effect_id] = _ActiveEffect(
                remaining=max(existing.remaining, duration_ticks),
                amplifier=max(existing.amplifier, amplifier),
            )
            return True
        self.active_effects[effect_id] = _ActiveEffect(
            remaining=duration_ticks,
            amplifier=amplifier,
        )
        return True

    def remove_effect(self, effect_id: str) -> bool:
        """Drop a status effect, returning whether one was present."""
        return self.active_effects.pop(effect_id, None) is not None

    def has_effect(self, effect_id: str) -> bool:
        return effect_id in self.active_effects

    def clear_effects(self) -> None:
        self.active_effects.clear()

    # -------------------------------------------------- combat staging / fall

    def cause_fall_damage(self, fall_distance: float, damage_multiplier: float = 1.0) -> bool:
        """Apply fall damage accumulated since the last physics tick.

        Called once per tick, after physics, when `self.fall_distance`
        would otherwise be reset. The multiplier is the per-block damage
        value times any item/armor modifier the next layer will read.

        Returns `damage`'s boolean so callers can chain
        (`if ent.cause_fall_damage(f): ...`).
        """
        # Vanilla skips fall damage when the entity has the slow_falling
        # effect, is in water, on a ladder, in cobweb, or is otherwise
        # exempt - those modifiers land via `damage_multiplier` once effects
        # exist. For now, no exemptions.
        if fall_distance <= 0.0:
            return False
        damage = (fall_distance - 1.0) * damage_multiplier
        if damage <= 0.0:
            return False
        return self.damage(amount=damage, source="minecraft:fall")

    # ----------------------------------------- interaction (the surface)

    def can_be_attacked_by(self, attacker: Optional["LivingEntity"]) -> bool:
        """Whether this entity is a valid damage target right now.

        Dead entities, `removed` entities, and self-attack are rejected.
        Team rules and PvP toggles land later.
        """
        if self.removed or self.dead:
            return False
        if attacker is not None and attacker is self:
            return False
        return True

    def interact(self, other: "LivingEntity") -> bool:
        """The default empty interaction.

        Pumpkin's `LivingEntity.interact` returns `InteractionResult.PASS`
        for anything the type-specific override does not consume. The
        boolean here is the same contract - True and the interaction
        was handled (and the caller should stop), False and it bubbled
        up to the higher-priority handler.

        Subclasses override (`Player.interact` for right-click,
        `Mob.interact` for breeding). The base body is empty on purpose:
        vanilla's `LivingEntity.interact` does nothing, and so does this
        round's server.
        """
        return False

    def attack(self, target: "LivingEntity", damage: float) -> bool:
        """Hit `target` with a melee swing of the given damage.

        Default implementation: applies `damage` via `target.damage`,
        and tags `last_attacking_id` on both sides for combat bookkeeping.
        Return is whether the swing landed (target refused or was
        already dead).

        A full combat handler will replace this with the `Attack` event
        + a knockback vector pulled from the attacker's velocity; this
        round's version is the minimum that lets the next layer wire up
        without re-inventing the bookkeeping.
        """
        if target is None or not target.can_be_attacked_by(self):
            return False
        landed = target.damage(
            amount=damage,
            source=("minecraft:player_attack" if _is_player(self)
                    else "minecraft:mob_attack"),
            attacker_id=self.entity_id,
        )
        return landed

    def knockback(self, source_x: float, source_z: float,
                  strength: float = 1.0) -> None:
        """Apply vanilla's knockback formula to this entity's velocity.

        `source_x/z` is the horizontal position of whatever pushed the
        entity (an attacker or an explosion). The entity moves away from
        that point along the horizontal plane. Resistance is the
        fraction of strength eaten before the velocity is applied.

        The wiring is deliberately light: this round has no physics,
        so the velocity write goes through `EntityPosition.set_velocity`
        and is consumed by the solver when it lands.
        """
        if self.dead:
            return
        effective = strength * (1.0 - self.knockback_resistance)
        if effective <= 0.0:
            return
        pos = self.position
        dx = pos.x - source_x
        dz = pos.z - source_z
        # Vanilla divides by sqrt(dx^2 + dz^2); we early-out if it is zero.
        denom = max((dx * dx + dz * dz) ** 0.5, 1e-6)
        strength_x = 0.4 * effective / denom
        # Vertical boost matches vanilla's upward kick (kept at least at
        # the current velocity so we don't cancel a jump in flight).
        vy = max(pos.velocity_y, 0.4 * effective)
        pos.set_velocity(pos.velocity_x + dx * strength_x,
                         vy,
                         pos.velocity_z + dz * strength_x)

    # --------------------------------------------------------- tick surface

    @property
    def entity_age(self) -> int:
        """Ticks since this entity existed.

        Read-only here; the live counter is `LivingEntity._age` and is
        bumped by `tick`. Used by combat (5-second kill credit window)
        and by anything else that wants a monotonically increasing
        "now" without a world reference.
        """
        return self._age

    def tick(self) -> None:
        """One game tick of living-entity behaviour.

        Order matches Pumpkin's `LivingEntity.baseTick`:

        1. Base entity tick (`Entity.tick`).
        2. Hurt-cooldown countdown.
        3. Death-time animation timer (vanilla counts up to 20 then
           leaves the entity in the "death pose" indefinitely).
        4. Status-effects countdown.
        5. Natural regeneration, if health is below max and not hungry
           (the player variant overrides; base is a no-op for non-hungry
           mobs).
        6. `entity_age++`.

        `Dimension.tick` calls this for every non-player living entity;
        the event loop calls it for players. The order is load-bearing:
        cooldown before effects, so an effect that adds `hurt_cooldown`
        applies AFTER this tick's decrement.
        """
        super().tick()
        if self.hurt_cooldown > 0:
            self.hurt_cooldown -= 1
        if self.dead and self.death_time < 20:
            self.death_time += 1
        self._tick_potion_effects()
        self._tick_regeneration()
        self._age += 1

    def _tick_potion_effects(self) -> None:
        """Decrement every active effect; drop the ones that hit zero.

        The real effect system lands in `entity/effect/`; this loop is the
        integration point and the only place that sees `active_effects`
        mutate during a tick.

        The order - decrement first, then prune - matters: a freshly added
        effect with `remaining=1` should expire this very tick, not linger
        for another one. The dict rebuild below is the only mutation, so
        reading from the original dict while building the new one is
        race-free.
        """
        if not self.active_effects:
            return
        decayed = {
            k: _ActiveEffect(remaining=eff.remaining - 1, amplifier=eff.amplifier)
            for k, eff in self.active_effects.items()
        }
        self.active_effects = {
            k: eff for k, eff in decayed.items() if eff.remaining > 0
        }

    def _tick_regeneration(self) -> None:
        """Base regeneration: nothing.

        Players have their own regen (food-driven). Mobs regen rarely and
        only with a `Regeneration` effect, which lands via
        `add_effect`, not here.
        """

    # ----------------------------------------------------------- misc

    def spawn_blob(self) -> bytes:
        """The metadata blob used at spawn.

        Currently identical to `Mob`'s: just health (index 9, float). The
        default lives here on `LivingEntity` because every living entity,
        not just mobs, sends this on spawn - and keeping the helper on the
        base class lets `Player.spawn_blob` override cleanly.
        """
        from entity.metadata import mob_spawn_blob
        return mob_spawn_blob(self.health)

    def __str__(self) -> str:
        type_name = self.entity_type.name if self.entity_type else "untyped-living"
        if self.dead:
            type_name = f"{type_name}(dead)"
        return f"LivingEntity({type_name}, id={self.entity_id})"


def _is_player(entity: LivingEntity) -> bool:
    """Whether `entity` is a `Player`, not just a player-shaped mob.

    Used by `LivingEntity.attack` to pick the right damage-type string.
    The class check is the only way we have today; duck-typing on
    `entity_type.name == "minecraft:player"` would also match a future
    "player-shaped NPC" (sitting-tagged zombie-horse towers, ...) and we
    do not want their hits to be `player_attack`.
    """
    cls = type(entity).__name__
    return cls == "Player"


__all__ = [
    "DEFAULT_MAX_HEALTH",
    "DEFAULT_HURT_COOLDOWN",
    "DEFAULT_KNOCKBACK_RESISTANCE",
    "LivingEntity",
]
