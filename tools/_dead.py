"""Remove the ported-but-never-called helpers, and the unreachable branch."""
import pathlib
import sys


def sub(path, old, new, count=1):
    p = pathlib.Path(path)
    s = p.read_text(encoding="utf-8")
    assert old in s, f"{path}: {old[:70]!r}"
    p.write_text(s.replace(old, new, count), encoding="utf-8")


# ---- node.py: five path-type predicates with no consumer ----
sub("entity/ai/pathfinding/node.py",
    '''_WATER_TYPES = frozenset((PathType.WATER, PathType.WATER_BORDER))
_DOOR_TYPES = frozenset((PathType.WALKABLE_DOOR, PathType.DOOR_OPEN,
                         PathType.DOOR_WOOD_CLOSED, PathType.DOOR_IRON_CLOSED))
_DANGEROUS_TYPES = frozenset((PathType.LAVA, PathType.DANGER_FIRE,
                              PathType.DAMAGE_FIRE, PathType.DANGER_OTHER,
                              PathType.DAMAGE_OTHER, PathType.DANGER_POWDER_SNOW,
                              PathType.DANGER_TRAPDOOR))
_PARTIAL_COLLISION_TYPES = frozenset((PathType.FENCE, PathType.WALKABLE_DOOR,
                                      PathType.DOOR_OPEN,
                                      PathType.DOOR_WOOD_CLOSED,
                                      PathType.DOOR_IRON_CLOSED))


def get_malus(path_type: PathType) -> float:
    """The class-default price of a path type."""
    return MALUS[int(path_type)]


def is_blocked(path_type: PathType) -> bool:
    """Whether no mob may stand here at any price."""
    return MALUS[int(path_type)] < 0.0


def is_water(path_type: PathType) -> bool:
    return path_type in _WATER_TYPES


def is_door(path_type: PathType) -> bool:
    return path_type in _DOOR_TYPES


def is_dangerous(path_type: PathType) -> bool:
    return path_type in _DANGEROUS_TYPES


def has_partial_collision(path_type: PathType) -> bool:
    """Whether the block is thinner than a cell - a fence, or any door.

    Partial collision is what makes the "can a 0.6-wide cow squeeze through
    this" question exist at all, and it is asked twice: once in the evaluator
    (`can_reach_without_collision`) and once in the diagonal rule
    (`is_diagonal_valid`, whose `can_pass_between_posts` exception exists only
    for two fences with a mob narrower than half a block).
    """
    return path_type in _PARTIAL_COLLISION_TYPES
''',
    '''def get_malus(path_type: PathType) -> float:
    """The class-default price of a path type.

    The only classifier this module exports. Vanilla's `PathType` has six
    predicates on top of it (`isBlocked`, `isWater`, `isDoor`, `isDangerous`,
    `hasPartialCollision`, `isPassable`) and none of them are carried here,
    because every question they answer is answered by the evaluator on the
    *mob's* reading of a cell rather than on the block's - see
    `WalkNodeEvaluator.does_block_have_partial_collision` for the one that
    looks most like a direct translation and is deliberately not one.
    """
    return MALUS[int(path_type)]
''')

sub("entity/ai/pathfinding/node.py",
    '''    def distance_xz_to(self, other: "Node") -> float:
        dx = other.pos[0] - self.pos[0]
        dz = other.pos[2] - self.pos[2]
        return sqrt(dx * dx + dz * dz)

''', "")

# ---- path.py: two accessors nothing reads ----
sub("entity/ai/pathfinding/path.py",
    '''    def get_previous_node(self) -> Optional[Node]:
        if self.next_node_index > 0:
            return self.get_node(self.next_node_index - 1)
        return None

''', "")
sub("entity/ai/pathfinding/path.py",
    '''    def get_remaining_nodes(self) -> List[Node]:
        if self.next_node_index >= len(self.nodes):
            return []
        return self.nodes[self.next_node_index:]

''', "")

# ---- navigation.py: an unreachable branch, and a setter nobody calls ----
sub("entity/ai/pathfinding/navigation.py",
    '''#: Ticks before a delayed recomputation actually happens (`mod.rs:981`).
#: Vanilla spreads recomputes out so a herd of mobs taking the same goal does
#: not all search on the same tick.
DELAYED_RECOMPUTE_TICKS = 20

''', "")
sub("entity/ai/pathfinding/navigation.py",
    '''                 "_repath_cooldown", "_has_delayed_recompute",
                 "_last_recompute_tick", "_ticks_on_current_node",''',
    '''                 "_repath_cooldown",
                 "_ticks_on_current_node",''')
sub("entity/ai/pathfinding/navigation.py",
    '''        self._repath_cooldown = 0
        self._has_delayed_recompute = False
        self._last_recompute_tick = 0
        self._ticks_on_current_node = 0''',
    '''        self._repath_cooldown = 0
        self._ticks_on_current_node = 0''')
sub("entity/ai/pathfinding/navigation.py",
    '''    def set_speed(self, speed: float) -> None:
        self.speed_modifier = speed

''', "")
sub("entity/ai/pathfinding/navigation.py",
    '''        self._repath_cooldown = REPATH_COOLDOWN
        self._has_delayed_recompute = False
        self._ticks_on_current_node = 0''',
    '''        self._repath_cooldown = REPATH_COOLDOWN
        self._ticks_on_current_node = 0''')
sub("entity/ai/pathfinding/navigation.py",
    '''        self._last_stuck_pos = None
        self._stuck = False
        self._has_delayed_recompute = False
        self._reset_stuck_timeout()''',
    '''        self._last_stuck_pos = None
        self._stuck = False
        self._reset_stuck_timeout()''')
sub("entity/ai/pathfinding/navigation.py",
    '''        self._last_recompute_tick = self.ticks
        self._has_delayed_recompute = False
        self._reset_stuck_timeout()''',
    '''        self._reset_stuck_timeout()''')
sub("entity/ai/pathfinding/navigation.py",
    '''    def _do_delayed_recompute(self, mob) -> None:
        """Recompute a path a few ticks *after* asking, not on the ask.

        Vanilla (`mod.rs:980`): a mob handed a new goal searches a fixed 20
        ticks later rather than immediately. The point is herd behaviour -
        twenty cows all released into the same pen on the same tick would
        otherwise run twenty searches on that tick, and the search is the most
        expensive thing a mob does. Spreading them costs the mob a fifth of a
        second of delay and buys a tick budget that scales with mob count.

        Off this round's path: Crotalus mobs are spread across chunks and the
        recompute path is not yet a measured cost. The branch is written and
        wired so that turning it on is a flag rather than a change.
        """
        if not self._has_delayed_recompute:
            return
        if self.ticks - self._last_recompute_tick < DELAYED_RECOMPUTE_TICKS:
            return
        self._recompute(mob)

    def request_delayed_recompute(self) -> None:
        """Ask for a path a little later rather than now (`setDelayedRecomputation`)."""
        self._has_delayed_recompute = True
        self._last_recompute_tick = self.ticks

''', "")
sub("entity/ai/pathfinding/navigation.py",
    '''        self._do_delayed_recompute(mob)

        if self.needs_new_path():''', '''        if self.needs_new_path():''')
sub("entity/ai/pathfinding/navigation.py",
    '''        self._repath_cooldown = REPATH_COOLDOWN
        self._last_recompute_tick = self.ticks
        return True''',
    '''        self._repath_cooldown = REPATH_COOLDOWN
        return True''')
sub("entity/ai/pathfinding/navigation.py",
    '''        self._repath_cooldown = REPATH_COOLDOWN
        self._last_recompute_tick = self.ticks
        self._has_delayed_recompute = False
    ''', '''        self._repath_cooldown = REPATH_COOLDOWN
    ''')

# ---- finder.py: a knob nothing turns ----
sub("entity/ai/pathfinding/finder.py",
    '''    def set_required_path_length(self, length: float) -> None:
        self._required_path_length = length

''', "")

sys.stdout.reconfigure(encoding="utf-8")
print("dead code removed")