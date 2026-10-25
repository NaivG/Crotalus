"""
Item registry - `minecraft:item`, all 1,505 items with their components.

Built from `assets/items.json`, which is a `{name: {id, components}}` map in
vanilla registry order. Numeric ids are the index the client uses, so this is an
`IndexedRegistry` and the recorded `id` is checked against the insertion
position at load time.

Components are kept as raw JSON. There are 58 distinct `minecraft:*` components
in 1.21.11 and the set changes every version; interpreting them into typed
fields would mean a registry that silently drops what it does not know. Instead
the registry exposes `component()` plus named accessors for the handful the
server reasons about (stack size, durability, food, rarity, item model) and a
reverse index by component so "every food item" is one call.

Like `minecraft:block`, `minecraft:item` is *not* a synchronized registry in
1.21.11: the client resolves item ids from its own built-in table through the
`minecraft:core` known pack. A plugin-registered item therefore gets a real
numeric id on the server, but a vanilla client cannot name it - see
`register_item`.
"""

from __future__ import annotations

import threading
from typing import Any, Dict, List, Mapping, Optional

from loguru import logger

from registry.core import (
    IndexedRegistry,
    Registry,
    RegistryError,
    RegistryFrozenError,
    ValidationError,
    canonical_id,
)
from registry.entries import ItemType

# A few components the server actually reads, named so callers do not have to
# spell them out. The full set lives in `assets/data_component.json`.
COMPONENT_MAX_STACK_SIZE = "minecraft:max_stack_size"
COMPONENT_MAX_DAMAGE = "minecraft:max_damage"
COMPONENT_FOOD = "minecraft:food"
COMPONENT_ITEM_NAME = "minecraft:item_name"
COMPONENT_ITEM_MODEL = "minecraft:item_model"
COMPONENT_EQUIPPABLE = "minecraft:equippable"
COMPONENT_CONSUMABLE = "minecraft:consumable"
COMPONENT_RARITY = "minecraft:rarity"
COMPONENT_ENCHANTABLE = "minecraft:enchantable"


class ItemRegistry:
    """The item registry, plus reverse indexes that plugins and recipes need."""

    def __init__(self) -> None:
        self.items: IndexedRegistry[ItemType] = IndexedRegistry(
            "minecraft:item",
            title="Item",
            description="Items and their data components",
        )
        self.component_types: Registry[int] = Registry(
            "minecraft:data_component_type",
            title="Data component type",
            description="Component type ids from assets/data_component.json",
        )
        self._by_component: Dict[str, List[ItemType]] = {}
        self._block_items: Dict[str, str] = {}
        self._lock = threading.RLock()
        self._frozen = False
        self._frozen_reason = ""

    # ----- identity -----

    def __len__(self) -> int:
        return len(self.items)

    def __contains__(self, name: str) -> bool:
        return name in self.items

    def __iter__(self):
        return iter(self.items)

    def __repr__(self) -> str:
        return f"<ItemRegistry items={len(self.items)} frozen={self._frozen}>"

    @property
    def frozen(self) -> bool:
        return self._frozen

    # ----- loading -----

    def load(
        self,
        items_data: Mapping[str, Mapping[str, Any]],
        component_data: Optional[Mapping[str, int]] = None,
    ) -> None:
        """Build the registry from `items.json` (and optionally component ids)."""
        expected = 0
        for name, record in items_data.items():
            numeric_id = int(record["id"])
            if numeric_id != expected:
                raise ValidationError(
                    f"item {name}: numeric id {numeric_id} does not follow the "
                    f"previous item (expected {expected}); item ids are not contiguous"
                )
            item = ItemType(
                name=canonical_id(name),
                numeric_id=numeric_id,
                components=dict(record.get("components", {})),
            )
            self._index_item(item)
            expected = numeric_id + 1

        if component_data is not None:
            for name, numeric_id in component_data.items():
                self.component_types.add(name, int(numeric_id), replace=True)

        logger.debug(
            f"Item registry: {len(self.items)} items, "
            f"{len(self._by_component)} distinct components"
        )

    def _index_item(self, item: ItemType) -> ItemType:
        with self._lock:
            if item.numeric_id != len(self.items):
                raise ValidationError(
                    f"item {item.name}: numeric id {item.numeric_id} does not match "
                    f"its position ({len(self.items)}) in the registry"
                )
            self.items.add(item.name, item, numeric_id=item.numeric_id)
            for component in item.components:
                self._by_component.setdefault(canonical_id(component), []).append(item)
        return item

    def link_blocks(self, blocks) -> None:
        """Record which item places which block.

        `blocks` is a `BlockRegistry` (or anything with `get_block`). Runs after
        `BlockRegistry.link_items`, which has already set `item.block` for the
        pairs the block data declares. The fallback here is the shared-name
        convention, so an item registered by a plugin without a matching block
        declaration still resolves when a block with its name exists.
        """
        lookup = getattr(blocks, "get_block", None) or blocks.get
        self._block_items.clear()
        for item in self.items.values():
            if item.block is None and lookup(item.name) is not None:
                item.block = item.name
            if item.block is not None:
                self._block_items[item.block] = item.name

    # ----- lookup -----

    def item(self, name: str) -> ItemType:
        """Look an item up, raising when it is unknown."""
        return self.items.require(name)

    def get(self, name: str) -> Optional[ItemType]:
        return self.items.get(name)

    def by_numeric_id(self, numeric_id: int) -> Optional[ItemType]:
        return self.items.by_numeric_id(numeric_id)

    def item_for_block(self, block: str) -> Optional[ItemType]:
        """The item a block is placed with, if it has one."""
        name = self._block_items.get(canonical_id(block))
        return self.items.get(name) if name else None

    def with_component(self, component: str) -> List[ItemType]:
        """Every item carrying a component, in registry order."""
        return list(self._by_component.get(canonical_id(component), ()))

    def foods(self) -> List[ItemType]:
        """Every item with a `minecraft:food` component."""
        return self.with_component(COMPONENT_FOOD)

    def damageable(self) -> List[ItemType]:
        """Every item with a `minecraft:max_damage` component."""
        return [item for item in self.items.values() if item.is_damageable]

    def find(self, predicate) -> List[ItemType]:
        return [item for item in self.items.values() if predicate(item)]

    def components_used(self) -> Dict[str, int]:
        """`{component_id: item_count}`, for reporting what was loaded."""
        return {name: len(items) for name, items in sorted(self._by_component.items())}

    # ----- plugin surface -----

    def register_item(
        self,
        name: str,
        components: Optional[Mapping[str, Any]] = None,
        *,
        after: Optional[str] = None,
        numeric_id: Optional[int] = None,
    ) -> ItemType:
        """Append an item to the registry.

        Appending is the only option: item ids index the client's own table, so
        inserting in the middle would renumber every later item. `after` is
        accepted for documentation value only and must be the last item, since
        the registry can only append.

        Note that a vanilla client cannot see a new item - `minecraft:item` is
        not synchronized. Server-side logic (drops, inventories, recipes) can
        use it immediately; showing it to a client needs a resource pack.
        """
        canonical = canonical_id(name)
        with self._lock:
            self._check_mutable(f"register item {canonical}")
            if canonical in self.items:
                raise RegistryError(f"item {canonical} is already registered")
            if after is not None:
                last = self.items.ids()[-1] if len(self.items) else None
                if canonical_id(after) != last:
                    raise RegistryError(
                        f"item {canonical}: `after={after}` is not supported; items "
                        f"are append-only and the last entry is {last}"
                    )

            resolved_id = len(self.items) if numeric_id is None else int(numeric_id)
            if resolved_id != len(self.items):
                raise RegistryError(
                    f"item {canonical}: numeric id {resolved_id} must be the next free "
                    f"id ({len(self.items)}); inserting would renumber later items"
                )
            item = ItemType(
                name=canonical,
                numeric_id=resolved_id,
                components=dict(components or {}),
            )
        return self._index_item(item)

    def modify_item(self, name: str, **components: Any) -> ItemType:
        """Set or replace data components on an existing item.

        Component keys may be written with or without the namespace; a value of
        None removes the component. No ids move, so this is legal after freeze.
        """
        item = self.items.require(name)
        with self._lock:
            for key, value in components.items():
                canonical = canonical_id(key)
                if value is None:
                    item.components.pop(canonical, None)
                    continue
                if canonical not in item.components:
                    self._by_component.setdefault(canonical, []).append(item)
                item.components[canonical] = value
        return item

    def remove_component(self, name: str, component: str) -> bool:
        """Remove a component; returns whether it was there."""
        item = self.items.require(name)
        canonical = canonical_id(component)
        if canonical not in item.components:
            return False
        with self._lock:
            item.components.pop(canonical)
            holders = self._by_component.get(canonical)
            if holders and item in holders:
                holders.remove(item)
        return True

    # ----- freezing -----

    def freeze(self, reason: str = "") -> None:
        with self._lock:
            self._frozen = True
            self._frozen_reason = reason
            self.items.freeze(reason)
            self.component_types.freeze(reason)

    def unfreeze(self) -> None:
        with self._lock:
            self._frozen = False
            self._frozen_reason = ""
            self.items.unfreeze()
            self.component_types.unfreeze()

    def _check_mutable(self, action: str) -> None:
        if self._frozen:
            detail = f" ({self._frozen_reason})" if self._frozen_reason else ""
            raise RegistryFrozenError(
                f"cannot {action}: item registry is frozen{detail}. "
                f"Register items from a plugin before the world loads."
            )

    # ----- integrity -----

    def validate(self) -> List[str]:
        problems: List[str] = []
        expected = 0
        for item in self.items.values():
            if item.numeric_id != expected:
                problems.append(
                    f"{item.name}: numeric id {item.numeric_id} != {expected}"
                )
            expected = item.numeric_id + 1
            if not item.components:
                problems.append(f"{item.name}: no components")
        return problems


__all__ = [
    "ItemRegistry",
    "COMPONENT_MAX_STACK_SIZE",
    "COMPONENT_MAX_DAMAGE",
    "COMPONENT_FOOD",
    "COMPONENT_ITEM_NAME",
    "COMPONENT_ITEM_MODEL",
    "COMPONENT_EQUIPPABLE",
    "COMPONENT_CONSUMABLE",
    "COMPONENT_RARITY",
    "COMPONENT_ENCHANTABLE",
]
