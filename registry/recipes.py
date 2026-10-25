"""
Recipe registry - all 1,470 recipes from `assets/recipes.json`.

Recipe identity
---------------
Pumpkin's recipe data is a bare list: unlike a datapack, its entries carry no
`minecraft:<name>` key. Vanilla recipe ids (`minecraft:oak_planks`,
`minecraft:baked_potato_from_smelting`) are therefore not recoverable, and this
registry synthesises a stable key for each recipe instead - see
`synthesise_key`. The key is what `replace_recipe`/`remove_recipe` take and what
shows up in logs, so it is part of the plugin API; explicit keys are always
preferred when a plugin registers its own.

Everything a crafting engine will need is preserved: ingredients stay typed
(`Ingredient`, so a `#tag` keeps resolving against live tag data), the shaped
`key` map and `pattern` are kept verbatim, and the untouched source record is
available as `Recipe.raw`.

The registry is also indexed three ways - by result, by type and by ingredient -
because those are the three questions the game asks (what does this craft into,
what are the smelting recipes, what can this item be used for).
"""

from __future__ import annotations

import threading
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from loguru import logger

from registry.core import (
    Registry,
    RegistryError,
    RegistryFrozenError,
    canonical_id,
    path_of,
)
from registry.entries import (
    COOKING_TYPES,
    Ingredient,
    ItemStack,
    Recipe,
    SMITHING_TYPES,
    SPECIAL_TYPES,
)

# Cooking type -> the suffix vanilla-ish names use. Kept in one place so a
# synthesised key is readable and cannot drift between builds.
_COOKING_SUFFIX = {
    "minecraft:smelting": "smelting",
    "minecraft:blasting": "blasting",
    "minecraft:smoking": "smoking",
    "minecraft:campfire_cooking": "campfire_cooking",
}


def _ingredient_path(ingredient: Optional[Ingredient]) -> Optional[str]:
    """A short, stable name for an ingredient, used in synthesised keys."""
    if ingredient is None:
        return None
    if ingredient.tag is not None and not ingredient.entries:
        return path_of(ingredient.tag)
    if ingredient.entries:
        return path_of(ingredient.entries[0])
    return None


def synthesise_key(recipe: Recipe, used: Iterable[str] = ()) -> str:
    """Build a stable recipe key from the recipe's own contents.

    The scheme, in order of preference:

      * cooking -> `<result>_from_<ingredient>_<smelting|blasting|smoking|campfire_cooking>`
      * stonecutting -> `<result>_from_<ingredient>_stonecutting`
      * shaped/shapeless/transmute -> `<result>`, or `<result>_from_<ingredient>`
        when another recipe already claimed the plain result name
      * smithing -> `<result>_smithing` (or the trim pattern for `smithing_trim`)
      * special -> the recipe type path, e.g. `crafting_special_armordye`

    Collisions get a `_2`, `_3`, ... suffix, so the result is deterministic for
    a given `recipes.json` and stable across restarts.
    """
    taken = used if isinstance(used, (set, frozenset)) else set(used)
    result_path = path_of(recipe.result.item) if recipe.result else None
    primary = _ingredient_path(recipe.ingredients[0]) if recipe.ingredients else None

    if recipe.type in COOKING_TYPES and result_path:
        base = f"{result_path}_from_{primary or 'unknown'}_{_COOKING_SUFFIX[recipe.type]}"
    elif recipe.type == "minecraft:stonecutting" and result_path:
        base = f"{result_path}_from_{primary or 'unknown'}_stonecutting"
    elif recipe.type == "minecraft:smithing_trim":
        pattern = recipe.raw.get("pattern")
        base = f"{path_of(pattern) if pattern else 'trim'}_smithing_trim"
    elif recipe.type in SMITHING_TYPES and result_path:
        base = f"{result_path}_smithing"
    elif result_path:
        base = result_path
        # `taken` holds canonical ids while `base` is a bare path, so the
        # comparison has to go through canonical_id too.
        if primary and canonical_id(base) in taken:
            base = f"{base}_from_{primary}"
    elif recipe.type in SPECIAL_TYPES:
        base = recipe.type_path
    else:
        base = f"recipe_{recipe.index}"

    key = base
    counter = 2
    while canonical_id(key) in taken:
        key = f"{base}_{counter}"
        counter += 1
    return canonical_id(key)


def parse_recipe(record: Mapping[str, Any], index: int) -> Recipe:
    """Build a `Recipe` from one `recipes.json` entry.

    Every shape Pumpkin's data uses is handled: `ingredient`/`ingredients`,
    a shaped `key` map, the smithing `template`/`base`/`addition` triple, the
    transmute `input`/`material` pair and the result-less special recipes.
    """
    recipe_type = canonical_id(record["type"])
    result_data = record.get("result")
    result = ItemStack.from_json(result_data) if result_data else None

    ingredients: List[Ingredient] = []
    key_map: Optional[Dict[str, Ingredient]] = None
    pattern: Optional[Tuple[str, ...]] = None

    raw_key = record.get("key")
    if raw_key:
        key_map = {key: Ingredient.from_json(value) for key, value in raw_key.items()}
    if "pattern" in record and recipe_type == "minecraft:crafting_shaped":
        pattern = tuple(record["pattern"])

    if "ingredients" in record:
        ingredients.extend(Ingredient.from_json(entry) for entry in record["ingredients"])
    if "ingredient" in record:
        ingredients.append(Ingredient.from_json(record["ingredient"]))
    for field in ("input", "material", "template", "base", "addition"):
        if field in record:
            ingredients.append(Ingredient.from_json(record[field]))

    return Recipe(
        key="",  # assigned by the registry once collisions are known
        type=recipe_type,
        index=index,
        result=result,
        ingredients=tuple(ingredients),
        category=record.get("category"),
        group=record.get("group"),
        pattern=pattern,
        key_map=key_map,
        cooking_time=record.get("cookingtime"),
        experience=record.get("experience"),
        show_notification=record.get("show_notification"),
        raw=dict(record),
    )


class RecipeRegistry:
    """The recipe registry with by-result, by-type and by-ingredient indexes."""

    def __init__(self) -> None:
        self.recipes: Registry[Recipe] = Registry(
            "minecraft:recipe",
            title="Recipe",
            description="Crafting, cooking, stonecutting and smithing recipes",
        )
        self._by_result: Dict[str, List[Recipe]] = {}
        self._by_type: Dict[str, List[Recipe]] = {}
        self._by_ingredient: Dict[str, List[Recipe]] = {}
        self._lock = threading.RLock()
        self._frozen = False
        self._frozen_reason = ""

    # ----- identity -----

    def __len__(self) -> int:
        return len(self.recipes)

    def __contains__(self, key: str) -> bool:
        return key in self.recipes

    def __iter__(self):
        return iter(self.recipes.values())

    def __repr__(self) -> str:
        return f"<RecipeRegistry recipes={len(self.recipes)} frozen={self._frozen}>"

    @property
    def frozen(self) -> bool:
        return self._frozen

    # ----- loading -----

    def load(self, records: Sequence[Mapping[str, Any]]) -> None:
        """Build the registry from `recipes.json`."""
        used: set = set()
        for index, record in enumerate(records):
            recipe = parse_recipe(record, index)
            recipe.key = synthesise_key(recipe, used)
            used.add(recipe.key)
            self._add_indexed(recipe)

        logger.debug(
            f"Recipe registry: {len(self.recipes)} recipes, "
            f"{len(self._by_result)} distinct results, "
            f"{len(self._by_type)} types"
        )

    def _add_indexed(self, recipe: Recipe) -> Recipe:
        with self._lock:
            self.recipes.add(recipe.key, recipe)
            if recipe.result is not None:
                self._by_result.setdefault(recipe.result.item, []).append(recipe)
            self._by_type.setdefault(recipe.type, []).append(recipe)
            for ingredient in recipe.ingredients:
                for key in self._ingredient_keys(ingredient):
                    self._by_ingredient.setdefault(key, []).append(recipe)
        return recipe

    @staticmethod
    def _ingredient_keys(ingredient: Ingredient) -> Tuple[str, ...]:
        if ingredient.tag is not None:
            return (f"#{ingredient.tag}",)
        return ingredient.entries

    # ----- lookup -----

    def recipe(self, key: str) -> Recipe:
        """Look a recipe up by key, raising when it is unknown."""
        return self.recipes.require(key)

    def get(self, key: str) -> Optional[Recipe]:
        return self.recipes.get(key)

    def by_result(self, item: str) -> List[Recipe]:
        """Every recipe producing an item, in registry order."""
        return list(self._by_result.get(canonical_id(item), ()))

    def by_type(self, recipe_type: str) -> List[Recipe]:
        """Every recipe of a type, e.g. `minecraft:smelting`."""
        return list(self._by_type.get(canonical_id(recipe_type), ()))

    def by_ingredient(self, item_or_tag: str) -> List[Recipe]:
        """Every recipe using an item, or a tag when given `#minecraft:planks`."""
        if item_or_tag.startswith("#"):
            key = f"#{canonical_id(item_or_tag[1:])}"
        else:
            key = canonical_id(item_or_tag)
        return list(self._by_ingredient.get(key, ()))

    def types(self) -> Dict[str, int]:
        """`{recipe_type: count}`, for reporting what was loaded."""
        return {name: len(items) for name, items in sorted(self._by_type.items())}

    def crafting_recipes(self) -> List[Recipe]:
        """Every shaped/shapeless/transmute recipe, in registry order."""
        return [recipe for recipe in self.recipes.values() if recipe.is_crafting]

    def cooking_recipes(self) -> List[Recipe]:
        """Every smelting/blasting/smoking/campfire recipe."""
        return [recipe for recipe in self.recipes.values() if recipe.is_cooking]

    def cooking_result(self, item: str, recipe_type: str) -> Optional[Recipe]:
        """The first recipe cooking an item in a given way."""
        wanted = canonical_id(item)
        for recipe in self._by_type.get(canonical_id(recipe_type), ()):
            if any(ingredient.matches(wanted) for ingredient in recipe.ingredients):
                return recipe
        return None

    def smelting_result(self, item: str) -> Optional[Recipe]:
        """What a furnace smelts an item into."""
        return self.cooking_result(item, "minecraft:smelting")

    def blasting_result(self, item: str) -> Optional[Recipe]:
        """What a blast furnace melts an item into."""
        return self.cooking_result(item, "minecraft:blasting")

    def smoking_result(self, item: str) -> Optional[Recipe]:
        """What a smoker cooks an item into."""
        return self.cooking_result(item, "minecraft:smoking")

    def stonecutting_results(self, item: str) -> List[Recipe]:
        """Every stonecutter recipe for an item."""
        wanted = canonical_id(item)
        return [
            recipe
            for recipe in self._by_type.get("minecraft:stonecutting", ())
            if any(ingredient.matches(wanted) for ingredient in recipe.ingredients)
        ]

    def replace_recipe(self, key: str, **fields: Any) -> Recipe:
        """Replace a recipe's fields in place, keeping its key and index.

        Only the fields passed are changed; everything else, including `raw`, is
        carried over. Recipe keys are not sent to the client today, but the order
        of the registry is what a future Update Recipes packet would number, so
        replacement never moves an entry.
        """
        existing = self.recipes.require(key)
        with self._lock:
            for field_name, value in fields.items():
                if field_name in ("key", "index"):
                    raise RegistryError(f"{field_name} cannot be changed")
                if not hasattr(existing, field_name):
                    raise RegistryError(f"recipe {existing.key} has no field {field_name!r}")
                setattr(existing, field_name, value)
        return existing

    def add_recipe(
        self,
        record: Mapping[str, Any],
        key: Optional[str] = None,
        *,
        index: Optional[int] = None,
    ) -> Recipe:
        """Append a recipe from a `recipes.json`-shaped mapping.

        Appending is the only option - an inserted recipe would shift the index
        of every later recipe, which is the order the client's recipe book is
        numbered by. `key` is optional; without one a stable key is synthesised.
        """
        with self._lock:
            self._check_mutable("add recipe")
            resolved_index = len(self.recipes) if index is None else int(index)
            if resolved_index != len(self.recipes):
                raise RegistryError(
                    f"recipe index {resolved_index} must be the next free index "
                    f"({len(self.recipes)}); recipes are append-only"
                )
            recipe = parse_recipe(record, resolved_index)
            recipe.key = (
                canonical_id(key)
                if key
                else synthesise_key(recipe, set(self.recipes.keys()))
            )
            if recipe.key in self.recipes:
                raise RegistryError(f"recipe {recipe.key} is already registered")
        return self._add_indexed(recipe)

    def remove_recipe(self, key: str) -> Recipe:
        """Unregister a recipe.

        Refused once frozen, because the recipe's index is what a future Update
        Recipes packet and any saved recipe-book state refer to. Prefer
        `replace_recipe` with a disabled recipe.
        """
        canonical = canonical_id(key)
        with self._lock:
            self._check_mutable(f"remove recipe {canonical}")
            recipe = self.recipes.require(canonical)
            self.recipes.remove(canonical)
            self._unindex(recipe)
        return recipe

    def _unindex(self, recipe: Recipe) -> None:
        if recipe.result is not None:
            self._drop(self._by_result.get(recipe.result.item), recipe)
        self._drop(self._by_type.get(recipe.type), recipe)
        for ingredient in recipe.ingredients:
            for key in self._ingredient_keys(ingredient):
                self._drop(self._by_ingredient.get(key), recipe)

    @staticmethod
    def _drop(bucket: Optional[List[Recipe]], recipe: Recipe) -> None:
        if bucket and recipe in bucket:
            bucket.remove(recipe)

    # ----- freezing -----

    def freeze(self, reason: str = "") -> None:
        with self._lock:
            self._frozen = True
            self._frozen_reason = reason
            self.recipes.freeze(reason)

    def unfreeze(self) -> None:
        with self._lock:
            self._frozen = False
            self._frozen_reason = ""
            self.recipes.unfreeze()

    def _check_mutable(self, action: str) -> None:
        if self._frozen:
            detail = f" ({self._frozen_reason})" if self._frozen_reason else ""
            raise RegistryFrozenError(
                f"cannot {action}: recipe registry is frozen{detail}. "
                f"Register recipes from a plugin before the world loads."
            )

    # ----- integrity -----

    def validate(self, items=None, tag_lookup=None) -> List[str]:
        """Check recipes against the item registry and tag registry.

        `items` is an `ItemRegistry` (or anything with `get`); `tag_lookup(tag)
        -> set[str] | None` resolves a tag's members. Both are optional, and a
        recipe referencing something unknown is reported rather than raising.
        """
        problems: List[str] = []
        seen: set = set()

        for recipe in self.recipes.values():
            if recipe.key in seen:
                problems.append(f"duplicate recipe key {recipe.key}")
            seen.add(recipe.key)

            if recipe.index != self.recipes.numeric_id(recipe.key):
                problems.append(
                    f"{recipe.key}: index {recipe.index} != registry position "
                    f"{self.recipes.numeric_id(recipe.key)}"
                )

            if recipe.result is not None and items is not None:
                if items.get(recipe.result.item) is None:
                    problems.append(
                        f"{recipe.key}: result {recipe.result.item} is not an item"
                    )

            for ingredient in recipe.ingredients:
                if ingredient.tag is not None:
                    if tag_lookup is not None and tag_lookup(ingredient.tag) is None:
                        problems.append(
                            f"{recipe.key}: ingredient tag #{ingredient.tag} is unknown"
                        )
                    continue
                if not ingredient.entries:
                    problems.append(f"{recipe.key}: an ingredient accepts nothing")
                    continue
                if items is not None:
                    for entry in ingredient.entries:
                        if items.get(entry) is None:
                            problems.append(
                                f"{recipe.key}: ingredient {entry} is not an item"
                            )

            if recipe.pattern is not None:
                # A space is vanilla's "empty slot" and is intentionally absent
                # from the key map, so it is not a missing symbol.
                symbols = set("".join(recipe.pattern)) - {" "}
                if recipe.key_map is None:
                    problems.append(f"{recipe.key}: shaped recipe has no key map")
                else:
                    missing = symbols - set(recipe.key_map)
                    if missing:
                        problems.append(
                            f"{recipe.key}: pattern uses {sorted(missing)} but the key "
                            f"map defines {sorted(recipe.key_map)}"
                        )
                if len(recipe.pattern) > 3 or any(len(row) > 3 for row in recipe.pattern):
                    problems.append(f"{recipe.key}: pattern is larger than 3x3")

        return problems


__all__ = ["RecipeRegistry", "parse_recipe", "synthesise_key"]
