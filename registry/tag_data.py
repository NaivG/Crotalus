"""
Tag adapter for the login path.

`network/client_handler.py` sends Update Tags from `get_tag_manager().build_update_tags_packet_data()`.
That call now builds the payload from `GameRegistry.tags`, which means tag
entries are resolved against blocks, items, fluids and entity types as well as
the synced registries. The old path resolved only against the synced table and
therefore dropped every block and item tag entry - see `registry/tags.py` for
the full account.

Nothing else in this module is load-bearing: the tag registry itself is
`registry.tags.TagRegistry`, reachable as `get_game_registry().tags`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

from loguru import logger

from registry.game_registry import GameRegistry, get_game_registry


class TagManager:
    """A view of the tag registries, in the shape the login path expects.

    Like `RegistryManager`, it follows the process-wide game registry unless one
    was passed explicitly.
    """

    _instance: Optional["TagManager"] = None

    def __init__(self, game_registry: Optional[GameRegistry] = None) -> None:
        self._explicit = game_registry
        self.last_report = None

    @classmethod
    def get_instance(cls) -> "TagManager":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @property
    def game_registry(self) -> GameRegistry:
        if self._explicit is not None:
            return self._explicit
        return get_game_registry()

    @property
    def tags(self):
        return self.game_registry.tags

    # ----- reading -----

    def get_tags_for_registry(self, registry_id: str) -> Dict[str, List[str]]:
        """`{tag_name: [entry ids]}`, as loaded (nested tags unexpanded)."""
        return {
            tag.name: list(tag.entries)
            for tag in self.tags.tags_of(registry_id)
        }

    def get_all_registry_ids(self) -> List[str]:
        return self.tags.registry_ids()

    def entries(self, registry_id: str, tag_name: str):
        """Resolved members of a tag, nested tags expanded."""
        return self.tags.entries(registry_id, tag_name)

    # ----- wire payload -----

    def build_update_tags_packet_data(self) -> List[Dict]:
        """The Update Tags payload body, as `CUpdateTags` expects it.

        Registries whose entries could not all be resolved are left out; see
        `GameRegistry.build_tag_payload`.
        """
        payload, report = self.game_registry.build_tag_payload()
        self.last_report = report
        logger.debug(
            f"Built Update Tags payload: {len(payload)} registries, "
            f"{report.resolved_entries}/{report.total_entries} entries resolved"
        )
        return payload

    # ----- plugin surface -----

    def add_custom_tag(self, registry_id: str, tag_name: str, entries: List[str]) -> None:
        """Add or extend a tag from plugin code."""
        self.tags.create_tag(registry_id, tag_name, entries)

    def initialize(self, tags_file: Optional[Path] = None) -> None:
        """Accepted for back-compat; the tag file is loaded with the registry."""
        if tags_file is not None:
            logger.debug(
                "TagManager.initialize(tags_file=...) is ignored: tags load with "
                "the game registry from assets/tags.json"
            )


# ==================== Global access ====================

def get_tag_manager() -> TagManager:
    """The tag manager bound to the process-wide game registry."""
    return TagManager.get_instance()


def get_tags_for_registry(registry_id: str) -> Dict[str, List[str]]:
    return get_tag_manager().get_tags_for_registry(registry_id)


def get_all_tag_registry_ids() -> List[str]:
    return get_tag_manager().get_all_registry_ids()


def build_update_tags_data() -> List[Dict]:
    return get_tag_manager().build_update_tags_packet_data()


__all__ = [
    "TagManager",
    "get_tag_manager",
    "get_tags_for_registry",
    "get_all_tag_registry_ids",
    "build_update_tags_data",
]
