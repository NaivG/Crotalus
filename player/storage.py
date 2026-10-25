"""
Player Storage

Handles loading and saving player data to disk.
"""

import json
import os
from pathlib import Path
from typing import Optional, Iterator
from loguru import logger

from player.player import Player
from player.cookie import CookieManager


class PlayerStorage:
    """Manages player data persistence.

    Stores player data in JSON format under a configured directory.
    Also manages player cookies with automatic loading/saving.
    """

    def __init__(self, storage_dir: str = "players"):
        """Initialize player storage.

        Args:
            storage_dir: Directory to store player data files
        """
        self.storage_dir = Path(storage_dir)
        self._players_cache: dict[str, Player] = {}
        self._cookies_cache: dict[str, CookieManager] = {}

    def _get_player_path(self, uuid: str) -> Path:
        """Get file path for player data."""
        # Use first 2 chars for subdirectory (like Minecraft)
        subdir = uuid[:2]
        return self.storage_dir / subdir / f"{uuid}.json"

    def _get_cookie_path(self, uuid: str) -> Path:
        """Get file path for player cookies."""
        subdir = uuid[:2]
        return self.storage_dir / subdir / f"{uuid}_cookies.json"

    def load_player(self, uuid: str) -> Optional[Player]:
        """Load player data from disk.

        Args:
            uuid: Player UUID

        Returns:
            Player instance or None if not found
        """
        # Check cache first
        if uuid in self._players_cache:
            return self._players_cache[uuid]

        file_path = self._get_player_path(uuid)

        if not file_path.exists():
            logger.debug(f"Player file not found: {file_path}")
            return None

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            player = Player.from_dict(data)
            self._players_cache[uuid] = player
            logger.info(f"Loaded player: {player.name} ({uuid[:8]}...)")
            return player

        except (json.JSONDecodeError, IOError) as e:
            logger.error(f"Failed to load player {uuid}: {e}")
            return None

    def save_player(self, player: Player) -> bool:
        """Save player data to disk.

        Args:
            player: Player instance to save

        Returns:
            True if saved successfully
        """
        file_path = self._get_player_path(player.uuid)

        try:
            # Ensure directory exists
            file_path.parent.mkdir(parents=True, exist_ok=True)

            # Write to temp file first, then rename (atomic)
            temp_path = file_path.with_suffix(".tmp")
            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump(player.to_dict(), f, indent=2)

            temp_path.replace(file_path)

            # Update cache
            self._players_cache[player.uuid] = player

            logger.debug(f"Saved player: {player.name}")
            return True

        except IOError as e:
            logger.error(f"Failed to save player {player.uuid}: {e}")
            return False

    def delete_player(self, uuid: str) -> bool:
        """Delete player data from disk.

        Args:
            uuid: Player UUID

        Returns:
            True if deleted successfully
        """
        file_path = self._get_player_path(uuid)

        try:
            if file_path.exists():
                file_path.unlink()

            # Also delete cookies
            cookie_path = self._get_cookie_path(uuid)
            if cookie_path.exists():
                cookie_path.unlink()

            # Clear cache
            self._players_cache.pop(uuid, None)
            self._cookies_cache.pop(uuid, None)

            logger.info(f"Deleted player: {uuid[:8]}...")
            return True

        except IOError as e:
            logger.error(f"Failed to delete player {uuid}: {e}")
            return False

    def player_exists(self, uuid: str) -> bool:
        """Check if player data exists on disk."""
        return self._get_player_path(uuid).exists()

    def load_cookies(self, uuid: str) -> CookieManager:
        """Load or create cookies for a player.

        Args:
            uuid: Player UUID

        Returns:
            CookieManager instance (creates new if not found)
        """
        # Check cache first
        if uuid in self._cookies_cache:
            return self._cookies_cache[uuid]

        file_path = self._get_cookie_path(uuid)

        if file_path.exists():
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                cookies = CookieManager.from_dict(uuid, data)
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"Failed to load cookies for {uuid[:8]}: {e}")
                cookies = CookieManager(uuid)
        else:
            cookies = CookieManager(uuid)

        self._cookies_cache[uuid] = cookies
        return cookies

    def save_cookies(self, cookies: CookieManager) -> bool:
        """Save cookies to disk.

        Args:
            cookies: CookieManager instance

        Returns:
            True if saved successfully
        """
        file_path = self._get_cookie_path(cookies.player_uuid)

        try:
            file_path.parent.mkdir(parents=True, exist_ok=True)

            temp_path = file_path.with_suffix(".tmp")
            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump(cookies.to_dict(), f, indent=2)

            temp_path.replace(file_path)

            logger.debug(f"Saved cookies for player: {cookies.player_uuid[:8]}...")
            return True

        except IOError as e:
            logger.error(f"Failed to save cookies: {e}")
            return False

    def list_players(self) -> Iterator[Player]:
        """Iterate over all saved players.

        Yields:
            Player instances
        """
        if not self.storage_dir.exists():
            return

        for file_path in self.storage_dir.rglob("*.json"):
            # Skip cookie files
            if "_cookies" in file_path.stem:
                continue

            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                yield Player.from_dict(data)
            except (json.JSONDecodeError, IOError):
                continue

    def get_player_count(self) -> int:
        """Get total number of saved players."""
        count = 0
        if self.storage_dir.exists():
            for file_path in self.storage_dir.rglob("*.json"):
                if "_cookies" not in file_path.stem:
                    count += 1
        return count

    def clear_cache(self) -> None:
        """Clear in-memory cache."""
        self._players_cache.clear()
        self._cookies_cache.clear()

    def get_cached_player(self, uuid: str) -> Optional[Player]:
        """Get player from cache without loading from disk."""
        return self._players_cache.get(uuid)

    def cache_player(self, player: Player) -> None:
        """Add player to cache without saving to disk."""
        self._players_cache[player.uuid] = player

    def evict_player(self, uuid: str, save: bool = True) -> bool:
        """Remove player from cache and optionally save first.

        Args:
            uuid: Player UUID
            save: If True, save before removing from cache

        Returns:
            True if successful
        """
        player = self._players_cache.get(uuid)
        if player is None:
            return False

        if save:
            self.save_player(player)

        del self._players_cache[uuid]
        return True


# Global storage instance
_player_storage: Optional[PlayerStorage] = None


def get_player_storage() -> PlayerStorage:
    """Get global player storage instance."""
    global _player_storage
    if _player_storage is None:
        _player_storage = PlayerStorage()
    return _player_storage


def set_player_storage(storage: PlayerStorage) -> None:
    """Set global player storage instance."""
    global _player_storage
    _player_storage = storage
