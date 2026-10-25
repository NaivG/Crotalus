"""
Cookie Manager

Handles Minecraft Cookie Request/Response packets.
Cookies can contain up to 5kiB (5120 bytes) of data per key.

Reference: https://minecraft.wiki/w/Minecraft_Wiki:Projects/cookie_specification
"""

import json
import base64
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path


# Maximum cookie payload size: 5kiB (5120 bytes)
MAX_COOKIE_SIZE = 5120


@dataclass
class CookieData:
    """Represents a single cookie entry."""

    key: str
    payload: bytes = field(default_factory=bytes)

    @property
    def size(self) -> int:
        """Get payload size in bytes."""
        return len(self.payload)

    @property
    def is_valid(self) -> bool:
        """Check if cookie payload is within size limit."""
        return self.size <= MAX_COOKIE_SIZE

    def to_base64(self) -> str:
        """Encode payload as base64 string."""
        return base64.b64encode(self.payload).decode("ascii")

    @classmethod
    def from_base64(cls, key: str, encoded: str) -> "CookieData":
        """Create CookieData from base64 encoded string."""
        payload = base64.b64decode(encoded.encode("ascii"))
        return cls(key=key, payload=payload)

    def to_dict(self) -> dict:
        """Serialize cookie to dict for storage."""
        return {
            "key": self.key,
            "payload": self.to_base64(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "CookieData":
        """Deserialize cookie from dict storage."""
        return cls.from_base64(
            key=data.get("key", ""),
            encoded=data.get("payload", "")
        )


class CookieManager:
    """Manages cookies for a player session.

    Handles the Cookie Request/Response flow:
    1. Server sends Cookie Request (CONFIG_STORE_COOKIE or PLAY_STORE_COOKIE)
    2. Client responds with Cookie Response containing the cookie data
    3. Server stores cookie data for future sessions
    """

    def __init__(self, player_uuid: str):
        self.player_uuid = player_uuid
        # cookies: key -> CookieData
        self._cookies: dict[str, CookieData] = {}
        # pending requests: key the server is waiting for
        self._pending_requests: set[str] = set()

    def get_cookie(self, key: str) -> Optional[bytes]:
        """Get cookie payload by key.

        Args:
            key: Cookie identifier

        Returns:
            Cookie payload bytes or None if not found
        """
        cookie = self._cookies.get(key)
        return cookie.payload if cookie else None

    def set_cookie(self, key: str, payload: bytes) -> bool:
        """Set a cookie for this player.

        Args:
            key: Cookie identifier
            payload: Cookie data (max 5kiB)

        Returns:
            True if cookie was set, False if payload too large
        """
        if len(payload) > MAX_COOKIE_SIZE:
            return False

        self._cookies[key] = CookieData(key=key, payload=payload)
        return True

    def set_cookie_from_dict(self, key: str, data: dict) -> bool:
        """Set a cookie from a dict (e.g., JSON).

        Args:
            key: Cookie identifier
            data: Dict to serialize as JSON

        Returns:
            True if cookie was set, False if data too large
        """
        json_bytes = json.dumps(data).encode("utf-8")
        return self.set_cookie(key, json_bytes)

    def get_cookie_as_json(self, key: str) -> Optional[dict]:
        """Get cookie payload and parse as JSON.

        Args:
            key: Cookie identifier

        Returns:
            Parsed JSON dict or None
        """
        payload = self.get_cookie(key)
        if payload is None:
            return None
        try:
            return json.loads(payload.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            return None

    def delete_cookie(self, key: str) -> bool:
        """Delete a cookie.

        Args:
            key: Cookie identifier

        Returns:
            True if cookie was deleted, False if not found
        """
        if key in self._cookies:
            del self._cookies[key]
            return True
        return False

    def has_cookie(self, key: str) -> bool:
        """Check if a cookie exists."""
        return key in self._cookies

    def list_cookies(self) -> list[str]:
        """List all cookie keys."""
        return list(self._cookies.keys())

    def request_cookie(self, key: str) -> None:
        """Mark a cookie as requested from server.

        The server will expect a response with this key.
        """
        self._pending_requests.add(key)

    def receive_cookie(self, key: str, payload: bytes) -> bool:
        """Handle received cookie data from client response.

        Args:
            key: Cookie identifier
            payload: Cookie data

        Returns:
            True if cookie was received successfully
        """
        # Remove from pending requests
        self._pending_requests.discard(key)

        # Validate and store
        if len(payload) > MAX_COOKIE_SIZE:
            return False

        self._cookies[key] = CookieData(key=key, payload=payload)
        return True

    def is_pending(self, key: str) -> bool:
        """Check if a cookie request is pending."""
        return key in self._pending_requests

    @property
    def pending_count(self) -> int:
        """Get number of pending cookie requests."""
        return len(self._pending_requests)

    def to_dict(self) -> dict:
        """Serialize all cookies for storage."""
        return {
            "player_uuid": self.player_uuid,
            "cookies": {
                key: cookie.to_dict()
                for key, cookie in self._cookies.items()
            }
        }

    def to_json(self) -> str:
        """Serialize cookies as JSON string."""
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, player_uuid: str, data: dict) -> "CookieManager":
        """Deserialize cookies from storage.

        Args:
            player_uuid: Player UUID
            data: Stored cookie data

        Returns:
            CookieManager instance with loaded cookies
        """
        manager = cls(player_uuid)

        cookies_data = data.get("cookies", {})
        for key, cookie_data in cookies_data.items():
            try:
                cookie = CookieData.from_dict(cookie_data)
                manager._cookies[key] = cookie
            except (KeyError, ValueError):
                # Skip invalid cookies
                continue

        return manager

    @classmethod
    def from_json(cls, player_uuid: str, json_str: str) -> "CookieManager":
        """Deserialize cookies from JSON string.

        Args:
            player_uuid: Player UUID
            json_str: JSON string

        Returns:
            CookieManager instance with loaded cookies
        """
        data = json.loads(json_str)
        return cls.from_dict(player_uuid, data)

    def load_from_file(self, file_path: Path) -> bool:
        """Load cookies from a file.

        Args:
            file_path: Path to cookie file

        Returns:
            True if loaded successfully
        """
        try:
            if not file_path.exists():
                return False

            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            cookies_data = data.get("cookies", {})
            for key, cookie_data in cookies_data.items():
                try:
                    cookie = CookieData.from_dict(cookie_data)
                    self._cookies[key] = cookie
                except (KeyError, ValueError):
                    continue

            return True
        except (json.JSONDecodeError, IOError):
            return False

    def save_to_file(self, file_path: Path) -> bool:
        """Save cookies to a file.

        Args:
            file_path: Path to save cookie file

        Returns:
            True if saved successfully
        """
        try:
            file_path.parent.mkdir(parents=True, exist_ok=True)

            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(self.to_dict(), f, indent=2)

            return True
        except IOError:
            return False

    def clear(self) -> None:
        """Clear all cookies."""
        self._cookies.clear()
        self._pending_requests.clear()

    def __len__(self) -> int:
        """Get number of stored cookies."""
        return len(self._cookies)

    def __contains__(self, key: str) -> bool:
        """Check if cookie exists."""
        return key in self._cookies

    def __getitem__(self, key: str) -> Optional[bytes]:
        """Get cookie by key."""
        return self.get_cookie(key)

    def __setitem__(self, key: str, value: bytes) -> bool:
        """Set cookie."""
        return self.set_cookie(key, value)
