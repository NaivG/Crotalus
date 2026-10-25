"""
Mojang Authentication Module

Handles session server verification for online mode.
Communicates with Mojang's session server to verify player authentication.
"""

import aiohttp
from typing import Optional, Tuple
from loguru import logger

from hashlib import sha1

MOJANG_SESSION_SERVER = "https://sessionserver.mojang.com/session/minecraft/hasJoined"


class MojangAuth:
    """Mojang authentication handler."""
    
    @staticmethod
    async def has_joined(
        username: str,
        server_id: str,
        shared_secret: bytes
    ) -> Tuple[bool, Optional[dict]]:
        """
        Verify with Mojang session server.
        
        Args:
            username: Player username
            server_id: Server ID from encryption request
            shared_secret: Decrypted shared secret
            
        Returns:
            (success, profile_data) tuple.
            If success is True, profile_data contains the player's profile.
            If success is False, profile_data is None.
        """
        # Generate SHA1 hash of server_id + shared_secret
        hash_data = server_id.encode('ascii') + shared_secret
        digest = sha1(hash_data).digest()
        
        # Convert to hexadecimal representation (Mojang's special format)
        hex_digest = _hexify(digest)
        
        params = {
            'username': username,
            'serverId': hex_digest
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(MOJANG_SESSION_SERVER, params=params) as resp:
                    if resp.status == 200:
                        profile = await resp.json()
                        logger.info(f"Mojang auth successful for {username}")
                        return True, profile
                    else:
                        logger.warning(
                            f"Mojang auth failed for {username}: "
                            f"HTTP {resp.status}"
                        )
                        return False, None
        except Exception as e:
            logger.error(f"Mojang auth error: {e}")
            return False, None


def _hexify(data: bytes) -> str:
    """
    Convert bytes to Mojang's special hexadecimal format.
    
    Mojang uses a signed hexadecimal representation where:
    - Positive numbers are represented as normal hex
    - Negative numbers (when first bit is 1) are prefixed with '-'
    
    Args:
        data: SHA1 digest bytes
        
    Returns:
        Hexadecimal string in Mojang's format
    """
    # Convert to integer (big-endian, signed)
    value = int.from_bytes(data, byteorder='big', signed=True)
    
    # Convert to hex
    result = hex(value)[2:]  # Remove '0x' prefix
    
    # Ensure even length
    if len(result) % 2 != 0:
        result = '0' + result
    
    return result
