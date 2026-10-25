"""
Serverbound packets (Client -> Server)

Every module below this package is safe to import for its side effect alone:
importing it registers the packets in network.packet.GLOBAL_REGISTRY.
"""

from network.packets.serverbound import (
    config,
    handshake,
    login,
    play,
    status,
)

__all__ = [
    "config",
    "handshake",
    "login",
    "play",
    "status",
]
