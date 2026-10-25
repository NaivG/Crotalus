"""
Clientbound packets (Server -> Client)

Every module below this package is safe to import for its side effect alone:
importing it registers the packets in network.packet.GLOBAL_REGISTRY.
"""

from network.packets.clientbound import config, login, play, status

__all__ = [
    "config",
    "login",
    "play",
    "status",
]
