"""Serverbound (Client -> Server) packets for the status state."""

from network.packets.serverbound.status.ping_request import SPingRequest
from network.packets.serverbound.status.status_request import SStatusRequest

__all__ = [
    "SPingRequest",
    "SStatusRequest",
]
