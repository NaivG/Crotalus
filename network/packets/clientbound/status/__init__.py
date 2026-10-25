"""Clientbound (Server -> Client) packets for the status state."""

from network.packets.clientbound.status.ping_response import CPingResponse
from network.packets.clientbound.status.status_response import CStatusResponse

__all__ = [
    "CPingResponse",
    "CStatusResponse",
]
