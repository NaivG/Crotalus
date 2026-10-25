"""Clientbound (Server -> Client) packets for the handshake state.

There are none. The handshake is client-initiated only: the server answers a
SHandshake by changing the connection state, never by sending a packet.
"""

__all__ = []
