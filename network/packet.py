"""
Base Packet Classes

Defines the interface for Minecraft packets.
"""

from __future__ import annotations
import zlib
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from loguru import logger

from network.buffer import ByteBuffer
from network.protocol import ProtocolState, Direction, PacketDirection, PROTOCOL_VERSION


class Packet(ABC):
    """Base class for all Minecraft packets."""

    # Packet ID - should be overridden by subclasses
    packet_id: int = 0
    
    # Direction and state this packet belongs to
    direction: PacketDirection
    
    # Friendly name for debugging
    name: str = "Unknown"

    def __init__(self):
        self.direction = PacketDirection(Direction.SERVER_TO_CLIENT, ProtocolState.PLAY)

    @abstractmethod
    def write(self, buf: ByteBuffer) -> None:
        """Write packet data to the buffer."""
        pass

    @abstractmethod
    def read(self, buf: ByteBuffer) -> None:
        """Read packet data from the buffer."""
        pass

    @classmethod
    def get_packet_id(cls, state: ProtocolState, direction: Direction) -> int:
        """Get the packet ID for a specific state and direction."""
        return cls.packet_id

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} id={self.packet_id}>"


class ServerBoundPacket(Packet):
    """Packet sent from client to server."""
    
    def __init__(self):
        super().__init__()
        self.direction = PacketDirection(Direction.CLIENT_TO_SERVER, ProtocolState.PLAY)


class ClientBoundPacket(Packet):
    """Packet sent from server to client."""
    
    def __init__(self):
        super().__init__()
        self.direction = PacketDirection(Direction.SERVER_TO_CLIENT, ProtocolState.PLAY)


class PacketCodec:
    """Handles encoding and decoding of packets."""

    def __init__(
        self,
        state: ProtocolState = ProtocolState.PLAY,
        compression_threshold: int = -1,
        encrypted: bool = False
    ):
        self.state = state
        self.compression_threshold = compression_threshold
        self.encrypted = encrypted
        self.protocol_version = PROTOCOL_VERSION

    def encode_packet(self, packet: Packet, buf: ByteBuffer) -> bytes:
        """Encode a packet into raw bytes.
        
        Format without compression: [length: VarInt] [packet_id: VarInt] [data: bytes]
        
        Format with compression: [length: VarInt] [data_length: VarInt] [compressed_data: bytes]
        where compressed_data = zlib(packet_id + packet_data) if data_length > 0
        
        Note: For zero-payload packets in CONFIG state (like CFinishConfiguration),
        we omit the data_length field to match Java client expectations.
        """
        packet_data = ByteBuffer()
        # Write packet ID first
        packet_data.write_varint(packet.packet_id)
        # Then write packet payload
        packet.write(packet_data)
        
        data_bytes = bytes(packet_data)
        
        # Build packet with length prefix
        result = ByteBuffer()

        # Check if this is a zero-payload packet (only packet ID, no extra payload)
        # For these packets, skip the data_length field to match Java client expectations
        # This is needed because the client doesn't send data_length for zero-payload packets
        is_zero_payload = len(data_bytes) == 1 and data_bytes[0] <= 0x7F
         
        # Apply compression if threshold is set (threshold >= 0 means enabled)
        # Skip data_length field for zero-payload packets (client bug compatibility)
        if self.compression_threshold >= 0 and not is_zero_payload:
            if len(data_bytes) >= self.compression_threshold:
                # Compress the data
                compressed = zlib.compress(data_bytes)
                
                # Write: data_length (uncompressed size) + compressed data
                result.write_varint(len(data_bytes))  # Uncompressed size
                result.write_bytes(compressed)
            else:
                # Not compressed enough - write data_length=0 + raw data
                result.write_varint(0)  # data_length = 0 means not compressed
                result.write_bytes(data_bytes)
        else:
            # Compression disabled or zero-payload packet - no data_length field
            result.write_bytes(data_bytes)
        
        # Add frame length prefix
        final = ByteBuffer()
        final.write_varint(len(result))
        final.write_bytes(bytes(result))
        
        return bytes(final)

    def decode_packet(self, data: bytes) -> tuple[int, bytes]:
        """Decode raw packet bytes into packet ID and data.
        
        Returns: (packet_id, packet_data_bytes)
        
        Note: The input data should NOT include the frame length prefix (packet length).
        The connection layer already reads and consumes the frame length.
        
        Input format when compression is enabled:
        [data_length: VarInt] [compressed/uncompressed_data: bytes]
        
        Where:
        - data_length = 0: data is NOT compressed (raw packet_id + payload)
        - data_length > 0: data IS compressed with zlib
        
        Important: We MUST read data_length first when compression is enabled.
        
        Compatibility: Some clients may send uncompressed packets without data_length
        field during transition periods. We detect this and handle it gracefully.
        """
        buf = ByteBuffer()
        buf.write(data)
        buf.reset_read()
        
        # When compression is enabled, first byte(s) should be data_length
        if self.compression_threshold >= 0:
            # Peek at the first byte to determine format
            # If treating it as data_length doesn't make sense, fall back to uncompressed format
            
            saved_offset = buf._read_offset
            
            try:
                # Try reading as data_length first
                potential_data_length = buf.read_varint()
                remaining_bytes = buf.readable_bytes()
                
                # Reset offset for re-reading
                buf._read_offset = saved_offset
                
                # Heuristic checks to determine if this looks like compressed format:
                # 1. If potential_data_length == 0, next byte should be packet_id (usually small)
                # 2. If potential_data_length > 0, it should be <= remaining_bytes
                # 3. For small packets (< 2 bytes), likely uncompressed
                
                if len(data) < 2:
                    # Too small to have data_length + packet_id, treat as uncompressed
                    packet_id = buf.read_varint()
                    packet_data = buf.read_bytes(buf.readable_bytes())
                    return packet_id, packet_data
                
                if potential_data_length == 0:
                    # Uncompressed format: [0x00] [packet_id] [payload]
                    # Verify: consume the 0, then check if remaining looks valid
                    data_length = buf.read_varint()  # Should be 0
                    if buf.readable_bytes() > 0:
                        # Normal uncompressed packet
                        packet_id = buf.read_varint()
                        packet_data = buf.read_bytes(buf.readable_bytes())
                        return packet_id, packet_data
                    else:
                        # Edge case: only data_length=0, no actual data
                        # This shouldn't happen, but handle it
                        return 0, b''
                
                elif potential_data_length <= remaining_bytes:
                    # Compressed format: [data_length] [compressed_data]
                    data_length = buf.read_varint()
                    compressed = buf.read_bytes(data_length)
                    
                    try:
                        decompressed = zlib.decompress(compressed)
                    except Exception as e:
                        # Decompression failed - might be uncompressed data that looks like compressed
                        # Fall back to uncompressed interpretation below
                        raise ValueError(f"Decompression failed: {e}")
                    
                    decomp_buf = ByteBuffer()
                    decomp_buf.write(decompressed)
                    decomp_buf.reset_read()
                    
                    packet_id = decomp_buf.read_varint()
                    packet_data = decomp_buf.read_bytes(decomp_buf.readable_bytes())
                    return packet_id, packet_data
                
                else:
                    # potential_data_length > remaining_bytes
                    # This doesn't make sense for compressed format
                    # Likely this is actually an uncompressed packet where the first byte
                    # is the packet_id, not data_length
                    
                    # Treat as uncompressed: [packet_id] [payload]
                    packet_id = buf.read_varint()
                    packet_data = buf.read_bytes(buf.readable_bytes())
                    return packet_id, packet_data
                    
            except Exception as e:
                # Fallback: treat as uncompressed packet
                logger.debug(f"Failed to parse as compressed packet, trying uncompressed: {e}")
                buf._read_offset = saved_offset
                packet_id = buf.read_varint()
                packet_data = buf.read_bytes(buf.readable_bytes())
                return packet_id, packet_data
        else:
            # No compression - packet_id + payload
            packet_id = buf.read_varint()
            packet_data = buf.read_bytes(buf.readable_bytes())
        
        return packet_id, packet_data

    def _decompress(self, data: bytes) -> bytes:
        """Decompress data using zlib."""
        return zlib.decompress(data)

    def _compress(self, data: bytes) -> bytes:
        """Compress data using zlib."""
        return zlib.compress(data)


class PacketRegistry:
    """Registry for looking up packet classes by ID."""

    def __init__(self):
        self._packets: dict[
            tuple[ProtocolState, Direction, int], type[Packet]
        ] = {}

    def register(
        self,
        packet_class: type[Packet],
        state: ProtocolState,
        direction: Direction,
        packet_id: int
    ) -> None:
        """Register a packet class."""
        self._packets[(state, direction, packet_id)] = packet_class

    def get(
        self,
        state: ProtocolState,
        direction: Direction,
        packet_id: int
    ) -> type[Packet] | None:
        """Get packet class by state, direction, and ID."""
        return self._packets.get((state, direction, packet_id))

    def create_packet(
        self,
        state: ProtocolState,
        direction: Direction,
        packet_id: int,
        data: bytes
    ) -> Packet | None:
        """Create a packet instance from raw data."""
        packet_class = self.get(state, direction, packet_id)
        if packet_class is None:
            return None

        packet = packet_class()
        buf = ByteBuffer()
        buf.write(data)
        buf.reset_read()
        packet.read(buf)
        return packet


# Global packet registry
GLOBAL_REGISTRY = PacketRegistry()


def register_packet(
    state: ProtocolState,
    direction: Direction,
    packet_id: int
):
    """Decorator to register a packet class."""
    def decorator(cls: type[Packet]) -> type[Packet]:
        cls.packet_id = packet_id
        cls.direction = PacketDirection(direction, state)
        cls.name = cls.__name__
        GLOBAL_REGISTRY.register(cls, state, direction, packet_id)
        return cls
    return decorator
