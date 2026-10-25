"""
Connection handling

Manages the TCP connection, packet encoding/decoding, and state transitions.
"""

import asyncio
import secrets
from typing import Callable, Optional
from loguru import logger

from network.buffer import ByteBuffer
from network.packet import PacketCodec, GLOBAL_REGISTRY, Packet
from network.protocol import ProtocolState, Direction, PROTOCOL_VERSION



class Connection:
    """Manages a single client connection.
    
    Handles:
    - TCP stream reading/writing
    - Packet framing (length prefix)
    - Compression
    - Encryption (for online mode)
    - Protocol state management
    """

    def __init__(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ):
        self.reader = reader
        self.writer = writer
        
        # Connection info
        self.remote_address = writer.get_extra_info("peername")
        
        # Protocol state
        self.state = ProtocolState.HANDSHAKE
        self.protocol_version = PROTOCOL_VERSION
        
        # Compression
        self.compression_enabled = False
        self.compression_threshold = -1  # -1 means disabled
        
        # Encryption (for online mode)
        self.encrypted = False
        self.shared_secret: Optional[bytes] = None
        self._encryptor = None
        self._decryptor = None
        
        # Packet codec
        self.codec = PacketCodec(self.state)
        
        # Packet handlers
        self.handlers: dict[int, Callable[[Packet], None]] = {}
        
        # Callback when connection closes
        self.on_close: Optional[Callable[[], None]] = None
        
        # Buffer for incoming data
        self._recv_buffer = ByteBuffer()
        
        # Running flag
        self._running = False

    async def start(self) -> None:
        """Start handling this connection."""
        self._running = True
        try:
            await self._handle_connection()
        except asyncio.CancelledError:
            logger.info(f"Connection {self.remote_address} cancelled")
        except ConnectionResetError:
            logger.info(f"Connection {self.remote_address} reset by peer")
        except Exception as e:
            logger.exception(f"Connection {self.remote_address} error: {e}")
        finally:
            await self.close()

    async def _handle_connection(self) -> None:
        """Main connection loop."""
        while self._running:
            try:
                # Wait for data
                data = await self.reader.read(8192)
                if not data:
                    break
                
                # Add to buffer
                self._recv_buffer.write(data)
                
                # Process packets
                await self._process_packets()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error handling connection: {e}")
                break

    async def _process_packets(self) -> None:
        """Process all complete packets in the buffer."""
        while self._recv_buffer.readable_bytes() > 0:
            # Need at least VarInt bytes to read packet length
            if self._recv_buffer.readable_bytes() < 1:
                break
            
            # Peek at the packet length
            # We need to decode VarInt to know the length
            # For now, read byte by byte until we have enough
            saved_offset = self._recv_buffer._read_offset
            
            try:
                packet_len = self._recv_buffer.read_varint()
            except BufferError:
                # Not enough data
                self._recv_buffer._read_offset = saved_offset
                break
            
            # Check if we have the full packet
            if self._recv_buffer.readable_bytes() < packet_len:
                # Not enough data yet
                self._recv_buffer._read_offset = saved_offset
                break
            
            # Read the packet data
            packet_data = self._recv_buffer.read(packet_len)
            
            # Decode packet
            await self._handle_packet(packet_data)

    async def _handle_packet(self, data: bytes) -> None:
        """Handle a single packet."""
        # Decode packet
        packet_id, packet_bytes = self.codec.decode_packet(data)
        
        # Look up packet class
        direction = self._get_read_direction()
        packet_class = GLOBAL_REGISTRY.get(self.state, direction, packet_id)
        
        if packet_class is None:
            # Include the raw bytes: an unknown packet cannot be recovered
            # later, and without them the next step is a guess. The log line
            # further down never runs for these, so this is the only place
            # the payload is recorded.
            logger.warning(
                f"Unknown packet: state={self.state.name}, "
                f"direction={direction.name}, id=0x{packet_id:02X}, "
                f"raw={data.hex()}"
            )
            return
        
        # Create and read packet
        try:
            packet = packet_class()
            buf = ByteBuffer()
            buf.write(packet_bytes)
            buf.reset_read()
            
            # Debug log before reading to see raw data
            logger.debug(f"Raw packet data: {data.hex()} (id=0x{packet_id:02X})")
            
            packet.read(buf)
            
            logger.debug(f"Received: {packet}")
            
            # Call handler if registered
            handler = self.handlers.get(packet_id)
            if handler:
                await handler(packet)
            else:
                await self.handle_packet(packet)
                
        except ConnectionResetError:
            # 连接丢失处理
            logger.warning(f"Connection {self.remote_address} reset by peer during packet handling.")
            await self.close()
        except Exception as e:
            logger.exception(f"Error parsing packet {packet_class.__name__}: {e}")

    async def handle_packet(self, packet: Packet) -> None:
        """Handle a received packet. Override in subclass."""
        logger.debug(f"Unhandled packet: {packet}")

    def _get_read_direction(self) -> Direction:
        """Get the direction for reading based on current state."""
        # For most states, we're receiving from client
        return Direction.CLIENT_TO_SERVER

    async def send_packet(self, packet: Packet) -> None:
        """Send a packet to the client."""
        # Update codec state
        self.codec.state = self.state
        
        # Encode packet
        data = self.codec.encode_packet(packet, ByteBuffer())
        
        logger.debug(f"Encoded packet: {data.hex()} (id=0x{packet.__class__.packet_id:02X})")
        
        # Send
        self.writer.write(data)
        await self.writer.drain()
        
        logger.debug(f"Sent: {packet}")

    async def send_raw(self, data: bytes) -> None:
        """Send raw data to the client."""
        self.writer.write(data)
        await self.writer.drain()

    def set_state(self, state: ProtocolState) -> None:
        """Change protocol state."""
        old_state = self.state
        self.state = state
        self.codec.state = state
        logger.info(f"State changed: {old_state.name} -> {state.name}")

    def enable_compression(self, threshold: int) -> None:
        """Enable compression."""
        self.compression_enabled = True
        self.compression_threshold = threshold
        self.codec.compression_threshold = threshold
        logger.info(f"Compression enabled (threshold: {threshold})")

    def enable_encryption(self, shared_secret: bytes) -> None:
        """
        Enable AES/CFB8 stream encryption using shared secret.
        
        Args:
            shared_secret: Decrypted shared secret from client
        """
        from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
        from cryptography.hazmat.backends import default_backend
        
        # Create AES/CFB8 cipher with IV of zeros (Minecraft protocol standard)
        aes_key = shared_secret
        iv = b'\x00' * 16
        
        self._encryptor = Cipher(
            algorithms.AES(aes_key),
            modes.CFB8(iv),
            backend=default_backend()
        ).encryptor()
        
        self._decryptor = Cipher(
            algorithms.AES(aes_key),
            modes.CFB8(iv),
            backend=default_backend()
        ).decryptor()
        
        self.encrypted = True
        self.shared_secret = shared_secret
        logger.debug("Encryption enabled (AES/CFB8)")

    async def close(self) -> None:
        """Close the connection."""
        if self._running:
            self._running = False
            
            # Call close callback
            if self.on_close:
                self.on_close()
            
            # Close writer
            self.writer.close()
            try:
                await self.writer.wait_closed()
            except Exception:
                pass
            
            logger.info(f"Connection {self.remote_address} closed")

    @property
    def is_connected(self) -> bool:
        """Check if connection is still active."""
        return not self.writer.is_closing()


class ServerConnection(Connection):
    """Connection from server perspective (receives client packets)."""
    
    def _get_read_direction(self) -> Direction:
        return Direction.CLIENT_TO_SERVER


class ClientConnection(Connection):
    """Connection from client perspective (receives server packets)."""
    
    def _get_read_direction(self) -> Direction:
        return Direction.SERVER_TO_CLIENT
