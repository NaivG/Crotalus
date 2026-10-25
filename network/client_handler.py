"""
Client Handler

Handles client-specific game logic and packet responses.
"""

import json
import time
import asyncio
import queue
from typing import Optional, List, Sequence, Callable, Any
from loguru import logger

from network.buffer import ByteBuffer
from network.connection import Connection
from network.packet import Packet
from network.packets import (
    SHandshake,
    CStatusRequest, CPingRequest, SStatusResponse, SPingResponse,
    SHello, SEncryptionAnswer, CDisconnect, CEncryptionRequest, CLoginSuccess,
    CSetCompression,
    SClientInformation, SFinishConfiguration, CFinishConfiguration, SKeepAliveConfig, SPong,
    CDisconnectConfig, CPluginMessageConfig, CKeepAliveConfig, CPingConfig,
    CRegistryData, CResourcePackPush, CUpdateEnabledFeatures,
    SCustomPayload,
    CKnownPacks, SKnownPacks, CUpdateTags,
    CLogin, CGameEvent, CSynchronizePlayerPosition, CKeepAlive, CDisconnectPlay,
    STeleportConfirm, SBundleItemSelected, SChangeDifficulty, SChatAck, SChatMessage,
    SChatCommand, SChatCommandSigned,
    SChunkBatchReceived, SCloseContainer, SClientCommand, SClientInformationPlay,
    SClientTickEnd, SPlayerInput, SPlayerLoaded, SKeepAlive,
    SMovementPacket, SMovePlayerPos, SMovePlayerPosRot, SMovePlayerRot,
    SMovePlayerStatusOnly,
    CPlayerAbilities, CPlayerInfoUpdate, PlayerInfoEntry, ACTION_UPDATE_GAME_MODE,
)
from network.protocol import ProtocolState, PROTOCOL_VERSION, CORE_PACK_VERSION, Direction
from network.buffer import ByteBuffer

from world.chunk_streamer import ChunkSink, PlayerViewMsg
from entity.metadata import player_spawn_blob

from registry.registry import get_registry_manager, RegistryDataPacketBuilder
from registry.tag_data import get_tag_manager

from datatypes import parse_mini_message

from core.auth import MojangAuth
from core.config import config
from player.player import GameMode, Player, PlayerState
from player.commands import dispatch_from_client


# Play-state heartbeat. Vanilla sends one every 15s and drops a client that
# has not echoed the previous one; these are the same numbers, so a vanilla
# client's own timeout and ours agree about when a peer has gone away.
#
# The interval is also the floor on how long a joined connection can sit in
# silence. Nothing else in PLAY is guaranteed to speak up -- an idle dimension
# streams no chunks -- so without this a server can be perfectly healthy and
# still look hung to the client.
KEEPALIVE_INTERVAL = 15.0
KEEPALIVE_TIMEOUT = 30.0


# How a protocol state maps onto the player's own lifecycle. Spelled out
# rather than derived from the enum names: the two agree on LOGIN, CONFIG and
# PLAY but not on the first one (HANDSHAKE vs HANDSHAKING) nor on the numbering
# (STATUS and TRANSFER have no player equivalent at all -- neither is somebody
# joining the world).
PLAYER_STATES = {
    ProtocolState.HANDSHAKE: PlayerState.HANDSHAKING,
    ProtocolState.LOGIN: PlayerState.LOGIN,
    ProtocolState.CONFIG: PlayerState.CONFIG,
    ProtocolState.PLAY: PlayerState.PLAY,
}

class ClientHandler(Connection):
    """Handles client-specific game logic."""

    def __init__(self, *args, actor_inbox: Optional["queue.Queue"] = None,
                 spawn_point: tuple[int, int, int] = (0, 64, 0),
                 broadcast: Optional[Callable[[str, Optional[str], int, str], Any]] = None,
                 system_broadcast: Optional[Callable[[str], Any]] = None,
                 player_info_add: Optional[Callable[[str, str], Any]] = None,
                 player_info_remove: Optional[Callable[[str], Any]] = None,
                 player_info_gamemode: Optional[Callable[[str, int], Any]] = None,
                 **kwargs):
        super().__init__(*args, **kwargs)

        # The player this connection belongs to. Built empty and filled in as
        # the handshake names them, so every consumer below -- commands, chat,
        # the tab list, chunk streaming -- reads one object instead of a set
        # of private fields somebody has to keep in step by hand.
        #
        # Anything the *pipe* owns rather than the person stays here: crypto,
        # keep-alive bookkeeping, the config phase. A Player can be saved to
        # disk and reloaded; an asyncio task cannot, which is exactly the
        # line between the two. See player/player.py.
        self.player = Player(ip_address=self._format_address(self.remote_address))

        # Server-wide chat relay, for lines this player typed. Called with
        # the sender's name, UUID and per-sender index, then the raw text;
        # the server assembles the Player Chat packet, which every recipient
        # needs its own copy of.
        self._broadcast = broadcast

        # Server-wide chat relay for lines the *server* originates -- join and
        # leave notices. Those have no sender to speak of, so they are system
        # messages and go out as such. None in a bare handler (tests, tools).
        self._system_broadcast = system_broadcast

        # Tab list. The client's Player Chat path resolves the sender against
        # this table, so the join add is a prerequisite for chat working at
        # all, and the leave remove is the only thing that clears an entry.
        # _tab_listed records whether the add actually went out, so the remove
        # is not sent for a connection that never made it into the list.
        self._player_info_add = player_info_add
        self._player_info_remove = player_info_remove
        self._player_info_gamemode = player_info_gamemode
        self._tab_listed: bool = False

        # Where to drop this player in. The server resolves it against the
        # generated terrain before accepting connections, so by the time a
        # client joins this is a real surface, not a guess.
        self._spawn_point = spawn_point

        # Where this player's chunk-view updates are posted. The dimension
        # actor thread owns that view state; we only ever put() messages on it.
        # None means the tick pipeline is not running, and streaming is off.
        self._actor_inbox = actor_inbox

        # The actor thread's handle on this socket. Built here, on the event
        # loop, because it captures the running loop.
        self._chunk_sink: Optional[ChunkSink] = None

        # This player's Set Entity Data spawn payload, built lazily on the
        # first view publish (it needs the profile name from login) and
        # cleared on disconnect. Prebuilt once rather than per message:
        # the blob is identical every time another client needs to see us.
        self._spawn_blob: Optional[bytes] = None

        # Crypto helpers (online mode)
        self._crypto = None
        self._verify_token: Optional[bytes] = None
        self._server_id: Optional[str] = None
        
        # Teleport tracking (PLAY state). `_pending_teleport_id` is the id the
        # client has been told about and has not confirmed yet. It is the one
        # sequence for *every* teleport, including the movement corrections the
        # dimension actor sends from its own thread (physics/player.py): the
        # handler publishes the current value on each view message, and
        # _handle_teleport_confirm mirrors a newer id back, so the two never
        # disagree about what the client is waiting on.
        self._pending_teleport_id: int = 0
        self._teleport_confirmed: bool = False

        # How many poses have been published for this player, stamped onto
        # every PlayerViewMsg so the actor-side movement validator can tell a
        # fresh message from a repeat.
        self._view_sequence: int = 0

        # Set when the leave broadcast fires, so close() announces at most
        # once across its re-entrant calls (timeout path + shutdown).
        self._left_announced: bool = False

        # Keep-alive tracking (CONFIG state)
        self._pending_keepalive_id: Optional[int] = None
        self._keepalive_sent_time: float = 0

        # Keep-alive tracking (PLAY state). Separate from the config cookie:
        # the two states have their own independent 0x04/0x1B pair and the
        # client starts a fresh clock on entering play.
        self._play_keepalive_task: Optional[asyncio.Task] = None
        self._play_keepalive_id: int = 0
        self._play_keepalive_sent_time: float = 0.0
        self._play_keepalive_acked: bool = True

        # Login flow tracking - must wait for Login Acknowledged before CONFIG
        self._awaiting_login_ack: bool = False

        # Config phase timeout tracking
        self._config_timeout_task: Optional[asyncio.Task] = None

        # Which step of the config handshake we are on, and whether the
        # client turned out to know minecraft:core. Both used to be created
        # on first write and read back through getattr(...), which is how a
        # phase name misspelled in one of the six call sites turns into a
        # silent no-op instead of a bug report.
        self._config_phase: str = ""
        self._client_knows_minecraft_core: bool = False
        self._server_known_packs: list = []

        # State-specific handlers
        self._handlers = {
            ProtocolState.HANDSHAKE: {
                0x00: self._handle_handshake,
            },
            ProtocolState.STATUS: {
                0x00: self._handle_status_request,
                0x01: self._handle_ping_request,
            },
            ProtocolState.LOGIN: {
                0x00: self._handle_login_start,
                0x01: self._handle_encryption_answer,
                0x03: self._handle_login_acknowledged,
            },
            ProtocolState.CONFIG: {
                0x00: self._handle_client_information,
                0x02: self._handle_custom_payload,
                0x03: self._handle_finish_configuration,
                0x04: self._handle_keep_alive_config,
                0x05: self._handle_pong,
                0x07: self._handle_known_packs,
            },
            ProtocolState.PLAY: {
                0x00: self._handle_teleport_confirm,
                0x02: self._handle_bundle_item_selected,
                0x03: self._handle_change_difficulty,
                0x05: self._handle_chat_ack,
                0x06: self._handle_chat_command,
                0x07: self._handle_chat_command_signed,
                0x08: self._handle_chat_message,
                0x0A: self._handle_chunk_batch_received,
                0x0B: self._handle_client_command,
                0x0C: self._handle_client_tick_end,
                0x0D: self._handle_client_information_play,
                0x12: self._handle_close_container,
                0x1B: self._handle_keep_alive_play,
                0x1D: self._handle_move_player_pos,
                0x1E: self._handle_move_player_pos_rot,
                0x1F: self._handle_move_player_rot,
                0x20: self._handle_move_player_status_only,
                0x2A: self._handle_player_input,
                0x2B: self._handle_player_loaded,
            },
        }

    @staticmethod
    def _format_address(addr) -> str:
        """Render a peername as a string, for the player's record.

        asyncio hands back a (host, port) tuple for TCP and a bare path for
        a unix socket; the player file wants one string either way.
        """
        if isinstance(addr, (tuple, list)):
            return ":".join(str(part) for part in addr)
        return str(addr) if addr else ""

    def set_state(self, state: ProtocolState) -> None:
        """Change protocol state, and mirror it onto the player.

        The player carries its own copy of the lifecycle so that anything
        holding a Player -- the tab list, a command, a future persistence
        pass -- can tell a joining player from a live one without reaching
        for the socket. States with no player-side equivalent leave it alone;
        see PLAYER_STATES.
        """
        super().set_state(state)
        player_state = PLAYER_STATES.get(state)
        if player_state is not None:
            self.player.state = player_state

    @property
    def in_game(self) -> bool:
        """True once this client is playable and must receive broadcasts.

        The handler is registered with the server the moment the TCP
        connection opens, long before a player exists, so broadcast filters
        need more than "is a connection": until PlayerLoaded the client is
        still applying its initial chunk set and would drop chat anyway.
        """
        return self.player.is_spawned()

    async def handle_packet(self, packet: Packet) -> None:
        """Route packet to appropriate handler."""
        # Get handler based on state and packet ID
        state_handlers = self._handlers.get(self.state, {})
        handler = state_handlers.get(packet.packet_id)
        
        if handler:
            await handler(packet)
        else:
            logger.debug(f"Unhandled packet: {packet} in state {self.state.name}")
    
    async def _handle_handshake(self, packet: SHandshake) -> None:
        """Handle handshake packet."""
        logger.info(f"Handshake from {self.remote_address}: protocol={packet.protocol_version}, "
                   f"address={packet.server_address}, next_state={packet.next_state}")
        
        # Set next state
        if packet.next_state == 1:
            self.set_state(ProtocolState.STATUS)
        elif packet.next_state == 2:
            # Check protocol version
            if packet.protocol_version == PROTOCOL_VERSION:
                self.set_state(ProtocolState.LOGIN)
            else:
                logger.warning(f"Unsupported protocol version: {packet.protocol_version}")
                # For login, we can disconnect
                if packet.protocol_version < PROTOCOL_VERSION:
                    await self.disconnect(f"Outdated client({packet.protocol_version})! Please use Minecraft 1.21.11.")
                else:
                    await self.disconnect(f"Incompatible client({packet.protocol_version})! This server is still on Minecraft 1.21.11.")
                return
        else:
            logger.warning(f"Invalid next state: {packet.next_state}")
            await self.close()
    
    async def _handle_status_request(self, packet: CStatusRequest) -> None:
        """Handle status request (server list ping)."""
        await self._send_status_response()

    async def _send_status_response(self) -> None:
        """Send status response to client."""

        version_name = "Crotalus 1.21.11"
        description = parse_mini_message(config.motd)
        
        status_data = {
                "version": {
                    "name": version_name,
                    "protocol": PROTOCOL_VERSION,
                },
                "players": {
                    "max": config.max_players,
                    "online": 0,  # TODO: Track connected players
                    "sample": [],
                },
                "description": description,
                # CamelCase key: the client's ServerStatus codec reads
                # "enforcesSecureChat" and defaults to false when absent.
                "enforcesSecureChat": config.get("server.enforce-secure-profile", False),
            }
        
        response = SStatusResponse(json.dumps(status_data))
        await self.send_packet(response)
    
    async def _handle_ping_request(self, packet: CPingRequest) -> None:
        """Handle ping request."""
        response = SPingResponse(packet.payload)
        await self.send_packet(response)
        await self.close()
    
    async def _handle_login_start(self, packet: SHello) -> None:
        """Handle login start."""
        self.player.name = packet.name
        logger.info(f"Login start: {self.player.name}")
        
        if config.online_mode:
            # Online mode: send encryption request
            await self._send_encryption_request()
        else:
            # Offline mode: skip encryption and proceed directly
            await self._proceed_to_login_success()
    
    async def _send_encryption_request(self) -> None:
        """Send encryption request for online mode."""
        from network.crypto import CryptoHelper
        
        # Initialize crypto helper
        self._crypto = CryptoHelper()
        self._verify_token = CryptoHelper.generate_verify_token()
        self._server_id = CryptoHelper.generate_server_id()
        
        # Send encryption request
        encryption_request = CEncryptionRequest(
            server_id=self._server_id,
            public_key=self._crypto.public_key_bytes,
            verify_token=self._verify_token
        )
        await self.send_packet(encryption_request)
        logger.debug(f"Sent encryption request to {self.player.name} (server_id={self._server_id})")
    
    async def _handle_encryption_answer(self, packet: SEncryptionAnswer) -> None:
        """Handle encryption response from client."""
        if self._crypto is None:
            logger.error(f"Received encryption answer without sending request from {self.player.name}")
            await self.disconnect("Unexpected encryption response")
            return
        
        try:
            # Decrypt shared secret
            shared_secret = self._crypto.decrypt_shared_secret(packet.shared_secret)
            
            # Verify token if present
            if packet.verify_token:
                decrypted_verify_token = self._crypto.decrypt_shared_secret(packet.verify_token)
                if decrypted_verify_token != self._verify_token:
                    logger.warning(f"Verify token mismatch for {self.player.name}")
                    await self.disconnect("Invalid verify token")
                    return
            
            # Enable encryption
            self.enable_encryption(shared_secret)
            logger.debug(f"Encryption enabled for {self.player.name}")
            
            # Verify with Mojang API
            success, profile = await MojangAuth.has_joined(
                self.player.name,
                self._server_id,
                shared_secret
            )
            
            if success:
                # Use Mojang-provided UUID and username
                self.player.uuid = profile['id'].replace('-', '')
                self.player.name = profile['name']
                logger.info(f"Mojang auth successful: {self.player.name} ({self.player.uuid})")
                await self._proceed_to_login_success()
            else:
                await self.disconnect("Failed to verify with Mojang")

        except Exception as e:
            logger.exception(f"Encryption error: {e}")
            await self.disconnect("Failed to enable encryption")
    
    async def _proceed_to_login_success(self) -> None:
        """Proceed to login success - MUST wait for Login Acknowledged before CONFIG."""
        import uuid

        # Enable compression if configured - it only takes effect after login success and in PLAY state
        if config.compression_threshold >= 0:
            logger.info(f"Compression will be enabled after login for {self.player.name} (threshold: {config.compression_threshold})")
            # Send compression threshold
            compression = CSetCompression(threshold=config.compression_threshold)
            await self.send_packet(compression)

        # Generate offline UUID if not already set (offline mode)
        if not self.player.uuid:
            self.player.uuid = str(uuid.uuid3(uuid.NAMESPACE_DNS, self.player.name))

        # Send login success
        login_success = CLoginSuccess(
            uuid=self.player.uuid,
            username=self.player.name,
            properties=[],
        )
        await self.send_packet(login_success)
        logger.debug(f"Sent Login Success to {self.player.name}")

        # Set flag and wait for Login Acknowledged
        self._awaiting_login_ack = True
        logger.info(f"Waiting for Login Acknowledged from {self.player.name}...")
    
    async def _send_server_config_packets(self) -> None:
        """Send server-side Config state packets according to standard Minecraft 1.20.2+ protocol.

        Flow:
        1. Send minecraft:brand (plugin message)
        2. Send Clientbound Known Packs (list of data packs server knows)
        3. Wait for Serverbound Known Packs (client's subset)
        4. Send Registry Data (for each registry)
        5. Send Update Tags
        6. Send Update Enabled Features
        7. Send Finish Configuration
        """
        logger.info(f"Starting config phase for {self.player.name}")

        # Step 1: Send minecraft:brand
        server_brand = "Crotalus"
        brand_packet = CPluginMessageConfig("minecraft:brand", server_brand.encode('utf-8'))
        await self.send_packet(brand_packet)
        logger.debug(f"Sent server brand '{server_brand}' to {self.player.name}")

        # Step 2: Send Known Packs - list all data packs the server knows.
        # Format: namespace + id + version as three separate strings.
        #
        # The version has to be exactly what the client's own built-in pack
        # reports, which is the game version rather than the protocol number.
        # The client keeps only packs it matches on all three fields, so
        # anything else here negotiates down to an empty reply and buys
        # nothing while still costing a round trip.
        known_packs = [
            {"namespace": "minecraft", "id": "core", "version": CORE_PACK_VERSION}
        ]
        known_packs_packet = CKnownPacks(known_packs)
        await self.send_packet(known_packs_packet)
        logger.info(f"Sent Known Packs to {self.player.name}: {known_packs}")

        # Step 3: Set state to wait for client's known packs response
        self._config_phase = 'waiting_known_packs'
        self._server_known_packs = known_packs
        logger.info(f"Waiting for client's Known Packs response from {self.player.name}...")

        # Start timeout timer
        self._config_timeout_task = asyncio.create_task(self._config_known_packs_timeout())

    async def _config_known_packs_timeout(self) -> None:
        """Timeout handler for waiting client's known packs response."""
        await asyncio.sleep(5.0)

        if self._config_phase == 'waiting_known_packs':
            logger.warning(f"Client {self.player.name} did not send Known Packs within 5s, proceeding with empty known packs...")
            # Proceed with empty known packs (no optimization)
            await self._continue_config_after_known_packs([])

    async def _handle_known_packs(self, packet: SKnownPacks) -> None:
        """Handle client's Known Packs response - compute mutual known packs and continue config."""
        logger.info(f"Received Known Packs from {self.player.name}: {packet.packs}")

        # Cancel timeout task
        if self._config_timeout_task:
            self._config_timeout_task.cancel()
            try:
                await self._config_timeout_task
            except asyncio.CancelledError:
                pass

        # Check if we're waiting for known packs
        if self._config_phase != 'waiting_known_packs':
            logger.debug("Known Packs received but not in waiting phase")
            return

        # The client replies with the intersection it computed against its own
        # packs, so it has already done the filtering -- asking again would
        # only get the same answer. What matters is whether minecraft:core
        # survived, matched on all three fields.
        self._client_knows_minecraft_core = packet.knows(
            "minecraft", "core", CORE_PACK_VERSION
        )
        if self._client_knows_minecraft_core:
            logger.info(
                f"Client {self.player.name} knows minecraft:core@{CORE_PACK_VERSION}, "
                f"registry entries may omit their NBT"
            )
        else:
            logger.info(
                f"Client {self.player.name} did not echo minecraft:core@{CORE_PACK_VERSION}, "
                f"sending full registry data"
            )

        # Continue with remaining config steps
        await self._continue_config_after_known_packs(packet.packs)

    async def _continue_config_after_known_packs(self, client_packs: list) -> None:
        """Continue config phase after known packs negotiation.

        Args:
            client_packs: The pack dicts the client echoed back, each with
                'namespace', 'id' and 'version' keys.
        """
        self._config_phase = 'sending_registry_data'
        logger.info(f"Continuing config phase for {self.player.name}")

        # Step 4: Send Registry Data
        registry_manager = get_registry_manager()

        # Whether each entry carries its NBT inline, or just its id and lets
        # the client fill the value in from the pack it just told us it has.
        #
        # This is the whole point of the negotiation: measured against
        # assets/synced_registries.json it takes 23 packets from 91,504 bytes
        # down to 4,500. Worth knowing before changing it -- the failure mode
        # is not an error but wrong data. If a client's core pack ever
        # disagrees with ours (an overridden resource pack, a modded client),
        # nothing is reported: recipes, enchantments and variant registries
        # just come out wrong. Pinning it back to True costs one login's
        # bandwidth and removes the possibility entirely.
        #
        # Generally speaking, most of the logical calculations in Java Edition
        # take place on the server side; therefore, if the core package is 
        # overridden in client side, the server-side behavior will be adopted 
        # in the best-case scenario. (Vanilla behavior)
        include_nbt = not self._client_knows_minecraft_core

        packet_builder = RegistryDataPacketBuilder(registry_manager, known_packs=client_packs)
        all_packets = packet_builder.build_all_packets(include_nbt=include_nbt)
        logger.info(f"Built {len(all_packets)} registry packets")

        # Send each registry packet
        for i, (registry_id, packet_data) in enumerate(all_packets):
            registry = registry_manager.get_registry(registry_id)
            num_entries = len(registry.entries) if registry else 0

            # Build packet: [packet_id: VarInt] + [data: bytes]
            buf = ByteBuffer()
            buf.write_varint(0x07)  # Registry Data packet ID
            buf.write_bytes(packet_data)

            # Add length prefix
            final = ByteBuffer()
            final.write_varint(len(buf))
            final.write_bytes(bytes(buf))

            await self.send_raw(bytes(final))
            logger.debug(f"Sent registry {i+1}/{len(all_packets)}: {registry_id} ({num_entries} entries)")
            await asyncio.sleep(0.01)  # Small delay to avoid overwhelming client

        logger.info(f"Sent {len(all_packets)} registry packets to {self.player.name}")

        # Step 5: Send Update Tags
        tag_manager = get_tag_manager()
        tag_data = tag_manager.build_update_tags_packet_data()
        update_tags = CUpdateTags(tag_data)
        await self.send_packet(update_tags)
        logger.info(f"Sent Update Tags to {self.player.name}: {len(tag_data)} registry tags")

        # Step 6: Send Update Enabled Features
        features = CUpdateEnabledFeatures()
        await self.send_packet(features)
        logger.debug(f"Sent Update Enabled Features to {self.player.name}")

        # Step 7: Send Finish Configuration
        finish = CFinishConfiguration()
        await self.send_packet(finish)
        logger.info(f"Sent Finish Configuration to {self.player.name}")

        self._config_phase = 'waiting_finish_config'
        logger.info(f"Config phase complete for {self.player.name}, waiting for Finish Configuration acknowledgment")
    
    async def _handle_login_acknowledged(self, packet) -> None:
        """Handle login acknowledged - CRITICAL step before CONFIG state.

        In Minecraft 1.20.2+, after this packet the client switches to CONFIG state
        and starts sending its configuration (brand, client info).
        """
        if not self._awaiting_login_ack:
            logger.warning(f"Unexpected Login Acknowledged from {self.player.name} (not expecting it)")
            return

        logger.info(f"Player {self.player.name} acknowledged login (Login Acknowledged received)")

        # Clear the flag
        self._awaiting_login_ack = False

        # Transition to config state
        # Client will now send:
        # - Serverbound Plugin Message (brand)
        # - Client Information
        # - Known Packs (after we send our Known Packs)
        self.set_state(ProtocolState.CONFIG)

        # Start sending server config immediately
        # The client will respond with its config data
        self._config_phase = 'waiting_client_info'
        logger.info(f"Starting config phase for {self.player.name}, waiting for client configuration...")

        # Start timeout timer to avoid waiting indefinitely
        self._config_timeout_task = asyncio.create_task(self._config_info_timeout())

    async def _config_info_timeout(self) -> None:
        """Timeout handler for waiting client configuration."""
        await asyncio.sleep(5.0)

        # Check if still waiting for client info
        if self._config_phase == 'waiting_client_info':
            logger.warning(f"Client {self.player.name} did not send configuration within 5s, starting server config anyway...")
            # Start sending server config (brand + known packs)
            await self._send_server_config_packets()

    async def _handle_client_information(self, packet: SClientInformation) -> None:
        """Handle client information packet - part of client configuration."""
        logger.info(f"Client info: locale={packet.locale}, view_distance={packet.view_distance}")

        # Store client settings. Every field the packet carries is kept, not
        # just the two this handler happens to read today: they are the
        # player's declared preferences, and Player already has a home for
        # each one.
        self.player.apply_client_settings(packet)

        # Check if we're still waiting for client info
        if self._config_phase != 'waiting_client_info':
            logger.debug("Client information received but not in waiting phase")
            return

        # Cancel timeout task since we received client info
        if self._config_timeout_task:
            self._config_timeout_task.cancel()
            try:
                await self._config_timeout_task
            except asyncio.CancelledError:
                pass

        # Now we can start sending server config
        await self._send_server_config_packets()
    
    async def _handle_keep_alive_config(self, packet: SKeepAliveConfig) -> None:
        """Handle keep alive response from client."""
        if self._pending_keepalive_id is not None:
            if packet.payload == self._pending_keepalive_id:
                logger.debug(f"Keepalive confirmed: expected={self._pending_keepalive_id}, got={packet.payload}")
                self._pending_keepalive_id = None  # Clear pending state - unblocks _send_config_packets()
            else:
                logger.warning(f"Keepalive ID mismatch: expected={self._pending_keepalive_id}, got={packet.payload}")
        # else: unsolicited keepalive, ignore (already handled by config phase)

    # ---------------- play-state heartbeat ----------------

    def _start_play_keepalive(self) -> None:
        """Begin the heartbeat that keeps a joined connection alive."""
        if self._play_keepalive_task and not self._play_keepalive_task.done():
            return
        self._play_keepalive_task = asyncio.create_task(self._play_keepalive_loop())

    async def _play_keepalive_loop(self) -> None:
        """Ping the client, and hang up on one that stops answering.

        Two jobs in one loop because they share a clock. The first is
        scheduled: a packet every KEEPALIVE_INTERVAL so an otherwise-idle
        server still proves it is there. The second is the deadline: a client
        that has not echoed the previous ping by KEEPALIVE_TIMEOUT is either
        gone or wedged, and keeping the socket open for it helps nobody.

        The echo is checked here rather than only in the packet handler
        because a client that stops talking produces silence, not a packet.
        """
        while True:
            await asyncio.sleep(KEEPALIVE_INTERVAL)

            if not self._running or self.state != ProtocolState.PLAY:
                return

            if not self._play_keepalive_acked:
                waited = time.monotonic() - self._play_keepalive_sent_time
                if waited >= KEEPALIVE_TIMEOUT:
                    logger.warning(
                        f"Keep-alive from {self.player.name} unanswered after "
                        f"{waited:.0f}s, disconnecting"
                    )
                    # Drop the handle first: close() cancels and awaits this
                    # task, and a task cannot await itself.
                    self._play_keepalive_task = None
                    await self.close()
                    return
                # Not dead yet -- keep pinging; the id below is the new one
                # and the client's next echo will match it.

            self._play_keepalive_id += 1
            self._play_keepalive_sent_time = time.monotonic()
            self._play_keepalive_acked = False
            await self.send_packet(CKeepAlive(self._play_keepalive_id))

    async def _handle_keep_alive_play(self, packet: SKeepAlive) -> None:
        """The client's echo of our play-state heartbeat."""
        if packet.payload == self._play_keepalive_id:
            self._play_keepalive_acked = True
        else:
            # A late echo of an id we have already replaced. Not fatal -- the
            # client is clearly alive -- but worth seeing, since it means
            # something upstream is reordering or duplicating.
            logger.debug(
                f"Stale keep-alive echo from {self.player.name}: "
                f"expected {self._play_keepalive_id}, got {packet.payload}"
            )
            self._play_keepalive_acked = True
    
    async def _handle_pong(self, packet: SPong) -> None:
        """Handle pong response."""
        pass  # Already handled by keep alive

    async def _handle_custom_payload(self, packet: SCustomPayload) -> None:
        """Handle custom payload/brand message from client (Step 16 of login.md)."""
        if packet.channel == "minecraft:brand":
            brand = packet.data.decode('utf-8', errors='replace')
            logger.info(f"Client brand: {brand}")
            self.player.brand = brand
        else:
            logger.debug(f"Received plugin message on channel '{packet.channel}'")

        # Check if we received client's brand while waiting for client info
        # Both brand and client info may arrive in any order, we need to wait for both
        if self._config_phase == 'waiting_client_info':
            # Cancel current timeout task since we're making progress
            if self._config_timeout_task:
                self._config_timeout_task.cancel()
                try:
                    await self._config_timeout_task
                except asyncio.CancelledError:
                    pass
            # Restart timeout in case client info never arrives
            self._config_timeout_task = asyncio.create_task(self._config_info_timeout())
            logger.debug(f"Client brand '{self.player.brand}' stored, waiting for Client Information...")

    async def _handle_finish_configuration(self, packet: SFinishConfiguration) -> None:
        """Handle finish configuration from client - client has switched to Play state."""
        logger.info(f"Player {self.player.name} acknowledged configuration, switching to PLAY state")
        
        # Server also switches to PLAY state now
        self.set_state(ProtocolState.PLAY)
        
        if config.compression_threshold >= 0:
            # NOW enable compression - it only takes effect in PLAY state
            # This matches the Minecraft protocol specification
            self.enable_compression(config.compression_threshold)
            logger.info(f"Compression enabled for PLAY state (threshold: {config.compression_threshold})")
        
        # Send Join Game packet
        login_packet = CLogin(
            # The client's id for its own player entity. Derived from the
            # player's UUID rather than hardcoded, so two clients on one
            # server do not both believe they are entity 1.
            entity_id=self.player.entity_id,
            is_hardcore=False,
            dimension_names=["minecraft:overworld", "minecraft:nether", "minecraft:the_end"],
            max_players=config.max_players,
            view_distance=config.view_distance,
            simulated_distance=config.simulation_distance,
            reduced_debug_info=False,
            enabled_respawn_screen=True,
            limited_crafting=False,
            dimension_type=0,
            dimension_name="minecraft:overworld",
            hashed_seed=0,
            game_mode=self.player.game_mode.value,
            previous_gamemode=-1,
            debug=False,
            is_flat=False,
            has_death_position=False,
            death_dimension=None,
            death_position=None,
            portal_cooldown=0,
            sealevel=63,
            enforce_secure_chat=config.get("server.enforce-secure-profile", False),
        )
        await self.send_packet(login_packet)
        
        logger.info(f"Sent Join Game packet to {self.player.name}")
        
        # Send Game Event 13 - Start waiting for level chunks (CRITICAL for 1.21+)
        # This tells the client to begin waiting for chunk data
        game_event = CGameEvent(
            event=CGameEvent.EVENT_START_WAITING_FOR_LEVEL_CHUNKS,
            value=0.0
        )
        await self.send_packet(game_event)
        
        logger.info(f"Sent Game Event (waiting for chunks) to {self.player.name}")
        
        # Send Synchronize Player Position - place player in the world
        # Increment teleport_id before sending
        self._pending_teleport_id += 1
        spawn_x, spawn_y, spawn_z = self._spawn_point
        sync_pos = CSynchronizePlayerPosition(
            teleport_id=self._pending_teleport_id,
            x=float(spawn_x) + 0.5,
            y=float(spawn_y),
            z=float(spawn_z) + 0.5,
            velocity_x=0.0,
            velocity_y=0.0,
            velocity_z=0.0,
            yaw=0.0,
            pitch=0.0,
            flags=0,  # Absolute position
        )
        await self.send_packet(sync_pos)

        # Seed the position from the teleport we just sent, so the first
        # chunk view is built around where the client actually is rather than
        # around the origin. Rotation and on_ground are left alone: this is a
        # position, and the client's first report is the first word on which
        # way it is facing.
        pos = self.player.position
        pos.x = float(spawn_x) + 0.5
        pos.y = float(spawn_y)
        pos.z = float(spawn_z) + 0.5

        logger.info(f"Sent Synchronize Player Position to {self.player.name} (teleport_id={self._pending_teleport_id})")

        # Start the play-state heartbeat now that the client is in the world
        # and something other than the login burst is on the wire.
        self._start_play_keepalive()
    
    # ==================== PLAY State Handlers ====================
    
    async def _handle_teleport_confirm(self, packet: STeleportConfirm) -> None:
        """Handle teleport confirmation.

        This is the point where the client stops waiting on the loading
        screen's promise and starts waiting for actual terrain. The streamer
        takes it from here: it sends UpdateViewDistance + UpdateViewPosition,
        then a batch of chunks, nearest first — the chunk under the player
        being first, which is what dismisses the loading screen.

        Any id at or above the pending one counts, and the pending id is moved
        up to it. The join teleport is not the only teleport any more: a
        movement correction is issued by the dimension actor thread
        (physics/player.py), which cannot increment this counter, so the id it
        used arrives here first and this is where the handler learns it. Ids
        only go up, and this connection issues exactly one of them itself (the
        join teleport), so "newer than what I last sent" is the only question
        the handshake has to answer — an id below the pending one is still a
        mismatch and is still refused.
        """
        # Verify teleport_id matches (or is newer than) the pending one
        if packet.teleport_id >= self._pending_teleport_id:
            self._pending_teleport_id = packet.teleport_id
            self._teleport_confirmed = True
            logger.info(f"Teleport confirmed by {self.player.name}: teleport_id={packet.teleport_id}")

            self._publish_view()

            # TODO: Continue with post-teleport login flow:
            # - Server Data (optional)
            # - Player Info Update (all players)
            # - Initialize World Border (optional)
            # - Update Time (optional)
            # - Set Default Spawn Position (optional)
        else:
            logger.warning(
                f"Teleport ID mismatch from {self.player.name}: "
                f"expected {self._pending_teleport_id}, got {packet.teleport_id}"
            )

    def _stream_radius(self) -> int:
        """How many chunks to stream to this player.

        The client reports its own render distance twice — once in CONFIG,
        once on entering play — and the server's own limit is the hard cap.
        Default to the server's if neither has been heard from.

        The clamp lives here, at the point of use, rather than at assignment:
        the player's view_distance is what the client asked for, and only
        this answer is a promise about what it will get.
        """
        if self.player.view_distance:
            return max(1, min(self.player.view_distance, config.view_distance))
        return max(1, config.view_distance)

    async def _send_packets(self, packets: Sequence["Packet"]) -> None:
        """Write a batch of packets. Runs on the event loop via ChunkSink."""
        for packet in packets:
            await self.send_packet(packet)

    def _publish_view(self) -> None:
        """Publish this player's current pose to the dimension actor.

        Called on every movement packet, plus the once-per-join moments
        (teleport confirmed, player loaded). That is a change from the old
        chunk-only world, where posting was worth only on a chunk-border
        crossing: the entity tracker needs the position snapshot every
        tick, because its delta codec compares "now" against "last sent".
        The cost is a frozen-dataclass queue put per movement packet, which
        is noise next to the chunk payloads on the same inbox, and the three
        consumers each dedupe their own expensive work (the streamer inside
        pump, the tracker by comparing baselines, the movement validator by
        finding the displacement to be zero).
        """
        # Only after the teleport is confirmed: that is the point the client
        # starts expecting terrain, and it keeps a stray pre-teleport move
        # packet from attaching a view for a position the client is leaving.
        if self._actor_inbox is None or not self._teleport_confirmed:
            return

        pos = self.player.position

        if self._chunk_sink is None:
            self._chunk_sink = ChunkSink(
                self._send_packets, asyncio.get_running_loop()
            )

        if self._spawn_blob is None:
            self._spawn_blob = player_spawn_blob(
                self.player.name, self.player.skin_parts
            )

        # Stamped here rather than at the call site: every publisher of a pose
        # goes through this method, so the sequence counts published poses and
        # nothing has to remember to pass a number.
        self._view_sequence += 1

        self._actor_inbox.put(PlayerViewMsg(
            player_id=str(id(self)),
            sink=self._chunk_sink,
            x=pos.x,
            y=pos.y,
            z=pos.z,
            radius=self._stream_radius(),
            # The simulation radius is the server's, not the client's: the
            # client has no say in how far entities tick. It rides on the same
            # message so the actor sees one ordered view of this player.
            simulation_radius=max(1, config.simulation_distance),
            entity_id=self.player.entity_id,
            uuid=self.player.uuid,
            yaw=pos.yaw,
            pitch=pos.pitch,
            on_ground=pos.on_ground,
            spawn_blob=self._spawn_blob,
            # Read fresh every publish rather than cached, because a
            # `/gamemode spectator` lands on the event loop between two
            # movement packets and the next publish is exactly when the
            # spawner should stop counting this player.
            spectator=self.player.game_mode == GameMode.SPECTATOR,
            # The actor's movement validator needs to tell a fresh pose from a
            # repeat, and its corrections have to join the *same* teleport id
            # sequence the client is already answering on. Both values are the
            # handler's, because the handler is what talks to the client.
            sequence=self._view_sequence,
            teleport_id=self._pending_teleport_id,
        ))

    def _stop_streaming(self) -> None:
        """Tell the actor to drop this player's view.

        The entity_id rides along so the tracker detaches the right player:
        removing them from everyone else's view is the tracker's half of a
        leave, and it broadcasts the Remove Entities for it.
        """
        if self._actor_inbox is not None and self._chunk_sink is not None:
            self._actor_inbox.put(PlayerViewMsg(
                player_id=str(id(self)),
                sink=None,
                entity_id=self.player.entity_id,
            ))
        self._chunk_sink = None
        self._spawn_blob = None
    
    async def _handle_bundle_item_selected(self, packet: SBundleItemSelected) -> None:
        """Handle bundle item selected."""
        logger.debug(f"Bundle item selected: slot={packet.slot_id}, index={packet.selected_item_index}")
    
    async def _handle_change_difficulty(self, packet: SChangeDifficulty) -> None:
        """Handle difficulty change."""
        logger.debug(f"Change difficulty: {packet.difficulty}")
    
    async def _handle_chat_ack(self, packet: SChatAck) -> None:
        """Handle chat acknowledgement."""
        logger.debug(f"Chat ack from {self.player.name}: offset={packet.offset}")
    
    async def _handle_chat_message(self, packet: SChatMessage) -> None:
        """Handle chat message."""
        logger.info(f"{self.player.name}: {packet.message}")
        if self._broadcast is not None:
            # The name and the "<name> " prefix are not ours to add: the
            # client builds the line from the sender name and the chat type.
            # Claim the index before awaiting, so two messages typed in one
            # tick cannot land on the same one.
            index = self.player.claim_sent_index()
            await self._broadcast(
                self.player.name, self.player.uuid, index, packet.message
            )

    async def _handle_chat_command(self, packet: SChatCommand) -> None:
        """Handle a command the player typed ("/gamemode creative" -> "gamemode creative").

        不在网络线程上执行：投递到统一命令线程（与控制台命令同一个
        调度方），由 PlayerCommandSender 把回复送回本玩家。
        """
        logger.info(f"{self.player.name} ran command: /{packet.command}")
        dispatch_from_client(self, packet.command)

    async def _handle_chat_command_signed(self, packet: SChatCommandSigned) -> None:
        """Handle a command with signed arguments.

        The signature block only verifies under an enforce-secure-chat server;
        offline, the command text itself is what matters, so dispatch is
        identical to the unsigned path.
        """
        logger.info(
            f"{self.player.name} ran command: /{packet.command} "
            f"(signed, {len(packet.argument_signatures)} argument signatures)"
        )
        dispatch_from_client(self, packet.command)

    async def set_gamemode(self, gamemode: int) -> None:
        """Switch this player's gamemode and tell every client that cares.

        Takes the bare wire value rather than a GameMode, because every
        caller has it in that form already: the tab list and Game Event both
        encode it as an integer, and /gamemode parses it from chat text. The
        player stores the parsed enum, so the stored state is a real mode and
        not a number that happens to be in range.

        Three packets, because each carries one facet of the change: Player
        Abilities grants what the mode permits (flight, invulnerability) and
        is what actually enables double-jump-to-fly; Player Info Update puts
        the right icon in every tab list; Game Event flips the client's own
        notion of its mode, which drives hotbar behaviour client-side. The
        order matches Pumpkin's set_gamemode, abilities first, and the Game
        Event lands last so it cannot race the abilities it should accompany.
        """
        mode = GameMode.from_wire(gamemode)
        if mode == self.player.game_mode:
            return
        self.player.game_mode = mode

        await self.send_packet(CPlayerAbilities.for_gamemode(gamemode))

        if self._player_info_gamemode is not None and self.player.uuid:
            await self._player_info_gamemode(self.player.uuid, gamemode)

        await self.send_packet(CGameEvent(
            event=CGameEvent.EVENT_CHANGE_GAME_MODE,
            value=float(gamemode),
        ))
    
    async def _handle_client_tick_end(self, packet: SClientTickEnd) -> None:
        """Handle the client's end-of-tick marker.

        Arrives ~20x/second, so it is counted rather than logged individually.
        A tick count that stops advancing is the signal that a client has gone
        quiet, which is what this packet exists to let a server notice.
        """
        self.player.client_ticks += 1
        if self.player.client_ticks % 20 == 0:
            pos = self.player.position
            logger.debug(
                f"Tick {self.player.client_ticks // 20}s from {self.player.name}: "
                f"pos=({pos.x:.2f}, {pos.y:.2f}, {pos.z:.2f})"
            )

    async def _handle_client_information_play(self, packet: SClientInformationPlay) -> None:
        """Handle client settings re-sent on entering play.

        The client re-reports its render distance once it knows the server's
        own, so this is the point where the two can be reconciled.
        """
        logger.info(
            f"Client information from {self.player.name}: "
            f"locale={packet.locale}, view_distance={packet.view_distance} "
            f"(server sends {config.view_distance})"
        )
        # Stored raw; _stream_radius does the clamping, at the one place the
        # number is actually turned into a promise.
        self.player.apply_client_settings(packet)

    async def _handle_move_player_pos_rot(self, packet: SMovePlayerPosRot) -> None:
        """Handle a position + rotation report from the client."""
        self._apply_movement(packet)
        self._log_movement("pos+rot", with_rotation=True)

    async def _handle_move_player_pos(self, packet: SMovePlayerPos) -> None:
        """Handle a position-only report — the client walked without turning."""
        self._apply_movement(packet)
        self._log_movement("pos")

    async def _handle_move_player_rot(self, packet: SMovePlayerRot) -> None:
        """Handle a rotation-only report — the client looked around in place."""
        self._apply_movement(packet)
        self._log_movement("rot", with_rotation=True)

    async def _handle_move_player_status_only(self, packet: SMovePlayerStatusOnly) -> None:
        """Handle a movement-state report — stepped off a ledge, or landed."""
        self._apply_movement(packet)

    def _apply_movement(self, packet: SMovementPacket) -> None:
        """Fold a movement packet into the server's view of the player.

        The client stays authoritative for its own position: the reported
        values are written here as-is and stored for whatever consumes them
        next (chunk streaming, entity broadcast, saving). The server's opinion
        about whether the *path* was possible is formed separately, on the
        dimension actor thread, from the pre/post positions this method
        publishes -- `physics/player.py` sweeps them and answers with a
        correction packet if the world says the move could not have happened.
        Nothing here waits for that answer, which is deliberate: the client's
        prediction is the position until the correction arrives.

        Every movement packet must go through here, not just the ones carrying
        a position. A packet that moves the player has to re-run the view
        computation, or the chunk stream stalls while they walk.
        """
        # Assigned field by field rather than through Player.update_position,
        # because that replaces the whole pose: a rotation-only packet carries
        # no coordinates, and folding it in as-is would teleport the player
        # back to the last teleport point.
        pos = self.player.position
        if packet.CARRIES_POSITION:
            pos.x = packet.x
            pos.y = packet.y
            pos.z = packet.z
        if packet.CARRIES_ROTATION:
            pos.yaw = packet.yaw
            pos.pitch = packet.pitch

        pos.on_ground = packet.on_ground

        # Publish every time, not just on a chunk-border crossing: the chunk
        # streamer dedupes on its own centre check, but the entity tracker
        # needs a snapshot of *this* pose each tick to diff against last
        # sent. Suppressing here on "same chunk" would freeze other clients'
        # view of a player who turns in place. The movement validator on the
        # actor thread is the third reader of the same message, and it needs
        # every packet too: a skipped pose is a step it cannot judge.
        self._publish_view()

        # TODO: broadcast to other players, and save to disk periodically

    def _log_movement(self, kind: str, with_rotation: bool = False) -> None:
        """Log where the player ended up after a movement packet.

        The client sends one of these every tick, so this stays at debug, and
        it names the packet kind rather than logging a generic "Move" — the
        four variants are otherwise indistinguishable in a capture.
        """
        pos = self.player.position
        parts = [
            f"{kind} from {self.player.name}",
            f"pos=({pos.x:.2f}, {pos.y:.2f}, {pos.z:.2f})",
        ]
        if with_rotation:
            parts.append(f"yaw={pos.yaw:.2f} pitch={pos.pitch:.2f}")
        parts.append(f"on_ground={pos.on_ground}")
        logger.debug(" ".join(parts))

    async def _handle_chunk_batch_received(self, packet: SChunkBatchReceived) -> None:
        """Handle the client's throughput report for the chunk stream.

        The value is what the client can absorb per tick. The streamer paces
        on its own for now, so this is recorded and logged rather than fed
        back into the send rate.
        """
        logger.debug(
            f"Chunk batch ack from {self.player.name}: "
            f"desired_chunks_per_tick={packet.desired_chunks_per_tick:.2f}"
        )

    async def _handle_player_input(self, packet: SPlayerInput) -> None:
        """Handle the set of movement keys the client is holding.

        The client only sends this when the set changes, so it is a good place
        to observe intent: a player holding forward and sprint is heading
        somewhere. Nothing replays it, and that is the deliberate deviation the
        plan records (`PHYSICS_PLAN.md` §八.1): movement is validated as a
        *path* against the collision engine (`physics/player.py`), not by
        simulating the keys, so the input stays a log line.
        """
        keys = packet.active_keys()
        logger.debug(
            f"Input from {self.player.name}: {', '.join(keys) if keys else '(none)'}"
        )

    async def _handle_player_loaded(self, packet: SPlayerLoaded) -> None:
        """Handle the client reporting it has finished loading into the world.

        This is the first point at which the client is actually playable, so it
        is the right moment to start streaming and to announce the join.
        The loaded flag is set before broadcasting: the joiner is a recipient
        of their own join message, vanilla-style.

        The tab list add goes out first and to everyone, this player included.
        It has to precede any chat: the client looks the sender of a Player
        Chat Message up in this table, and a miss means the message is thrown
        away and reported to the player as a chat validation error.
        """
        logger.info(f"{self.player.name} joined the game.")
        self.player.loaded = True
        if self._player_info_add is not None and self.player.uuid:
            await self._player_info_add(self.player.uuid, self.player.name)
            self._tab_listed = True
        self._publish_view()
        if self._system_broadcast is not None:
            await self._system_broadcast(f"{self.player.name} joined the game.")

    async def close(self) -> None:
        """Close the connection, releasing the actor's view of this player."""
        # The heartbeat calls close() itself on a timeout, so this has to be
        # safe to re-enter from inside the task being cancelled.

        # Vanilla announces "X left the game." on every disconnect, mirroring
        # the join broadcast, so every joined player leaves behind exactly one
        # join/leave pair. Announce here, before the keepalive teardown: the
        # cancelled task can re-enter close() from its timeout branch, so the
        # once-only guard must be claimed before any await below.
        
        if self.player.loaded and not self._left_announced:
            logger.info(f"{self.player.name} left the game.")
            self.player.loaded = False

            if self._system_broadcast is not None:
                await self._system_broadcast(f"{self.player.name} left the game.")
            self._left_announced = True

            # After the leave notice, so the removal is not what makes the
            # player vanish. Guarded on having actually been added: a player
            # who dropped during login never got a tab list entry, and
            # removing one that was never there would drop somebody else's.
            if self._tab_listed and self._player_info_remove is not None:
                self._tab_listed = False
                await self._player_info_remove(self.player.uuid)

        task = self._play_keepalive_task
        self._play_keepalive_task = None
        if task and not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        self._stop_streaming()
        await super().close()

    async def _handle_close_container(self, packet: SCloseContainer) -> None:
        """Handle container close."""
        logger.debug(f"Close container: {packet.window_id}")

    async def _handle_client_command(self, packet: SClientCommand) -> None:
        """Handle a client command."""
        logger.debug(f"Client command: action={packet.action_id}")
        # Handle respawn request
        if packet.action_id == SClientCommand.ACTION_RESPAWN:
            logger.info(f"Player {self.player.name} requested respawn")
    
    async def disconnect(self, reason: str) -> None:
        """Send the state-appropriate disconnect packet, then close.

        Every exit path that has something to tell the client goes through
        here, send and close as one step: splitting them invites exactly one
        of the two being forgotten (the old call sites each had to remember
        their own close(), and server stop() remembered only the close). The
        packet is chosen by the state the client is currently in — the same
        JSON payload exists at a different ID in login, config and play, and
        the client resolves IDs against its current state. Sending close()
        would have been the reverse integration, but close() also runs on
        dead sockets and status pings where a send is meaningless.

        A failed send (socket already dead) must not skip the close: the
        teardown is the part that matters once the client is gone.
        """
        packet = {
            ProtocolState.LOGIN: CDisconnect,
            ProtocolState.CONFIG: CDisconnectConfig,
            ProtocolState.PLAY: CDisconnectPlay,
        }.get(self.state)
        if packet is not None:
            try:
                await self.send_packet(packet(json.dumps({"text": reason})))
            except Exception:
                logger.debug(f"Failed to deliver disconnect to {self.player.name}")
        await self.close()
