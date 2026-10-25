"""
Protocol States and Direction definitions

Based on Minecraft 1.21.11 protocol
"""

from enum import IntEnum
from typing import Protocol, runtime_checkable


class ProtocolState(IntEnum):
    """Minecraft connection protocol states."""
    HANDSHAKE = 0
    STATUS = 1
    LOGIN = 2
    CONFIG = 3  # Added in 1.20.2
    PLAY = 4
    TRANSFER = 5  # For server transfers


class Direction(IntEnum):
    """Packet direction."""
    SERVER_TO_CLIENT = 0
    CLIENT_TO_SERVER = 1


class PacketDirection:
    """Represents the direction of a packet."""
    
    def __init__(self, direction: Direction, state: ProtocolState):
        self.direction = direction
        self.state = state

    def __repr__(self) -> str:
        direction_str = "ServerBound" if self.direction == Direction.CLIENT_TO_SERVER else "ClientBound"
        return f"PacketDirection({direction_str}, {self.state.name})"


# Protocol version for 1.21.11
PROTOCOL_VERSION = 774

# The version string the client's built-in data pack reports for itself.
# Known Packs negotiation matches on (namespace, id, version) with exact
# equality, so this is the game version spelled out, not the protocol number
# and not a bare "1" -- a mismatch here is invisible on the wire and simply
# makes the client reply with an empty list.
CORE_PACK_VERSION = "1.21.11"


# Packet ID mappings for each state
# Format: {state: {direction: {packet_id: packet_class}}}

PACKET_IDS = {
    ProtocolState.HANDSHAKE: {
        Direction.CLIENT_TO_SERVER: {
            0x00: "SHandshake",  # Handshake
        },
        Direction.SERVER_TO_CLIENT: {},
    },
    ProtocolState.STATUS: {
        Direction.CLIENT_TO_SERVER: {
            0x00: "CStatusRequest",     # Status Request
            0x01: "CPingRequest",       # Ping Request
        },
        Direction.SERVER_TO_CLIENT: {
            0x00: "SStatusResponse",    # Status Response
            0x01: "SPingResponse",      # Ping Response
        },
    },
    ProtocolState.LOGIN: {
        Direction.CLIENT_TO_SERVER: {
            0x00: "SHello",             # Hello
            0x01: "SEncryptionAnswer", # Encryption Answer (Online mode)
            0x02: "SCustomQueryAnswer",# Custom Query Answer
            0x03: "SLoginAcknowledged", # Login Acknowledged (1.19.3+)
            0x04: "SCookieResponse",    # Cookie Response (1.20.5+)
        },
        Direction.SERVER_TO_CLIENT: {
            0x00: "CDisconnect",        # Disconnect (login)
            0x01: "CEncryptionRequest", # Encryption Request
            0x02: "CLoginSuccess",       # Login Success
            0x03: "CSetInitialPosition",# Set Initial Position (deprecated in 1.20.2+)
            0x04: "CCustomPayload",     # Custom Payload (plugin channels)
            0x05: "CPlayBuiltin",       # Play Builtin (1.21.5+)
            0x06: "CKnownPack",         # Known Packs (1.20.5+)
        },
    },
    ProtocolState.CONFIG: {
        Direction.CLIENT_TO_SERVER: {
            0x00: "SClientInformation",      # Client Information
            0x01: "SKnownPack",              # Known Packs
            0x02: "SCustomPayload",          # Custom Payload
            0x04: "SKeepAlive",             # Keep Alive (config)
            0x05: "SPong",                  # Pong
            0x06: "SResourcePackResponse",  # Resource Pack Response
            0x07: "SHandshake",             # Handshake (transfers)
            0x08: "SCookieResponse",        # Cookie Response (1.21.6+)
            0x09: "SAcceptCodeOfConduct",   # Accept Code of Conduct (1.21.9+)
        },
        Direction.SERVER_TO_CLIENT: {
            0x00: "CDisconnect",            # Disconnect
            0x01: "CFeatureFlags",           # Feature Flags
            0x02: "CRegistryData",          # Registry Data
            0x03: "CPluginMessage",         # Plugin Message (BungeeCord)
            0x04: "CResourcePack",          # Resource Pack
            0x05: "CKeepAlive",             # Keep Alive
            0x06: "CPing",                  # Ping
            0x0E: "CKnownPacks",            # Known Packs
            0x08: "CPlayBuiltin",           # Play Builtin (1.21.5+)
            0x09: "CCustomPayload",         # Custom Payload
        },
    },
    ProtocolState.PLAY: {
        Direction.CLIENT_TO_SERVER: {
            # Reference only - the live registry is GLOBAL_REGISTRY in
            # network/packet.py, built by the @register_packet decorators.
            # Listed here so an ID can be looked up without a client.
            0x00: "STeleportConfirm",            # Accept Teleportation
            0x01: "SBlockEntityTagQuery",        # Query Block Entity Tag
            0x02: "SBundleItemSelected",         # Bundle Item Selected
            0x03: "SChangeDifficulty",           # Change Difficulty
            0x04: "SChangeGameMode",             # Creative Inventory Action
            0x05: "SChatAck",                    # Message Acknowledgment
            0x06: "SChatCommand",                # Chat Command
            0x07: "SChatCommandSigned",          # Chat Command Signed
            0x08: "SChatMessage",                # Chat Message
            0x09: "SChatSessionUpdate",          # Player Session
            0x0A: "SChunkBatchReceived",         # Chunk Batch Received
            0x0B: "SClientCommand",              # Client Command
            0x0C: "SClientTickEnd",              # Client Tick End
            0x0D: "SClientInformationPlay",      # Client Information
            0x0E: "SCommandSuggestion",          # Command Suggestions Request
            0x0F: "SConfigurationAcknowledged",  # Acknowledge Configuration
            0x10: "SContainerButtonClick",       # Click Container Button
            0x11: "SContainerClick",             # Click Container
            0x12: "SCloseContainer",             # Close Container
            0x13: "SContainerSlotStateChanged",  # Container Slot State Changed
            0x14: "SCookieResponse",             # Cookie Response
            0x15: "SCustomPayload",              # Custom Payload
            0x16: "SDebugSubscriptionRequest",   # Debug Subscription Request
            0x17: "SEditBook",                   # Edit Book
            0x18: "SEntityTagQuery",             # Query Entity Tag
            0x19: "SInteract",                   # Interact
            0x1A: "SJigsawGenerate",             # Jigsaw Generate
            0x1B: "SKeepAlive",                  # Keep Alive
            0x1C: "SLockDifficulty",             # Lock Difficulty
            0x1D: "SMovePlayerPos",              # Set Player Position
            0x1E: "SMovePlayerPosRot",           # Set Player Position and Rotation
            0x1F: "SMovePlayerRot",              # Set Player Rotation
            0x20: "SMovePlayerStatusOnly",       # Set Player Movement Flags
            0x21: "SMoveVehicle",                # Move Vehicle
            0x22: "SPaddleBoat",                 # Paddle Boat
            0x23: "SPickItemFromBlock",          # Pick Item From Block
            0x24: "SPickItemFromEntity",         # Pick Item From Entity
            0x25: "SPingRequest",                # Ping Request
            0x26: "SPlaceRecipe",                # Place Recipe
            0x27: "SPlayerAbilities",            # Player Abilities
            0x28: "SPlayerAction",               # Player Action
            0x29: "SPlayerCommand",              # Player Command
            0x2A: "SPlayerInput",                # Player Input
            0x2B: "SPlayerLoaded",               # Player Loaded
            0x2C: "SPong",                       # Pong
        },
        Direction.SERVER_TO_CLIENT: {
            # ... see the same source for the clientbound table
            0x00: "CSpawnEntity",             # Spawn Entity
            0x01: "CSpawnExperienceOrb",      # Spawn Experience Orb
            0x02: "CSpawnLivingEntity",       # Spawn Living Entity
            0x03: "CSpawnPainting",           # Spawn Painting
            0x04: "CSpawnPlayer",             # Spawn Player
            0x05: "CEntityAnimation",         # Entity Animation
            # Chunk streaming. IDs verified against Pumpkin's 1.21.11 commit
            # (42c903493); its working tree tracks a later protocol.
            0x0B: "CChunkBatchFinished",      # Chunk Batch Finished
            0x0C: "CChunkBatchStart",         # Chunk Batch Start
            0x25: "CUnloadChunk",             # Forget Level Chunk
            0x2C: "CLevelChunkWithLight",     # Chunk Data and Update Light
            0x2F: "CLightUpdate",             # Light Update
            0x5C: "CSetChunkCacheCenter",     # Update View Position
            0x5D: "CSetChunkCacheRadius",     # Update View Distance
            # ... many more
        },
    },
}

