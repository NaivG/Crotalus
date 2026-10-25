"""
Packet definitions for Minecraft 1.21.11

Layout mirrors the protocol rather than the wire format:

    network/packets/
        clientbound/          server -> client
            handshake/  status/  login/  config/  play/
        serverbound/          client -> server
            handshake/  status/  login/  config/  play/

The two axes are orthogonal, so they are two levels of directory. Every class
lives in the folder matching its direction, and its `C`/`S` prefix repeats that
direction: `C` for clientbound, `S` for serverbound.

Packet IDs are *not* globally unique - they are only meaningful within a
(state, direction) pair, which is exactly how GLOBAL_REGISTRY keys them.
"""

# Serverbound (Client -> Server)
from network.packets.serverbound.handshake import SHandshake
from network.packets.serverbound.status import SStatusRequest, SPingRequest
from network.packets.serverbound.login import (
    SHello,
    SEncryptionAnswer,
    SCustomQueryAnswer,
    SLoginAcknowledged,
)
from network.packets.serverbound.config import (
    SClientInformation,
    SCookieResponse,
    SCustomPayload,
    SFinishConfiguration,
    SKeepAliveConfig,
    SKnownPacks,
    SPong,
)
from network.packets.serverbound.play import (
    STeleportConfirm,
    SBundleItemSelected,
    SChangeDifficulty,
    SChatAck,
    SChatCommand,
    SChatCommandSigned,
    SChatMessage,
    SChunkBatchReceived,
    SClientCommand,
    SClientInformationPlay,
    SClientStatus,
    SClientTickEnd,
    SCloseContainer,
    SContainerClick,
    SClickContainerButton,
    SInteract,
    SKeepAlive,
    SMovementPacket,
    SMovePlayerPos,
    SMovePlayerPosRot,
    SMovePlayerRot,
    SMovePlayerStatusOnly,
    SPlayerAbilities,
    SPlayerAction,
    SPlayerCommand,
    SPlayerInput,
    SPlayerLoaded,
    SSetHeldItem,
    SSwingArm,
    SUseItem,
    SUseItemOn,
)

# Clientbound (Server -> Client)
from network.packets.clientbound.status import CStatusResponse, CPingResponse
from network.packets.clientbound.login import (
    CDisconnect,
    CEncryptionRequest,
    CLoginSuccess,
    CSetCompression,
    CSetInitialPosition,
)
from network.packets.clientbound.config import (
    CCookieRequest,
    CPluginMessageConfig,
    CDisconnectConfig,
    CFinishConfiguration,
    CKeepAliveConfig,
    CKnownPacks,
    CPingConfig,
    CRegistryData,
    CResourcePackPush,
    CStoreCookie,
    CUpdateEnabledFeatures,
    CUpdateTags,
)
from network.packets.clientbound.play import (
    CDisconnectPlay,
    CGameEvent,
    CKeepAlive,
    CLogin,
    CPlayerAbilities,
    CPlayerChatMessage,
    CPlayerInfoRemove,
    CPlayerInfoUpdate,
    PlayerInfoEntry,
    ACTION_ADD_PLAYER,
    ACTION_INITIALIZE_CHAT,
    ACTION_UPDATE_GAME_MODE,
    ACTION_UPDATE_LISTED,
    ACTION_UPDATE_LATENCY,
    ACTION_UPDATE_DISPLAY_NAME,
    ACTION_UPDATE_LIST_ORDER,
    ACTION_UPDATE_HAT,
    JOIN_ACTIONS,
    CSystemChatMessage,
    CSynchronizePlayerPosition,
    CUpdateTime,
    # Bundle
    CBundleDelimiterPacket,
    # Block layer
    CAcknowledgeBlock,
    CBlockUpdate,
    # Effects
    CClearTitles,
    CParticle,
    CPickupItem,
    CSetActionBarText,
    CSetSubtitleText,
    CSetTitleText,
    CSetTitlesAnimation,
    CSoundEffect,
    # Player state
    CDamageEvent,
    CRespawn,
    CSetDefaultSpawnPosition,
    CSetEntityMotion,
    CSetEquipment,
    CSetExperience,
    CSetHealth,
    CSetHeldItem,
    # Entity layer
    CAddEntity,
    CEntityAnimation,
    CEntityEvent,
    CEntityPositionSync,
    CHurtAnimation,
    CMoveEntityPos,
    CMoveEntityPosRot,
    CMoveEntityRot,
    CRemoveEntities,
    CRotateHead,
    CSetEntityData,
    CTeleportEntity,
    CLevelChunkWithLight,
    CSetChunkCacheCenter,
    CSetChunkCacheRadius,
    CUnloadChunk,
    CLightUpdate,
    CChunkBatchStart,
    CChunkBatchFinished,
)

# ---------------------------------------------------------------------------
# Backwards-compatible aliases
#
# The status packets used to be named after their *sender* rather than their
# destination, the opposite of every other state. They are now renamed to the
# convention above; these aliases keep older imports working. "Client Status"
# joins them for the same reason: it became "Client Command" in 1.21.5.
# ---------------------------------------------------------------------------
CStatusRequest = SStatusRequest
CPingRequest = SPingRequest
SStatusResponse = CStatusResponse
SPingResponse = CPingResponse

__all__ = [
    # Handshake (serverbound)
    "SHandshake",

    # Status
    "SStatusRequest",
    "SPingRequest",
    "CStatusResponse",
    "CPingResponse",

    # Login
    "SHello",
    "SEncryptionAnswer",
    "SCustomQueryAnswer",
    "SLoginAcknowledged",
    "CDisconnect",
    "CEncryptionRequest",
    "CLoginSuccess",
    "CSetCompression",
    "CSetInitialPosition",

    # Config
    "SClientInformation",
    "SCookieResponse",
    "SCustomPayload",
    "SFinishConfiguration",
    "SKeepAliveConfig",
    "SKnownPacks",
    "SPong",
    "CCookieRequest",
    "CPluginMessageConfig",
    "CDisconnectConfig",
    "CFinishConfiguration",
    "CKeepAliveConfig",
    "CKnownPacks",
    "CPingConfig",
    "CRegistryData",
    "CResourcePackPush",
    "CStoreCookie",
    "CUpdateEnabledFeatures",
    "CUpdateTags",

    # Play
    "STeleportConfirm",
    "SBundleItemSelected",
    "SChangeDifficulty",
    "SChatAck",
    "SChatMessage",
    "SChatCommand",
    "SChatCommandSigned",
    "SChunkBatchReceived",
    "SClientCommand",
    "SClientInformationPlay",
    "SClientTickEnd",
    "SCloseContainer",
    "SContainerClick",
    "SClickContainerButton",
    "SInteract",
    "SKeepAlive",
    "SMovementPacket",
    "SMovePlayerPos",
    "SMovePlayerPosRot",
    "SMovePlayerRot",
    "SMovePlayerStatusOnly",
    "SPlayerAbilities",
    "SPlayerAction",
    "SPlayerCommand",
    "SPlayerInput",
    "SPlayerLoaded",
    "SSetHeldItem",
    "SSwingArm",
    "SUseItem",
    "SUseItemOn",
    "CDisconnectPlay",
    "CGameEvent",
    "CKeepAlive",
    "CLogin",
    "CPlayerChatMessage",
    "CPlayerAbilities",
    "CPlayerInfoRemove",
    "CPlayerInfoUpdate",
    "PlayerInfoEntry",
    "ACTION_ADD_PLAYER",
    "ACTION_INITIALIZE_CHAT",
    "ACTION_UPDATE_GAME_MODE",
    "ACTION_UPDATE_LISTED",
    "ACTION_UPDATE_LATENCY",
    "ACTION_UPDATE_DISPLAY_NAME",
    "ACTION_UPDATE_LIST_ORDER",
    "ACTION_UPDATE_HAT",
    "JOIN_ACTIONS",
    "CSystemChatMessage",
    "CSynchronizePlayerPosition",
    "CUpdateTime",
    # Entity layer
    "CAddEntity",
    "CEntityAnimation",
    "CEntityEvent",
    "CEntityPositionSync",
    "CHurtAnimation",
    "CMoveEntityPos",
    "CMoveEntityPosRot",
    "CMoveEntityRot",
    "CRemoveEntities",
    "CRotateHead",
    "CSetEntityData",
    "CTeleportEntity",
    "CPickupItem",

    # Block layer (clientbound play)
    "CAcknowledgeBlock",
    "CBlockUpdate",

    # Effects (clientbound play)
    "CClearTitles",
    "CParticle",
    "CSetActionBarText",
    "CSetSubtitleText",
    "CSetTitleText",
    "CSetTitlesAnimation",
    "CSoundEffect",

    # Player state (clientbound play)
    "CDamageEvent",
    "CRespawn",
    "CSetDefaultSpawnPosition",
    "CSetEntityMotion",
    "CSetEquipment",
    "CSetExperience",
    "CSetHealth",
    "CSetHeldItem",

    # Bundle (clientbound play)
    "CBundleDelimiterPacket",

    # Chunk streaming (clientbound play)
    "CLevelChunkWithLight",
    "CSetChunkCacheCenter",
    "CSetChunkCacheRadius",
    "CUnloadChunk",
    "CLightUpdate",
    "CChunkBatchStart",
    "CChunkBatchFinished",

    # Deprecated aliases (see above)
    "CStatusRequest",
    "CPingRequest",
    "SStatusResponse",
    "SPingResponse",
    "SClientStatus",
]
