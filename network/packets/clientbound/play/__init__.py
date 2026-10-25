"""Clientbound (Server -> Client) packets for the play state."""

from network.packets.clientbound.play.chunk import (
    CLevelChunkWithLight,
    CSetChunkCacheCenter,
    CSetChunkCacheRadius,
    CUnloadChunk,
    CLightUpdate,
    CChunkBatchStart,
    CChunkBatchFinished,
)
from network.packets.clientbound.play.disconnect import CDisconnectPlay
from network.packets.clientbound.play.game_event import CGameEvent
from network.packets.clientbound.play.keep_alive import CKeepAlive
from network.packets.clientbound.play.login import CLogin
# Entity layer: spawn, move, despawn. Grouped together because they are one
# feature, not because of any dependency between them.
from network.packets.clientbound.play.add_entity import CAddEntity
from network.packets.clientbound.play.entity_animation import CEntityAnimation
from network.packets.clientbound.play.entity_event import CEntityEvent
from network.packets.clientbound.play.entity_position_sync import (
    CEntityPositionSync,
)
from network.packets.clientbound.play.hurt_animation import CHurtAnimation
from network.packets.clientbound.play.move_entity_pos import CMoveEntityPos
from network.packets.clientbound.play.move_entity_pos_rot import (
    CMoveEntityPosRot,
)
from network.packets.clientbound.play.move_entity_rot import CMoveEntityRot
from network.packets.clientbound.play.remove_entities import CRemoveEntities
from network.packets.clientbound.play.rotate_head import CRotateHead
from network.packets.clientbound.play.set_entity_data import CSetEntityData
from network.packets.clientbound.play.teleport_entity import CTeleportEntity
# Block-layer: changes, acks, bulk updates.
from network.packets.clientbound.play.acknowledge_block import CAcknowledgeBlock
from network.packets.clientbound.play.block_update import CBlockUpdate
# Effects: sound, particles, titles, action bar.
from network.packets.clientbound.play.clear_titles import CClearTitles
from network.packets.clientbound.play.particle import CParticle
from network.packets.clientbound.play.pickup_item import CPickupItem
from network.packets.clientbound.play.set_action_bar_text import CSetActionBarText
from network.packets.clientbound.play.set_subtitle_text import CSetSubtitleText
from network.packets.clientbound.play.set_title_text import CSetTitleText
from network.packets.clientbound.play.set_titles_animation import CSetTitlesAnimation
from network.packets.clientbound.play.sound import CSoundEffect
# Player state: health, xp, held slot, equipment, motion, respawn, damage.
from network.packets.clientbound.play.damage_event import CDamageEvent
from network.packets.clientbound.play.respawn import CRespawn
from network.packets.clientbound.play.set_default_spawn_position import (
    CSetDefaultSpawnPosition,
)
from network.packets.clientbound.play.set_entity_motion import CSetEntityMotion
from network.packets.clientbound.play.set_equipment import CSetEquipment
from network.packets.clientbound.play.set_experience import CSetExperience
from network.packets.clientbound.play.set_health import CSetHealth
from network.packets.clientbound.play.set_held_item import CSetHeldItem
from network.packets.clientbound.play.player_abilities import CPlayerAbilities
from network.packets.clientbound.play.player_chat import CPlayerChatMessage
from network.packets.clientbound.play.player_info_update import (
    ACTION_ADD_PLAYER,
    ACTION_INITIALIZE_CHAT,
    ACTION_UPDATE_DISPLAY_NAME,
    ACTION_UPDATE_GAME_MODE,
    ACTION_UPDATE_HAT,
    ACTION_UPDATE_LATENCY,
    ACTION_UPDATE_LISTED,
    ACTION_UPDATE_LIST_ORDER,
    CPlayerInfoRemove,
    CPlayerInfoUpdate,
    JOIN_ACTIONS,
    PlayerInfoEntry,
)
from network.packets.clientbound.play.system_chat import CSystemChatMessage
from network.packets.clientbound.play.synchronize_player_position import (
    CSynchronizePlayerPosition,
)
from network.packets.clientbound.play.update_time import CUpdateTime
# Bundle delimiter: brackets a group of packets for same-tick delivery.
from network.packets.clientbound.play.bundle_delimiter import CBundleDelimiterPacket

__all__ = [
    # Chunk streaming
    "CLevelChunkWithLight",
    "CSetChunkCacheCenter",
    "CSetChunkCacheRadius",
    "CUnloadChunk",
    "CLightUpdate",
    "CChunkBatchStart",
    "CChunkBatchFinished",
    "CDisconnectPlay",
    "CGameEvent",
    "CKeepAlive",
    "CLogin",
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
    # Block layer
    "CAcknowledgeBlock",
    "CBlockUpdate",
    # Effects
    "CClearTitles",
    "CParticle",
    "CPickupItem",
    "CSetActionBarText",
    "CSetSubtitleText",
    "CSetTitleText",
    "CSetTitlesAnimation",
    "CSoundEffect",
    # Player state
    "CDamageEvent",
    "CRespawn",
    "CSetDefaultSpawnPosition",
    "CSetEntityMotion",
    "CSetEquipment",
    "CSetExperience",
    "CSetHealth",
    "CSetHeldItem",
    "CPlayerAbilities",
    "CPlayerChatMessage",
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
    # Bundle
    "CBundleDelimiterPacket",
]