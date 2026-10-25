"""Serverbound (Client -> Server) packets for the play state."""

from network.packets.serverbound.play.bundle_item_selected import SBundleItemSelected
from network.packets.serverbound.play.change_difficulty import SChangeDifficulty
from network.packets.serverbound.play.chat_ack import SChatAck
from network.packets.serverbound.play.chat_command import SChatCommand
from network.packets.serverbound.play.chat_command_signed import SChatCommandSigned
from network.packets.serverbound.play.chat_message import SChatMessage
from network.packets.serverbound.play.chunk_batch_received import SChunkBatchReceived
from network.packets.serverbound.play.client_information import SClientInformationPlay
from network.packets.serverbound.play.client_status import SClientCommand
from network.packets.serverbound.play.client_tick_end import SClientTickEnd
from network.packets.serverbound.play.close_container import SCloseContainer
from network.packets.serverbound.play.click_container_button import SClickContainerButton
from network.packets.serverbound.play.container_click import SContainerClick
from network.packets.serverbound.play.interact import SInteract
from network.packets.serverbound.play.keep_alive import SKeepAlive
from network.packets.serverbound.play.movement import SMovementPacket
from network.packets.serverbound.play.move_player_pos import SMovePlayerPos
from network.packets.serverbound.play.move_player_rot import SMovePlayerRot
from network.packets.serverbound.play.move_player_status_only import SMovePlayerStatusOnly
from network.packets.serverbound.play.player_abilities import SPlayerAbilities
from network.packets.serverbound.play.player_action import SPlayerAction
from network.packets.serverbound.play.player_command import SPlayerCommand
from network.packets.serverbound.play.player_input import SPlayerInput
from network.packets.serverbound.play.player_loaded import SPlayerLoaded
from network.packets.serverbound.play.player_position_rotation import SMovePlayerPosRot
from network.packets.serverbound.play.set_held_item import SSetHeldItem
from network.packets.serverbound.play.swing_arm import SSwingArm
from network.packets.serverbound.play.teleport_confirm import STeleportConfirm
from network.packets.serverbound.play.use_item import SUseItem
from network.packets.serverbound.play.use_item_on import SUseItemOn

# Renamed in 1.21.5, when "Client Status" became "Client Command".
SClientStatus = SClientCommand

__all__ = [
    "SBundleItemSelected",
    "SChangeDifficulty",
    "SChatAck",
    "SChatCommand",
    "SChatCommandSigned",
    "SChatMessage",
    "SChunkBatchReceived",
    "SClientCommand",
    "SClientInformationPlay",
    "SClientStatus",
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
    "STeleportConfirm",
    "SUseItem",
    "SUseItemOn",
]