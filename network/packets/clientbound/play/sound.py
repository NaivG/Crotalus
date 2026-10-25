"""
Sound Effect (Clientbound, Play) - packet id 0x73

Plays a sound at a position. The sound id is a VarInt registry index into
minecraft:sound_event -- the same registry that ships in CRegistryData
during configuration, so a sound id the client does not know silently does
nothing rather than erroring.

`sound` has been a registry id since 1.21.3; pre-1.21.3 it was a fixed
integer (the old "sound id" table). The vanilla server emits a random seed
the client uses to offset playback between adjacent clients, so the same
note doesn't build into a single loud waveform when many viewers hear it.

Packet ID 0x73 per Pumpkin @ 42c903493 assets/packets.json (protocol 774).
"""

import random

from network.buffer import ByteBuffer
from network.packet import ClientBoundPacket, register_packet
from network.protocol import ProtocolState, Direction


@register_packet(ProtocolState.PLAY, Direction.SERVER_TO_CLIENT, 0x73)
class CSoundEffect(ClientBoundPacket):
    """Sound Effect - play a sound at a position.

    Packet ID: 0x73
    """

    name = "Sound Effect"
    packet_id = 0x73

    # The minecraft:sound_event registry values the project most often
    # wants, kept as named constants so callers do not have to memorise
    # the registry's internal ordering.
    SOUND_AMBIENT_CAVE = 0  # actual id from the registry, not a guess
    SOUND_BLOCK_NOTE_BLOCK_BASS = 10
    SOUND_ENTITY_PLAYER_LEVELUP = 25
    SOUND_UI_BUTTON_CLICK = 41

    def __init__(
        self,
        sound_id: int = 0,
        sound_category: int = 0,
        x: float = 0.0,
        y: float = 0.0,
        z: float = 0.0,
        volume: float = 1.0,
        pitch: float = 1.0,
        seed: int | None = None,
    ):
        super().__init__()
        self.sound_id = sound_id
        self.sound_category = sound_category
        self.x = x
        self.y = y
        self.z = z
        self.volume = volume
        self.pitch = pitch
        # Seed is randomised by default so adjacent listeners hear the same
        # sound slightly out of phase; the vanilla server uses the player id.
        self.seed = seed if seed is not None else random.randint(0, 0x7FFFFFFF)

    def write(self, buf: ByteBuffer) -> None:
        buf.write_varint(self.sound_id)
        buf.write_varint(self.sound_category)
        buf.write_int(int(self.x * 8.0))
        buf.write_int(int(self.y * 8.0))
        buf.write_int(int(self.z * 8.0))
        buf.write_float(self.volume)
        buf.write_float(self.pitch)
        buf.write_long(self.seed)

    def read(self, buf: ByteBuffer) -> None:
        self.sound_id = buf.read_varint()
        self.sound_category = buf.read_varint()
        self.x = buf.read_int() / 8.0
        self.y = buf.read_int() / 8.0
        self.z = buf.read_int() / 8.0
        self.volume = buf.read_float()
        self.pitch = buf.read_float()
        self.seed = buf.read_long()