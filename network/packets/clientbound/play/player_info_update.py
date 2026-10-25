"""
Player Info Update (Clientbound, Play)

The tab list. On its face it is a list of names, but it is also the client's
lookup table for the *signed chat* pipeline, which is what makes it a
prerequisite for showing player chat rather than a cosmetic feature.

When a ClientboundPlayerChatMessage (0x3F) arrives, the client first resolves
the message's sender UUID against its Player Info map, and only then builds the
message. A miss is not a rendering detail -- the client logs "Received player
chat packet for unknown player with ID" and drops the line into a
`chat.validation_error` toast, which is the "chat validation error" the player
sees. The bytes of the chat packet are irrelevant at that point; a server can
get every field of 0x3F right and still have the client refuse the message.

So this packet is what makes chat work at all. It is sent on join, before any
Player Chat, to the joining player and to everyone already online.

Fields are selected by a bitmask, and each entry then carries only the selected
ones, in the order of the bits. The mask is a *byte*, not a VarInt -- the client
reads it with readEnumSet, which is a single byte. The entry list is a VarInt
count followed by that many entries.

Written for 1.21.11. The action bits were read off the client's own enum, whose
declaration order fixes both the bit values and the field order:

    ordinal 0  ADD_PLAYER          0x01
    ordinal 1  INITIALIZE_CHAT     0x02
    ordinal 2  UPDATE_GAME_MODE    0x04
    ordinal 3  UPDATE_LISTED       0x08
    ordinal 4  UPDATE_LATENCY      0x10
    ordinal 5  UPDATE_DISPLAY_NAME 0x20
    ordinal 6  UPDATE_LIST_ORDER   0x40
    ordinal 7  UPDATE_HAT          0x80

The bit values here differ from other sources for nearby versions; they are
transcribed from this client rather than from the wiki or from a protocol
library, both of which were checked and did not match.
"""

from network.buffer import ByteBuffer
from network.packet import ClientBoundPacket, register_packet
from network.protocol import ProtocolState, Direction

# Action bits, in the order the client reads them. This list is the on-wire
# field order: an entry writes its fields in exactly this sequence, so the
# ordinals are load-bearing and not just an implementation detail.
ACTION_ADD_PLAYER = 0x01
ACTION_INITIALIZE_CHAT = 0x02
ACTION_UPDATE_GAME_MODE = 0x04
ACTION_UPDATE_LISTED = 0x08
ACTION_UPDATE_LATENCY = 0x10
ACTION_UPDATE_DISPLAY_NAME = 0x20
ACTION_UPDATE_LIST_ORDER = 0x40
ACTION_UPDATE_HAT = 0x80

# The set a vanilla server sends for a player who has just joined: enough for
# the tab list to render a name, a game mode icon and a latency bar, and to
# make the entry resolvable for chat.
JOIN_ACTIONS = (
    ACTION_ADD_PLAYER
    | ACTION_UPDATE_GAME_MODE
    | ACTION_UPDATE_LISTED
    | ACTION_UPDATE_LATENCY
    | ACTION_UPDATE_HAT
)

GAME_MODE_SURVIVAL = 0
GAME_MODE_ADVENTURE = 2


def write_byte_array(buf: ByteBuffer, data: bytes | None) -> None:
    """Write a length-prefixed byte array, the shape readByteArray expects.

    Length 0 is the encoding for "absent" here -- an empty public key or
    signature is a real value on the wire, not a missing one.
    """
    payload = bytes(data or b"")
    buf.write_varint(len(payload))
    buf.write_bytes(payload)


class PlayerInfoEntry:
    """One player's entry in a Player Info Update.

    Only the fields whose action bit is set in the packet's mask are written.
    A field left at its default while its bit is set writes the default, which
    is a legitimate value -- there is no "absent" state once a bit is chosen.

    Fields:
    - uuid: the player's profile UUID, the key the client indexes by
    - name: display name, only written with ADD_PLAYER
    - properties: profile properties (textures), written with ADD_PLAYER
    - game_mode: 0 survival .. 3 spectator
    - listed: whether the player shows in the tab list at all
    - latency: milliseconds, shown as the ping bar
    - hat: whether to render the player's hat layer
    - chat_session: (session_id, expires_at, public_key, signature), only
      written with INITIALIZE_CHAT. Left None on an offline server, where
      there is no key to sign with.
    """

    def __init__(
        self,
        uuid: str,
        name: str = "",
        properties: list | None = None,
        game_mode: int = GAME_MODE_SURVIVAL,
        listed: bool = True,
        latency: int = 0,
        hat: bool = True,
        chat_session: tuple | None = None,
    ):
        self.uuid = uuid
        self.name = name
        self.properties = properties or []
        self.game_mode = game_mode
        self.listed = listed
        self.latency = latency
        self.hat = hat
        self.chat_session = chat_session

    def write(self, buf: ByteBuffer, actions: int) -> None:
        """Write the entry, emitting only the fields `actions` selects."""
        buf.write_uuid(self.uuid)

        if actions & ACTION_ADD_PLAYER:
            buf.write_string(self.name)
            # Property list: each property is a name, a value and an optional
            # signature. Offline there are none, so this is a count of zero.
            buf.write_varint(len(self.properties))
            for prop in self.properties:
                prop_name, prop_value, prop_sig = (list(prop) + [None, None, None])[:3]
                buf.write_string(prop_name)
                buf.write_string(prop_value)
                buf.write_bool(prop_sig is not None)
                if prop_sig is not None:
                    buf.write_string(prop_sig)

        if actions & ACTION_INITIALIZE_CHAT:
            # Not a prefixed optional, unlike most optional-looking fields:
            # choosing this bit commits to writing the whole session. The key
            # and the signature are each a length-prefixed byte array, and
            # length 0 is how "no key" is spelled -- a server with nothing to
            # sign with still has to send the session if it sets this bit.
            if self.chat_session is None:
                raise ValueError(
                    "INITIALIZE_CHAT is set but the entry has no chat session; "
                    "omit the bit instead of sending an empty one"
                )
            session_id, expires_at, public_key, signature = self.chat_session
            buf.write_uuid(session_id)
            buf.write_long(expires_at)
            write_byte_array(buf, public_key)
            write_byte_array(buf, signature)

        if actions & ACTION_UPDATE_GAME_MODE:
            buf.write_varint(self.game_mode)

        if actions & ACTION_UPDATE_LISTED:
            buf.write_bool(self.listed)

        if actions & ACTION_UPDATE_LATENCY:
            buf.write_varint(self.latency)

        if actions & ACTION_UPDATE_DISPLAY_NAME:
            # Text component, absent here: the name from ADD_PLAYER is used.
            buf.write_bool(False)

        if actions & ACTION_UPDATE_LIST_ORDER:
            buf.write_varint(0)

        if actions & ACTION_UPDATE_HAT:
            buf.write_bool(self.hat)


@register_packet(ProtocolState.PLAY, Direction.SERVER_TO_CLIENT, 0x44)
class CPlayerInfoUpdate(ClientBoundPacket):
    """Player Info Update.

    Fields:
    - actions: bitmask of which per-entry fields follow
    - entries: the PlayerInfoEntry list
    """

    name = "Player Info Update"
    packet_id = 0x44

    def __init__(self, entries: list, actions: int = JOIN_ACTIONS):
        super().__init__()
        self.entries = entries
        self.actions = actions

    @classmethod
    def add_player(cls, entry: PlayerInfoEntry) -> "CPlayerInfoUpdate":
        """One entry, with the full set a joining player gets."""
        return cls([entry], JOIN_ACTIONS)

    def write(self, buf: ByteBuffer) -> None:
        # A byte, not a VarInt: the client reads this as an EnumSet.
        buf.write_byte(self.actions)
        buf.write_varint(len(self.entries))
        for entry in self.entries:
            entry.write(buf, self.actions)

    def read(self, buf: ByteBuffer) -> None:
        # Not implementing read for server-side
        pass


@register_packet(ProtocolState.PLAY, Direction.SERVER_TO_CLIENT, 0x43)
class CPlayerInfoRemove(ClientBoundPacket):
    """Player Info Remove.

    Without this, a player who disconnects stays in every other player's tab
    list forever: the client only ever adds entries, and the only thing that
    takes one out is this packet. It is also what keeps a UUID reusable -- a
    player who rejoins under the same UUID is otherwise still resolvable from
    the stale entry, which is harmless for chat but visibly wrong in the tab
    list.

    Fields:
    - uuids: the players to drop from the tab list
    """

    name = "Player Info Remove"
    packet_id = 0x43

    def __init__(self, uuids: list):
        super().__init__()
        self.uuids = uuids

    def write(self, buf: ByteBuffer) -> None:
        buf.write_varint(len(self.uuids))
        for uuid in self.uuids:
            buf.write_uuid(uuid)

    def read(self, buf: ByteBuffer) -> None:
        # Not implementing read for server-side
        pass
