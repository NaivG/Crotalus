"""
Player Chat Message (Clientbound, Play)

A line a player actually typed. System Chat (0x77) carries no sender at all,
so the client renders it as an anonymous server notice. This packet carries
the sender's UUID and display name and is decorated on the client from the
minecraft:chat_type registry, which is what produces the "<name> message"
line -- with the sender's hover card and the reply affordances -- rather
than a bare server line.

The 1.19+ signing scheme is bypassed the way a vanilla offline-mode server
bypasses it. Message Signature is a *prefixed optional* field, so carrying no
signature is a normal state on the wire rather than a failed verification, and
the last-seen-signature chain is sent empty. The client marks the line as
unverified, which is expected on a server with no player public keys to verify
against, not an error.

The layout is transcribed from the 1.21.11 client itself, whose record has
eight components -- globalIndex, sender, index, signature, body,
unsignedContent, filterMask, chatType -- and not from the wiki, which still
documents the older thirteen-field shape. The two agree on the first four and
on the four sub-fields of `body`, which is what makes the difference so easy to
miss: a packet built to the documented shape decodes cleanly for most of its
length and only fails once a byte is missing somewhere no error points at.
"""

import random
import time
from typing import Optional

from network.buffer import ByteBuffer
from network.packet import ClientBoundPacket, register_packet
from network.protocol import ProtocolState, Direction
from datatypes import write_nbt

# The all-zero profile. Doubles as the fallback when a login arrived without
# a UUID, which the client always displays regardless of the disableChat
# launch option.
NIL_UUID = "00000000-0000-0000-0000-000000000000"

# minecraft:chat_type registry index of "chat", whose decoration is
# "<sender> content". Crotalus syncs that registry in this order, so 0 is the
# ordinary player-to-player line. This is a *registry* index; the wire value is
# one higher, see INLINE_CHAT_TYPE.
CHAT_TYPE_CHAT = 0

# Chat Type is an "ID or inline" field: a zero means an inline definition
# follows on the wire, and any other value is a registry index offset by one.
# Writing 0 for registry entry 0 therefore does not select minecraft:chat -- it
# asks the client to parse the rest of the packet as an inline chat type, and
# the decode dies at the end of the buffer.
INLINE_CHAT_TYPE = 0


@register_packet(ProtocolState.PLAY, Direction.SERVER_TO_CLIENT, 0x3F)
class CPlayerChatMessage(ClientBoundPacket):
    """Player Chat Message.

    Fields:
    - global_index: counter of the messages sent to *this* client alone,
      starting at 0. Every recipient needs its own packet instance.
    - sender_uuid: sender's profile UUID
    - index: counter of the messages sent by *this sender* alone, starting
      at 0, which is what keeps their messages in order
    - sender_name: display name, used as the `sender` parameter of chat_type
    - message: the text, used as the `content` parameter of chat_type
    - chat_type: minecraft:chat_type *registry index*
    """

    name = "Player Chat Message"
    packet_id = 0x3F

    def __init__(
        self,
        sender_uuid: Optional[str] = None,
        sender_name: str = "",
        message: str = "",
        index: int = 0,
        global_index: int = 0,
        chat_type: int = CHAT_TYPE_CHAT,
    ):
        super().__init__()
        self.sender_uuid = sender_uuid or NIL_UUID
        self.sender_name = sender_name
        self.message = message
        self.index = index
        self.global_index = global_index
        self.chat_type = chat_type

        # These belong to the signed body -- the plaintext a signature would
        # have covered. Nothing on an offline server ever checks them, but the
        # timestamp is compared against the client's own clock, so it has to be
        # real or the line is treated as stale.
        self.timestamp = int(time.time() * 1000)
        self.salt = random.getrandbits(63)

    def write(self, buf: ByteBuffer) -> None:
        buf.write_varint(self.global_index)
        buf.write_uuid(self.sender_uuid)
        buf.write_varint(self.index)

        # Message Signature: absent, so the line reads as unverified rather
        # than as a signature that failed to check out.
        buf.write_bool(False)

        # Signed body: the plaintext, when it was composed, its salt, and the
        # chain of signatures the sender claims to have seen. Only signatures
        # ever fill the last one, and offline there are none.
        buf.write_string(self.message)
        buf.write_long(self.timestamp)
        buf.write_long(self.salt)
        buf.write_varint(0)

        # Unsigned Content: present, because nothing was signed. This is the
        # component the client actually renders, decorated from the chat type
        # -- which is what keeps the sender's name hoverable and the line
        # quotable. It is a *prefixed optional*: the boolean in front of it is
        # a real field, and omitting it shifts every byte after it.
        buf.write_bool(True)
        buf.write(write_nbt({"text": self.message}))

        # Message Filter: 0 = pass-through, so no BitSet follows.
        buf.write_varint(0)

        # Chat Type: registry index + 1, because 0 is the inline sentinel.
        buf.write_varint(self.chat_type + 1)

        # Sender Name, then Target Name: absent, this is not a whisper.
        buf.write(write_nbt({"text": self.sender_name}))
        buf.write_bool(False)

    def read(self, buf: ByteBuffer) -> None:
        # Not implementing read for server-side
        pass
