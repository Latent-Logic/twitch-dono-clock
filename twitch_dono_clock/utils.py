import logging
from abc import ABCMeta

from twitchAPI.chat import ChatCommand
from twitchAPI.type import TwitchAPIException, UnauthorizedException

log = logging.getLogger(__name__)


class Singleton(ABCMeta):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        try:
            return cls._instances[cls]
        except KeyError:
            cls._instances[cls] = super().__call__(*args, **kwargs)
            return cls._instances[cls]


class ChatReplies(metaclass=Singleton):
    def __init__(self) -> None:
        self._bot_user_id: str | None = None
        self._app_auth_enabled: bool = True
        self._mode_logged: bool = False

    def disable_app_auth_replies(self) -> None:
        log.warning(f"App auth unavailable; chat replies will use the IRC fallback")
        self._app_auth_enabled = False

    def set_bot_user_id(self, bot_user_id: str) -> None:
        self._bot_user_id = bot_user_id

    async def send_reply(self, cmd: ChatCommand, message: str) -> None:
        """Send message as a reply to cmd, trying first for_source_only w/ fallback to cmd.reply"""
        if self._app_auth_enabled and self._bot_user_id is not None:
            try:
                assert cmd.room is not None
                response = await cmd.chat.twitch.send_chat_message(
                    broadcaster_id=cmd.room.room_id,
                    sender_id=self._bot_user_id,
                    message=message,
                    reply_parent_message_id=cmd.id,
                    for_source_only=True,
                )
                log.debug(f"Message reply got result {response!r}")
                if not self._mode_logged:
                    log.info("Chat replies using App-Auth for_source_only mode (send_chat_message)")
                    self._mode_logged = True
                return
            except UnauthorizedException as err:
                log.warning(
                    f"Chat replies falling back to IRC (App Auth failed: {err!r}); latching for rest of runtime"
                )
                self._app_auth_enabled = False
                self._mode_logged = True
            except (TwitchAPIException, KeyError) as err:
                log.warning(f"Chat app-auth reply failed: {err!r})")
        await cmd.reply(message)
