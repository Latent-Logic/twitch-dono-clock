"""Twitch donothon clock based on reading chat"""
import asyncio
import logging
from datetime.datetime import fromisoformat
from pathlib import Path

import toml
from twitchAPI.chat import Chat, ChatCommand, ChatMessage, ChatSub, EventData
from twitchAPI.oauth import UserAuthenticator
from twitchAPI.twitch import Twitch
from twitchAPI.type import AuthScope, ChatEvent, TwitchAPIException

# "chat:read chat:edit"
USER_SCOPE = [AuthScope.CHAT_READ, AuthScope.CHAT_EDIT]
log = logging.getLogger("test_tracker")
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s\t%(levelname)s\t%(name)s\t%(message)s"
)


# this will be called when the event READY is triggered, which will be on bot start
async def on_ready(ready_event: EventData):
    log.info("Bot is ready for work, joining channels")
    # join our target channel, if you want to join multiple, either call join for each individually
    # or even better pass a list of channels as the argument
    await ready_event.chat.join_room(SETTINGS["twitch"]["channel"])
    # you can do other bot initialization things in here


# this will be called whenever a message in a channel was send by either the bot OR another user
async def on_message(msg: ChatMessage):
    log.info(f"in {msg.room.name}, {msg.user.name} said: {msg.text}")
    log.debug(f"{msg._parsed=}")


# this will be called whenever someone subscribes to a channel
async def on_sub(sub: ChatSub):
    log.info(
        f"New subscription in {sub.room.name}:\\n"
        f"  Type: {sub.sub_plan}\\n"
        f"  Message: {sub.sub_message}"
    )
    log.debug(f"{sub._parsed=}")


# this will be called whenever the !reply command is issued
async def pause_command(cmd: ChatCommand):
    if len(cmd.parameter) == 0:
        await cmd.reply("you did not tell me what to reply with")
    else:
        await cmd.reply(f"{cmd.user.name}: {cmd.parameter}")


# this is where we set up the bot
async def main(settings: dict):
    # set up twitch api instance and add user authentication with some scopes
    twitch = await Twitch(
        settings["twitch"]["app_id"], settings["twitch"]["app_secret"]
    )
    usr_token_file = Path(settings["twitch"]["user_token_file"])
    if usr_token_file.is_file():
        user_auth = toml.load(usr_token_file)
        token, refresh_token = user_auth["token"], user_auth.get("refresh_token")
        if not refresh_token:
            twitch.auto_refresh_auth = False
    else:
        auth = UserAuthenticator(twitch, USER_SCOPE, url=settings["twitch"]["auth_url"])
        token, refresh_token = await auth.authenticate(browser_name="google-chrome")
        usr_token_file.write_text(
            toml.dumps({"token": token, "refresh_token": refresh_token})
        )
    try:
        await twitch.set_user_authentication(token, USER_SCOPE, refresh_token)
    except TwitchAPIException:
        log.error(f"Invalid token, please remove {usr_token_file} try again")
        raise

    # create chat instance
    chat = await Chat(twitch)

    # register the handlers for the events you want

    # listen to when the bot is done starting up and ready to join channels
    chat.register_event(ChatEvent.READY, on_ready)
    # listen to chat messages
    chat.register_event(ChatEvent.MESSAGE, on_message)
    # listen to channel subscriptions
    chat.register_event(ChatEvent.SUB, on_sub)
    # there are more events, you can view them all in this documentation

    # you can directly register commands and their handlers
    chat.register_command("tpause", pause_command)

    # we are done with our setup, lets start this bot up!
    chat.start()

    # lets run till we press enter in the console
    try:
        input("press ENTER to stop\n")
    finally:
        # now we can close the chat bot and the twitch api client
        chat.stop()
        await twitch.close()


if __name__ == "__main__":
    SETTINGS = toml.load("settings.toml")
    START_TIME = fromisoformat(SETTINGS["start"]["time"])

    asyncio.run(main(SETTINGS))
