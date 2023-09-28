"""Twitch donothon clock based on reading chat"""
import asyncio
import csv
import logging
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Optional, Tuple

import toml
from twitchAPI.chat import Chat, ChatCommand, ChatMessage, ChatSub, EventData
from twitchAPI.oauth import UserAuthenticator
from twitchAPI.twitch import Twitch
from twitchAPI.type import AuthScope, ChatEvent, TwitchAPIException

# "chat:read chat:edit"
USER_SCOPE = [AuthScope.CHAT_READ, AuthScope.CHAT_EDIT]
CSV_COLUMNS = ["time", "user", "target", "type", "amount"]
log = logging.getLogger("test_tracker")
logging.basicConfig(level=logging.INFO, format="%(asctime)s\t%(levelname)s\t%(name)s\t%(message)s")


# this will be called when the event READY is triggered, which will be on bot start
async def on_ready(ready_event: EventData):
    log.info("Bot is ready for work, joining channels")
    # join our target channel, if you want to join multiple, either call join for each individually
    # or even better pass a list of channels as the argument
    await ready_event.chat.join_room(SETTINGS["twitch"]["channel"])
    # you can do other bot initialization things in here


# this will be called whenever a message in a channel was send by either the bot OR another user
async def on_message(msg: ChatMessage):
    log.debug(f"{msg.user.name=} {msg._parsed=}")
    if msg.bits:
        log.info(f"in {msg.room.name}, {msg.user.name} sent bits: {msg.bits}")
        LIVE_STATS["donos"]["bits"] += int(msg.bits)
        append_csv(
            Path(SETTINGS["db"]["events"]),
            ts=msg.sent_timestamp,
            user=msg.user.name,
            target=None,
            type="bits",
            amount=msg.bits,
        )
    for user, regex, target in MSG_MAGIC:
        if msg.user.name.lower() == user.lower():
            match = regex.match(msg.text)
            if match:
                log.info(f"in {msg.room.name}, {match['user']} sent {target}: {match['amount']}")
                if target == "bits":
                    LIVE_STATS["donos"]["bits"] += int(match["amount"])
                elif target == "direct":
                    LIVE_STATS["donos"]["direct"] += float(match["amount"])
                append_csv(
                    Path(SETTINGS["db"]["events"]),
                    ts=msg.sent_timestamp,
                    user=match["user"],
                    target=None,
                    type=target,
                    amount=float(match["amount"]),
                )


# this will be called whenever someone subscribes to a channel
async def on_sub(sub: ChatSub):
    log.info(
        f"New subscription in {sub.room.name}:"
        f"\tType: {sub.sub_plan}\\n"
        f'\tFrom: {sub._parsed["tags"]["display-name"]}'
        f'\tTo: {sub._parsed["tags"].get("msg-param-recipient-user-name", sub._parsed["tags"]["display-name"])}'
    )
    log.debug(f"{sub._parsed=}")
    tier = SETTINGS["subs"]["plan"][sub.sub_plan]
    LIVE_STATS["donos"]["subs"][tier] += 1
    append_csv(
        Path(SETTINGS["db"]["events"]),
        ts=sub._parsed["tags"]["tmi-sent-ts"],
        user=sub._parsed["tags"]["display-name"],
        target=sub._parsed["tags"].get("msg-param-recipient-user-name"),
        type=f"subs_{tier}",
        amount=1,
    )


# this will be called whenever the !reply command is issued
async def pause_command(cmd: ChatCommand):
    if not cmd.user.mod:
        log.warning(f"Non-mod user {cmd.user.name} just tried to !tpause")
        return
    if LIVE_STATS["pause_start"] is not None:
        await cmd.reply(f"Pause already started at {LIVE_STATS['pause_start']}")
    else:
        LIVE_STATS["pause_start"] = datetime.now(tz=timezone.utc)
        await cmd.reply("Pause started")


async def resume_command(cmd: ChatCommand):
    if not cmd.user.mod:
        log.warning(f"Non-mod user {cmd.user.name} just tried to !tresume")
        return
    if LIVE_STATS["pause_start"] is None:
        await cmd.reply("Pause not started")
    else:
        LIVE_STATS["pause_start"] = None
        LIVE_STATS["pause_min"] += (datetime.now(tz=timezone.utc) - LIVE_STATS["pause_start"]).total_seconds() / 60
        await cmd.reply(
            f"Pause resumed with an addition of "
            f"{(datetime.now(tz=timezone.utc) - LIVE_STATS['pause_start']).total_seconds() / 60:.02f} "
            f"minutes for a total of {LIVE_STATS['pause_min']:.02f} minutes"
        )
        Path(SETTINGS["db"]["pause"]).write_text(f"{LIVE_STATS['pause_min']:.02f}")


def load_pause(file_path: Path):
    if not file_path.is_file():
        log.warning(f"No pause file found at {file_path}, creating one")
        file_path.parent.mkdir(exist_ok=True, parents=True)
        file_path.write_text("0.0")
        return
    LIVE_STATS["pause_min"] = float(file_path.read_text())


# Load CSV log file for refreshing stats
def load_csv(file_path: Path):
    if not file_path.is_file():
        log.warning(f"No CSV file found at {file_path}, creating one")
        file_path.parent.mkdir(exist_ok=True, parents=True)
        file_path.write_text(",".join(CSV_COLUMNS))
        return
    with file_path.open("r") as f:
        reader = csv.DictReader(f, delimiter=",")
        assert reader.fieldnames == CSV_COLUMNS
        for row in reader:
            if row["type"] == "bits":
                LIVE_STATS["donos"]["bits"] += int(row["amount"])
            elif row["type"] == "direct":
                LIVE_STATS["donos"]["direct"] += float(row["amount"])
            elif row["type"].startswith("subs_"):
                if row["type"].endswith("_t1"):
                    LIVE_STATS["donos"]["subs"]["t1"] += int(row["amount"])
                elif row["type"].endswith("_t2"):
                    LIVE_STATS["donos"]["subs"]["t2"] += int(row["amount"])
                elif row["type"].endswith("_t3"):
                    LIVE_STATS["donos"]["subs"]["t3"] += int(row["amount"])
    log.info(f"Loaded CSV file and got: {LIVE_STATS}")


def append_csv(file_path: Path, ts: int, user: str, type: str, amount: float, target: Optional[str] = None):
    if not file_path.is_file():
        raise FileNotFoundError(f"No CSV file found at {file_path}, Should have been created earlier?!?")
    with file_path.open("a") as f:
        csv.DictWriter(f, CSV_COLUMNS).writerow(
            {"time": ts, "user": user, "target": target or "", "type": type, "amount": amount}
        )


def cal_minutes() -> float:
    minutes = 0
    donos = LIVE_STATS["donos"]
    minutes += donos["bits"] * SETTINGS["bits"]["min"]
    minutes += donos["direct"] * SETTINGS["direct"]["min"]
    subs = donos["subs"]
    minutes += subs["t1"] * SETTINGS["subs"]["tier"]["t1"]["min"]
    minutes += subs["t2"] * SETTINGS["subs"]["tier"]["t2"]["min"]
    minutes += subs["t3"] * SETTINGS["subs"]["tier"]["t3"]["min"]
    return minutes


def calc_dollars() -> float:
    dollars = 0
    donos = LIVE_STATS["donos"]
    dollars += donos["bits"] * SETTINGS["bits"]["money"]
    dollars += donos["direct"] * SETTINGS["direct"]["money"]
    subs = donos["subs"]
    dollars += subs["t1"] * SETTINGS["subs"]["tier"]["t1"]["money"]
    dollars += subs["t2"] * SETTINGS["subs"]["tier"]["t2"]["money"]
    dollars += subs["t3"] * SETTINGS["subs"]["tier"]["t3"]["money"]
    return dollars


def calc_timer() -> str:
    global LIVE_STATS
    if LIVE_STATS["pause_start"] is not None:
        cur_time: datetime = LIVE_STATS["pause_start"]
    else:
        cur_time = datetime.now(tz=timezone.utc)
    time_so_far = cur_time - START_TIME
    corrected_tsf = time_so_far - timedelta(minutes=LIVE_STATS["pause_min"])
    accrued_time = timedelta(minutes=cal_minutes())
    remaining = accrued_time - corrected_tsf
    hours = int(remaining.total_seconds() / 60 / 60)
    minutes = int(remaining.total_seconds() / 60) % 60
    seconds = int(remaining.total_seconds()) % 60
    time_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    if LIVE_STATS["pause_start"] is not None:
        return f"PAUSED: {time_str}"
    else:
        return time_str


def write_files():
    out_dict = SETTINGS["output"]
    out_dir = Path(out_dict["dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / out_dict["bits"]).write_text(f'{LIVE_STATS["donos"]["bits"]}')
    (out_dir / out_dict["direct"]).write_text(f'{LIVE_STATS["donos"]["direct"]:.02f}')
    (out_dir / out_dict["subs"]).write_text(
        str(LIVE_STATS["donos"]["subs"]["t1"] + LIVE_STATS["donos"]["subs"]["t2"] + LIVE_STATS["donos"]["subs"]["t3"])
    )
    (out_dir / out_dict["countdown"]).write_text(calc_timer())
    (out_dir / out_dict["money"]).write_text(f"{calc_dollars():0.02f}")


async def write_every_second():
    while True:
        await asyncio.sleep(1)
        write_files()


# this is where we set up the bot
async def main(settings: dict):
    # set up twitch api instance and add user authentication with some scopes
    twitch = await Twitch(settings["twitch"]["app_id"], settings["twitch"]["app_secret"])
    usr_token_file = Path(settings["twitch"]["user_token_file"])
    if usr_token_file.is_file():
        user_auth = toml.load(usr_token_file)
        token, refresh_token = user_auth["token"], user_auth.get("refresh_token")
        if not refresh_token:
            twitch.auto_refresh_auth = False
    else:
        auth = UserAuthenticator(twitch, USER_SCOPE, url=settings["twitch"]["auth_url"])
        token, refresh_token = await auth.authenticate(browser_name="google-chrome")
        usr_token_file.write_text(toml.dumps({"token": token, "refresh_token": refresh_token}))
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
    chat.register_command("tresume", resume_command)

    # we are done with our setup, lets start this bot up!
    chat.start()

    try:
        try:
            await write_every_second()
        except (KeyboardInterrupt, asyncio.CancelledError):
            pass
    finally:
        # now we can close the chat bot and the twitch api client
        chat.stop()
        await twitch.close()


def regex_compile(settings: dict) -> List[Tuple[str, re.Pattern, str]]:
    msg_magic = []
    for user, obj in settings["bits"].get("msg", {}).items():
        msg_magic.append((user, re.compile(obj["regex"]), "bits"))
    for user, obj in settings["direct"].get("msg", {}).items():
        msg_magic.append((user, re.compile(obj["regex"]), "direct"))
    return msg_magic


if __name__ == "__main__":
    SETTINGS = toml.load("settings.toml")
    START_TIME = datetime.fromisoformat(SETTINGS["start"]["time"])
    LIVE_STATS = {
        "pause_min": 0.0,
        "pause_start": None,
        "donos": {
            "bits": 0,
            "subs": {"t1": 0, "t2": 0, "t3": 0},
            "direct": 0,
        },
    }
    load_pause(Path(SETTINGS["db"]["pause"]))
    MSG_MAGIC = regex_compile(SETTINGS)
    log.info(f"{MSG_MAGIC}")
    load_csv(Path(SETTINGS["db"]["events"]))

    asyncio.run(main(SETTINGS))
