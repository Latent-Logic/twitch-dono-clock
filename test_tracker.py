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

LIVE_STATS = {
    "pause_min": 0.0,
    "pause_start": None,  # type: Optional[datetime]
    "donos": {
        "bits": 0,
        "subs": {"t1": 0, "t2": 0, "t3": 0},
        "tips": 0,
    },
    "end": {},
}


def config_logging(level: str = "INFO"):
    stream_handler = logging.StreamHandler()
    file_handler = logging.FileHandler(datetime.now().strftime("%Y.%m.%d-%H.%M.%S.log"))
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s\t%(levelname)s\t%(name)s\t%(message)s",
        handlers=[stream_handler, file_handler],
    )
    stream_handler.setLevel(level=level)
    chat_logger = logging.getLogger("twitchAPI.chat")

    def oauth_filter(record):
        if record.msg.startswith('> "PASS oauth:'):
            return False
        return True

    chat_logger.addFilter(oauth_filter)


config_logging()


# this will be called when the event READY is triggered, which will be on bot start
async def on_ready(ready_event: EventData):
    log.info(f"Bot is ready for work, joining channel {SETTINGS['twitch']['channel']}")
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
            user=msg.user.display_name,
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
                    amount = int(match["amount"])
                elif target == "tips":
                    amount = float(match["amount"])
                LIVE_STATS["donos"][target] += amount
                append_csv(
                    Path(SETTINGS["db"]["events"]),
                    ts=msg.sent_timestamp,
                    user=match["user"],
                    target=None,
                    type=target,
                    amount=amount,
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
        target=sub._parsed["tags"].get("msg-param-recipient-display-name"),
        type=f"subs_{tier}",
        amount=1,
    )


def save_pause_file():
    pause_min, pause_start = LIVE_STATS["pause_min"], LIVE_STATS["pause_start"]
    if pause_start:
        Path(SETTINGS["db"]["pause"]).write_text(f"{pause_min:.02f};{pause_start.isoformat()}")
    else:
        Path(SETTINGS["db"]["pause"]).write_text(f"{LIVE_STATS['pause_min']:.02f}")


# this will be called whenever the !reply command is issued
async def pause_command(cmd: ChatCommand):
    fmt_dict = {
        "user": cmd.user.name,
        "cmd": "tpause",
        "pause_min": LIVE_STATS["pause_min"],
        "pause_start": LIVE_STATS["pause_start"],
        "end_ts": LIVE_STATS["end"].get("end_ts"),
        "end_min": LIVE_STATS["end"].get("end_min"),
    }
    if not (cmd.user.mod or cmd.user.name.lower() == SETTINGS["twitch"]["channel"].lower()):
        log.warning(SETTINGS["fmt"]["cmd_blocked"].format(**fmt_dict))
        return
    elif LIVE_STATS["end"]:
        await cmd.reply(SETTINGS["fmt"]["cmd_after_end"].format(**fmt_dict))
        return
    if LIVE_STATS["pause_start"] is not None:
        await cmd.reply(SETTINGS["fmt"]["tpause_failure"].format(**fmt_dict))
    else:
        pause_start = datetime.now(tz=timezone.utc)
        LIVE_STATS["pause_start"] = pause_start
        fmt_dict["pause_start"] = pause_start
        save_pause_file()
        with open(SETTINGS["db"]["pause_log"], "a") as f:
            f.write(f"{pause_start.isoformat()}\t{LIVE_STATS['pause_min']:.2f}\tPause Started\n")
        log.info(SETTINGS["fmt"]["tpause_success"].format(**fmt_dict))
        await cmd.reply(SETTINGS["fmt"]["tpause_success"].format(**fmt_dict))


async def resume_command(cmd: ChatCommand):
    fmt_dict = {
        "user": cmd.user.name,
        "cmd": "tresume",
        "pause_min": LIVE_STATS["pause_min"],
        "pause_start": LIVE_STATS["pause_start"],
    }
    if not (cmd.user.mod or cmd.user.name.lower() == SETTINGS["twitch"]["channel"].lower()):
        log.warning(SETTINGS["fmt"]["cmd_blocked"].format(**fmt_dict))
        return
    if LIVE_STATS["pause_start"] is None:
        await cmd.reply(SETTINGS["fmt"]["tresume_failure"].format(**fmt_dict))
    else:
        now = datetime.now(tz=timezone.utc)
        added_min = (now - LIVE_STATS["pause_start"]).total_seconds() / 60
        LIVE_STATS["pause_min"] += added_min
        fmt_dict["pause_min"] = LIVE_STATS["pause_min"]
        fmt_dict["added_min"] = added_min
        fmt_dict["pause_start"] = None
        LIVE_STATS["pause_start"] = None
        save_pause_file()
        with open(SETTINGS["db"]["pause_log"], "a") as f:
            f.write(f"{now.isoformat()}\t{LIVE_STATS['pause_min']:.2f}\tPause Ended & added {added_min:.2f}\n")
        log.info(SETTINGS["fmt"]["tresume_success"].format(**fmt_dict))
        await cmd.reply(SETTINGS["fmt"]["tresume_success"].format(**fmt_dict))


async def parse_time_from_cmd(cmd: ChatCommand, cmd_name: str):
    fmt_dict = {
        "user": cmd.user.name,
        "cmd": cmd_name,
        "pause_min": LIVE_STATS["pause_min"],
        "pause_start": LIVE_STATS["pause_start"],
        "end_ts": LIVE_STATS["end"].get("end_ts"),
        "end_min": LIVE_STATS["end"].get("end_min"),
    }
    if not (cmd.user.mod or cmd.user.name.lower() == SETTINGS["twitch"]["channel"].lower()):
        log.warning(SETTINGS["fmt"]["cmd_blocked"].format(**fmt_dict))
        return
    elif LIVE_STATS["end"]:
        await cmd.reply(SETTINGS["fmt"]["cmd_after_end"].format(**fmt_dict))
        return
    try:
        raw = cmd.parameter.split()[0]
        if raw.endswith("h") or raw.endswith("hr") or raw.endswith("hrs"):
            num_str = raw[: raw.find("h")]
            mins = float(num_str) * 60
        elif raw.endswith("m") or raw.endswith("min") or raw.endswith("mins"):
            num_str = raw[: raw.find("m")]
            mins = float(num_str)
        elif raw.endswith("s") or raw.endswith("sec") or raw.endswith("secs"):
            num_str = raw[: raw.find("s")]
            mins = float(num_str) / 60
        else:
            raise ValueError(f"Didn't recognize {raw}. Type a number followed by s, m, or h")
        if mins <= 0:
            raise ValueError(f"Only use positive numbers with this command")
    except IndexError as err:
        fmt_dict |= {"err": str(err), "err_type": str(type(err))}
        log.error(cmd.reply(SETTINGS["fmt"]["missing_time_parameter_failure"].format(**fmt_dict)))
        await cmd.reply(SETTINGS["fmt"]["missing_time_parameter_failure"].format(**fmt_dict))
        raise
    except ValueError as err:
        fmt_dict |= {"err": str(err), "err_type": str(type(err))}
        log.error(SETTINGS["fmt"]["invalid_time_parameter_failure"].format(**fmt_dict))
        await cmd.reply(SETTINGS["fmt"]["invalid_time_parameter_failure"].format(**fmt_dict))
        raise
    return mins


async def add_time_command(cmd: ChatCommand):
    try:
        minutes = await parse_time_from_cmd(cmd, "tadd")
    except (IndexError, ValueError):
        return
    fmt_dict = {
        "user": cmd.user.name,
        "cmd": "tadd",
        "pause_min": LIVE_STATS["pause_min"],
        "pause_start": LIVE_STATS["pause_start"],
        "pause_delta": minutes,
    }
    LIVE_STATS["pause_min"] += minutes
    save_pause_file()
    now = datetime.now(tz=timezone.utc)
    with open(SETTINGS["db"]["pause_log"], "a") as f:
        f.write(f"{now.isoformat()}\t{LIVE_STATS['pause_min']:.2f}\tPause time increased by !tadd {minutes}m\n")
    log.info(SETTINGS["fmt"]["tadd_success"].format(**fmt_dict))
    await cmd.reply(SETTINGS["fmt"]["tadd_success"].format(**fmt_dict))


async def remove_time_command(cmd: ChatCommand):
    try:
        minutes = await parse_time_from_cmd(cmd, "tremove")
    except (IndexError, ValueError):
        return
    fmt_dict = {
        "user": cmd.user.name,
        "cmd": "tremove",
        "pause_min": LIVE_STATS["pause_min"],
        "pause_start": LIVE_STATS["pause_start"],
        "pause_delta": minutes,
    }
    LIVE_STATS["pause_min"] -= minutes
    save_pause_file()
    now = datetime.now(tz=timezone.utc)
    with open(SETTINGS["db"]["pause_log"], "a") as f:
        f.write(f"{now.isoformat()}\t{LIVE_STATS['pause_min']:.2f}\tPause time reduced by !tremove {minutes}m\n")
    log.info(SETTINGS["fmt"]["tremove_success"].format(**fmt_dict))
    await cmd.reply(SETTINGS["fmt"]["tremove_success"].format(**fmt_dict))


async def raised_command(cmd: ChatCommand):
    fmt_dict = {
        "user": cmd.user.name,
        "cmd": "traised",
    }
    if not (cmd.user.mod or cmd.user.name.lower() == SETTINGS["twitch"]["channel"].lower()):
        log.warning(SETTINGS["fmt"]["cmd_blocked"].format(**fmt_dict))
        return
    so_far_total_min = calc_time_so_far().total_seconds() / 60
    fmt_dict = {
        **fmt_dict,
        "so_far_total_min": so_far_total_min,
        "so_far_hrs": so_far_total_min // 60,
        "so_far_min": so_far_total_min % 60,
        "min_paid_for": calc_chat_minutes(),
        "min_end_at": calc_end().total_seconds() / 60,
        "end_ts": LIVE_STATS["end"].get("end_ts"),
        "end_min": LIVE_STATS["end"].get("end_min"),
        "total_value": calc_dollars(),
        "countdown": calc_timer(),
        "bits": LIVE_STATS["donos"]["bits"],
        "tips": LIVE_STATS["donos"]["tips"],
        "subs": sum(LIVE_STATS["donos"]["subs"].values()),
        "subs_t1": LIVE_STATS["donos"]["subs"]["t1"],
        "subs_t2": LIVE_STATS["donos"]["subs"]["t2"],
        "subs_t3": LIVE_STATS["donos"]["subs"]["t3"],
        "pause_min": LIVE_STATS["pause_min"],
        "pause_start": LIVE_STATS["pause_start"] or "Not Currently Paused",
    }
    log.info(SETTINGS["fmt"]["traised_success"].format(**fmt_dict))
    await cmd.reply(SETTINGS["fmt"]["traised_success"].format(**fmt_dict))


def load_pause(file_path: Path):
    if not file_path.is_file():
        log.warning(f"No pause file found at {file_path}, creating one")
        file_path.parent.mkdir(exist_ok=True, parents=True)
        file_path.write_text("0.0")
        return
    raw = file_path.read_text()
    if ";" in raw:
        time, pause_time = raw.strip().split(";", maxsplit=1)
        LIVE_STATS["pause_start"] = datetime.fromisoformat(pause_time)
    else:
        time = raw
    LIVE_STATS["pause_min"] = float(time)
    log.debug(f"Loaded Pause file and got {LIVE_STATS['pause_min']=} {LIVE_STATS['pause_start']=}")


# Load CSV log file for refreshing stats
def load_csv(file_path: Path):
    if not file_path.is_file():
        log.warning(f"No CSV file found at {file_path}, creating one")
        file_path.parent.mkdir(exist_ok=True, parents=True)
        file_path.write_text(",".join(CSV_COLUMNS))
        return
    with file_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=",")
        assert reader.fieldnames == CSV_COLUMNS
        for row in reader:
            if row["type"] == "bits":
                LIVE_STATS["donos"]["bits"] += int(row["amount"])
            elif row["type"] in {"direct", "tips"}:
                LIVE_STATS["donos"]["tips"] += float(row["amount"])
            elif row["type"].startswith("subs_"):
                if row["type"].endswith("_t1"):
                    LIVE_STATS["donos"]["subs"]["t1"] += int(row["amount"])
                elif row["type"].endswith("_t2"):
                    LIVE_STATS["donos"]["subs"]["t2"] += int(row["amount"])
                elif row["type"].endswith("_t3"):
                    LIVE_STATS["donos"]["subs"]["t3"] += int(row["amount"])
    log.debug(f"Loaded CSV file and got: {LIVE_STATS['donos']=}")


def append_csv(file_path: Path, ts: int, user: str, type: str, amount: float, target: Optional[str] = None):
    if not file_path.is_file():
        raise FileNotFoundError(f"No CSV file found at {file_path}, Should have been created earlier?!?")
    with file_path.open("a", encoding="utf-8") as f:
        csv.DictWriter(f, CSV_COLUMNS, lineterminator="\n").writerow(
            {"time": ts, "user": user, "target": target or "", "type": type, "amount": amount}
        )


def calc_chat_minutes() -> float:
    """Total number of minutes paid for by chat"""
    minutes = 0
    donos = LIVE_STATS["donos"]
    minutes += donos["bits"] * SETTINGS["bits"]["min"]
    minutes += donos["tips"] * SETTINGS["tips"]["min"]
    subs = donos["subs"]
    minutes += subs["t1"] * SETTINGS["subs"]["tier"]["t1"]["min"]
    minutes += subs["t2"] * SETTINGS["subs"]["tier"]["t2"]["min"]
    minutes += subs["t3"] * SETTINGS["subs"]["tier"]["t3"]["min"]
    return minutes


def calc_end() -> timedelta:
    """Find the timedelta to use for final calculations"""
    if LIVE_STATS["end"].get("end_min"):
        return timedelta(minutes=LIVE_STATS["end"]["end_min"])
    minutes = calc_chat_minutes()
    if SETTINGS["end"].get("max_minutes"):
        minutes = min(minutes, SETTINGS["end"]["max_minutes"])
    return timedelta(minutes=minutes)


def calc_minutes_over() -> float:
    """How many minutes over the final calculation we are"""
    if SETTINGS["end"].get("max_minutes"):
        return calc_chat_minutes() - SETTINGS["end"]["max_minutes"]
    else:
        return 0.0


def calc_dollars() -> float:
    """Total financial gain from chat donations"""
    dollars = 0
    donos = LIVE_STATS["donos"]
    dollars += donos["bits"] * SETTINGS["bits"]["money"]
    dollars += donos["tips"] * SETTINGS["tips"]["money"]
    subs = donos["subs"]
    dollars += subs["t1"] * SETTINGS["subs"]["tier"]["t1"]["money"]
    dollars += subs["t2"] * SETTINGS["subs"]["tier"]["t2"]["money"]
    dollars += subs["t3"] * SETTINGS["subs"]["tier"]["t3"]["money"]
    return dollars


def calc_time_so_far() -> timedelta:
    """How much time has been counted down since the start"""
    if LIVE_STATS["end"].get("end_ts"):
        cur_time = LIVE_STATS["end"]["end_ts"]
    elif LIVE_STATS["pause_start"] is not None:
        cur_time: datetime = LIVE_STATS["pause_start"]
    else:
        cur_time = datetime.now(tz=timezone.utc)
    time_so_far = cur_time - SETTINGS["start"]["time"]
    corrected_tsf = time_so_far - timedelta(minutes=LIVE_STATS["pause_min"])
    return corrected_tsf


def calc_timer() -> str:
    """Generate the timer string from the difference between paid and run minutes"""
    remaining = calc_end() - calc_time_so_far()
    hours = int(remaining.total_seconds() / 60 / 60)
    minutes = int(remaining.total_seconds() / 60) % 60
    seconds = int(remaining.total_seconds()) % 60
    time_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    if LIVE_STATS["pause_start"] is not None:
        pause_format = SETTINGS["fmt"]["countdown_pause"]
        return pause_format.format(clock=time_str)
    else:
        return time_str


def handle_end(initial_run: bool = False):
    if initial_run and not LIVE_STATS["end"]:
        end_file = Path(SETTINGS["db"]["end_mark"])
        if end_file.is_file():
            LIVE_STATS["end"] = toml.load(end_file)
            assert "end_min" in LIVE_STATS["end"] and "end_ts" in LIVE_STATS["end"]
            log.debug(f"Loaded end marker file and got {LIVE_STATS['end']=}")
    if LIVE_STATS["end"]:
        return  # We've already reached an end state, no need for further calculations
    time_so_far = calc_time_so_far()
    available_time = calc_end()
    if time_so_far < available_time:
        return  # We've not reached our ending time yet everyone still run normally
    now = datetime.now(tz=timezone.utc)
    end_time = now - (time_so_far - available_time)
    LIVE_STATS["end"] = {
        "end_min": available_time.total_seconds() / 60,
        "end_ts": end_time,
        "ended_at_max": calc_chat_minutes() >= SETTINGS["end"].get("max_minutes", 0),
    }
    Path(SETTINGS["db"]["end_mark"]).write_text(toml.dumps(LIVE_STATS["end"]))


def write_files():
    out_dict = SETTINGS["output"]
    out_dir = Path(out_dict["dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    handle_end()
    (out_dir / out_dict["bits"]).write_text(f'{LIVE_STATS["donos"]["bits"]}')
    (out_dir / out_dict["tips"]).write_text(f'{LIVE_STATS["donos"]["tips"]:.02f}')
    (out_dir / out_dict["subs"]).write_text(
        str(LIVE_STATS["donos"]["subs"]["t1"] + LIVE_STATS["donos"]["subs"]["t2"] + LIVE_STATS["donos"]["subs"]["t3"])
    )
    (out_dir / out_dict["countdown"]).write_text(calc_timer())
    (out_dir / out_dict["total_value"]).write_text(f"{calc_dollars():0.02f}")


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
    if settings["twitch"].get("enable_cmds", True):
        chat.register_command("tpause", pause_command)
        chat.register_command("tresume", resume_command)
        chat.register_command("tadd", add_time_command)
        chat.register_command("tremove", remove_time_command)
        chat.register_command("traised", raised_command)

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
    for user, obj in settings["tips"].get("msg", {}).items():
        msg_magic.append((user, re.compile(obj["regex"]), "tips"))
    log.debug(f"Loaded regex matches: {msg_magic}")
    return msg_magic


if __name__ == "__main__":
    SETTINGS = toml.load("settings.toml")
    if isinstance(SETTINGS["start"]["time"], str):
        SETTINGS["start"]["time"] = datetime.fromisoformat(SETTINGS["start"]["time"])
    load_pause(Path(SETTINGS["db"]["pause"]))
    MSG_MAGIC = regex_compile(SETTINGS)
    load_csv(Path(SETTINGS["db"]["events"]))
    handle_end(initial_run=True)
    log.info(f"Finished loading files and got {LIVE_STATS=}")

    asyncio.run(main(SETTINGS))
