"""Twitch donothon clock based on reading chat"""
import asyncio
import json
import logging
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Literal, Optional
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import toml
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.encoders import jsonable_encoder
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse
from twitchAPI.chat import Chat, ChatCommand, ChatMessage, ChatSub, EventData
from twitchAPI.eventsub.websocket import EventSubWebsocket
from twitchAPI.helper import first
from twitchAPI.oauth import UserAuthenticator
from twitchAPI.twitch import Twitch
from twitchAPI.type import AuthScope, ChatEvent, TwitchAPIException
from websockets import ConnectionClosedOK

from twitch_dono_clock.config import SETTINGS
from twitch_dono_clock.donos import BITS, CSV_COLUMNS, CSV_TYPES, TIPS, Donos
from twitch_dono_clock.end import End
from twitch_dono_clock.pause import Pause, PauseException
from twitch_dono_clock.spins import Spins

# "chat:read chat:edit"
USER_SCOPE = [AuthScope.CHAT_READ, AuthScope.CHAT_EDIT]

log = logging.getLogger("test_tracker")


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
    log.info(f"Bot is ready for work, joining channel {SETTINGS.twitch.channel}")
    # join our target channel, if you want to join multiple, either call join for each individually
    # or even better pass a list of channels as the argument
    await ready_event.chat.join_room(SETTINGS.twitch.channel)
    # you can do other bot initialization things in here


# this will be called whenever a message in a channel was send by either the bot OR another user
async def on_message(msg: ChatMessage):
    log.debug(f"{msg.user.name=} {msg._parsed=}")
    if msg.bits:
        log.info(f"in {msg.room.name}, {msg.user.name} sent bits: {msg.bits}")
        Donos().add_event(
            ts=msg.sent_timestamp,
            user=msg.user.display_name,
            target=None,
            type=BITS,
            amount=int(msg.bits),
        )
    for user, regex, target in SETTINGS.compiled_re:
        if msg.user.name.lower() == user.lower():
            match = regex.match(msg.text)
            if match:
                log.info(f"in {msg.room.name}, {match['user']} sent {target}: {match['amount']}")
                if target == BITS:
                    amount = int(match["amount"])
                elif target == TIPS:
                    amount = float(match["amount"])
                else:
                    raise ValueError(f"Unknown target from msg parsing {target}")
                Donos().add_event(
                    ts=msg.sent_timestamp,
                    user=match["user"],
                    target=None,
                    type=target,
                    amount=amount,
                )


# this will be called whenever someone subscribes to a channel
async def on_sub(sub: ChatSub):
    log_msg = (
        f"New subscription in {sub.room.name}:"
        f"\tType: {sub.sub_plan}"
        f'\tFrom: {sub._parsed["tags"]["display-name"]}'
        f'\tTo: {sub._parsed["tags"].get("msg-param-recipient-user-name", sub._parsed["tags"]["display-name"])}'
    )
    if SETTINGS.subs.count_multimonth:
        months = int(sub._parsed["tags"].get("msg-param-multimonth-duration", 0))
        if not months and SETTINGS.subs.count_multimonth_gift:
            months = int(sub._parsed["tags"].get("msg-param-gift-months", 0))
        if not months:
            months = 1
        log_msg += f"\t Months: {months}"
    else:
        months = 1
    log.info(log_msg)
    log.debug(f"{sub._parsed=}")
    Donos().add_event(
        ts=sub._parsed["tags"]["tmi-sent-ts"],
        user=sub._parsed["tags"]["display-name"],
        target=sub._parsed["tags"].get("msg-param-recipient-display-name"),
        type=Donos.sub_from_twitch_plan(sub.sub_plan),
        amount=months,
    )


async def on_raid(raid: dict):
    raider = raid.get("tags", {}).get("msg-param-displayName", "Unknown")
    raider_count = int(raid.get("tags", {}).get("msg-param-viewerCount", -1))
    _raid_timestamp = raid.get("tags", {}).get("tmi-sent-ts")
    log.info(f"RAID from {raider} with {raider_count} raiders just happened!")
    log.debug(f"{raid}")


async def spin_done_command(cmd: ChatCommand):
    fmt_dict = {
        "user": cmd.user.name,
        "cmd": "tspin",
        "spins_done": Spins().performed,
        "old_spins_done": Spins().performed,
        "spins_to_do": int(Spins().calc_todo(Donos().calc_dollars())),
    }
    if not (cmd.user.mod or cmd.user.name.lower() in SETTINGS.twitch.admin_users):
        log.warning(SETTINGS.fmt.cmd_blocked.format(**fmt_dict))
        return
    parameters = cmd.parameter.split()
    if not parameters:
        Spins().spin_performed()
        fmt_dict["spins_done"] = Spins().performed
        response = "Spin counter increased by 1, now {spins_done}/{spins_to_do}".format(**fmt_dict)
    elif len(parameters) == 1:
        if parameters[0].lower() == "check":
            response = "Spin counter is at {spins_done}/{spins_to_do}".format(**fmt_dict)
            log.info(response)
            await cmd.reply(response)
            return
        try:
            Spins().set_performed(int(parameters[0]))
        except ValueError:
            await cmd.reply(f"The new total `{parameters[0]}` must be parsable as an integer")
            return
        fmt_dict["spins_done"] = Spins().performed
        response = "Spin counter set from {old_spins_done} to {spins_done} out of {spins_to_do}".format(**fmt_dict)
    else:
        await cmd.reply("Command format !{cmd} (check|<new_total>)".format(**fmt_dict))
        return
    log.info(response)
    await cmd.reply(response)


async def pause_command(cmd: ChatCommand):
    fmt_dict = {
        "user": cmd.user.name,
        "cmd": "tpause",
        "pause_min": Pause().minutes,
        "pause_start": Pause().start,
        "end_ts": End().end_ts,
        "end_min": End().end_min,
    }
    if not (cmd.user.mod or cmd.user.name.lower() in SETTINGS.twitch.admin_users):
        log.warning(SETTINGS.fmt.cmd_blocked.format(**fmt_dict))
        return
    elif End().is_ended():
        await cmd.reply(SETTINGS.fmt.cmd_after_end.format(**fmt_dict))
        return
    if Pause().is_paused():
        await cmd.reply(SETTINGS.fmt.tpause_failure.format(**fmt_dict))
    else:
        Pause().start_pause("!{cmd}".format(**fmt_dict))
        fmt_dict["pause_start"] = Pause().start
        log.info(SETTINGS.fmt.tpause_success.format(**fmt_dict))
        await cmd.reply(SETTINGS.fmt.tpause_success.format(**fmt_dict))


async def resume_command(cmd: ChatCommand):
    fmt_dict = {
        "user": cmd.user.name,
        "cmd": "tresume",
        "pause_min": Pause().minutes,
        "pause_start": Pause().start,
    }
    if not (cmd.user.mod or cmd.user.name.lower() in SETTINGS.twitch.admin_users):
        log.warning(SETTINGS.fmt.cmd_blocked.format(**fmt_dict))
        return
    if not Pause().is_paused():
        await cmd.reply(SETTINGS.fmt.tresume_failure.format(**fmt_dict))
    else:
        added_min = Pause().resume("!{cmd}".format(**fmt_dict))
        fmt_dict["pause_min"] = Pause().minutes
        fmt_dict["added_min"] = added_min
        fmt_dict["pause_start"] = None
        log.info(SETTINGS.fmt.tresume_success.format(**fmt_dict))
        await cmd.reply(SETTINGS.fmt.tresume_success.format(**fmt_dict))


async def parse_time_from_cmd(cmd: ChatCommand, cmd_name: str):
    fmt_dict = {
        "user": cmd.user.name,
        "cmd": cmd_name,
        "pause_min": Pause().minutes,
        "pause_start": Pause().start,
        "end_ts": End().end_ts,
        "end_min": End().end_min,
    }
    if not (cmd.user.mod or cmd.user.name.lower() in SETTINGS.twitch.admin_users):
        log.warning(SETTINGS.fmt.cmd_blocked.format(**fmt_dict))
        return
    elif End().is_ended():
        await cmd.reply(SETTINGS.fmt.cmd_after_end.format(**fmt_dict))
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
        log.error(cmd.reply(SETTINGS.fmt.missing_time_parameter_failure.format(**fmt_dict)))
        await cmd.reply(SETTINGS.fmt.missing_time_parameter_failure.format(**fmt_dict))
        raise
    except ValueError as err:
        fmt_dict |= {"err": str(err), "err_type": str(type(err))}
        log.error(SETTINGS.fmt.invalid_time_parameter_failure.format(**fmt_dict))
        await cmd.reply(SETTINGS.fmt.invalid_time_parameter_failure.format(**fmt_dict))
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
        "pause_min": Pause().minutes,
        "pause_start": Pause().start,
        "pause_delta": minutes,
    }
    Pause().pause_increase(minutes, "!{cmd}".format(**fmt_dict))
    log.info(SETTINGS.fmt.tadd_success.format(**fmt_dict))
    await cmd.reply(SETTINGS.fmt.tadd_success.format(**fmt_dict))


async def remove_time_command(cmd: ChatCommand):
    try:
        minutes = await parse_time_from_cmd(cmd, "tremove")
    except (IndexError, ValueError):
        return
    fmt_dict = {
        "user": cmd.user.name,
        "cmd": "tremove",
        "pause_min": Pause().minutes,
        "pause_start": Pause().start,
        "pause_delta": minutes,
    }
    Pause().pause_reduce(minutes, "!{cmd}".format(**fmt_dict))
    log.info(SETTINGS.fmt.tremove_success.format(**fmt_dict))
    await cmd.reply(SETTINGS.fmt.tremove_success.format(**fmt_dict))


async def raised_command(cmd: ChatCommand):
    fmt_dict = {
        "user": cmd.user.name,
        "cmd": "traised",
    }
    if not (cmd.user.mod or cmd.user.name.lower() in SETTINGS.twitch.admin_users):
        log.warning(SETTINGS.fmt.cmd_blocked.format(**fmt_dict))
        return
    so_far_total_min = calc_time_so_far().total_seconds() / 60
    fmt_dict = {
        **fmt_dict,
        "so_far_total_min": so_far_total_min,
        "so_far_hrs": so_far_total_min // 60,
        "so_far_min": so_far_total_min % 60,
        "min_paid_for": Donos().calc_chat_minutes(),
        "min_total": Donos().calc_total_minutes(),
        "min_end_at": calc_end().total_seconds() / 60,
        "end_ts": End().end_ts,
        "end_min": End().end_min,
        "total_value": Donos().calc_dollars(),
        "countdown": calc_timer(),
        "bits": Donos().bits,
        "tips": Donos().tips,
        "subs": Donos().subs,
        "subs_t1": Donos().subs_t1,
        "subs_t2": Donos().subs_t2,
        "subs_t3": Donos().subs_t3,
        "pause_min": Pause().minutes,
        "pause_start": Pause().start or "Not Currently Paused",
    }
    log.info(SETTINGS.fmt.traised_success.format(**fmt_dict))
    await cmd.reply(SETTINGS.fmt.traised_success.format(**fmt_dict))


async def add_tip_command(cmd: ChatCommand):
    fmt_dict = {
        "user": cmd.user.name,
        "cmd": "taddtip",
    }
    if not (cmd.user.mod or cmd.user.name.lower() in SETTINGS.twitch.admin_users):
        log.warning(SETTINGS.fmt.cmd_blocked.format(**fmt_dict))
        return
    elif End().is_ended():
        await cmd.reply(SETTINGS.fmt.cmd_after_end.format(**fmt_dict))
        return

    parameters = cmd.parameter.split()
    if len(parameters) == 2:
        giver, amount_str = parameters
        reason = None
    elif len(parameters) == 3:
        giver, amount_str, reason = parameters
    else:
        await cmd.reply("Command format !{cmd} [donor] [amount] <type-of-tip>".format(**fmt_dict))
        return
    if giver.startswith("@"):
        giver = giver[1:]
    if amount_str.startswith("$"):
        amount_str = amount_str[1:]
    try:
        amount = float(amount_str)
    except ValueError:
        await cmd.reply("Parameter [amount] must be parsable as numbers to be recorded")
        return
    log.info(f"in {cmd.room.name}, {cmd.user.name} added tip from {giver}: {amount:.02f} w/ type {reason}")
    Donos().add_event(
        ts=cmd.sent_timestamp,
        user=giver,
        target=reason,
        type=TIPS,
        amount=amount,
    )
    if reason:
        await cmd.reply(f"Recorded tip from {giver} of ${amount:.02f} with type {reason}")
    else:
        await cmd.reply(f"Recorded tip from {giver} of ${amount:.02f}")


def calc_end() -> timedelta:
    """Find the timedelta to use for final calculations"""
    if End().is_ended():
        return timedelta(minutes=End().end_min)
    minutes = Donos().calc_total_minutes()
    if SETTINGS.end.max_minutes:
        minutes = min(minutes, SETTINGS.end.max_minutes)
    return timedelta(minutes=minutes)


def calc_time_so_far() -> timedelta:
    """How much time has been counted down since the start"""
    if End().is_ended():
        cur_time = End().end_ts
    elif Pause().is_paused():
        cur_time: datetime = Pause().start
    else:
        cur_time = datetime.now(tz=timezone.utc)
    time_so_far = cur_time - SETTINGS.start.time
    corrected_tsf = time_so_far - timedelta(minutes=Pause().minutes)
    return corrected_tsf


def calc_timer() -> str:
    """Generate the timer string from the difference between paid and run minutes"""
    remaining = calc_end() - calc_time_so_far()
    hours = int(remaining.total_seconds() / 60 / 60)
    minutes = int(remaining.total_seconds() / 60) % 60
    seconds = int(remaining.total_seconds()) % 60
    time_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    if Pause().is_paused():
        pause_format = SETTINGS.fmt.countdown_pause
        return pause_format.format(clock=time_str)
    else:
        return time_str


async def channel_offline(_event):
    now = datetime.now(tz=timezone.utc)
    if Pause().is_paused():
        log.info(f"Channel went offline at {now.isoformat()}, already was paused at {Pause().start.isoformat()}")
    elif SETTINGS.twitch.pause_on_offline:
        now = Pause().start_pause("channel went offline")
        log.info(f"Channel went offline at {now.isoformat()}, pause started")
    else:
        log.info(f"Channel went offline at {now.isoformat()}, but pause not started, timer is still running")


async def channel_online(_event):
    now = datetime.now(tz=timezone.utc)
    if Pause().is_paused() and SETTINGS.twitch.unpause_on_online:
        added_min = Pause().resume("channel went online")
        log.info(
            f"Pause resumed with an addition of {added_min:.02f} minutes"
            f" for a total of {Pause().minutes:.02f} minutes"
        )
    elif Pause().is_paused():
        delta = (now - Pause().start).total_seconds() / 60.0
        log.info(
            f"Channel went online at {now.isoformat()}, pause started at {Pause().start.isoformat()} or {delta} min ago"
        )
    else:
        log.info(f"Channel went online at {now.isoformat()}, but time was not paused?!?")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # set up twitch api instance and add user authentication with some scopes
    twitch = await Twitch(SETTINGS.twitch.app_id, SETTINGS.twitch.app_secret.get_secret_value())
    usr_token_file = Path(SETTINGS.twitch.user_token_file)
    if usr_token_file.is_file():
        user_auth = toml.load(usr_token_file)
        token, refresh_token = user_auth["token"], user_auth.get("refresh_token")
        if not refresh_token:
            twitch.auto_refresh_auth = False
    else:
        auth = UserAuthenticator(twitch, USER_SCOPE, url=SETTINGS.twitch.auth_url)
        token, refresh_token = await auth.authenticate(browser_name="google-chrome")
        usr_token_file.write_text(toml.dumps({"token": token, "refresh_token": refresh_token}))
    try:
        await twitch.set_user_authentication(token, USER_SCOPE, refresh_token)
    except TwitchAPIException:
        log.error(f"Invalid token, please remove {usr_token_file} try again")
        raise

    # Get id for twitch channel
    channel = await first(twitch.get_users(logins=[SETTINGS.twitch.channel]))

    # create eventsub websocket instance and start the client.
    eventsub = None
    if SETTINGS.twitch.eventsub:
        eventsub = EventSubWebsocket(twitch)
        eventsub.start()
        await eventsub.listen_stream_offline(channel.id, channel_offline)
        await eventsub.listen_stream_online(channel.id, channel_online)

    # create chat instance
    chat = await Chat(twitch, callback_loop=asyncio.get_running_loop(), no_message_reset_time=6)

    # listen to when the bot is done starting up and ready to join channels
    chat.register_event(ChatEvent.READY, on_ready)
    # listen to chat messages
    chat.register_event(ChatEvent.MESSAGE, on_message)
    # listen to channel subscriptions
    chat.register_event(ChatEvent.SUB, on_sub)
    # listen to channel raids
    chat.register_event(ChatEvent.RAID, on_raid)

    # you can directly register commands and their handlers
    if SETTINGS.twitch.enable_cmds:
        chat.register_command("tpause", pause_command)
        chat.register_command("tresume", resume_command)
        chat.register_command("tadd", add_time_command)
        chat.register_command("tremove", remove_time_command)
        chat.register_command("traised", raised_command)
        chat.register_command("taddtip", add_tip_command)
        if Spins.enabled:
            chat.register_command("tspin", spin_done_command)

    # we are done with our setup, lets start this bot up!
    chat.start()

    yield  # Run FastAPI stuff

    if eventsub:
        await eventsub.stop()
    # now we can close the chat bot and the twitch api client
    chat.stop()
    await twitch.close()
    log.info("Done shutting down TwitchAPI")


app = FastAPI(lifespan=lifespan)


class JSONResponse(JSONResponse):
    def render(self, content) -> bytes:
        return json.dumps(content, ensure_ascii=False, allow_nan=False, indent=4, separators=(",", ":")).encode("utf-8")


@app.get("/live_stats", response_class=JSONResponse)
async def get_live_stats():
    End().handle_end(calc_time_so_far, calc_end, Donos().calc_total_minutes)
    full_stats = {"pause": Pause().to_dict(), "donos": Donos().to_dict(), "end": End().to_dict()}
    return jsonable_encoder(full_stats)


@app.get("/live_stats/bits", response_class=PlainTextResponse)
async def get_live_stats_bits():
    return f"{Donos().bits}"


@app.get("/live_stats/tips", response_class=PlainTextResponse)
async def get_live_stats_tips():
    return f"${Donos().tips:.02f}"


@app.get("/live_stats/subs", response_class=PlainTextResponse)
async def get_live_stats_subs():
    return str(Donos().subs)


@app.get("/live_stats/total_value", response_class=PlainTextResponse)
async def get_total_value():
    return f"${Donos().calc_dollars():0.02f}"


if Spins.enabled:

    @app.get("/live_stats/spins", response_class=PlainTextResponse)
    async def get_total_value():
        return f"{Spins().performed}/{Spins().calc_todo(Donos().calc_dollars()):0.1f}"


@app.get("/calc_timer", response_class=PlainTextResponse)
async def get_calc_timer():
    End().handle_end(calc_time_so_far, calc_end, Donos().calc_total_minutes)
    return calc_timer()


@app.get("/events", response_class=HTMLResponse)
async def get_events(timezone: Optional[str] = None):
    if timezone is None:
        tz = ZoneInfo("UTC")
    else:
        try:
            tz = ZoneInfo(timezone)
        except ZoneInfoNotFoundError as e:
            return f"<html><body><xmp>{e}</xmp></body></html>"

    events_per_day = {}
    for row in Donos.csv_iter():
        row["time"]: datetime = datetime.fromtimestamp(int(row["time"]) / 1000).astimezone(tz)
        day = row["time"].date()
        day_list = events_per_day.setdefault(day, [])
        day_list.append(row)

    build_table = ""
    for date, rows in events_per_day.items():
        build_table += f"\n<h1>{date:%Y-%m-%d}</h1>\n<table>\n"
        build_table += "<tr>" + "".join(f"<th>{s}</th>" for s in CSV_COLUMNS) + "</tr>\n"
        for row in rows:
            build_table += "<tr>" + "".join(f"<td>{row[s]}</td>" for s in CSV_COLUMNS) + "</tr>\n"
        build_table += "</table>\n"
    style = """table {border: 2px solid rgb(140 140 140);}
    th,td {border: 1px solid rgb(160 160 160);}"""
    return f"<html><head><style>{style}</style></head><body>{build_table}</body></html>"


@app.get("/donors", response_class=HTMLResponse)
async def get_donors(sort: str = "total"):
    donor_keys = {
        "name": (lambda x: x["name"].lower(), False),
        "total": (lambda x: x["total"], True),
        **{k: (lambda x, y=k: x[y], True) for k in CSV_TYPES},
    }
    if sort not in donor_keys:
        return f"<html><body><xmp>{sort} not in {tuple(donor_keys.keys())}</xmp></body></html>"
    donor_db = {}
    for row in Donos.csv_iter():
        user_db = donor_db.setdefault(
            row["user"].lower(), {"name": row["user"], "total": 0, **{k: 0 for k in CSV_TYPES}}
        )
        amount = float(row["amount"]) if row["type"] == TIPS else int(row["amount"])
        user_db[row["type"]] += amount
        user_db["total"] += amount * SETTINGS.get_value(row["type"])

    build_table = "<table>\n"
    build_table += "<tr>" + "".join(f"<th><a href='?sort={s}'>{s}</a></th>" for s in donor_keys) + "</tr>\n"
    for row in sorted(donor_db.values(), key=donor_keys[sort][0], reverse=donor_keys[sort][1]):
        build_table += "<tr>" + "".join(f"<td>{row[s]}</td>" for s in donor_keys) + "</tr>\n"
    build_table += "</table>\n"
    style = """table {border: 2px solid rgb(140 140 140);}
    th,td {border: 1px solid rgb(160 160 160);}"""
    return f"<html><head><style>{style}</style></head><body>{build_table}</body></html>"


websocket_html = """
<!DOCTYPE html>
<html>
    <head>
        <title>{name}</title>
        <style>{css}</style>
    </head>
    <body>
        <div id='text'>Not Yet Connected</div>
        <script>
            var ws
            function connect() {{
                ws = new WebSocket("{hostname}/{path}");
                ws.onmessage = function(event) {{ document.getElementById('text').innerText = event.data }}
            }}
            connect()
            var interval = setInterval(function() {{
                    if (ws.readyState === WebSocket.CLOSED) {{ connect() }}
            }}, 60000);
        </script>
    </body>
</html>
"""


@app.get("/live", response_class=HTMLResponse)
async def get_live_timer():
    return websocket_html.format(name="countdown", css=SETTINGS.output.css, hostname=SETTINGS.output.public, path="ws")


@app.get("/live_counter", response_class=HTMLResponse)
async def get_live_counter(item: Optional[Literal["tips", "bits", "subs", "subs_t1", "subs_t2", "subs_t3"]] = None):
    if item is None:
        return (
            "<html><body>"
            ", ".join(f"<a href='?item={s}'>{s}</a>" for s in ("tips", "bits", "subs", "subs_t1", "subs_t2", "subs_t3"))
            + "</body></html>"
        )
    else:
        return websocket_html.format(
            name=f"{item} counter",
            css=SETTINGS.output.css,
            hostname=SETTINGS.output.public,
            path=f"ws_counter?item={item}",
        )


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            End().handle_end(calc_time_so_far, calc_end, Donos().calc_total_minutes)
            try:
                await websocket.send_text(calc_timer())
                await asyncio.sleep(0.5)
            except ConnectionClosedOK:
                break
    except WebSocketDisconnect:
        pass


@app.websocket("/ws_counter")
async def websocket_endpoint(
    websocket: WebSocket, item: Literal["tips", "bits", "subs", "subs_t1", "subs_t2", "subs_t3"]
):
    last_sent = None
    await websocket.accept()
    try:
        while True:
            try:
                cur_msg = str(getattr(Donos(), item))
                if cur_msg != last_sent:
                    await websocket.send_text(cur_msg)
                    last_sent = cur_msg
                await asyncio.sleep(1)
            except ConnectionClosedOK:
                break
    except WebSocketDisconnect:
        pass


@app.put("/admin/pause/begin", response_class=JSONResponse)
async def put_pause_begin(password: str, time: Optional[datetime] = None):
    SETTINGS.raise_on_bad_password(password)
    old_values = Pause().to_dict()
    try:
        if time is None:
            Pause().start_pause("/admin/pause/start")
            return {"old": old_values, "new": Pause().to_dict()}
        else:
            Pause().set_pause_start(time, "/admin/pause/start")
            return {"old": old_values, "new": Pause().to_dict()}
    except PauseException as e:
        raise HTTPException(status_code=409, detail=str(e))


@app.put("/admin/pause/resume")
async def put_pause_resume(password: str, time: Optional[datetime] = None):
    SETTINGS.raise_on_bad_password(password)
    old_values = Pause().to_dict()
    try:
        if time is None:
            Pause().resume("/admin/pause/resume")
            return {"old": old_values, "new": Pause().to_dict()}
        else:
            Pause().resumed_at(time, "/admin/pause/resume")
            return {"old": old_values, "new": Pause().to_dict()}
    except PauseException as e:
        raise HTTPException(status_code=409, detail=str(e))


@app.put("/admin/pause/abort")
async def put_pause_abort(password: str):
    SETTINGS.raise_on_bad_password(password)
    old_values = Pause().to_dict()
    try:
        Pause().abort_current("/admin/pause/resume")
        return {"old": old_values, "new": Pause().to_dict()}
    except PauseException as e:
        raise HTTPException(status_code=409, detail=str(e))


if __name__ == "__main__":
    Pause.load_pause()
    Spins.load_spins()
    Donos.load_csv()
    End.load_end()
    End().handle_end(calc_time_so_far, calc_end, Donos().calc_total_minutes)
    log.info(f"Users who can run cmds in addition to mods {SETTINGS.twitch.admin_users}")

    import uvicorn

    try:
        uvicorn.run(app, host=SETTINGS.output.listen, port=SETTINGS.output.port)
    except KeyboardInterrupt:
        pass
