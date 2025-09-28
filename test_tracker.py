"""Twitch donothon clock based on reading chat"""
import asyncio
import json
import logging
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Literal, Optional
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import toml
import twitchAPI.oauth
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.encoders import jsonable_encoder
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse
from twitchAPI.chat import Chat, ChatCommand, ChatMessage, ChatSub, EventData
from twitchAPI.chat.middleware import BaseCommandMiddleware
from twitchAPI.eventsub.websocket import EventSubWebsocket
from twitchAPI.helper import first
from twitchAPI.oauth import UserAuthenticator
from twitchAPI.twitch import Twitch
from twitchAPI.type import AuthScope, ChatEvent, TwitchAPIException
from websockets import ConnectionClosedOK

from twitch_dono_clock.config import SETTINGS, load_overrides, override_value
from twitch_dono_clock.donos import (
    BITS,
    CSV_COLUMNS,
    CSV_TYPES,
    TIPS,
    Donos,
    add_tip_command,
)
from twitch_dono_clock.end import End, EndException
from twitch_dono_clock.pause import (
    Pause,
    PauseException,
    add_time_command,
    pause_command,
    remove_time_command,
    resume_command,
)
from twitch_dono_clock.spins import Spins, spin_done_command

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


class OnlyMyChat(BaseCommandMiddleware):
    async def can_execute(self, command: ChatCommand) -> bool:
        """Only allow execution of command is from this chat"""
        if "source-room-id" in command._parsed["tags"]:
            if command._parsed["tags"]["source-room-id"] != command._parsed["tags"]["room-id"]:
                return False
        return True

    async def was_executed(self, command: ChatCommand):
        pass  # Nothing to track


# this will be called when the event READY is triggered, which will be on bot start
async def on_ready(_ready_event: EventData):
    log.info(f"Bot is ready for work, should have already joined channel {SETTINGS.twitch.channel}")


# this will be called whenever a message in a channel was send by either the bot OR another user
async def on_message(msg: ChatMessage):
    if "source-room-id" in msg._parsed["tags"]:
        if msg._parsed["tags"]["source-room-id"] != msg._parsed["tags"]["room-id"]:
            log.debug(f"Skipping shared-chat message from {msg._parsed['tags']['source-room-id']} saying {msg.text}")
            return
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
    fancy_msg = msg._parsed["tags"].get("msg-id")
    if fancy_msg and msg.user.name.lower() != msg.room.name.lower():
        if fancy_msg == "gigantified-emote-message":
            log.info(f"{msg.sent_timestamp} {msg.user.display_name} sent a Giant Emote {msg.text}")
            Donos().add_event(
                ts=msg.sent_timestamp,
                user=msg.user.display_name,
                target="giant-emote",
                type=BITS,
                amount=SETTINGS.bits.giant_emote_bits,
            )
        elif fancy_msg == "animated-message":
            animation_type = msg._parsed["tags"].get("animation-id")
            log.info(f"{msg.sent_timestamp} {msg.user.display_name} sent {animation_type} Animated Message: {msg.text}")
            Donos().add_event(
                ts=msg.sent_timestamp,
                user=msg.user.display_name,
                target="animated-msg",
                type=BITS,
                amount=SETTINGS.bits.animated_message_bits,
            )
        else:
            log.info(f"{msg.sent_timestamp} {msg.user.display_name} sent an unknown {fancy_msg=} - {msg.text}")
    for user, regex, dono_type, target in SETTINGS.compiled_re:
        if msg.user.name.lower() == user.lower():
            match = regex.match(msg.text)
            if match:
                via = f" via {target}" if target else ""
                log.info(f"in {msg.room.name}, {match['user']} sent {dono_type}{via}: {match['amount']}")
                if dono_type == BITS:
                    amount = int(match["amount"].replace(",", ""))
                elif dono_type == TIPS:
                    amount = float(match["amount"].replace(",", ""))
                else:
                    raise ValueError(f"Unknown target from msg parsing {dono_type}")
                if not amount:  # If we have a record with 0 bits / 0 tip we shouldn't record
                    log.debug(f"Skipping recording {amount} {dono_type}")
                    continue
                Donos().add_event(
                    ts=msg.sent_timestamp,
                    user=match["user"],
                    target=target,
                    type=dono_type,
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
        "points": Donos().calc_points(),
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
    response = SETTINGS.fmt.traised_success.format(**fmt_dict)
    log.info(response)
    await cmd.reply(response)


def calc_end() -> timedelta:
    """Find the timedelta to use for final calculations"""
    if End().end_min:
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


def calc_timer(handle_end: bool = True) -> str:
    """Generate the timer string from the difference between paid and run minutes"""
    if handle_end:
        End().handle_end(calc_time_so_far, calc_end, Donos().calc_total_minutes)
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


async def store_user_token(user_auth_token, user_auth_refresh_token):
    usr_token_file = Path(SETTINGS.twitch.user_token_file)
    usr_token_file.write_text(toml.dumps({"token": user_auth_token, "refresh_token": user_auth_refresh_token}))
    log.info(f"Wrote updated {usr_token_file}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # set up twitch api instance and add user authentication with some scopes
    twitch = await Twitch(SETTINGS.twitch.app_id, SETTINGS.twitch.app_secret.get_secret_value())
    twitch.user_auth_refresh_callback = store_user_token
    usr_token_file = Path(SETTINGS.twitch.user_token_file)
    if usr_token_file.is_file():
        user_auth = toml.load(usr_token_file)
        token, refresh_token = user_auth["token"], user_auth.get("refresh_token")
        if not refresh_token:
            twitch.auto_refresh_auth = False
    else:
        auth = UserAuthenticator(twitch, USER_SCOPE, url=SETTINGS.twitch.auth_url)

        class FakeWebBrowser:
            @staticmethod
            def get(*_args, **_kwargs):
                class FakeBrowser:
                    @staticmethod
                    def open(url, *_args, **_kwargs):
                        log.info(f"Please open {url}")

                return FakeBrowser()

        # Patch the webbrowser import to print the url to load
        twitchAPI.oauth.webbrowser = FakeWebBrowser()

        token, refresh_token = await auth.authenticate()
        await store_user_token(token, refresh_token)
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
        eventsub = EventSubWebsocket(twitch, callback_loop=asyncio.get_running_loop())
        eventsub.start()
        await eventsub.listen_stream_offline(channel.id, channel_offline)
        await eventsub.listen_stream_online(channel.id, channel_online)

    # create chat instance
    chat = await Chat(
        twitch,
        initial_channel=[SETTINGS.twitch.channel],
        callback_loop=asyncio.get_running_loop(),
        no_message_reset_time=6,
    )

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
        chat.register_command_middleware(OnlyMyChat())
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


@app.get("/live_stats/points", response_class=PlainTextResponse)
async def get_live_stats_points():
    return f"{Donos().calc_points():.02f}"


@app.get("/live_stats/total_value", response_class=PlainTextResponse)
async def get_total_value():
    return f"${Donos().calc_dollars():0.02f}"


if Spins.enabled:

    @app.get("/live_stats/spins", response_class=PlainTextResponse)
    async def get_total_value():
        return f"{Spins().performed}/{Spins().calc_todo(Donos().calc_dollars()):0.1f}"


@app.get("/calc_timer", response_class=PlainTextResponse)
async def get_calc_timer():
    return calc_timer()


@app.get("/traised", response_class=JSONResponse)
async def traised_fields():
    so_far_total_min = calc_time_so_far().total_seconds() / 60
    return {
        "so_far_total_min": so_far_total_min,
        "so_far_hrs": so_far_total_min // 60,
        "so_far_min": so_far_total_min % 60,
        "min_paid_for": Donos().calc_chat_minutes(),
        "min_total": Donos().calc_total_minutes(),
        "min_end_at": calc_end().total_seconds() / 60,
        "end_ts": End().end_ts,
        "end_min": End().end_min,
        "total_value": Donos().calc_dollars(),
        "points": Donos().calc_points(),
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


@app.get("/events", response_class=HTMLResponse)
async def get_events(timezone: Optional[str] = None):
    if timezone is None:
        tz = ZoneInfo("UTC")
    else:
        try:
            tz = ZoneInfo(timezone)
        except ZoneInfoNotFoundError as e:
            return f"<html><body><xmp>{e}</xmp></body></html>"

    conversions = ""
    if SETTINGS.tips.convert:
        conversions += (
            "Tips of the following types are in a non-USD currency\n <table>\n"
            "<tr><th>Type</th><th>From</th><th>To</th><th>Ratio</th></tr>"
        )
        for tip_type, conv in SETTINGS.tips.convert.items():
            conversions += (
                f"<tr><td>{tip_type}</td><td>{conv.src}</td><td>{conv.target}</td><td>{conv.ratio}</td></tr>\n"
            )
        conversions += "</table>\n\n"

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
    return f"<html><head><style>{style}</style></head><body>{conversions}{build_table}</body></html>"


@app.get("/events_csv", response_class=PlainTextResponse)
async def get_events_csv():
    return Donos.dono_path.read_text()


@app.get("/events_targets", response_class=JSONResponse)
async def get_events_targets():
    donation_targets = {"tips": {}, "bits": {}}
    for row in Donos.csv_iter():
        if row["type"] not in donation_targets:
            continue
        target = row["target"] or "<untagged>"
        if row["type"] == TIPS:
            amount = float(row["amount"])
            conv = SETTINGS.tips.convert.get(row["target"])
            if conv:
                amount *= conv.ratio
        else:
            amount = int(row["amount"])
        if target in donation_targets[row["type"]]:
            donation_targets[row["type"]][target] += amount
        else:
            donation_targets[row["type"]][target] = amount
    return donation_targets


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
        if row["type"] == TIPS:
            amount = float(row["amount"])
            conv = SETTINGS.tips.convert.get(row["target"])
            if conv:
                amount *= conv.ratio
        else:
            amount = int(row["amount"])
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
async def get_live_counter(
    item: Optional[Literal["tips", "bits", "subs", "subs_t1", "subs_t2", "subs_t3", "total", "points"]] = None
):
    if item is None:
        return (
            "<html><body>"
            ", ".join(
                f"<a href='?item={s}'>{s}</a>"
                for s in ("tips", "bits", "subs", "subs_t1", "subs_t2", "subs_t3", "total", "points")
            )
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
            try:
                await websocket.send_text(calc_timer())
                await asyncio.sleep(0.5)
            except ConnectionClosedOK:
                break
    except WebSocketDisconnect:
        pass


@app.websocket("/ws_counter")
async def websocket_counter_endpoint(
    websocket: WebSocket, item: Literal["tips", "bits", "subs", "subs_t1", "subs_t2", "subs_t3", "total", "points"]
):
    last_sent = None
    money = {"tips", "total"}
    await websocket.accept()
    try:
        while True:
            try:
                if item in money:
                    cur_msg = f"${getattr(Donos(), item):.02f}"
                else:
                    cur_msg = str(getattr(Donos(), item))
                if cur_msg != last_sent:
                    await websocket.send_text(cur_msg)
                    last_sent = cur_msg
                await asyncio.sleep(1)
            except ConnectionClosedOK:
                break
    except WebSocketDisconnect:
        pass


@app.put("/admin/end/clear", response_class=JSONResponse)
async def put_end_clear(password: str):
    SETTINGS.raise_on_bad_password(password)
    old_values = End().to_dict()
    try:
        End().clear("/admin/end/clear")
        return {"old": old_values, "new": End().to_dict()}
    except EndException as e:
        raise HTTPException(status_code=409, detail=str(e))


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


@app.put("/admin/pause/resume", response_class=JSONResponse)
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


@app.put("/admin/pause/abort", response_class=JSONResponse)
async def put_pause_abort(password: str):
    SETTINGS.raise_on_bad_password(password)
    old_values = Pause().to_dict()
    try:
        Pause().abort_current("/admin/pause/resume")
        return {"old": old_values, "new": Pause().to_dict()}
    except PauseException as e:
        raise HTTPException(status_code=409, detail=str(e))


@app.put("/admin/pause/minutes", response_class=JSONResponse)
async def put_set_minutes(password: str, minutes: float):
    SETTINGS.raise_on_bad_password(password)
    old_values = Pause().to_dict()
    try:
        Pause().pause_set(minutes, "/admin/pause/minutes")
        return {"old": old_values, "new": Pause().to_dict()}
    except PauseException as e:
        raise HTTPException(status_code=409, detail=str(e))


@app.put("/admin/donos/reload", response_class=JSONResponse)
async def put_donos_reload(password: str):
    SETTINGS.raise_on_bad_password(password)
    old_values = Donos().to_dict()
    Donos().reload_csv()
    return {"old": old_values, "new": Donos().to_dict()}


@app.put("/admin/donos/wipe", response_class=JSONResponse)
async def put_donos_wipe(password: str, are_you_sure: bool = False):
    SETTINGS.raise_on_bad_password(password)
    old_values = Donos().to_dict()
    filename = None
    if are_you_sure:
        filename = Donos().clear_csv()
    return {"old": old_values, "new": Donos().to_dict(), "backup": filename}


@app.put("/admin/settings/overrides", response_class=JSONResponse)
async def put_settings_overrides(password: str):
    SETTINGS.raise_on_bad_password(password)
    try:
        return load_overrides()
    except Exception as e:
        raise HTTPException(status_code=409, detail=str(e))


@app.put("/admin/settings/override_value", response_class=JSONResponse)
async def put_settings_overrides(password: str, key: str, value: Any):
    SETTINGS.raise_on_bad_password(password)
    try:
        return override_value(key, value)
    except Exception as e:
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
