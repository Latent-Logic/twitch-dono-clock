import asyncio
import json
import logging
import random
import time
import uuid
from collections.abc import Callable, Iterator
from datetime import datetime

import websockets
from websockets import ClientConnection, ConnectionClosed, InvalidHandshake

from twitch_dono_clock.utils import BitRedeem, BitType

logging.getLogger("websockets.client").setLevel(logging.INFO)

PUBSUB_URL = "wss://pubsub-edge.twitch.tv"
# Keep websocket alive ping every 5 minutes
PING_INTERVAL = 5 * 60
MAX_JITTER = 0.25
# Timespan after a PING to expect a PONG, longer than this reconnect
PONG_TIMEOUT = 30
# After 10 minutes of now messages assume it broke and reconnect
NO_MESSAGE_TIMEOUT = 10 * 60
# Reconnect backoff
RECONNECT_DELAY_STEPS = [0, 1, 2, 4, 8, 16, 32, 64, 128]
# Only consider a connection stable if it has run for a minute plus
STABILITY_THRESHOLD = 60

POINT_REDEMPTION_TOPICS = {"reward-redeemed", "automatic-reward-redeemed"}


log = logging.getLogger(__name__)


class PubsubSocketClosed(Exception):
    pass


class PubsubReconnect(PubsubSocketClosed):
    pass


def parse_bit_redemption(data: dict) -> BitRedeem:
    redemption = data["redemption"]
    reward = redemption.get("reward", {})
    assert reward.get("pricing_type") == "BITS"
    user = redemption.get("user", {})

    reward_type = reward.get("reward_type")
    if reward_type == "SEND_GIGANTIFIED_EMOTE":
        bit_enum = BitType.GIGANTIFIED_EMOTE
        title = "Gigantify an Emote"
    elif reward_type == "SEND_ANIMATED_MESSAGE":
        bit_enum = BitType.ANIMATED_MESSAGE
        title = "Message Effects"
    elif reward_type == "CELEBRATION":
        bit_enum = BitType.CELEBRATION
        title = "On-Screen Celebration"
    else:
        bit_enum = BitType.POWER_UP
        title = reward.get("title", "-unknown-")

    return BitRedeem(
        type=bit_enum,
        title=title,
        timestamp=datetime.fromisoformat(data["timestamp"]),
        user=user.get("display_name") or user.get("login"),
        cost=reward.get("bits_cost") or reward.get("default_bits_cost", 0),
    )


def handle_message(frame: dict, topic: str, callback: Callable[[BitRedeem], None]) -> None:
    frame_data = frame.get("data", {})
    if frame_data.get("topic") != topic:
        return

    raw_msg = frame_data.get("message")  # message is a JSON string
    if not (isinstance(raw_msg, str) and raw_msg):
        log.warning(f"MESSAGE frame missing viable data: {raw_msg!r}")
        return
    try:
        inner = json.loads(raw_msg)
    except json.JSONDecodeError as e:
        log.warning(f"Failed to parse inner message: {e}")
        return
    if not isinstance(inner, dict):
        log.warning(f"Inner message was not a JSON dict: {inner!r}")
        return

    log.debug(f"Got MESSAGE from server of type {inner.get('type')}")
    if inner.get("type") not in POINT_REDEMPTION_TOPICS:
        return
    data = inner["data"]
    if data.get("redemption", {}).get("reward", {}).get("pricing_type") != "BITS":
        return
    log.debug(f"{data=}")
    try:
        event = parse_bit_redemption(data)
    except (KeyError, TypeError) as e:
        log.warning(f"Failed to parse redemption: {e!r}")
        return

    callback(event)


async def ping_keepalive(ws: ClientConnection, pong_received: asyncio.Event) -> None:
    while True:
        await asyncio.sleep(PING_INTERVAL - random.uniform(0, MAX_JITTER))
        pong_received.clear()
        try:
            await ws.send(json.dumps({"type": "PING"}))
        except ConnectionClosed:
            log.warning("Could not send PING, resetting connection")
            await ws.close(4000, "ping failed")
            return
        try:
            await asyncio.wait_for(pong_received.wait(), timeout=PONG_TIMEOUT)
        except TimeoutError:
            log.warning(f"No PONG within {PONG_TIMEOUT}s, resetting connection")
            await ws.close(4000, "pong timeout")
            return


def parse_ws_blob(raw: str) -> Iterator[dict]:
    for line in raw.split("\r\n"):
        if not line:
            continue
        try:
            yield json.loads(line)
        except json.JSONDecodeError:
            log.warning(f"Dropping a malformed (non-JSON) frame: {line}")
            continue


async def run_session(
    ws: ClientConnection, topic: str, success: asyncio.Event, callback: Callable[[BitRedeem], None]
) -> None:
    pong_received = asyncio.Event()
    ping_task = asyncio.create_task(ping_keepalive(ws, pong_received))
    try:
        nonce = str(uuid.uuid4())
        listen = {"type": "LISTEN", "nonce": nonce, "data": {"topics": [topic]}}
        try:
            await ws.send(json.dumps(listen))
        except ConnectionClosed as e:
            log.warning(f"Failed to send LISTEN: {e!r}")
            raise
        log.info(f"LISTEN sent for {topic}")

        while True:
            try:
                raw = await asyncio.wait_for(ws.recv(decode=True), timeout=NO_MESSAGE_TIMEOUT)
            except TimeoutError:
                log.warning(f"Silent for {NO_MESSAGE_TIMEOUT}s; assuming a dead socket.")
                raise PubsubSocketClosed

            # One text frame can hold several JSON objects; handle them in turn.
            for frame in parse_ws_blob(raw):
                ftype = frame.get("type")

                if ftype == "PONG":
                    pong_received.set()
                    log.debug("Received websocket PONG to our PING")
                    continue
                if ftype == "RECONNECT":
                    log.info("Server requested RECONNECT, closing and restarting")
                    raise PubsubReconnect
                if ftype == "RESPONSE":  # LISTEN response
                    if frame.get("error"):
                        log.warning(f"RESPONSE error: {frame['error']}")
                    if frame.get("nonce") == nonce:
                        success.set()
                        log.debug("PubSub connection successfully LISTEN->RESPONSE connected")
                    continue
                if ftype == "AUTH_REVOKED":  # No auth used for community-points-channel-v1?
                    log.warning(f"Server sent AUTH_REVOKED?? {frame=}")
                    raise PubsubSocketClosed
                if ftype != "MESSAGE":
                    log.warning(f"Unhandled frame type: {frame=}")
                    continue

                handle_message(frame, topic, callback)
    except ConnectionClosed as e:
        log.warning(f"Connection closed: {e!r}")
        raise PubsubSocketClosed
    finally:
        ping_task.cancel()
        try:
            await ping_task
        except asyncio.CancelledError:
            pass
        await ws.close(1000, "session ended")


async def subscribe_redemptions(channel_id: str | int, callback: Callable[[BitRedeem], None]) -> None:
    topic = f"community-points-channel-v1.{channel_id}"
    connect_success = asyncio.Event()
    backoff = 0
    while True:
        connect_success.clear()
        ws = None
        while ws is None:
            try:
                log.debug("PubSub Connecting...")
                ws = await websockets.connect(PUBSUB_URL, close_timeout=1)
            except (OSError, InvalidHandshake, TimeoutError) as e:
                delay = RECONNECT_DELAY_STEPS[min(backoff, len(RECONNECT_DELAY_STEPS) - 1)]
                log.warning(f"Connection failed ({e!r}): retry in {delay}s")
                backoff += 1
                await asyncio.sleep(delay)
        log.debug("PubSub Connected.")
        connect_start = time.monotonic()
        try:
            await run_session(ws, topic, connect_success, callback)
        except PubsubReconnect:
            backoff = 0
            log.warning("Reopening the PubSub socket now (server requested reconnect).")
        except PubsubSocketClosed:
            log.warning("Reopening the PubSub socket after a delay")
        time_running = time.monotonic() - connect_start
        if connect_success.is_set() and time_running >= STABILITY_THRESHOLD:
            backoff = 0
        delay = RECONNECT_DELAY_STEPS[min(backoff, len(RECONNECT_DELAY_STEPS) - 1)]
        backoff += 1
        log.info(f"PubSub reconnecting in {delay}s...")
        await asyncio.sleep(delay)


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.DEBUG, format="%(asctime)s\t%(levelname)s\t%(name)s\t%(message)s")

    def print_redemption(event: BitRedeem) -> None:
        ts = int(event.timestamp.timestamp() * 1000)
        log.info(f"[{ts}] {event.user} redeemed '{event.title}' ({event.cost} bits)")

    CHANNEL_ID = "43825251"  # LatentLogic user id for testing
    if len(sys.argv) >= 2:
        CHANNEL_ID = sys.argv[1]
    try:
        asyncio.run(subscribe_redemptions(CHANNEL_ID, print_redemption))
    except KeyboardInterrupt:
        print("Interrupted; shutting down.")
