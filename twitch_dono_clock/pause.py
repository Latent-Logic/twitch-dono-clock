import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from twitchAPI.chat import ChatCommand

from twitch_dono_clock.config import SETTINGS
from twitch_dono_clock.end import End
from twitch_dono_clock.utils import Singleton

log = logging.getLogger(__name__)


class PauseException(Exception):
    pass


class PauseNotPaused(PauseException):
    pass


class PauseInProgress(PauseException):
    pass


class Pause(metaclass=Singleton):
    pause_file = Path(SETTINGS.db.pause)
    log_file = Path(SETTINGS.db.pause_log)

    def __init__(self, minutes: float = 0.0, start: Optional[datetime] = None):
        self._minutes = minutes
        self._start = start

    @classmethod
    def load_pause(cls) -> "Pause":
        if not cls.pause_file.is_file():
            log.warning(f"No pause file found at {cls.pause_file}, creating one")
            cls.pause_file.parent.mkdir(exist_ok=True, parents=True)
            cls.pause_file.write_text("0.0")
            return cls()
        raw = cls.pause_file.read_text()
        if ";" in raw:
            time_str, pause_time = raw.strip().split(";", maxsplit=1)
            start = datetime.fromisoformat(pause_time)
        else:
            time_str = raw
            start = None
        time = float(time_str)
        log.info(f"Loaded Pause file and got {time=} {start=}")
        return cls(time, start)

    def save(self):
        if self.is_paused():
            self.pause_file.write_text(f"{self.minutes:.02f};{self._start.isoformat()}")
        else:
            self.pause_file.write_text(f"{self.minutes:.02f}")

    def to_dict(self) -> dict[str, Any]:
        return {"minutes": self.minutes, "start": self.start}

    def log_pause_change(self, why: str):
        now = datetime.now(timezone.utc)
        line = f"{now.isoformat()}\t{self.minutes:.2f}\t{why}\n"
        log.info(line.rstrip("\n"))
        with self.log_file.open("a") as f:
            f.write(line)

    @property
    def minutes(self) -> float:
        return self._minutes

    def pause_set(self, minutes: float, reason: Optional[str] = None):
        old_min = self._minutes
        self._minutes = minutes
        self.save()
        reason_msg = f" because {reason}" if reason else ""
        self.log_pause_change(f"Hard changed minutes from {old_min}{reason_msg}")

    def pause_increase(self, minutes: float, reason: Optional[str] = None):
        old_min = self._minutes
        self._minutes += minutes
        self.save()
        reason_msg = f" because {reason}" if reason else ""
        self.log_pause_change(f"Pause time increased by {minutes}min from {old_min}{reason_msg}")

    def pause_reduce(self, minutes: float, reason: Optional[str] = None):
        old_min = self._minutes
        self._minutes -= minutes
        self.save()
        reason_msg = f" because {reason}" if reason else ""
        self.log_pause_change(f"Pause time reduced by {minutes}min from {old_min}{reason_msg}")

    @property
    def start(self) -> Optional[datetime]:
        return self._start

    def is_paused(self):
        return self._start is not None

    def start_pause(self, reason: Optional[str] = None) -> datetime:
        if self.is_paused():
            raise PauseInProgress(f"Can't start pause as it already started at {self.start}")
        pause_start = datetime.now(tz=timezone.utc)
        self._start = pause_start
        self.save()
        self.log_pause_change(f"Pause Started because {reason}" if reason else "Pause Started")
        return pause_start

    def set_pause_start(self, time: datetime, reason: Optional[str] = None):
        existing_start = self._start
        self._start = time
        self.save()
        reason_msg = f" because {reason}" if reason else ""
        if existing_start:
            self.log_pause_change(
                f"Pause start changed from {existing_start.isoformat()} to {time.isoformat()}{reason_msg}"
            )
        else:
            self.log_pause_change(f"Pause started backdated to {time.isoformat()}{reason_msg}")

    def resume(self, reason: Optional[str] = None) -> float:
        if not self.is_paused():
            raise PauseNotPaused("Can't resume a paused timer if we're not paused")
        added_min = (datetime.now(tz=timezone.utc) - self._start).total_seconds() / 60
        self._minutes += added_min
        self._start = None
        self.save()
        reason_msg = f"because {reason} " if reason else ""
        self.log_pause_change(f"Pause Ended {reason_msg}& added {added_min:.2f}")
        return added_min

    def resumed_at(self, time: datetime, reason: Optional[str] = None) -> float:
        if not self.is_paused():
            raise PauseNotPaused("Can't resume a paused timer if we're not paused")
        if time <= self._start:
            raise PauseException(f"Can't resume timer before the pause started (at {self._start.isoformat()})")
        added_min = (time - self._start).total_seconds() / 60
        self._minutes += added_min
        self._start = None
        self.save()
        reason_msg = f"because {reason} " if reason else ""
        self.log_pause_change(f"Pause Ended at {time.isoformat()} {reason_msg}& added {added_min:.2f}")
        return added_min

    def abort_current(self, reason: Optional[str] = None) -> datetime:
        if not self.is_paused():
            raise PauseNotPaused("Can't remove a paused from the timer if we're not paused")
        old_start = self.start
        self._start = None
        self.save()
        reason_msg = f" because {reason}" if reason else ""
        self.log_pause_change(f"Pause start of {old_start.isoformat()} aborted{reason_msg}")
        return old_start


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
        response = SETTINGS.fmt.tpause_success.format(**fmt_dict)
        log.info(response)
        await cmd.reply(response)


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
        response = SETTINGS.fmt.tresume_success.format(**fmt_dict)
        log.info(response)
        await cmd.reply(response)


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
    response = SETTINGS.fmt.tadd_success.format(**fmt_dict)
    log.info(response)
    await cmd.reply(response)


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
    response = SETTINGS.fmt.tremove_success.format(**fmt_dict)
    log.info(response)
    await cmd.reply(response)
