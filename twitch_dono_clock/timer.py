from datetime import datetime, timedelta, timezone
from typing import Any

from twitch_dono_clock.config import SETTINGS
from twitch_dono_clock.donos import Donos
from twitch_dono_clock.end import End
from twitch_dono_clock.pause import Pause


def calc_end() -> timedelta:
    """Find the timedelta to use for final calculations"""
    end = End()
    if end.end_min:
        return timedelta(minutes=end.end_min)
    minutes = Donos().calc_total_minutes()
    if SETTINGS.end.max_minutes:
        minutes = min(minutes, SETTINGS.end.max_minutes)
    return timedelta(minutes=minutes)


def calc_time_so_far() -> timedelta:
    """How much time has been counted down since the start"""
    end = End()
    pause = Pause()
    if end.is_ended():
        cur_time = end.end_ts
    elif pause.is_paused():
        cur_time = pause.start
    else:
        cur_time = datetime.now(tz=timezone.utc)
    assert cur_time is not None
    time_so_far = cur_time - SETTINGS.start.time
    corrected_tsf = time_so_far - timedelta(minutes=pause.minutes)
    return corrected_tsf


def calc_timer_dict(handle_end: bool = True) -> dict[str, Any]:
    if handle_end:
        End().handle_end(calc_time_so_far, calc_end, Donos().calc_total_minutes)
    remaining = calc_end() - calc_time_so_far()
    return {
        "hours": int(remaining.total_seconds() / 60 / 60),
        "minutes": int(remaining.total_seconds() / 60) % 60,
        "seconds": int(remaining.total_seconds()) % 60,
        "is_locked": bool(SETTINGS.end.max_minutes and Donos().calc_total_minutes() >= SETTINGS.end.max_minutes),
        "is_paused": bool(Pause().is_paused()),
    }


def calc_timer(handle_end: bool = True) -> str:
    """Generate the timer string from the difference between paid and run minutes"""
    timer_info = calc_timer_dict(handle_end)
    time_str = "{hours:02d}:{minutes:02d}:{seconds:02d}".format(**timer_info)
    if timer_info["is_locked"]:
        time_str = SETTINGS.fmt.countdown_max.format(clock=time_str)
    if timer_info["is_paused"]:
        time_str = SETTINGS.fmt.countdown_pause.format(clock=time_str)
    return time_str
