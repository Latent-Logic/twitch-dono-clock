import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import toml

from twitch_dono_clock.config import SETTINGS
from twitch_dono_clock.utils import Singleton

log = logging.getLogger(__name__)


class EndException(Exception):
    pass


class EndNotEnded(EndException):
    pass


class End(metaclass=Singleton):
    end_file = Path(SETTINGS.db.end_mark)

    def __init__(
        self, end_min: Optional[float] = None, end_ts: Optional[datetime] = None, ended_at_max: Optional[bool] = None
    ):
        self.end_min = end_min
        self.end_ts = end_ts
        self.ended_at_max = ended_at_max

    @classmethod
    def load_end(cls) -> "End":
        if cls.end_file.is_file():
            end_dict = toml.load(cls.end_file)
            log.info(f"Loaded end marker file and got {end_dict=}")
            return cls(**end_dict)
        return cls()

    def is_ended(self):
        return self.end_ts is not None

    def to_dict(self) -> Dict[str, Any]:
        if self.is_ended():
            return {
                "end_min": self.end_min,
                "end_ts": self.end_ts,
                "ended_at_max": self.ended_at_max,
            }
        return {}

    def save(self):
        if not self.is_ended():
            raise EndNotEnded("Can't save an end if we've not ended")
        self.end_file.write_text(toml.dumps(self.to_dict()))

    def handle_end(
        self,
        calc_time_so_far: Callable[[], timedelta],
        calc_end: Callable[[], timedelta],
        calc_total_minutes: Callable[[], float],
    ):
        if self.is_ended():
            return  # We've already reached an end state, no need for further calculations
        time_so_far = calc_time_so_far()
        available_time = calc_end()
        if time_so_far < available_time:
            return  # We've not reached our ending time yet everyone still run normally
        now = datetime.now(tz=timezone.utc)
        self.end_min = available_time.total_seconds() / 60
        self.end_ts = now - (time_so_far - available_time)
        if SETTINGS.end.max_minutes == 0:
            self.ended_at_max = False
        else:
            self.ended_at_max = calc_total_minutes() >= SETTINGS.end.max_minutes

        log.info(f"Timer has ended! {self.end_ts.isoformat()} w/ {self.end_min:.2f}min")
        self.save()

    def clear(self, reason: Optional[str] = None):
        if not self.is_ended():
            raise EndNotEnded("Can't clear an end if we've not ended!")
        old = self.to_dict()
        self.end_ts = None
        self.end_min = None
        self.ended_at_max = None
        self.end_file.unlink()
        reason_msg = f" because {reason}" if reason else ""
        log.info(f"End of {old} cleared{reason_msg}")
