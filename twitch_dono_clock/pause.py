import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from twitch_dono_clock.config import SETTINGS
from twitch_dono_clock.utils import Singleton

log = logging.getLogger(__name__)


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

    def log_pause_change(self, why: str):
        now = datetime.now(timezone.utc)
        line = f"{now.isoformat()}\t{self.minutes:.2f}\t{why}\n"
        log.debug(line)
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
        pause_start = datetime.now(tz=timezone.utc)
        self._start = pause_start
        self.save()
        self.log_pause_change(f"Pause Started because {reason}" if reason else "Pause Started")
        return pause_start

    def set_pause_start(self, time: datetime):
        existing_start = self._start
        if time > datetime.now(tz=timezone.utc):
            raise ValueError(f"{time} is in the future, not starting a pause before now.")
        self._start = time
        self.save()
        if existing_start:
            self.log_pause_change(f"Pause start changed from {existing_start.isoformat()} to {time.isoformat()}")
        else:
            self.log_pause_change(f"Pause started backdated to {time.isoformat()}")

    def resume(self, reason: Optional[str] = None):
        assert self.is_paused(), "Can't resume a paused timer if we're not paused"
        added_min = (datetime.now(tz=timezone.utc) - self._start).total_seconds() / 60
        self._minutes += added_min
        self._start = None
        self.save()
        reason_msg = f"because {reason} " if reason else ""
        self.log_pause_change(f"Pause Ended {reason_msg}& added {added_min:.2f}")
        return added_min
