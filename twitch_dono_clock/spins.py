import logging
from pathlib import Path

from twitch_dono_clock.config import SETTINGS
from twitch_dono_clock.utils import Singleton

log = logging.getLogger(__name__)


class Spins(metaclass=Singleton):
    spin_file = Path(SETTINGS.db.spins)
    enabled = SETTINGS.spins.enabled
    div_val = SETTINGS.spins.value_div

    def __init__(self, starting_spins: int = 0):
        self._performed = starting_spins

    @classmethod
    def load_spins(cls):
        if not cls.enabled:
            log.debug("Spins not enabled, not loading")
            return
        if not cls.spin_file.is_file():
            log.warning(f"No spin file found at {cls.spin_file}, creating one")
            cls.spin_file.parent.mkdir(exist_ok=True, parents=True)
            cls.spin_file.write_text("0")
            return cls()
        performed = int(cls.spin_file.read_text())
        log.info(f"Loaded Spin file and got {performed=}")
        return cls(performed)

    def save(self):
        self.spin_file.write_text(str(self._performed))

    @property
    def performed(self) -> int:
        return self._performed

    def spin_performed(self):
        self._performed += 1
        self.save()

    def set_performed(self, new: int):
        self._performed = new
        self.save()

    def calc_todo(self, value: float) -> float:
        return value / self.div_val
