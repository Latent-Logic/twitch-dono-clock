import logging
from pathlib import Path

from twitchAPI.chat import ChatCommand

from twitch_dono_clock.config import SETTINGS
from twitch_dono_clock.donos import Donos
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

    def spin_performed(self, inc_amount: int = 1):
        self._performed += inc_amount
        log.info(f"Spin counter incremented by {inc_amount} to {self._performed}")
        self.save()

    def set_performed(self, new_value: int):
        log.info(f"Spin counter set from {self._performed} to {new_value}")
        self._performed = new_value
        self.save()

    def calc_todo(self, value: float) -> float:
        return value / self.div_val


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
    elif parameters[0].lower() == "check":
        response = "Spin counter is at {spins_done}/{spins_to_do}".format(**fmt_dict)
    elif len(parameters) == 1:
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
