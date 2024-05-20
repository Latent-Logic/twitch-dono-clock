import csv
import logging
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Optional, Union

from twitchAPI.chat import ChatCommand

from twitch_dono_clock.config import SETTINGS
from twitch_dono_clock.end import End
from twitch_dono_clock.utils import Singleton

log = logging.getLogger(__name__)

SUBS, T1, T2, T3 = "subs", "t1", "t2", "t3"
CSV_COLUMNS = ["time", "user", "target", "type", "amount"]
CSV_TYPES = ["bits", "tips", f"{SUBS}_{T1}", f"{SUBS}_{T2}", f"{SUBS}_{T3}"]
BITS, TIPS, SUBS_T1, SUBS_T2, SUBS_T3 = CSV_TYPES


class Donos(metaclass=Singleton):
    dono_path = Path(SETTINGS.db.events)

    def __init__(self, new_dict: Optional[dict[str, Any]] = None):
        if new_dict is None:
            self.donos = {BITS: 0, SUBS: {T1: 0, T2: 0, T3: 0}, TIPS: 0}
        else:
            assert BITS in new_dict and TIPS in new_dict and SUBS in new_dict
            self.donos = new_dict

    @staticmethod
    def sub_from_twitch_plan(plan: str) -> str:
        return f"{SUBS}_{SETTINGS.subs.plan[plan]}"

    @classmethod
    def csv_iter(cls) -> Iterable[dict[str, str]]:
        with cls.dono_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter=",")
            assert reader.fieldnames == CSV_COLUMNS
            for row in reader:
                yield row

    @classmethod
    def read_csv(cls) -> dict[str, Any]:
        donos = {BITS: 0, SUBS: {T1: 0, T2: 0, T3: 0}, TIPS: 0}
        for row in cls.csv_iter():
            if row["type"] == BITS:
                donos[BITS] += int(row["amount"])
            elif row["type"] == TIPS:
                donos[TIPS] += float(row["amount"])
            elif row["type"].startswith("subs_"):
                if row["type"].endswith("_t1"):
                    donos[SUBS][T1] += int(row["amount"])
                elif row["type"].endswith("_t2"):
                    donos[SUBS][T2] += int(row["amount"])
                elif row["type"].endswith("_t3"):
                    donos[SUBS][T3] += int(row["amount"])
        return donos

    @classmethod
    def load_csv(cls):
        if not cls.dono_path.is_file():
            log.warning(f"No CSV file found at {cls.dono_path}, creating one")
            cls.dono_path.parent.mkdir(exist_ok=True, parents=True)
            cls.dono_path.write_text(",".join(CSV_COLUMNS) + "\n")
            return cls()
        donos = cls.read_csv()
        log.info(f"Loaded dono events CSV file and got: {donos}")
        return cls(donos)

    def to_dict(self):
        to_return = dict(self.donos)
        to_return[SUBS] = dict(to_return[SUBS])
        return to_return

    def reload_csv(self):
        old_dict = self.to_dict()
        self.donos = self.read_csv()
        log.info(f"Reloaded CSV file, went from {old_dict} to {self.donos}")

    def add_event(self, ts: int, user: str, type: str, amount: Union[int, float], target: Optional[str] = None):
        if not self.dono_path.is_file():
            raise FileNotFoundError(f"No CSV file found at {self.dono_path}, Should have been created earlier?!?")
        if type == BITS:
            self.donos[BITS] += amount
        elif type == TIPS:
            self.donos[TIPS] += amount
        elif type == SUBS_T1:
            self.donos[SUBS][T1] += amount
        elif type == SUBS_T2:
            self.donos[SUBS][T2] += amount
        elif type == SUBS_T3:
            self.donos[SUBS][T3] += amount
        else:
            raise ValueError(f"Add event w/ type {type} is not recognized")
        with self.dono_path.open("a", encoding="utf-8") as f:
            csv.DictWriter(f, CSV_COLUMNS, lineterminator="\n").writerow(
                {"time": ts, "user": user, "target": target or "", "type": type, "amount": amount}
            )

    def calc_chat_minutes(self) -> float:
        """Total number of minutes paid for by chat"""
        return sum(
            (
                self.bits * SETTINGS.bits.min,
                self.tips * SETTINGS.tips.min,
                self.subs_t1 * SETTINGS.subs.tier.t1.min,
                self.subs_t2 * SETTINGS.subs.tier.t2.min,
                self.subs_t3 * SETTINGS.subs.tier.t3.min,
            )
        )

    def calc_total_minutes(self):
        return self.calc_chat_minutes() + SETTINGS.start.minutes

    def calc_minutes_over(self) -> float:
        """How many minutes over the final calculation we are"""
        if SETTINGS.end.max_minutes:
            return self.calc_total_minutes() - SETTINGS.end.max_minutes
        else:
            return 0.0

    def calc_dollars(self) -> float:
        """Total financial gain from chat donations"""
        return sum(
            (
                self.bits * SETTINGS.bits.money,
                self.tips * SETTINGS.tips.money,
                self.subs_t1 * SETTINGS.subs.tier.t1.money,
                self.subs_t2 * SETTINGS.subs.tier.t2.money,
                self.subs_t3 * SETTINGS.subs.tier.t3.money,
            )
        )

    @property
    def tips(self) -> float:
        return self.donos[TIPS]

    @property
    def bits(self) -> int:
        return self.donos[BITS]

    @property
    def subs(self) -> int:
        return sum(self.donos[SUBS].values())

    @property
    def subs_t1(self) -> int:
        return self.donos[SUBS][T1]

    @property
    def subs_t2(self) -> int:
        return self.donos[SUBS][T2]

    @property
    def subs_t3(self) -> int:
        return self.donos[SUBS][T3]


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
