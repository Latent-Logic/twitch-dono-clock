"""Print the time on clock for every resumed timestamp"""
import sys
from datetime import datetime, timezone, tzinfo
from pathlib import Path
from typing import Iterable, Tuple
from zoneinfo import ZoneInfo

from test_tracker import calc_timer
from twitch_dono_clock.donos import BITS, SUBS, T1, T2, T3, TIPS, Donos
from twitch_dono_clock.end import End
from twitch_dono_clock.pause import Pause

TIMEZONE = "America/Los_Angeles"


def pause_generator(pause_file: Path) -> Iterable[Tuple[datetime, float]]:
    for line in pause_file.read_text().strip().split("\n"):
        if not line.strip():
            continue
        time_str, pause_str, why_str = line.split("\t")
        time = datetime.fromisoformat(time_str)
        cur_pause = float(pause_str)
        if "Ended" in why_str and "online" in why_str:
            yield time, cur_pause


def load_csv(pause_file: Path, tz: tzinfo):
    if not pause_file.is_file():
        print(f"No Pause log file found at {pause_file}")
        sys.exit(2)
    pause_end_itr = pause_generator(pause_file)
    pause_end, pause = next(pause_end_itr)
    for row in Donos.csv_iter():
        abs_time = datetime.fromtimestamp(int(row["time"]) / 1000, tz=timezone.utc)
        if abs_time > pause_end:
            Pause()._minutes = pause
            End().end_ts = pause_end
            loc_time = pause_end.astimezone(tz)
            print(f"{loc_time.isoformat()}\t{pause}  \t{calc_timer(handle_end=False)}")
            try:
                pause_end, pause = next(pause_end_itr)
            except StopIteration:
                return
        if row["type"] == BITS:
            Donos().donos[BITS] += int(row["amount"])
        elif row["type"] == TIPS:
            Donos().donos[TIPS] += float(row["amount"])
        elif row["type"].startswith("subs_"):
            if row["type"].endswith("_t1"):
                Donos().donos[SUBS][T1] += int(row["amount"])
            elif row["type"].endswith("_t2"):
                Donos().donos[SUBS][T2] += int(row["amount"])
            elif row["type"].endswith("_t3"):
                Donos().donos[SUBS][T3] += int(row["amount"])


if __name__ == "__main__":
    load_csv(Pause.log_file, ZoneInfo(TIMEZONE))
