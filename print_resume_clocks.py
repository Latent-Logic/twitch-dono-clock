"""Print the time on clock for every resumed timestamp"""
import csv
import sys
from datetime import datetime, timezone, tzinfo
from pathlib import Path
from typing import Iterable, Tuple
from zoneinfo import ZoneInfo

import toml

import test_tracker
from test_tracker import CSV_COLUMNS, LIVE_STATS, calc_timer

TIMEZONE = "America/Los_Angeles"


def pause_generator(pause_file: Path) -> Iterable[Tuple[datetime, float]]:
    for line in pause_file.read_text().strip().split("\n"):
        if not line.strip():
            continue
        time_str, pause_str, why_str = line.split("\t")
        time = datetime.fromisoformat(time_str)
        cur_pause = float(pause_str)
        if "Ended" in why_str:
            yield time, cur_pause


def load_csv(csv_file: Path, pause_file: Path, tz: tzinfo):
    if not csv_file.is_file():
        print(f"No CSV file found at {csv_file}")
        sys.exit(1)
    if not pause_file.is_file():
        print(f"No Pause log file found at {pause_file}")
        sys.exit(2)
    pause_end_itr = pause_generator(pause_file)
    pause_end, pause = next(pause_end_itr)
    with csv_file.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=",")
        assert reader.fieldnames == CSV_COLUMNS
        for row in reader:
            abs_time = datetime.fromtimestamp(int(row["time"]) / 1000, tz=timezone.utc)
            if abs_time > pause_end:
                LIVE_STATS["pause_min"] = pause
                LIVE_STATS["end"] = {"end_ts": pause_end}
                loc_time = pause_end.astimezone(tz)
                print(f"{loc_time.isoformat()}\t{pause}  \t{calc_timer()}")
                try:
                    pause_end, pause = next(pause_end_itr)
                except StopIteration:
                    return
            if row["type"] == "bits":
                LIVE_STATS["donos"]["bits"] += int(row["amount"])
            elif row["type"] in {"direct", "tips"}:
                LIVE_STATS["donos"]["tips"] += float(row["amount"])
            elif row["type"].startswith("subs_"):
                if row["type"].endswith("_t1"):
                    LIVE_STATS["donos"]["subs"]["t1"] += int(row["amount"])
                elif row["type"].endswith("_t2"):
                    LIVE_STATS["donos"]["subs"]["t2"] += int(row["amount"])
                elif row["type"].endswith("_t3"):
                    LIVE_STATS["donos"]["subs"]["t3"] += int(row["amount"])


if __name__ == "__main__":
    SETTINGS = toml.load("settings.toml")
    test_tracker.SETTINGS = SETTINGS
    load_csv(Path(SETTINGS["db"]["events"]), Path(SETTINGS["db"]["pause_log"]), ZoneInfo(TIMEZONE))
