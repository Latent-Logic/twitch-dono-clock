import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import toml
from fastapi import HTTPException
from pydantic import BaseModel, Field, SecretStr, model_validator

log = logging.getLogger(__name__)


class SetTwitch(BaseModel):
    app_id: str
    app_secret: SecretStr
    channel: str
    auth_url: str
    user_token_file: str
    enable_cmds: bool
    admin_users: List[str]
    eventsub: bool
    pause_on_offline: bool
    unpause_on_online: bool

    @model_validator(mode="after")
    def streamer_in_admin(self):
        """Make sure streamer is in admin_users & that all admin_users are lowercase"""
        streamer = self.channel.lower()
        found_streamer = False
        for i, name in enumerate(self.admin_users):
            if streamer == name.lower():
                found_streamer = True
            self.admin_users[i] = name.lower()
        if not found_streamer:
            self.admin_users.append(streamer)
        return self


class SetStart(BaseModel):
    minutes: int
    time: datetime


class SetEnd(BaseModel):
    max_minutes: int

    @model_validator(mode="after")
    def valid_max(self):
        assert self.max_minutes >= 0
        return self


class SetDB(BaseModel):
    events: str = "db/events.csv"
    pause: str = "db/pause.txt"
    pause_log: str = "db/pause_log.txt"
    end_mark: str = "db/end.toml"
    spins: str = "db/spins.txt"


class SetOutput(BaseModel):
    listen: str
    port: int
    public: str
    css: str
    admin_pass: SecretStr


class SetSpins(BaseModel):
    enabled: bool = False
    value_div: float = 25

    @model_validator(mode="after")
    def div_valid(self):
        assert self.value_div > 0, f"{self.div_valid=} must be greater than 0"
        return self


class SetRegex(BaseModel):
    regex: str
    target: Optional[str] = None

    @property
    def re(self) -> re.Pattern:
        return re.compile(self.regex)


class Conversions(BaseModel):
    src: str
    target: str
    ratio: float


class SetTips(BaseModel):
    min: float
    money: float
    points: float = 1.0
    msg: Dict[str, SetRegex] = Field(default_factory=dict)
    convert: dict[str, Conversions] = Field(default_factory=dict)


class SetBits(BaseModel):
    min: float
    money: float
    points: float = 0.01
    animated_message_bits: int = 20
    giant_emote_bits: int = 30
    on_screen_bits: int = 40
    msg: Dict[str, SetRegex] = Field(default_factory=dict)


class SetSubValue(BaseModel):
    min: float
    money: float
    points: float = 1.0


class SetSubsTiers(BaseModel):
    t1: SetSubValue
    t2: SetSubValue
    t3: SetSubValue


class SetSubs(BaseModel):
    count_multimonth: bool
    count_multimonth_gift: bool
    plan: Dict[str, str]
    tier: SetSubsTiers


class SetFmt(BaseModel):
    countdown_pause: str
    cmd_blocked: str
    cmd_after_end: str
    tpause_success: str
    tpause_failure: str
    tresume_success: str
    tresume_failure: str
    traised_success: str
    missing_time_parameter_failure: str
    invalid_time_parameter_failure: str
    tadd_success: str
    tremove_success: str


class Settings(BaseModel):
    twitch: SetTwitch
    start: SetStart
    end: SetEnd
    db: SetDB = Field(default_factory=SetDB)
    output: SetOutput
    spins: SetSpins = Field(default_factory=SetSpins)
    tips: SetTips
    bits: SetBits
    subs: SetSubs
    fmt: SetFmt

    _compiled_re: List[Tuple[str, re.Pattern, str, Optional[str]]] = []

    @model_validator(mode="after")
    def compile_regex(self):
        if self.end.max_minutes:
            assert self.end.max_minutes > self.start.minutes, "end.max_minutes must be larger than start.minutes"
        for user, obj in self.bits.msg.items():
            self._compiled_re.append((user, obj.re, "bits", obj.target))
        for user, obj in self.tips.msg.items():
            self._compiled_re.append((user, obj.re, "tips", obj.target))
        return self

    @property
    def compiled_re(self) -> List[Tuple[str, re.Pattern, str, Optional[str]]]:
        return self._compiled_re

    def get_value(self, type_name: str) -> Union[float, int]:
        if type_name == "bits":
            return self.bits.money
        if type_name == "tips":
            return self.tips.money
        if type_name == "subs_t1":
            return self.subs.tier.t1.money
        if type_name == "subs_t2":
            return self.subs.tier.t2.money
        if type_name == "subs_t3":
            return self.subs.tier.t3.money

    def raise_on_bad_password(self, to_check: str):
        if to_check != self.output.admin_pass.get_secret_value().format(channel=self.twitch.channel):
            raise HTTPException(status_code=401, detail="Password is invalid")


OVERRIDE_FILE = Path("db/overrides.toml")
OVERRIDES_KEY = "overrides"


def load_overrides() -> dict[str, Any]:
    if not OVERRIDE_FILE.is_file():
        log.warning(f"No override file found at {OVERRIDE_FILE}, creating one")
        OVERRIDE_FILE.parent.mkdir(exist_ok=True, parents=True)
        with OVERRIDE_FILE.open("w") as f:
            toml.dump({OVERRIDES_KEY: {}}, f)
    try:
        override_blob = toml.load(OVERRIDE_FILE)
        overrides = override_blob[OVERRIDES_KEY]
    except Exception:
        raise
    return overrides


def _pydantic_cludge_coerce(obj: BaseModel, key: str, value: Any):
    blob = obj.model_dump()
    blob[key] = value
    new_obj = obj.model_validate(blob)
    return getattr(new_obj, key)


def implement_overrides(overrides: dict[str, Any], settings: Settings):
    for key, value in overrides.items():
        *parts, final_key = key.split(".")
        obj = settings
        try:
            for part in parts:
                obj = getattr(obj, part)
            setattr(obj, final_key, _pydantic_cludge_coerce(obj, final_key, value))
        except (AttributeError, KeyError, TypeError):
            log.error(f"Failure to process override of {key} with {value}")
            raise


try:
    SETTINGS = Settings.model_validate(toml.load("settings.toml"), strict=True)
    implement_overrides(load_overrides(), SETTINGS)
except Exception:
    raise Exception("Failed to load settings from settings.toml, copy settings.toml.example and edit to needs")


def override_value(key: str, value: Any) -> dict[str, Any]:
    *parts, final_key = key.split(".")
    obj = SETTINGS
    try:
        existing_overrides = load_overrides()
        for part in parts:
            obj = getattr(obj, part)
        value = _pydantic_cludge_coerce(obj, final_key, value)
        setattr(obj, final_key, value)
        existing_overrides[key] = value
        with OVERRIDE_FILE.open("w") as f:
            toml.dump({OVERRIDES_KEY: existing_overrides}, f)
    except (AttributeError, KeyError, TypeError):
        log.warning(f"Failed to on-the-fly override {key} with {value}")
        raise
    return existing_overrides
