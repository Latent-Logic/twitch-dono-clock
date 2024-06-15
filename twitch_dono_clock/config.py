import re
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union

import toml
from fastapi import HTTPException
from pydantic import BaseModel, Field, SecretStr, model_validator


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


class SetItemValue(BaseModel):
    min: float
    money: float
    msg: Dict[str, SetRegex] = Field(default_factory=dict)


class SetBits(BaseModel):
    min: float
    money: float
    animated_message_bits: int = 20
    giant_emote_bits: int = 30
    on_screen_bits: int = 40
    msg: Dict[str, SetRegex] = Field(default_factory=dict)


class SetSubsTiers(BaseModel):
    t1: SetItemValue
    t2: SetItemValue
    t3: SetItemValue


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
    tips: SetItemValue
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


try:
    SETTINGS = Settings.model_validate(toml.load("settings.toml"), strict=True)
except Exception:
    raise Exception("Failed to load settings from settings.toml, copy settings.toml.example and edit to needs")
