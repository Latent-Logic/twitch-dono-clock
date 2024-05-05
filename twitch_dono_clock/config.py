import re
from datetime import datetime
from typing import Dict, List, Tuple, Union

import toml
from pydantic import BaseModel, Field, model_validator


class SetTwitch(BaseModel):
    app_id: str
    app_secret: str
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


class SetDB(BaseModel):
    events: str
    pause: str
    pause_log: str
    end_mark: str


class SetOutput(BaseModel):
    listen: str
    port: int
    public: str
    css: str


class SetRegex(BaseModel):
    regex: str

    @property
    def re(self) -> re.Pattern:
        return re.compile(self.regex)


class SetBitsTips(BaseModel):
    min: float
    money: float
    msg: Dict[str, SetRegex] = Field(default_factory=dict)


class SetSubsTiers(BaseModel):
    t1: SetBitsTips
    t2: SetBitsTips
    t3: SetBitsTips


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
    db: SetDB
    output: SetOutput
    tips: SetBitsTips
    bits: SetBitsTips
    subs: SetSubs
    fmt: SetFmt

    _compiled_re: List[Tuple[str, re.Pattern, str]] = []

    @model_validator(mode="after")
    def compile_regex(self):
        for user, obj in self.bits.msg.items():
            self._compiled_re.append((user, obj.re, "bits"))
        for user, obj in self.tips.msg.items():
            self._compiled_re.append((user, obj.re, "tips"))
        return self

    @property
    def compiled_re(self) -> List[Tuple[str, re.Pattern, str]]:
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


try:
    SETTINGS = Settings.model_validate(toml.load("settings.toml"), strict=True)
except Exception:
    raise Exception("Failed to load settings from settings.toml, copy settings.toml.example and edit to needs")
