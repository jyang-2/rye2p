import typing
import pydantic
from typing import List
from pathlib import Path


class Fly(pydantic.BaseModel):
    date_imaged: str
    fly_num: int
    genotype: str
    date_eclosed: str
    sex: str


class ThorImage(pydantic.BaseModel):
    date_time: str
    utime: int
    name: str
    ori_path: str
    rel_path: str


class ThorSync(pydantic.BaseModel):
    name: str
    ori_path: str
    rel_path: str


class LinkedThorAcquisition(pydantic.BaseModel):
    """ Holds info for looking up fly, thorsync, and thorimage file ids"""
    thorimage: str = pydantic.Field(...)
    thorsync: str = pydantic.Field(...)
    olf_config: str = pydantic.Field(...)
    notes: typing.Optional[str]


class OldFlatOdorSpaceAcquisition(pydantic.BaseModel):
    date_imaged: str = pydantic.Field(...)
    fly_num: int = pydantic.Field(...)
    thorimage: str = pydantic.Field(...)
    thorsync: typing.Optional[str]
    # fly_id: int = pydantic.Field(...)
    # expt_id: int = pydantic.Field(...)
    olf_config: typing.Optional[str]
    panel: typing.Optional[str]
    pin_sequence: typing.Optional[List[int]]
    bad_trials: typing.Optional[List]
    imaging_type: typing.Optional[str]
    s2p_stat_file: typing.Optional[str]

    def filename_base(self):
        return f"{self.date_imaged}__fly{self.fly_num:02}__{self.thorimage}"

    # def expt_dir(self, relative_to=None):
    #     if relative_to is None:
    #         parent_dir = Path(".")
    #     else:
    #         parent_dir = relative_to
    #     return parent_dir.joinpath(str(self.fly_id), str(self.expt_id))
    #
    # def fly_dir(self, relative_to=None):
    #     if relative_to is None:
    #         parent_dir = Path(".")
    #     else:
    #         parent_dir = relative_to
    #     return parent_dir.joinpath(str(self.fly_id))

    def fly_dir(self, relative_to=None):
        if relative_to is None:
            parent_dir = Path(".")
        else:
            parent_dir = relative_to
        fly_dir = parent_dir.joinpath(self.date_imaged, self.fly_num)
        return fly_dir

    def mov_dir(self, relative_to=None):
        if relative_to is None:
            parent_dir = Path(".")
        else:
            parent_dir = relative_to
        mov_dir = parent_dir.joinpath(self.date_imaged, self.fly_num, self.thorimage)
        return mov_dir

    def stat_file(self, relative_to=None):
        if self.s2p_stat_file is None:
            stat_file = None
        else:
            if relative_to is None:
                parent_dir = Path(".")
            else:
                parent_dir = relative_to
            stat_file = parent_dir / Path(self.s2p_stat_file)
        return stat_file


# class ThorSyncLineTimes(pydantic.BaseModel):
#     frame_times: list[float]
#     stack_times: list[float]
#     scope_ict: list[float]
#     scope_fct: list[float]
#     olf_ict: list[float]
#     olf_fct: list[float]


class PinOdor(pydantic.BaseModel):
    name: str
    log10_conc: typing.Optional[float]
    abbrev: typing.Optional[str]
