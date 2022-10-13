import typing
import pydantic



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


class FlyWithAcquisitions(pydantic.BaseModel):
    """ Fly info, including list of linked thor acquisitions.

    Used to parse data in {NAS_PRJ_DIR}/manifestos/linked_thor_acquisitions.yaml

    Example:
    -------

        # load linked thorlabs acquisitions (by fly) from yaml manifest
        with open(NAS_PRJ_DIR.joinpath("manifestos/linked_thor_acquisitions.yaml"), 'r') as f:
            linked_acq = yaml.safe_load(f)

        # parse w/ pydantic model
        flies_with_acq = [FlyWithAcquisitions(**item) for item in linked_acq]

    """
    date_imaged: str = pydantic.Field(...)
    fly_num: int = pydantic.Field(...)
    linked_thor_acquisitions: typing.List[LinkedThorAcquisition]


class ThorSyncLineTimes(pydantic.BaseModel):
    frame_times: list[float]
    stack_times: list[float]
    scope_ict: list[float]
    scope_fct: list[float]
    olf_ict: list[float]
    olf_fct: list[float]


class PinOdor(pydantic.BaseModel):
    name: str
    log10_conc: typing.Optional[float]
    abbrev: typing.Optional[str]