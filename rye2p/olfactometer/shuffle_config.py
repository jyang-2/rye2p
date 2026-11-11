from collections import OrderedDict
from pathlib import Path
import random
import pydantic
from pydantic import BaseModel
import pendulum
from typing import Union, List, Dict, TypedDict, Optional
import yaml
import pprint as pp
from pydantic import BaseModel
import itertools
from more_itertools import partition
import shutil

# OLF_CONFIG_DIR = Path(r"C:\Users\user\src\tom_olfactometer_configs")
OLF_CONFIG_DIR = Path("/local/tensor/Haoru/rotation_project/olfactometer_configs")


class Odor(BaseModel):
    name: str
    log10_conc: float
    abbrev: str = None
    # inventory: Optional[str] = None
    # alias: Optional[str] = None

    def __hash__(self):
        return hash((type(self),) + tuple(self.__dict__.values()))

    def __eq__(self, other):
        if other.__class__ is self.__class__:
            return self.dict() == other.dict()


def load_experiment_type(yaml_file):
    """ex: kiwi.yaml"""
    with open(OLF_CONFIG_DIR.joinpath(yaml_file), 'r') as f:
        olf_type = yaml.safe_load(f)
    return olf_type


def load_experiment_config(yaml_file):
    with open(OLF_CONFIG_DIR.joinpath(yaml_file), 'r') as f:
        olf_config = yaml.safe_load(f)
    return olf_config


def is_paraffin(odor_vial):
    return odor_vial.name in ['pfo', 'paraffin']


def new_yaml_file():
    nt = pendulum.now()
    now_str = nt.format('YYYYMMDD_HHmmss')
    new_yaml_filename = f"{now_str}_stimuli_.yaml"
    return new_yaml_filename


def shuffle_odor_list(odors, paraffin_first=True):
    odor_vials, pfo_vials = partition(is_paraffin, list(odors))
    pfo_vials = list(pfo_vials)
    odor_vials = list(odor_vials)

    random.shuffle(odor_vials)
    new_stim_order = pfo_vials + odor_vials
    return new_stim_order


def get_reference_pins2odors(yaml_config):
    yaml_config_data = load_experiment_config(yaml_config)

    pins_2_odors = {k: Odor.parse_obj(v)
                    for k, v in yaml_config_data['pins2odors'].items()
                    }
    return pins_2_odors


def get_pin_list(stim_order_list, ref_pin2odor):
    ref_odor2pin = {v: k for k, v in ref_pin2odor.items()}  # no repeats
    new_pin_list = [ref_odor2pin[item] for item in stim_order_list]
    return new_pin_list


def copy_yaml_file(yaml_file):
    if isinstance(yaml_file, str):
        src = OLF_CONFIG_DIR.joinpath(yaml_file)
    else:
        src = Path(yaml_file)

    new_yaml = new_yaml_file()
    dst = OLF_CONFIG_DIR.joinpath(new_yaml)

    shutil.copyfile(src, dst)
    print(f"\ncopying yaml file:")
    print(f"\tsrc: {src.name}")
    print(f"\tdst: {dst.name}")
    return dst


def most_recent_file(folder=None):
    if folder is None:
        folder = OLF_CONFIG_DIR
    return max(OLF_CONFIG_DIR.iterdir(), key=lambda x: x.stat().st_ctime)


# %% SECTION 00: GET ODORS FROM MOVIE CONFIG (like kiwi.yaml)
"""
movie_yaml should be one of the following:
files = [
'validation1.yaml', 
'odorspace0.yaml'
]
"""
prj = 'haoru_rotation'
movie_yaml = 'kiwi.yaml'

stimuli_to_remove = None

movie_settings = load_experiment_type(movie_yaml)
movie_stimuli = [Odor.parse_obj(item) for item in movie_settings['odors']]
# movie_stimuli = pydantic.parse_obj_as(List[Odor], movie_settings['odors'])
print(f'\n{movie_yaml} --> odor stimuli:\n----------------------------------')

pp.pprint(movie_stimuli)

if stimuli_to_remove is not None:
    for item in stimuli_to_remove:
        movie_stimuli.remove(item)

print(f'\n{movie_yaml} --> odor stimuli:\n----------------------------------')
pp.pprint(movie_stimuli)
# %% SECTION 01: LOAD REFERENCE PINS2ODORS
""" Load reference pin mapping """

ref_yaml = "20250115_164737_stimuli.yaml"
ref_pin_map = get_reference_pins2odors(ref_yaml)

print(f'\n\n{ref_yaml} --> ref. pins2odors:\n----------------------------------')
pp.pprint(ref_pin_map)
# %% SECTION 02: SHUFFLE STIMULI


if prj in ['mb_odor_space', 'haoru_rotation']:
    n_blocks = 3
    consecutive_odor_repeats = False

    if consecutive_odor_repeats:  # grouped stimulus presentation
        new_stim_list = [item for item in shuffle_odor_list(movie_stimuli)
                         for i in range(n_blocks)]
    else:  # non-consecutive, shuffled stimuli
        if n_blocks > 1:
            block_stim = [shuffle_odor_list(movie_stimuli) for i in range(n_blocks)]
            new_stim_list = list(itertools.chain(*block_stim))
        else:
            new_stim_list = shuffle_odor_list(movie_stimuli)

print('new_stim_list:')
pp.pprint(new_stim_list)

# %% SECTION 03: IF HAPPY WITH `new_stim_list` convert to pin_groups dict

# TODO: Add `pin_2_balance` values

balance_2_odor_pins = {42: [37, 38, 39, 40, 41],
                       50: [45, 46, 47, 48, 49]}

odor_pins_2_balance = {v: k for k, v in balance_2_pins.items()}

new_stim_pins = get_pin_list(new_stim_list, ref_pin_map)

new_pin_groups = [dict(pins=[item]) for item in new_stim_pins]

# %% Print yaml-formatted pin groups
print(yaml.dump(new_pin_groups))

# %% Write new yaml config (only change the `pin_sequence` item)

yaml_data = load_experiment_config(ref_yaml)

new_yaml = new_yaml_file()
yaml_data['pin_sequence']['pin_groups'] = new_pin_groups

with open(OLF_CONFIG_DIR.joinpath(new_yaml), 'w') as f:
    yaml.dump(yaml_data, f, sort_keys=False)

print(f"New olf_config created:")
print(f"\tref_yaml: {ref_yaml}")
print(f"\tnew_yaml: {new_yaml}")
