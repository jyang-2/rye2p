"""Module for parsing olfactometer configs from the old format (see odor_space_collab/old_olfactometer_configs)"""

from pathlib import Path
from rye2p import datadirs
from rye2p.pydantic_models import OldFlatOdorSpaceAcquisition
import pydantic
from typing import List
import yaml
import json


def load_old_olf_config(olf_config_name, olf_config_dir=datadirs.OLF_CONFIG_DIR):
    olf_config_file = olf_config_dir.joinpath(olf_config_name)
    with olf_config_file.open('r') as f:
        olf_config_data = yaml.safe_load(f)
    return olf_config_data


def load_old_reference_panel():
    olf_ref_yaml = Path("/local/matrix/Remy-Data/projects/odor_space_collab/old_olfactometer_configs"
                        "/all_panels_reference.yaml")
    with olf_ref_yaml.open('r') as f:
        ref_odors = yaml.safe_load(f)
    return ref_odors


def construct_old_olf_config_filename(flat_acq):
    return f"_{flat_acq.date_imaged.replace('-', '')}_panels.yaml"


# %%
if __name__ == '__main__':
    ################
    # load manifest
    ################
    prj = 'mb_odor_space'
    datadirs.set_prj(prj)

    ref_odor_panel = load_old_reference_panel()
    # %%
    # make a dictionary mapping reference odor abbreviations to the full odor definition
    abbrev2odor = {odor['abbrev']: odor for odor in ref_odor_panel['odors']}

    # change odor key abbreviations so that they are all lowercase
    abbrev2odor = {k.lower(): v for k, v in abbrev2odor.items()}
    print(abbrev2odor)
    # %%

    MANIFEST_FILE = datadirs.NAS_PRJ_DIR.joinpath('manifestos',
                                                  'flat_linked_thor_acquisitions_old.json'
                                                  )
    all_flat_acqs = pydantic.parse_file_as(List[OldFlatOdorSpaceAcquisition],
                                           MANIFEST_FILE)

    for flacq in all_flat_acqs:
        # load olf config data
        print(f'\n---')
        print(flacq)

        old_olf_config_name = construct_old_olf_config_filename(flacq)
        print(f"\t- old olf_config: {old_olf_config_name}")
        print(f"\t- old panel: {flacq.panel}")
        print(f"\t- pin_sequence: {flacq.pin_sequence}")

        olf_config = load_old_olf_config(old_olf_config_name, olf_config_dir=datadirs.OLF_CONFIG_DIR)

        # edit abbrev so that they're all lowercase, and convert
        pins2abbrev = olf_config['panels'][flacq.panel]
        pins2abbrev = {k: v.lower() for k, v in pins2abbrev.items()}
        print(pins2abbrev)

        # map pins2abbrev * abbrev2odor --> pins2odors
        pins2odors = {k: abbrev2odor[v] for k, v in pins2abbrev.items()}

        # build new olf config dict
        pin_list = flacq.pin_sequence
        pin_groups = [dict(pins=[item]) for item in pin_list]

        olf_config_dict = dict(pin_sequence=dict(pin_groups=pin_groups),
                               pins2odors=pins2odors)

        # construct new olf_config name
        new_olf_config_file = "{}-fly{:02d}-{}_.yaml".format(flacq.date_imaged.replace('-', ''),
                                                             flacq.fly_num,
                                                             flacq.thorimage)
        print(f"\t- new olf_config: {old_olf_config_name}")
        with datadirs.OLF_CONFIG_DIR.joinpath(new_olf_config_file).open('w') as f:
            yaml.dump(olf_config_dict, f)
    #%%
