"""Module for parsing olfactometer configs from the old format (see
odor_space_collab/old_olfactometer_configs)"""

from pathlib import Path
from rye2p import datadirs
# from rye2p.pydantic_models import OldFlatOdorSpaceAcquisition
# import pydantic
from typing import List
import yaml
import json
import pandas as pd


def parse_olf_config_file(olf_config_file):
    with olf_config_file.open('r') as f:
        olf_config_data = yaml.safe_load(f)

    df_pins2odors = pd.DataFrame.from_dict(olf_config_data['pins2odors'], orient='index')
    df_pins2odors.index.name = 'pin'
    df_pins2odors = df_pins2odors.astype({'log10_conc': float})
    df_pins2odors['stim'] = df_pins2odors['abbrev'] + ' @ ' + df_pins2odors['log10_conc'].astype(
            str)

    single_pin_sequence = (pd.DataFrame(olf_config_data['pin_sequence']['pin_groups'])['pins']
                           .explode()
                           .to_list()
                           )
    df_stim = df_pins2odors.loc[single_pin_sequence, :].reset_index()
    df_stim.index.name = 'trial_idx'

    return df_pins2odors, df_stim


def parse_olf_config_file_downstairs(olf_config_file, guess_balance_pins=True, balance_pins=None,
                                     is_two_component_ramp=False):
    with olf_config_file.open('r') as f:
        olf_config_data = yaml.safe_load(f)

    df_pins2odors = pd.DataFrame.from_dict(olf_config_data['pins2odors'], orient='index')
    df_pins2odors.index.name = 'pin'
    df_pins2odors = df_pins2odors.loc[:, ['name', 'log10_conc', 'abbrev']]
    df_pins2odors = df_pins2odors.astype({'log10_conc': float})
    df_pins2odors['stim'] = df_pins2odors['abbrev'] + ' @ ' + df_pins2odors['log10_conc'].astype(
            str)

    single_pin_sequence = (pd.DataFrame(olf_config_data['pin_sequence']['pin_groups'])['pins']
                           .explode()
                           .to_list()
                           )
    if guess_balance_pins and balance_pins is None:
        balance_pins = list(set(single_pin_sequence) - set(df_pins2odors.index))

    pin_sequence = pd.DataFrame(olf_config_data['pin_sequence']['pin_groups'])
    pin_sequence.index.name = 'trial_idx'
    filtered_pin_sequence = pin_sequence.explode('pins').query("pins not in @balance_pins")

    if is_two_component_ramp:
        df_stim_by_trial = filtered_pin_sequence.join(df_pins2odors, on='pins')
        df_two_component_stim = (df_stim_by_trial.loc[:, ['abbrev', 'stim']]
                                 .set_index('abbrev', append=True)
                                 .unstack('abbrev')
                                 .droplevel(0, axis=1)
                                 )
        for col in df_two_component_stim.columns:
            df_two_component_stim.loc[df_two_component_stim[col].str.contains('nan'), col] = ""
        df_two_component_stim["stim"] = df_two_component_stim[df_two_component_stim.columns].apply(
            ", ".join, axis=1)
        df_two_component_stim['stim'] = df_two_component_stim['stim'].str.strip(" ,")
        df_two_component_stim['stim'] = df_two_component_stim['stim'].replace("", 'pfo @ 0.0')

        df_stim = df_two_component_stim
    else:
        df_stim = df_pins2odors.loc[filtered_pin_sequence['pins'], :].reset_index()
        df_stim.index.name = 'trial_idx'

    return df_pins2odors, df_stim


def load_old_olf_config(olf_config_name, olf_config_dir=datadirs.OLF_CONFIG_DIR):
    olf_config_file = olf_config_dir.joinpath(olf_config_name)
    with olf_config_file.open('r') as f:
        olf_config_data = yaml.safe_load(f)
    return olf_config_data


def load_old_reference_panel():
    olf_ref_yaml = Path(
            "/local/matrix/Remy-Data/projects/odor_space_collab/old_olfactometer_configs"
            "/all_panels_reference.yaml")
    with olf_ref_yaml.open('r') as f:
        ref_odors = yaml.safe_load(f)
    return ref_odors


def construct_old_olf_config_filename(flat_acq):
    return f"_{flat_acq.date_imaged.replace('-', '')}_panels.yaml"

# %%
# def _convert_old_olf_configs():
#     ################
#     # load manifest
#     ################
#     prj = 'mb_odor_space'
#     datadirs.set_prj(prj)
#
#     ref_odor_panel = load_old_reference_panel()
#     # %%
#     # make a dictionary mapping reference odor abbreviations to the full odor definition
#     abbrev2odor = {odor['abbrev']: odor for odor in ref_odor_panel['odors']}
#
#     # change odor key abbreviations so that they are all lowercase
#     abbrev2odor = {k.lower(): v for k, v in abbrev2odor.items()}
#     print(abbrev2odor)
#     # %%
#
#     MANIFEST_FILE = datadirs.NAS_PRJ_DIR.joinpath('manifestos',
#                                                   'flat_linked_thor_acquisitions_old.json'
#                                                   )
#     all_flat_acqs = pydantic.parse_file_as(List[OldFlatOdorSpaceAcquisition],
#                                            MANIFEST_FILE)
#
#     for flacq in all_flat_acqs:
#         # load olf config data
#         print(f'\n---')
#         print(flacq)
#
#         old_olf_config_name = construct_old_olf_config_filename(flacq)
#         print(f"\t- old olf_config: {old_olf_config_name}")
#         print(f"\t- old panel: {flacq.panel}")
#         print(f"\t- pin_sequence: {flacq.pin_sequence}")
#
#         olf_config = load_old_olf_config(old_olf_config_name,
#                                          olf_config_dir=datadirs.OLF_CONFIG_DIR)
#
#         # edit abbrev so that they're all lowercase, and convert
#         pins2abbrev = olf_config['panels'][flacq.panel]
#         pins2abbrev = {k: v.lower() for k, v in pins2abbrev.items()}
#         print(pins2abbrev)
#
#         # map pins2abbrev * abbrev2odor --> pins2odors
#         pins2odors = {k: abbrev2odor[v] for k, v in pins2abbrev.items()}
#
#         # build new olf config dict
#         pin_list = flacq.pin_sequence
#         pin_groups = [dict(pins=[item]) for item in pin_list]
#
#         olf_config_dict = dict(pin_sequence=dict(pin_groups=pin_groups),
#                                pins2odors=pins2odors)
#
#         # construct new olf_config name
#         new_olf_config_file = "{}-fly{:02d}-{}_.yaml".format(flacq.date_imaged.replace('-', ''),
#                                                              flacq.fly_num,
#                                                              flacq.thorimage)
#         print(f"\t- new olf_config: {old_olf_config_name}")
#         with datadirs.OLF_CONFIG_DIR.joinpath(new_olf_config_file).open('w') as f:
#             yaml.dump(olf_config_dict, f)
