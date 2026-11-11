from pathlib import Path
from rye2p import datadirs
import pydantic
from typing import List
import yaml
import json
import pandas as pd
import re


def load_olf_config(olf_config_file):
    with olf_config_file.open('r') as f:
        olf_config_data = yaml.safe_load(f)
    return olf_config_data


def olf_config_data_2_pins2odors(olf_config_data, available_pins=None):
    df_pins2odors = pd.DataFrame.from_dict(
            dict(
                    odor_pin_full=olf_config_data['odor_pin_full'],
                    odor_pin=olf_config_data['odor_pin'])
            )

    pins_used = list(set(olf_config_data['odor_stim_seq']))
    df_pins2odors.index = df_pins2odors.index + 1

    df_pins2odors = df_pins2odors.loc[pins_used, :]

    odor_col, odor_conc_col = zip(
            *df_pins2odors['odor_pin_full'].apply(odor_str_2_pin_odor).to_list())

    odor_col, odor_conc_col = zip(
            *df_pins2odors['odor_pin_full'].apply(odor_str_2_pin_odor).to_list())

    abbrev_col, abbrev_conc_col = zip(
            *df_pins2odors['odor_pin'].apply(odor_str_2_pin_odor).to_list())

    df_pins2odors['odor'] = odor_col
    df_pins2odors['odor_conc'] = odor_conc_col

    df_pins2odors['abbrev'] = abbrev_col
    df_pins2odors['abbrev_conc'] = abbrev_conc_col

    if (df_pins2odors['abbrev_conc'] == df_pins2odors['odor_conc']).all():
        df_pins2odors = df_pins2odors.loc[:, ['odor', 'abbrev', 'odor_conc']]
        df_pins2odors.rename(columns={'odor_conc': 'conc'}, inplace=True)
        df_pins2odors['stim'] = df_pins2odors['abbrev'] + ' @ ' + df_pins2odors['conc'].astype(str)
    df_pins2odors.index.name = 'pin'
    return df_pins2odors


def odor_str_2_pin_odor(odor_str):
    pattern = "(.+)\((.*)\)"

    if '(' in odor_str and ')' in odor_str:
        odor, conc = re.search(pattern, odor_str).groups()
        conc = float(conc)
        odor = odor.strip()
    else:
        odor, conc = (odor_str, 0.0)
    return odor, conc


def main(olf_config_name):
    olf_config_dir = Path('/local/matrix/Remy-Data/for_others/for_yang/olfactometer_configs')
    olf_config_data = load_olf_config(olf_config_dir.joinpath(olf_config_name))

    df_pins2odors = olf_config_data_2_pins2odors(olf_config_data)
    df_stim = df_pins2odors.loc[olf_config_data['odor_stim_seq'], :]

    return df_pins2odors, df_stim
# %% make


# %%
