from pathlib import Path
import numpy as np
import json
from rye2p import convert_raw_fn_thorimage, process_thorsync, datadirs, thormatch, olf_configs
import yaml


def has_preprocessing_files(mov_dir, required_files=None):
    """Checks if mov_dir contains the required files, and returns a readable string."""

    files = sorted(list(mov_dir.iterdir()))
    file_names = [item.name for item in files]

    if required_files is None:
        required_files = ['Image_001_001.tif',
                          'stk',
                          'Experiment.xml',
                          'timestamps.npy',
                          'pid.npz']

    has_files = [item in file_names for item in required_files]
    checklist = np.where(np.array(has_files), "x", " ")

    # txt = f"""\n{mov_dir.relative_to(NAS_PRJ_DIR)}:"""
    txt = f"""\n{mov_dir}:"""

    for x, file in zip(checklist, required_files):
        txt = txt + f"""\n\t- [{x}] {file}"""

    print(txt)
    return txt


# datadirs.set_prj('odor_space_collab')
# datadirs.set_prj('for_yang')
# datadirs.set_prj('natural_mixtures')
# datadirs.set_prj('lexi_rotation')
datadirs.set_prj('haoru_rotation')
# %% SECTION 01: CONVERT ALL.RAW MOVIES IN FOLDER TO TIFF FILES

convert_raw_data = True

if convert_raw_data:
    folder = datadirs.RAW_DATA_DIR.joinpath('2025-02-11')
    print(f"folder: {folder}")

    convert_raw_fn_thorimage.main(folder,
                                  datadirs.RAW_DATA_DIR,
                                  datadirs.NAS_PROC_DIR)

# %% match thorsync and thorimage files
# folder = datadirs.RAW_DATA_DIR.joinpath('2023-03-28/1')
#
# proc_fly_dir = thormatch.main(folder, datadirs.RAW_DATA_DIR, datadirs.NAS_PROC_DIR,
#                               match_files=True)

# %% SECTION 02: Extract timestamps from corresponding ThorSync files for ThorImage movies

# load dataset manifesto
with open(datadirs.NAS_PRJ_DIR.joinpath('manifestos', 'flat_linked_thor_acquisitions.json'),
          'r') as f:
    all_flat_acqs = json.load(f)

for flat_acq in all_flat_acqs[-4:]:
    print(flat_acq)
# %% TIMESTAMPS
for flat_acq in all_flat_acqs[-4:]:
    timestamps_file, timestamps = process_thorsync.main(flat_acq,
                                                        raw_dir=datadirs.RAW_DATA_DIR,
                                                        proc_dir=datadirs.NAS_PROC_DIR,
                                                        match_stack_times_only=True)
    
# %% Get odor stimulus info
for flat_acq in all_flat_acqs[-4:]:
    mov_dir = datadirs.NAS_PROC_DIR.joinpath(flat_acq['date_imaged'],
                                             str(flat_acq['fly_num']),
                                             flat_acq['thorimage'])

    # get stim info
    olf_config_file = datadirs.OLF_CONFIG_DIR.joinpath(flat_acq['olf_config'])
    # df_pins2odors, df_stim = olf_configs.parse_olf_config_file(olf_config_file)
    df_pins2odors, df_stim = olf_configs.parse_olf_config_file_downstairs(
            olf_config_file, is_two_component_ramp=False)

    # save stim info
    df_pins2odors.to_csv(mov_dir.joinpath('df_pins2odors.csv'), sep="\t")
    df_stim.to_csv(mov_dir.joinpath('df_stim.csv'), sep="\t")

    stim_list = dict(stim_list_flatstr=df_stim['stim'].to_list())
    with open(mov_dir.joinpath('stim_list.json'), "w") as f:
        json.dump(stim_list, f)
# %%
mov_dir = Path("/local/tensor/Haoru/rotation_project/processed_data/2025-02-18/pid/SyncData003_old")

olf_config_file = datadirs.OLF_CONFIG_DIR.joinpath("20250217_144430_stimuli_.yaml")

df_pins2odors, df_stim = olf_configs.parse_olf_config_file_downstairs(
        olf_config_file, is_two_component_ramp=False)

# save stim info
df_pins2odors.to_csv(mov_dir.joinpath('df_pins2odors.csv'), sep="\t")
df_stim.to_csv(mov_dir.joinpath('df_stim.csv'), sep="\t")

stim_list = dict(stim_list_flatstr=df_stim['stim'].to_list())
with open(mov_dir.joinpath('stim_list.json'), "w") as f:
    json.dump(stim_list, f)
# %% for yang only
# from rye2p import olf_configs_yang
#
# for flat_acq in all_flat_acqs[-1:]:
#     mov_dir = datadirs.NAS_PROC_DIR.joinpath(flat_acq['date_imaged'],
#                                              str(flat_acq['fly_num']),
#                                              flat_acq['thorimage'])
#     # get stim info
#     olf_config_file = datadirs.OLF_CONFIG_DIR.joinpath(flat_acq['olf_config'])
#     df_pins2odors, df_stim = olf_configs_yang.main(olf_config_file)
#
#     # save stim info
#     df_pins2odors.to_csv(mov_dir.joinpath('df_pins2odors.csv'))
#     df_stim.to_csv(mov_dir.joinpath('df_stim.csv'))
#
#     stim_list = dict(stim_list_flatstr=df_stim['stim'].to_list()
#                      )
#     with open(mov_dir.joinpath('stim_list.json'), "w") as f:
#         json.dump(stim_list, f)
# %%


