import pprint
from pathlib import Path
import utils2p
import shutil
import numpy as np
from tqdm import tqdm
import json
from collections import OrderedDict
import pprint as pp
from rye2p import convert_raw_fn_thorimage, process_thorsync, pid, datadirs, thormatch


def has_preprocessing_files(mov_dir, required_files=None):
    """Checks if mov_dir contains the required files, and returns a readable string."""

    files = sorted(list(mov_dir.iterdir()))
    file_names = [item.name for item in files]

    if required_files is None:
        required_files = ['Image_001_001.tif', 'stk', 'Experiment.xml', 'timestamps.npy', 'pid.npz']

    has_files = [item in file_names for item in required_files]
    checklist = np.where(np.array(has_files), "x", " ")

    # txt = f"""\n{mov_dir.relative_to(NAS_PRJ_DIR)}:"""
    txt = f"""\n{mov_dir}:"""

    for x, file in zip(checklist, required_files):
        txt = txt + f"""\n\t- [{x}] {file}"""

    print(txt)
    return txt


datadirs.set_prj('for_yang')


# %% SECTION 01: CONVERT ALL.RAW MOVIES IN FOLDER TO TIFF FILES


convert_raw_data = True
if convert_raw_data:
    folder = datadirs.RAW_DATA_DIR.joinpath('2023-02-14')
    print(f"folder: {folder}")

    convert_raw_fn_thorimage.main(folder,
                                  datadirs.RAW_DATA_DIR,
                                  datadirs.NAS_PROC_DIR)
#%% match thorsync and thorimage files
folder = datadirs.RAW_DATA_DIR.joinpath('2023-02-14/1')

proc_fly_dir = thormatch.main(folder, datadirs.RAW_DATA_DIR, datadirs.NAS_PROC_DIR, match_files=True)
# %% SECTION 02: Extract timestamps from corresponding ThorSync files for ThorImage movies

# load dataset manifesto
with open(datadirs.NAS_PRJ_DIR.joinpath('manifestos', 'flat_linked_thor_acquisitions.json'), 'r') as f:
    all_flat_acqs = json.load(f)

for item in all_flat_acqs:
    print(item)
# %%
for flat_acq in [all_flat_acqs[-1]]:
    timestamps_file, timestamps = process_thorsync.main(flat_acq,
                                                        raw_dir=datadirs.RAW_DATA_DIR,
                                                        proc_dir=datadirs.NAS_PROC_DIR,
                                                        match_stack_times_only=True)
# %%
