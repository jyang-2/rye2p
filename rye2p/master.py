import pprint
from pathlib import Path
import utils2p
import shutil
import numpy as np
from tqdm import tqdm
import json
from collections import OrderedDict
import pprint as pp

from rye2p import convert_raw_fn_thorimage, process_thorsync, pid


def _get_directories(prj=None):
    if prj is None:
        prj = 'odor_space_collab'

    if prj == 'natural_mixtures':
        _RAW_DATA_DIR = Path("/media/remy/remy-storage/Remy's Dropbox Folder/HongLab @ Caltech Dropbox/Remy"
                            "/natural_mixtures/raw_data")
        _PROC_DATA_DIR = Path("/local/matrix/Remy-Data/projects/natural_mixtures/processed_data")
    elif prj == 'narrow_odors':
        _RAW_DATA_DIR = Path("/local/storage/Remy/narrow_odors/raw_data")
        _PROC_DATA_DIR = Path("/local/storage/Remy/narrow_odors/processed_data")
    elif prj == 'odor_space_collab':
        _RAW_DATA_DIR = Path("/media/remy/remy-storage/Remy's Dropbox Folder/HongLab @ Caltech "
                            "Dropbox/Remy/odor_space_collab/raw_data")
        _PROC_DATA_DIR = Path("/local/matrix/Remy-Data/projects/odor_space_collab/processed_data")

    _NAS_PRJ_DIR = _PROC_DATA_DIR.parent

    return _RAW_DATA_DIR, _PROC_DATA_DIR, _NAS_PRJ_DIR


RAW_DATA_DIR, PROC_DATA_DIR, NAS_PRJ_DIR = _get_directories('odor_space_collab')


def has_preprocessing_files(mov_dir, required_files=None):
    """Checks if mov_dir contains the required files, and returns a readable string."""

    files = sorted(list(mov_dir.iterdir()))
    file_names = [item.name for item in files]

    if required_files is None:
        required_files = ['Image_001_001.tif', 'stk', 'Experiment.xml', 'timestamps.npy', 'pid.npz']

    has_files = [item in file_names for item in required_files]
    checklist = np.where(np.array(has_files), "x", " ")

    txt = f"""\n{mov_dir.relative_to(NAS_PRJ_DIR)}:"""

    for x, file in zip(checklist, required_files):
        txt = txt + f"""\n\t- [{x}] {file}"""

    print(txt)
    return txt


def main(flacq):
    timestamps_file, timestamps = process_thorsync.main(flacq)
    pid_npz_file = pid.main(flacq)


# %% SECTION 01: CONVERT .RAW MOVIES IN FOLDER TO TIFF FILES
convert_raw_data = True
if convert_raw_data:
    folder = RAW_DATA_DIR.joinpath('2022-09-22')
    print(f"folder: {folder}")

    convert_raw_fn_thorimage.main(folder)

# %%

# load dataset manifesto
with open(NAS_PRJ_DIR.joinpath('manifestos', 'flat_linked_thor_acquisitions.json'), 'r') as f:
    flat_lacq_list = json.load(f)

for flat_acq in flat_lacq_list[-3:]:
    main(flat_acq)
# %%
