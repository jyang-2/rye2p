"""Convert .raw movies and move ThorImage files from raw data directory to processed data
directory"""

import pprint
from pathlib import Path
import utils2p
import shutil
# import numpy as np
from tqdm import tqdm


def copy_raw_dir_2_proc_dir(src_file, raw_dir, proc_dir):
    if isinstance(src_file, str):
        src_file = Path(src_file)

    dest_file = proc_dir.joinpath(*src_file.relative_to(raw_dir).parts)
    shutil.copy(src_file, dest_file)
    return dest_file


def convert_raw_to_tiff(raw_file, raw_dir, proc_dir):
    """ Convert raw file to single tif stack

    Args:
        proc_dir ():
        raw_dir ():
        raw_file ():

    Returns:
        Path: filepath to saved tiff stack

    """
    meta_file = raw_file.with_name('Experiment.xml')
    meta = utils2p.Metadata(meta_file)

    # .raw --> .tif
    stack1, = utils2p.load_raw(raw_file, meta)

    # convert to relative path
    rel_path = raw_file.relative_to(raw_dir)
    print(f"\nRaw image file: {rel_path}")

    # build tiff file name - save full movie as Image_001_001.tif
    save_name = proc_dir.joinpath(rel_path.with_suffix(".tif"))

    save_name.parent.mkdir(parents=True, exist_ok=True)
    print(f"Saving to: {save_name}")

    # save tiff file to project directory on network drive
    utils2p.save_img(save_name, stack1)
    print(f"...saved successfully.\n")
    return save_name
