"""Types of ThorImage files:
Raw image files
---------------
  -
  - folder will contain "Image_001_001.raw"
  - Experiment.xml file will contain :

      - ... <CaptureMode mode="1" /> ...

      - <ExperimentNotes text="odors: mixed 2/10&#xD;&#xA;surgeon:
              Remy&#xD;&#xA;20220211_143711_stimuli\20220211_143711_stimuli_0.yaml" />

  - Has a corresponding ThorSync recording that must also be parsed
  - Streaming, stimulus-triggered functional recordings
  - Can be a single or multi-plane recording

Piezo previews
---------------
  - Movies captured using "Preview" in the piezo controller dialog box of ThorImage
  - Place these in /previews and ignore

z-stacks:
---------
  - Experiment.xml file will contain : <CaptureMode mode="0" />
  - No raw image file in directory
  - Has no associated ThorSync file
  - Convert to 3D tiff stack

# %% SECTION 00:  CONVERT RAW IMAGE FILES TO TIFF STACKS

- Find all raw image files in day's data folder
- Compute ThorImage/ThorSync file associations based on creation times
- Convert .raw file into .tif stack
- Other formats include:
    - split .tif stacks (<4 GB, for suite2p and caiman)
    -
"""

import pprint
from pathlib import Path
import utils2p
import shutil
import numpy as np
from tqdm import tqdm
import argparse


# %%

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


def split_tiff_files(tiff_file, frames_per_batch, save_dir=Path('stk')):
    """ Split tiff files into smaller stacks (timestamps)

    Args:
        tiff_file (Path): Path to .tif stack<br><br>
        frames_per_batch (int): # of timepoints to include in each batch<br><br>
        save_dir (Path): where split tif stacks are saved.<br>
                        -  If `save_dir` is a relative path, then the save path is relative to tiff_file.parent<br>
                        -  If `save_dir` is an absolute path, then files are saved there.<br><br>

    Returns:
        List[Path]: List of saved tiff files

    """
    if isinstance(tiff_file, str):
        tiff_file = Path(tiff_file)

    if isinstance(save_dir, str):
        save_dir = Path(save_dir)

    STK_DIR = tiff_file.parent.joinpath(save_dir)

    if not STK_DIR.is_dir():
        saved_files = []
        STK_DIR.mkdir()

        batch_gen = utils2p.load_stack_batches(tiff_file, frames_per_batch)

        for i, batch in enumerate(batch_gen):
            save_name = STK_DIR.joinpath(f"stk_{i:03d}.tif")
            utils2p.save_img(save_name, batch)
            saved_files.append(save_name)
    else:
        saved_files = None

    return saved_files


# %%
def main(imaging_folder, raw_dir, proc_dir):
    if isinstance(imaging_folder, str):
        imaging_folder = Path(imaging_folder)

    all_raw_files = sorted(list(imaging_folder.rglob("Image_001_001.raw")))

    if len(all_raw_files) == 0:
        all_raw_files = sorted(list(imaging_folder.rglob("Image_0001_0001.raw")))

    # -----------------------------------------
    # convert .raw files to 1 large tiff stack
    # -----------------------------------------
    saved_tiff_files = []

    for file in tqdm(all_raw_files):
        saved_tiff_files.append(convert_raw_to_tiff(file, raw_dir, proc_dir))

    # -------------------------------------------------------------
    # split recently converted tiff stacks into smaller tiff stacks
    # -------------------------------------------------------------
    saved_split_tiffs = []

    for file in tqdm(saved_tiff_files):
        print(f"\nFull .tif file: {file}")
        saved_tiffs = split_tiff_files(file, frames_per_batch=1000)
        saved_split_tiffs.append(saved_tiffs)
        pprint.pprint(saved_tiffs, indent=4)

    return saved_tiff_files, saved_split_tiffs


if __name__ == '__main__':
    None
