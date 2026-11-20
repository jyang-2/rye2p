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

from tqdm import tqdm

# Implementations moved to rye2p.thorimage.io (roadmap step 1); re-exported
# here so existing imports keep working until the step-5 deprecation shims.
from rye2p.thorimage.io import (
    copy_raw_dir_2_proc_dir,
    convert_raw_to_tiff,
    split_tiff_files,
)


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
        saved_tiffs = split_tiff_files(file, frames_per_batch=500)
        saved_split_tiffs.append(saved_tiffs)
        pprint.pprint(saved_tiffs, indent=4)

    return saved_tiff_files, saved_split_tiffs


if __name__ == '__main__':
    None
