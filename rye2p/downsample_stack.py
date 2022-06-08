from pathlib import Path
import dateparser
import utils2p
import shutil
import numpy as np
from tqdm import tqdm

steps = \
"""
movie:
  acqs:use 
    
"""


# Types of ThorImage files:
# Raw image files
# ---------------
#   -
#   - folder will contain "Image_001_001.raw"
#   - Experiment.xml file will contain :

#       - ... <CaptureMode mode="1" /> ...
#
#       - <ExperimentNotes text="odors: mixed 2/10&#xD;&#xA;surgeon:
#               Remy&#xD;&#xA;20220211_143711_stimuli\20220211_143711_stimuli_0.yaml" />
#
#   - Has a corresponding ThorSync recording that must also be parsed
#   - Streaming, stimulus-triggered functional recordings
#   - Can be a single or multi-plane recording
#
# Piezo previews
# ---------------
#   - Movies captured using "Preview" in the piezo controller dialog box of ThorImage
#   - Place these in /previews and ignore
#
# z-stacks:
# ---------
#   - Experiment.xml file will contain : <CaptureMode mode="0" />
#   - No raw image file in directory
#   - Has no associated ThorSync file
#   - Convert to 3D tiff stack

# %% SECTION 00:  CONVERT RAW IMAGE FILES TO TIFF STACKS
"""
- Find all raw image files in day's data folder
- Compute ThorImage/ThorSync file associations based on creation times
- Convert .raw file into .tif stack
- Other formats include:    
    - split .tif stacks (<4 GB, for suite2p and caiman)
    - 
"""
imaging_date = '2022-03-27'

RAW_DATA_DIR = Path("/media/remy/remy-storage/Remy's Dropbox Folder/HongLab @ Caltech Dropbox/Remy/"
                    "natural_mixtures/raw_data")

PROC_DATA_DIR = Path("/local/storage/Remy/natural_mixtures/processed_data")
# %%

all_meta_files = Path.rglob("Experiment.xml")
for meta_file in all_meta_files:
    img_meta = utils2p.Metadata(meta_file)
    raw_file = meta_file.with_name("Image_001_001.tif")

# %% SECTION 00: RUN THOR_FILES_GET_STAT.PY ON DATA DIRECTORY
# Run script to get file creation/modification date and timestamps
file_types = ["Image_001_001.raw",
              "Episode001.h5",
              "Experiment.xml",
              "ThorRealTimeDataSettings.xml"]

all_raw_files = sorted(list(RAW_DATA_DIR.joinpath(imaging_date, '1').rglob("Image_001_001.raw")))
rel_raw_files = [item.relative_to(RAW_DATA_DIR) for item in all_raw_files]
# %%
for raw_file in all_raw_files:
    # ThorImage metadata
    meta_file = raw_file.with_name('Experiment.xml')
    meta = utils2p.Metadata(meta_file)

    # .raw --> .tif
    stack1, = utils2p.load_raw(raw_file, meta)

    # convert to relative path
    rel_path = raw_file.relative_to(RAW_DATA_DIR)
    print(f"\nRaw image file: {rel_path}")

    # build tiff file name
    save_name = PROC_DATA_DIR.joinpath(rel_path.with_suffix(".tif"))

    #
    save_name.parent.mkdir(parents=True, exist_ok=True)
    print(f"Saving to: {save_name}")

    # save tiff file to project directory on network drive
    utils2p.save_img(save_name, stack1)
    print(f"...saved successfully.\n")

# %% SPLIT STACKS
# divide .tif files for easy loading (also, necessary to run suite2p with non-bigtiff formatted .tif files)

PROC_DATA_DIR = Path("/local/storage/Remy/narrow_odors/processed_data")
frames_per_batch = 500
tiff_files = sorted(list(PROC_DATA_DIR.joinpath(imaging_date).rglob("Image_001_001.tif")))

for file in tiff_files:
    print(f"\nFull .tif file: {file}")

    STK_DIR = file.with_name("stk")

    if not STK_DIR.is_dir():

        print(f"splitting movie --> {STK_DIR}")
        STK_DIR.mkdir()

        # load tiff file in batches and save
        batch_gen = utils2p.load_stack_batches(file, frames_per_batch)

        for i, batch in enumerate(batch_gen):
            save_name = STK_DIR.joinpath(f"stk_{i:03d}.tif")
            utils2p.save_img(save_name, batch)
            print(f"\t{save_name} saved successfully.")


# %% DOWNSAMPLE IN TIME IF SIGNAL IS WEAK (fast-z resonance scans will likely require this)


def split_acquisitions_to_stacks(file):
    print(f"\n---")
    print(f"loading {file}")
    stack1 = utils2p.load_img(file, memmap=True)

    timestamps = np.load(file.with_name('timestamps.npy'), allow_pickle=True).item()
    frames_per_batch = timestamps['n_stack_times_per_pulse']

    split_stacks = np.split(stack1, np.cumsum(frames_per_batch), axis=0)
    split_stacks = list(filter(lambda x: x.size > 0, split_stacks))

    save_dir = Path(file.with_name('acqs'))
    save_dir.mkdir(exist_ok=True)

    saved_files = []
    for i, substack in enumerate(split_stacks):
        save_file = f"acq_{i:04d}.tif"
        utils2p.save_img(save_dir.joinpath(save_file), substack.astype('uint16'))
        print(f"  - {save_file} saved.")
        saved_files.append(save_dir.joinpath(save_file))
    return saved_files


for img_file in tiff_files[1:]:
    print(split_acquisitions_to_stacks(img_file))







#%%

    # %% DOWNSAMPLE IN TIME (SUM)


# naming convention is to insert "_tsum<...>" between the file name stem and file suffix

# allow downsampling only by a divisor
allow_nondivisor = True

tiff_files = sorted(list(PROC_DATA_DIR.joinpath(imaging_date).rglob("Image_001_001.tif")))

dsub_time = 3

for file in tiff_files:
    print(f"\nFull .tif file: {file}")

    stack1 = utils2p.load_img(file, memmap=True)
    n_frames = stack1.shape[0]

    idx = np.arange(0, n_frames, dsub_time)

    print(f"\t...Downsampling movie")
    split_stacks = np.split(stack1, idx[1:])
    ds_split_stacks = [item.sum(axis=0, keepdims=True) for item in split_stacks if item.shape[0] == dsub_time]
    stack1_ds = np.concatenate(ds_split_stacks, axis=0)

    # save to .tif file in the same folder as the original, full-length movie
    save_name = f"{file.stem}_tsum{dsub_time:02d}.tif"
    folder = file.with_name(f"downsampled_{dsub_time}")
    folder.mkdir(exist_ok=True)

    print(f"...Saving downsampled movie")
    utils2p.save_img(folder.joinpath(save_name), stack1_ds.astype('uint16'))
    print(f"\t...{save_name} saved successfully.")
# %%

tiff_files = PROC_DATA_DIR.rglob("Image_001_001.tif")
tiff_files = sorted(list(tiff_files))

# %% For each Experiment.xml file, determine what type of recording it is


# %%

DATA_DIR = Path("/media/remy/remy-storage/Remy's Dropbox Folder/HongLab @ Caltech Dropbox/Remy/"
                "natural_mixtures/raw_data/2022-02-10")

DB_PRJ_DIR = Path("/media/remy/remy-storage/Remy's Dropbox Folder/HongLab @ Caltech Dropbox/Remy/natural_mixtures")

RAW_DIR = DB_PRJ_DIR.joinpath('raw_data')

COPY_DIR = Path("/local/storage/Remy/natural_mixtures/processed_data")

all_meta_files = sorted(list(DATA_DIR.rglob("Experiment.xml")))
all_meta_files = list(filter(lambda x: "previews" not in x.parts, all_meta_files))

saved_files = []
for meta_file in all_meta_files:
    meta = utils2p.Metadata(meta_file)
    capture_mode = meta.get_metadata_value('CaptureMode', 'mode')

    # fastz
    if capture_mode == '1':
        print(meta_file)
        raw_file = meta_file.with_name("Image_001_001.raw")
        stack1, = utils2p.load_raw(raw_file, meta)

    elif capture_mode == '0':
        print(meta_file)
# %%
allow_nondivisor = True

tiff_files = sorted(list(PROC_DATA_DIR.joinpath(imaging_date).rglob("Image_001_001.tif")))

dsub_time = 3

for file in tiff_files:
    print(f"\nFull .tif file: {file}")

    stack1 = utils2p.load_img(file, memmap=True)
    n_frames = stack1.shape[0]

    idx = np.arange(0, n_frames, dsub_time)

    print(f"\t...Downsampling movie")
    split_stacks = np.split(stack1, idx[1:])
    ds_split_stacks = [item.sum(axis=0, keepdims=True) for item in split_stacks if item.shape[0] == dsub_time]
    stack1_ds = np.concatenate(ds_split_stacks, axis=0)

    # save to .tif file in the same folder as the original, full-length movie
    save_name = f"{file.stem}_tsum{dsub_time:02d}.tif"
    folder = file.with_name(f"downsampled_{dsub_time}")
    folder.mkdir(exist_ok=True)

    print(f"...Saving downsampled movie")
    utils2p.save_img(folder.joinpath(save_name), stack1_ds.astype('uint16'))
    print(f"\t...{save_name} saved successfully.")

# #%%
# saved_files = list(DATA_DIR.rglob("Image_001_001.tif"))
# for file in saved_files:
#     save_name = COPY_DIR.joinpath(file.relative_to(RAW_DIR))
#     print(save_name)
#     save_name.parent.mkdir(parents=True, exist_ok=True)
#     shutil.move(str(file), str(save_name.parent))
#
# #%% SPLIT STACKS
# from pathlib import Path
# PROC_DIR = Path("/local/storage/Remy/natural_mixtures/processed_data/2022-02-10")
#
# tiff_files = list(PROC_DIR.rglob("Image_001_001.tif"))
# for file in tiff_files:
#     SAVE_DIR = file.with_name("stk")
#
#     # save planes
#     if not SAVE_DIR.is_dir():
#         SAVE_DIR.mkdir()
#         batch_gen = utils2p.load_stack_batches(file, 1000)
#         for i, batch in enumerate(batch_gen):
#             utils2p.save_img(SAVE_DIR.joinpath(f"stk_{i:03d}.tif"), batch)

# %%
# elif capture_mode == '0':   # zstack
#
#
#
#     stack1, = utils2p.load_z_stack(meta_file.folder, meta)
#     file =
#     utils2p.save_img("ChannelA.tif", stack1)
#     utils2p.save_img("ChannelB.tif", stack2)
#
#     SAVE_DIR = meta_file.relative_to(DB_PRJ_DIR.joinpath('raw_data'))
#     raw_file = utils2p.find_raw_file(meta_file.parent)
#     stack1, stack2 = utils2p.load_raw(raw_file, metadata)
