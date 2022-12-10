import copy
from pathlib import Path
import dateparser
import utils2p
import shutil
import numpy as np
from tqdm import tqdm
import xarray as xr
import tifffile
from rye2p import convert_raw_fn_thorimage


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


def edit_timestamps(timestamps, tsub, drop_frames=None):
    da_timestamps = xr.DataArray(data=timestamps['stack_times'],
                                 dims=['time'])


# %%
def imread_stk_dir(stk_dir):
    tiff_files = sorted(list(stk_dir.glob("*.tif")))

    tiff_stacks = []
    axes_list = []
    ij_meta_list = []

    for file in tiff_files:
        print(f"loading {file}")
        with tifffile.TiffFile(file) as tif:
            stack0 = tif.asarray()
            axes = tif.series[0].axes
            imagej_metadata = tif.imagej_metadata

        tiff_stacks.append(stack0)
        axes_list.append(axes)
        ij_meta_list.append(imagej_metadata)

    return tiff_stacks, axes_list, ij_meta_list


def downsample_from_tiff(tiff_file, tsub, frames_to_drop=None, agg='sum'):
    stack0 = utils2p.load_img(tiff_file, memmap=True)

    if frames_to_drop is not None:
        stack_trimmed = np.delete(stack0, frames_to_drop, axis=0)
    else:
        stack_trimmed = stack0

    if tiff_file.with_name('timestamps.npy').is_file():
        timestamps = np.load(tiff_file.with_name('timestamps.npy'), allow_pickle=True).item()
        ts = timestamps['stack_times']
        timestamps_loaded = True
    else:
        ts = np.arange(t)
        timestamps_loaded = False

    da_stack0 = xr.DataArray(stack0,
                             dims=dims,
                             coords=dict(
                                     t=ts,
                                     z=np.arange(z),
                                     y=np.arange(y),
                                     x=np.arange(x),
                                     )
                             )
    if frames_to_drop is not None:
        da_stack = da_stack0.drop_isel(t=frames_to_drop)
    else:
        da_stack = copy.deepcopy(da_stack0)

    if agg == 'sum':
        da_stack = da_stack.coarsen(t=tsub, boundary='trim').sum()
    elif agg == 'mean':
        da_stack = da_stack.coarsen(t=tsub, boundary='trim').mean()

    if timestamps_loaded:
        timestamps_ds = copy.deepcopy(timestamps)
        timestamps_ds['stack_times'] = da_stack['t'].to_numpy()
        timestamps_ds['frames_dropped'] = frames_to_drop
        timestamps_ds['tsub'] = tsub,
    else:
        timestamps_ds = None
    return da_stack, timestamps_ds


def downsample_from_stk_dir(stk_dir, tsub, frames_to_drop=None, agg='mean'):
    tiff_list = sorted(list(stk_dir.glob("*.tif")))
    stack0 = np.concatenate([tifffile.imread(x) for x in tiff_list], axis=0)
    axes = 'TZYX'
    dims = [item for item in str.lower(axes)]
    t, z, y, x = stack0.shape

    if stk_dir.with_name('timestamps.npy').is_file():
        timestamps = np.load(stk_dir.with_name('timestamps.npy'), allow_pickle=True).item()
        ts = timestamps['stack_times']
        timestamps_loaded = True
    else:
        ts = np.arange(t)
        timestamps_loaded = False

    da_stack0 = xr.DataArray(stack0,
                             dims=dims,
                             coords=dict(
                                     t=ts,
                                     z=np.arange(z),
                                     y=np.arange(y),
                                     x=np.arange(x),
                                     )
                             )
    if frames_to_drop is not None:
        da_stack = da_stack0.drop_isel(t=frames_to_drop)
    else:
        da_stack = copy.deepcopy(da_stack0)

    if agg == 'sum':
        da_stack = da_stack.coarsen(t=tsub, boundary='trim').sum().astype('uint16')
    elif agg == 'mean':
        da_stack = da_stack.coarsen(t=tsub, boundary='trim').mean()
        da_stack = da_stack.astype('float32')

    if timestamps_loaded:
        timestamps_ds = copy.deepcopy(timestamps)
        timestamps_ds['stack_times'] = da_stack['t'].to_numpy()
        timestamps_ds['frames_dropped'] = frames_to_drop
        timestamps_ds['tsub'] = tsub,
    else:
        timestamps_ds = None
    return da_stack, timestamps_ds


def downsample_and_save_acquisition(stk_dir, tsub, frames_to_drop, agg='mean'):
    da_stack, timestamps_ds = downsample_from_stk_dir(stk_dir, tsub, frames_to_drop=frames_to_drop, agg=agg)
    src_dir = stk_dir.parent
    mov_name = src_dir.name

    # make new folder for downsampled acquisition
    target_dir = src_dir.with_name(f"{mov_name}_tsub{tsub:02d}")
    target_dir.mkdir(exist_ok=True)

    stack_ds = da_stack.to_numpy()
    utils2p.save_img(target_dir.joinpath("Image_001_001.tif"), stack_ds)
    np.save(target_dir.joinpath('timestamps.npy'), timestamps_ds)

    convert_raw_fn_thorimage.split_tiff_files(target_dir.joinpath("Image_001_001.tif"), frames_per_batch=500)
    return target_dir


def edit_movie_and_timestamps(file, tsub, frames_to_drop=None, agg='sum'):
    """
    stk_dir = Path("/local/matrix/Remy-Data/projects/odor_space_collab/processed_data_old/2019-03-06/4/_003/stk")

    Args:
        file ():
        tsub ():
        frames_to_drop ():
        agg ():

    Returns:

    """
    if file.is_dir():
        stk_dir = file
    else:
        with tifffile.TiffFile(file) as tif:
            stack0 = tif.asarray()
            axes = tif.series[0].axes
            imagej_metadata = tif.imagej_metadata
        if axes == 'TZYX':
            t, z, y, x = stack0.shape
        dims = [item for item in str.lower(axes)]

    if file.with_name('timestamps.npy').is_file():
        timestamps = np.load(file.with_name('timestamps.npy'), allow_pickle=True).item()
        ts = timestamps['stack_times']
        timestamps_loaded = True
    else:
        ts = np.arange(t)
        timestamps_loaded = False

    da_stack0 = xr.DataArray(stack0,
                             dims=dims,
                             coords=dict(
                                     t=ts,
                                     z=np.arange(z),
                                     y=np.arange(y),
                                     x=np.arange(x),
                                     )
                             )
    # drop frames
    if frames_to_drop is not None:
        da_stack = da_stack0.drop_isel(t=frames_to_drop)
    else:
        da_stack = copy.deepcopy(da_stack0)

    # downsample
    if agg == 'sum':
        da_stack = da_stack.coarsen(t=tsub, boundary='trim').sum()
    elif agg == 'mean':
        da_stack = da_stack.coarsen(t=tsub, boundary='trim').mean()

    if timestamps_loaded:
        timestamps_ds = copy.deepcopy(timestamps)
        timestamps_ds['stack_times'] = da_stack['t'].to_numpy()
        timestamps_ds['frames_dropped'] = frames_to_drop
        timestamps_ds['tsub'] = tsub,
    else:
        timestamps_ds = None
    return da_stack, timestamps_ds


# %%
if __name__ == '__main__':
    for thorimage_name in ['fn_0001']:
        tsub = 5
        folder_tsub = downsample_and_save_acquisition(
                Path("/local/matrix/Remy-Data/projects/odor_space_collab/processed_data_old/2019-04-26/3")
                .joinpath(thorimage_name, 'stk'),
                frames_to_drop=None,
                tsub=tsub,
                )
