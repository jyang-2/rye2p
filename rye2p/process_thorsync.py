import itertools
import os
import sys
from pathlib import Path
import dateparser
import shutil
import numpy as np
import pydantic
import h5py
import utils2p
import utils2p.synchronization
from re import sub

import yaml
import json
import dotenv
import pprint as pp
import math
from scipy.ndimage import gaussian_filter1d
from scipy.stats import zscore

import h5py
from fuzzywuzzy import fuzz
from rye2p import fix_frame_times


# # load .env variables
# prj = 'odor_unpredictability'
#
# if prj == 'natural_mixtures':
#     dotenv.load_dotenv(Path.home().joinpath('dotenv_files/natural_mixtures/.env'))
#     # config = dotenv.dotenv_values('dotenv_files/natural_mixtures/.env')
#
#     DB_PRJ_DIR = Path(os.getenv("DB_PRJ_DIR"))
#     DB_RAW_DIR = Path(os.getenv("DB_RAW_DIR"))
#
#     NAS_PRJ_DIR = Path(os.getenv("NAS_PRJ_DIR"))
#     NAS_PROC_DIR = Path(os.getenv("NAS_PROC_DIR"))
#     NAS_OLFCONFIG_DIR = Path(os.getenv("NAS_OLFCONFIG_DIR"))
#
#     # NAS_PRJ_DIR = Path("/local/matrix/Remy-Data/natural_mixtures/natural_mixtures")
#     # NAS_PROC_DIR = Path("/local/matrix/Remy-Data/natural_mixtures/natural_mixtures/processed_data")
#     # NAS_OLFCONFIG_DIR = Path("/local/matrix/Remy-Data/natural_mixtures/natural_mixtures/olfactometer_configs")
#
# elif prj == 'narrow_odors':
#     NAS_PRJ_DIR = Path("/local/storage/Remy/narrow_odors")
#     NAS_PROC_DIR = Path("/local/storage/Remy/narrow_odors/processed_data")
#     DB_RAW_DIR = Path("/local/storage/Remy/narrow_odors/raw_data")
#
# elif prj == 'odor_space_collab':
#     DB_RAW_DIR = Path("/media/remy/remy-storage/Remy's Dropbox Folder/HongLab @ Caltech "
#                       "Dropbox/Remy/odor_space_collab/raw_data")
#     NAS_PRJ_DIR = Path("/local/matrix/Remy-Data/projects/odor_space_collab")
#     NAS_PROC_DIR = Path("/local/matrix/Remy-Data/projects/odor_space_collab/processed_data")
#
# elif prj == 'odor_unpredictability':
#     DB_RAW_DIR = Path("/media/remy/remy-storage/Remy's Dropbox Folder/HongLab @ Caltech Dropbox/"
#                       "Remy/odor_unpredictability/raw_data")
#     NAS_PRJ_DIR = Path("/local/matrix/Remy-Data/projects/odor_unpredictability")
#     NAS_PROC_DIR = Path("/local/matrix/Remy-Data/projects/odor_unpredictability/processed_data")

# %%
def get_thorsync_line_names(h5_file):
    thorsync_groups = ['AI', 'CI', 'DI', 'Global']

    line_names = []
    with h5py.File(h5_file, 'r') as f:
        for grp in f.keys():
            for item in f[grp]:
                line_names.append(item)
    line_names.remove('GCtr')
    return line_names


# %%
# class ThorSyncLineTimes(pydantic.BaseModel):
#     frame_times: list[float]
#     stack_times: list[float]
#     scope_ict: list[float]
#     scope_fct: list[float]
#     olf_ict: list[float]
#     olf_fct: list[float]
# %%
def np_in_range(x, x_start, x_end):
    return np.logical_and(x >= x_start, x < x_end)


def in_range(val, x_start, x_end, inclusive=False):
    if inclusive:
        is_in_range = (val >= x_start) & (val <= x_end)
    else:
        is_in_range = (val >= x_start) & (val < x_end)
    return is_in_range


def snake_case(s):
    """ Function for converting variable names to snake case
    
    Args:
        s (str): variable name, like "linkedThorAcquisitions"

    Returns:
        sc (str): variable name in snake_case

    Examples:

        >>>     snake_case("linkedThorAcquisitions")
        Out[83]: 'linked_thor_acquisitions'

    """
    sc = '_'.join(
            sub('([A-Z][a-z]+)', r' \1',
                sub('([A-Z]+)', r' \1',
                    s.replace('-', ' '))).split()).lower()
    return sc


def flacq2rawfiles(flat_lacq, raw_dir):
    meta_file = raw_dir.joinpath(flat_lacq['date_imaged'],
                                 str(flat_lacq['fly_num']),
                                 flat_lacq['thorimage'],
                                 'Experiment.xml')

    h5_file = raw_dir.joinpath(flat_lacq['date_imaged'],
                               str(flat_lacq['fly_num']),
                               flat_lacq['thorsync'],
                               'Episode001.h5')

    sync_meta_file = h5_file.with_name('ThorRealTimeDataSettings.xml')
    return h5_file, sync_meta_file, meta_file


def flacq2dir(flat_lacq, proc_dir):
    folder = proc_dir.joinpath(flat_lacq['date_imaged'],
                               str(flat_lacq['fly_num']),
                               flat_lacq['thorimage'])
    return folder


def get_pulse_idx(h5_file, line_name, voltage_threshold=2.5):
    """ Finds index of rising and falling edges."""
    pulse_line, = utils2p.synchronization.get_lines_from_sync_file(h5_file, [line_name])

    ici = utils2p.synchronization.edges(pulse_line, size=voltage_threshold)[0].tolist()
    fci = utils2p.synchronization.edges(pulse_line, size=(-np.inf, -1 * voltage_threshold))[0].tolist()
    return ici, fci


def get_matched_thorsync_lines(h5_file, var_names=None):
    """ThorSync line names vary slightly across recordings (PiezoMonitor, piezoMonitor, piezo_monitor, etc)."""

    if var_names is None:
        var_names = (
                'piezo_monitor',
                'pockels_monitor',
                'lightpath_shutter',
                'olf_disp_pin',
                'pid',
                'scope_pin',
                'frame_counter',
                'frame_in',
                'frame_out')

    cleaned_var_names = [clean_line_name(item) for item in var_names]

    line_names_from_file = get_thorsync_line_names(h5_file)
    cleaned_line_names = [clean_line_name(item) for item in line_names_from_file]

    match_ratio = np.array([[fuzz.ratio(x, y) for x in cleaned_var_names] for y in cleaned_line_names])
    matched_thorsync_lines = np.array(line_names_from_file)[np.argmax(match_ratio, axis=1)].tolist()

    return matched_thorsync_lines


def clean_line_name(line_name):
    return line_name.lower().replace(' ', '').replace('_', '')


def get_thorsync_lines(h5_file):
    sync_meta_file = h5_file.with_name('ThorRealTimeDataSettings.xml')
    sync_meta = utils2p.synchronization.SyncMetadata(sync_meta_file)

    var_names = (
            'piezo_monitor',
            'pockels_monitor',
            'lightpath_shutter',
            'olf_disp_pin',
            'pid',
            'scope_pin',
            'frame_counter',
            'frame_in',
            'frame_out')

    # match `line_names_from_file` to desired `var_names`
    matched_line_names = get_matched_thorsync_lines(h5_file, var_names=var_names)

    (piezo_monitor,
     pockels_monitor,
     lightpath_shutter,
     olf_disp_pin,
     pid,
     scope_pin,
     frame_counter,
     frame_in,
     frame_out) = utils2p.synchronization.get_lines_from_sync_file(h5_file, matched_line_names)

    sync_times = utils2p.synchronization.get_times(len(frame_counter), sync_meta.get_freq())
    frame_out = (frame_out > 0) * 1

    olf_disp_pin = utils2p.synchronization.correct_split_edges(olf_disp_pin)
    scope_pin = utils2p.synchronization.correct_split_edges(scope_pin)
    lightpath_shutter = utils2p.synchronization.correct_split_edges(lightpath_shutter)

    sync_data = dict(sync_times=sync_times,
                     piezo_monitor=piezo_monitor,
                     pockels_monitor=pockels_monitor,
                     lightpath_shutter=lightpath_shutter,
                     olf_disp_pin=olf_disp_pin,
                     pid=pid,
                     scope_pin=scope_pin,
                     frame_counter=frame_counter,
                     frame_in=frame_in,
                     frame_out=frame_out)
    return sync_data


def extract_timestamp_data(h5_file, meta_file):
    sync_data = get_thorsync_lines(h5_file)
    meta = utils2p.Metadata(meta_file)

    # check if single plane
    fast_z = bool(int(meta.get_metadata_value("Streaming", 'zFastEnable')))

    # timepoints for every individual frame
    # --------------------------------------

    processed_frame_counter = utils2p.synchronization.process_frame_counter(sync_data['frame_counter'],
                                                                            steps_per_frame=1)
    mask = np.logical_and(sync_data['scope_pin'] > 1, processed_frame_counter >= 0)
    lines_to_crop = [sync_data['sync_times'], processed_frame_counter]
    cropped_sync_times, cropped_frame_counter = utils2p.synchronization.crop_lines(mask, lines_to_crop)
    frame_times = utils2p.synchronization.get_start_times(cropped_frame_counter, cropped_sync_times)

    # timepoints for each volumetric scan
    # ------------------------------------------
    steps_per_frame = meta.get_n_z() + meta.get_n_flyback_frames()
    processed_stack_counter = utils2p.synchronization.process_frame_counter(sync_data['frame_counter'],
                                                                            steps_per_frame=steps_per_frame)
    mask = np.logical_and(sync_data['scope_pin'] > 1, processed_stack_counter >= 0)
    lines_to_crop = [sync_data['sync_times'], processed_stack_counter]
    cropped_sync_times, cropped_stack_counter = utils2p.synchronization.crop_lines(mask, lines_to_crop)
    stack_times = utils2p.synchronization.get_start_times(cropped_stack_counter, cropped_sync_times)

    voltage_threshold = 2.5
    sync_times = sync_data['sync_times']

    # scope ict/fct
    # ------------
    scope_pin = sync_data['scope_pin']
    scope_ici, = utils2p.synchronization.edges(scope_pin, size=voltage_threshold)
    scope_fci, = utils2p.synchronization.edges(scope_pin, size=(-np.inf, -1 * voltage_threshold))
    scope_ict = sync_times[scope_ici]
    scope_fct = sync_times[scope_fci]

    # olf ict/fct
    # ------------
    olf_disp_pin = sync_data['olf_disp_pin']
    olf_ici, = utils2p.synchronization.edges(olf_disp_pin, size=voltage_threshold)
    olf_fci, = utils2p.synchronization.edges(olf_disp_pin, size=(-np.inf, -1 * voltage_threshold))
    olf_ict = sync_times[olf_ici]
    olf_fct = sync_times[olf_fci]

    timestamp_data = dict(frame_times=frame_times,
                          stack_times=stack_times,
                          scope_ict=scope_ict,
                          scope_fct=scope_fct,
                          olf_ict=olf_ict,
                          olf_fct=olf_fct
                          )
    return timestamp_data


def fast_z_enabled(meta_file):
    meta = utils2p.Metadata(meta_file)
    return bool(int(meta.get_metadata_value("Streaming", 'zFastEnable')))


def convert_thorsync_to_timestamps_npy(flacq, raw_dir, proc_dir):
    """ Extracts timing info for frame acquisitions, stimuli, and scope acquisitions.

    Args:
        proc_dir ():
        raw_dir ():
        flacq (dict): dict from dataset manifesto `flat_linked_thor_acquisitions.json`

    Returns:
        timestamps (dict): contains fields 'stack_times', 'frame_times', 'scope_ici', 'scope_ict', 'olf_ici', 'olf_ict'
        """

    # make initial timestamps0.npy
    h5_file, sync_meta_file, meta_file = flacq2rawfiles(flacq, raw_dir)
    SAVE_DIR = flacq2dir(flacq, proc_dir)

    print(f'\nConverting ThorSync to timestamps:')
    print(f"\t- SAVE_DIR: {SAVE_DIR}")

    meta = utils2p.Metadata(meta_file)
    fast_z = bool(int(meta.get_metadata_value("Streaming", 'zFastEnable')))

    timestamps0 = extract_timestamp_data(h5_file, meta_file)
    np.save(SAVE_DIR.joinpath('timestamps0.npy'), timestamps0)

    timestamps = fix_frame_times.fix_timestamps0(SAVE_DIR.joinpath('timestamps0.npy'))

    # correct timestamps file
    # -----------------------
    # if volumetric
    if fast_z:
        steps_per_frame = meta.get_n_flyback_frames() + meta.get_n_z()

        stack_times_correct = timestamps['stack_times'].size == meta.get_n_time_points()
        frame_times_correct = timestamps['frame_times'].size == steps_per_frame * meta.get_n_time_points()

        print(f"\t- correct # of stack_times: {stack_times_correct}")
        print(f"\t- correct # of frame_times: {frame_times_correct}")

    else:
        n_frames_averaged = meta.get_n_averaging()
        print(f'\tsingle plane movie, {n_frames_averaged} frame average')

        timestamps['stack_times'] = timestamps['stack_times'][::n_frames_averaged]

        stack_times_correct = timestamps['stack_times'].size == meta.get_n_time_points()
        print(f"\t- correct # of stack_times: {stack_times_correct}")

    save_file = SAVE_DIR.joinpath('timestamps.npy')
    np.save(save_file, timestamps)
    print(f"timestamps.npy saved")
    print(f"- {save_file}")

    return save_file, timestamps


def edit_timestamps(timestamps_ori, frames_to_drop, tsub):
    """

    Args:
        timestamps_ori (): should have correct # of stack times
        frames_to_drop (): frames to drop before downsampling (list of indices)
        tsub ():

    Returns:

    """


def create_proc_dir(flacq, proc_dir):
    """Creates movie folder in processed data directory `PROC_DATA_DIR` for FlatFlyAcquisitions."""

    mov_dir = flacq2dir(flacq, proc_dir)
    print(f"\n{mov_dir}")
    print(f"\tmovie folder exists: {mov_dir.is_dir()}")

    if not mov_dir.is_dir():
        mov_dir.mkdir(parents=True)
        print(f"\tfolder created.")

    return mov_dir


def copy_experiment_xml(flacq, raw_dir, proc_dir):
    """Copies 'Experiment.xml' file from the raw data  to processed_data.

    Source and destination filepaths are determined from information contained in flacq.

    Args:
        proc_dir ():
        raw_dir ():
        flacq (dict): FlatFlyAcquisition

    Returns:
        dest_file (Path): filepath where Experiment.xml was copied
    """
    h5_file, sync_meta_file, meta_file = flacq2rawfiles(flacq, raw_dir)
    dest_file = flacq2dir(flacq, proc_dir).joinpath('Experiment.xml')

    if not dest_file.exists():
        shutil.copy(meta_file, dest_file)

        print(f"\nCopied meta_file:"
              f"\n\t- src:{meta_file}"
              f"\n\t- dest: {dest_file}")

    return meta_file, dest_file


def main(flacq, raw_dir, proc_dir):
    print(f'\nProcessing ThorSync files:')
    print(f'---------------------------')
    pp.pprint(flacq)

    mov_dir = create_proc_dir(flacq, proc_dir)

    copy_experiment_xml(flacq, raw_dir, proc_dir)

    timestamps_file, timestamps = convert_thorsync_to_timestamps_npy(flacq, raw_dir, proc_dir)
    print('Done processing ThorSync files.')
    return timestamps_file, timestamps

# %%  Move this to another script ------------------------------------------

# for lacq in db.linked_thor_acq_collection.find():
#
#     # look up relevant info
#     thorimage = db.thorimage_collection.find_one({'_id': lacq['thorimage_id']})
#     thorsync = db.thorsync_collection.find_one({'_id': lacq['thorsync_id']})
#
#     # print linked documents
#     print(f"\nLINKED_THOR_ACQUISITION")
#     pp.pprint(lacq)
#
#     print("f\nTHORIMAGE")
#     pp.pprint(thorimage)
#
#     print(f"\nTHORSYNC")
#     pp.pprint(thorsync)
#     #print(thorsync)
#
#     meta_file = Path(thorimage['rel_path'])
#     sync_meta_file = Path(thorsync['rel_path'])
#     h5_file = sync_meta_file.with_name('Episode001.h5')
#
#     meta = utils2p.Metadata(DB_RAW_DIR.joinpath(meta_file))
#     sync_meta = utils2p.synchronization.SyncMetadata(DB_RAW_DIR.joinpath(sync_meta_file))
#
#     # LOAD THORSYNC H5 FILE AND COMPUTE FRAME TIMES
#     # thorsync line names
#     line_names = ['PiezoMonitor', 'Pockels1Monitor', 'lightpathshutter',
#                   'olfDispPin', 'pid', 'scopePin', 'FrameCounter',
#                   'FrameIn', 'FrameOut', ]
#
#     (piezo_monitor,
#      pockels_monitor,
#      lightpath_shutter,
#      olf_disp_pin,
#      pid,
#      scope_pin,
#      frame_counter,
#      frame_in,
#      frame_out) = utils2p.synchronization.get_lines_from_sync_file(DB_RAW_DIR.joinpath(h5_file),
#                                                                  line_names)
#     sync_times = utils2p.synchronization.get_times(len(frame_counter), sync_meta.get_freq())
#     frame_out = (frame_out > 0) * 1
#
#     # timepoints for each volume
#     # ---------------------------
#     steps_per_frame = meta.get_n_z() + meta.get_n_flyback_frames()
#
#     processed_stack_counter = utils2p.synchronization.process_frame_counter(frame_counter,
#                                                                             steps_per_frame=steps_per_frame)
#     stack_times = utils2p.synchronization.get_start_times(processed_stack_counter, sync_times)
#
#     # timepoints for every individual frame
#     # --------------------------------------
#     processed_frame_counter = utils2p.synchronization.process_frame_counter(frame_counter,
#                                                                             steps_per_frame=1)
#     frame_times = utils2p.synchronization.get_start_times(processed_frame_counter, sync_times)
#
#
#     # scope ict/fct
#     # ------------
#     scope_ici, = utils2p.synchronization.edges(scope_pin, size=voltage_threshold)
#     scope_fci, = utils2p.synchronization.edges(scope_pin, size=(-np.inf, -1 * voltage_threshold))
#     scope_ict = sync_times[scope_ici]
#     scope_fct = sync_times[scope_fci]
#
#     # olf ict/fct
#     # ------------
#     olf_ici, = utils2p.synchronization.edges(olf_disp_pin, size=voltage_threshold)
#     olf_fci, = utils2p.synchronization.edges(olf_disp_pin, size=(-np.inf, -1 * voltage_threshold))
#     olf_ict = sync_times[olf_ici]
#     olf_fct = sync_times[olf_fci]
#
#     print(f'\n------------------------------------')
#     print(lacq)
#     print(f"\nmeta.get_n_time_points() = {meta.get_n_time_points()}")
#     print(f"stack_times.shape = {stack_times.shape}")
#     print(f"frame_times.shape = {frame_times.shape}")
#
#
#     db.linked_thor_acq_collection.update_one(lacq,
#                                              {'$set': {'stack_times': stack_times.tolist(),
#                                                        'frame_times': frame_times.tolist(),
#                                                        'scope_ict': scope_ict.tolist(),
#                                                        'scope_fct': scope_fct.tolist(),
#                                                        'olf_ict': olf_ict.tolist(),
#                                                        'olf_fct': olf_fct.tolist(),
#                                                        }})

# %%
# from typing import Optional
#
# class PinOdor(pydantic.BaseModel):
#     name: str
#     log10_conc: Optional[float]
#     abbrev: Optional[str]
#
# class MultiChannelStimulus(pydantic.BaseModel):
#     pin_list: tuple
#     no_pins: set # pins for normally-open solenoids (balance flow)
#
#
# def pin_odor_str(x):
#     if x.log10_conc is None:
#         s = ''
#     else:
#         s = f"{x.name} [{x.log10_conc}]"
#     return s
# #%%
# for lacq in db.linked_thor_acq_collection.find():
#     # load the olfactometer config file
#     olf_config_yaml = NAS_OLFCONFIG_DIR.joinpath(Path(*lacq['olf_config'].split('\\')))
#
#     with olf_config_yaml.open('r') as f:
#         olf_config = yaml.safe_load(f)
#
#     # get the pin2odor mapping
#     pin_odor_map = pin_odor_map = {k:PinOdor(**v) for k, v in olf_config['pins2odors'].items()}
#
#     open_pins = {2, 42}
#     pin_sequence = [set(item['pins']) - open_pins for item in olf_config['pin_sequence']['pin_groups']]
#     pin_sequence = [tuple(item) for item in pin_sequence]
#
#     stim_str_list = []
#     for pins in pin_sequence:
#         stim_str = [pin_odor_str(pin_odor_map[pin]) for pin in pins]
#         stim_str = tuple(stim_str)
#     # for a, b in pin_sequence:
#     #     stim_str = [pin_odor_str(pin_odor_map[a]),
#     #                      pin_odor_str(pin_odor_map[b])]
#     #     stim_str = tuple(stim_str)
#         stim_str_list.append(", ".join(stim_str))
#
#
#     print(f'\n{olf_config_yaml.relative_to(NAS_OLFCONFIG_DIR)}')
#     for item in stim_str_list:
#         print(item)
#
#     db.linked_thor_acq_collection.update_one(lacq,
#                                              {'$set': {'stim_str_list': stim_str_list
#                                                        }
#                                               }
#                                              ),
#     #print(f"{a}, {b} : {stim_str}")
# #%% update trimmed stack_times
# for lacq in db.linked_thor_acq_collection.find():
#     thorimage = db.thorimage_collection.find_one({'_id': lacq['thorimage_id']})
#     thorsync = db.thorsync_collection.find_one({'_id': lacq['thorsync_id']})
#     meta_file = Path(thorimage['rel_path'])
#     meta = utils2p.Metadata(DB_RAW_DIR.joinpath(meta_file))
#
#     fixed_stack_times = lacq['stack_times'][:meta.get_n_time_points()]
#
#     db.linked_thor_acq_collection.update_one(lacq,
#                                              {'$set': {'fixed_stack_times': fixed_stack_times
#                                                        }
#                                               }
#                                              ),
#
# # %%  to find index locations to plot stimulus lines, do the following:
# dsub_time = 3
# stack_times_ds = stack_times[::dsub_time]
# stim_ici_ds = np.interp(lacq['olf_ict'], stack_times_ds, np.arange(len(stack_times_ds)))
# #%%%
#
#
#
# pin_odors = [list(item) for item in pin_odors]
#
# pins_used = list(set(itertools.chain.from_iterable(pin_odors)))
# for pin in pins_used:
#     pp.pprint(olf_config['pins2odors'][pin])
#
# for pin1, pin2 in pin_odors:
#     str1 = f"{olf_config['pins2odors'][pin1][]'"
# #%%
# stage_lookup_thorimage = {
#    "$lookup": {
#          "from": "thorimage_collection",
#          "localField": "thorimage_id",
#          "foreignField": "_id",
#          "as": "thorimage",
#    }
# }
#
# stage_lookup_thorsync = {
#    "$lookup": {
#          "from": "thorsync_collection",
#          "localField": "thorsync_id",
#          "foreignField": "_id",
#          "as": "thorsync",
#    }
# }
# stage_unwind_thorsync = { '$unwind': '$thorsync' }
# stage_unwind_thorimage = { '$unwind': '$thorimage' }
#
# pipeline = [stage_lookup_thorimage, stage_unwind_thorimage, stage_lookup_thorsync, stage_unwind_thorsync]
# results = list(db.linked_thor_acq_collection.aggregate(pipeline))
# #%%
# imaging_date = '2022-02-10'
#
# RAW_DATA_DIR = Path("/media/remy/remy-storage/Remy's Dropbox Folder/HongLab @ Caltech Dropbox/Remy/"
#                 "natural_mixtures/raw_data")
#
# PROC_DATA_DIR = Path("/local/storage/Remy/natural_mixtures/processed_data")
# #%%
# notes_file = sorted(list(RAW_DATA_DIR.rglob("stimulus_info_*.yaml")))
#
#
# #%%
# folder = Path("/media/remy/remy-storage/Remy's Dropbox Folder/HongLab @ Caltech
# Dropbox/Remy/natural_mixtures/raw_data/2022-02-11/1/SyncData001")
# h5_file = folder.joinpath("Episode001.h5")
# sync_meta_file = h5_file.with_name("ThorRealTimeDataSettings.xml")
# raw_file = Path("/media/remy/remy-storage/Remy's Dropbox Folder/HongLab @ Caltech
# Dropbox/Remy/natural_mixtures/raw_data/2022-02-11/3/kiwi/Image_001_001.raw")
# meta_file = raw_file.with_name("Experiment.xml")
#
# meta = utils2p.Metadata(meta_file)
# sync_meta = utils2p.synchronization.SyncMetadata(sync_meta_file)
#
# #%%
# f = h5py.File(h5_file, 'r')
#
# line_names = []
# for item in f.keys():
#     if item != 'Freq':
#         print(item)
#         for line in f[item].keys():
#             line_names.append(line)
#             print(f"\t{line}")
#
# #%%
#
# line_names = ['PiezoMonitor',
#  'Pockels1Monitor',
#  'lightpathshutter',
#  'olfDispPin',
#  'pid',
#  'scopePin',
#  'FrameCounter',
#  'FrameIn',
#  'FrameOut',
# ]
#
# (piezo_monitor, pockels_monitor,
#  lightpath_shutter, olf_disp_pin, pid, scope_pin,
#  frame_counter, frame_in, frame_out) = utils2p.synchronization.get_lines_from_h5_file(h5_file, line_names)
#
# processed_frame_counter = utils2p.synchronization.process_frame_counter(frame_counter,
#                                                                         steps_per_frame=meta.get_n_z() +
#                                                                                         meta.get_n_flyback_frames(),
#                                                                         )
#
# single_plane_counter = utils2p.synchronization.process_frame_counter(frame_counter,
#                                                                         steps_per_frame=1)
# frame_times =
# stack_counter = np.floor_divide()
#
# sync_times = utils2p.synchronization.get_times(len(processed_frame_counter), sync_meta.get_freq())
#
# mask = (scope_pin>0.5)
# frame_times = utils2p.synchronization.get_start_times(single_plane_counter[mask],
#                                                       sync_times[mask],
#                                                       zero_based_counter=True)
#
#
#
# (cropped_frame_counter, cropped_sync_times) = \
#     utils2p.synchronization.crop_lines(mask,
#                                        (processed_frame_counter, sync_times))
#
# frame_times_2p = utils2p.synchronization.get_start_times(cropped_sync_times, cropped_sync_times)
#
#
# #%%
# stack_counter = np.floor_divide(processed_frame_counter, meta.get_n_z() + meta.get_n_flyback_frames())
# plane_counter = np.mod(processed_frame_counter, meta.get_n_z() + meta.get_n_flyback_frames())
#
# sync_times = utils2p.synchronization.get_times(len(processed_frame_counter), sync_meta.get_freq())
#
#
#
#
# #frame_times_2p = utils2p.synchronization.get_start_times(processed_frame_counter, sync_times)
#
#
# #%%
#
# #%%
# sync_file = h5_file
# sync_meta_file = folder.joinpath('ThorRealTimeDataSettings.xml')
# meta_file = Path("/media/remy/remy-storage/Remy's Dropbox Folder/HongLab @ Caltech "
#                         "Dropbox/Remy/natural_mixtures/raw_data/2022-02-11/3/kiwi/Experiment.xml")
#
# meta = utils2p.Metadata(meta_file)
#
# processed_lines = utils2p.synchronization.get_processed_lines(sync_file, sync_meta_file, meta_file)
#
# sync_times = utils2p.synchronization.get_times(len(processed_frame_counter), sync_meta.get_freq())
# unique_frames, frame_idx = np.unique(processed_frame_counter, return_index=True)
#
# unique_frames, frame_idx = np.unique(processed_frame_counter, return_index=True)
#
# frame_times = sync_times[frame_idx[unique_frames>=0 & frame_idx<]]
#
# frame_times = sync_times[processed_frame_counter>=0]
# utils2p.synchronization.get_start_times(frame_times>=0, times)
# #%%
#
# set(frame_counter)
# {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30}
#
# metadata_file = utils2p.find_metadata_file("data/mouse_kidney_z_stack")
#
# metadata = utils2p.Metadata(metadata_file)
#
# processed_frame_counter = utils2p.synchronization.process_frame_counter(frame_counter, metadata)
#
# set(processed_frame_counter)
# {0, -9223372036854775808}
#
# steps_per_frame = metadata.get_n_z() * metadata.get_n_averaging()
#
# steps_per_frame
# 30
#
# processed_frame_counter = utils2p.synchronization.process_frame_counter(frame_counter,
# steps_per_frame=steps_per_frame)
#
# set(processed_frame_counter)
# {0, -9223372036854775808}
#
# #%%
