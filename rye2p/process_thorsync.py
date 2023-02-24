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


def get_thorsync_line_names(h5_file):
    thorsync_groups = ['AI', 'CI', 'DI', 'Global']

    line_names = []
    with h5py.File(h5_file, 'r') as f:
        for grp in f.keys():
            for item in f[grp]:
                line_names.append(item)
    line_names.remove('GCtr')
    return line_names


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

    # line names to search for
    target_var_names = [clean_line_name(item) for item in var_names]

    line_names_from_file = get_thorsync_line_names(h5_file)
    cleaned_line_names_from_file = [clean_line_name(item) for item in line_names_from_file]

    # match ratio: dims = (# target_vars, # vars in file)
    match_ratio = np.array([[fuzz.ratio(x, y)
                             for x in target_var_names]
                            for y in cleaned_line_names_from_file]
                           )
    matched_thorsync_lines = np.array(line_names_from_file)[np.argmax(match_ratio, axis=0)].tolist()

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


def convert_thorsync_to_timestamps_npy(flacq, raw_dir, proc_dir, match_stack_times_only=True):
    """ Extracts timing info for frame acquisitions, stimuli, and scope acquisitions.

    Args:
        proc_dir ():
        raw_dir ():
        flacq (dict): dict from dataset manifesto `flat_linked_thor_acquisitions.json`
        match_stack_times_only (bool): only require the # of stack_times extracted to match the # of timepoints in
            Experiment.xml

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

    if fast_z:
        if match_stack_times_only:
            # remove olf times outside of scope trigger pulses
            timestamps = fix_frame_times.fix_olf_times(timestamps0)
        else:
            timestamps = fix_frame_times.fix_timestamps0(SAVE_DIR.joinpath('timestamps0.npy'))
    else:
        # if movie is single plane, it's important to match frame times and stack times
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


def main(flacq, raw_dir, proc_dir, match_stack_times_only=True):
    print(f'\nProcessing ThorSync files:')
    print(f'---------------------------')
    pp.pprint(flacq)

    mov_dir = create_proc_dir(flacq, proc_dir)

    copy_experiment_xml(flacq, raw_dir, proc_dir)

    timestamps_file, timestamps = convert_thorsync_to_timestamps_npy(flacq, raw_dir, proc_dir,
                                                                     match_stack_times_only=match_stack_times_only)
    print('Done processing ThorSync files.')
    return timestamps_file, timestamps
