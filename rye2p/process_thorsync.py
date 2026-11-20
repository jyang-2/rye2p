# import itertools
# import os
# import sys
# from pathlib import Path
# import dateparser
import shutil
import numpy as np
# import pydantic
# import h5py
import utils2p
import utils2p.synchronization
from re import sub

# import yaml
# import json
# import dotenv
import pprint as pp
# import math
# from scipy.ndimage import gaussian_filter1d
# from scipy.stats import zscore

import h5py
from fuzzywuzzy import fuzz
from rye2p import fix_frame_times
from rye2p.pathguard import guard_write_dir


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
    if not h5_file.is_file():
        h5_file = raw_dir.joinpath(flat_lacq['date_imaged'],
                                   str(flat_lacq['fly_num']),
                                   flat_lacq['thorsync'],
                                   'Episode_0000.h5')

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


def extract_timestamp_data(h5_file, meta_file, microscope=None, strict=True):
    """Extract timestamp arrays from a ThorSync h5 + paired Experiment.xml.

    Rewritten in roadmap step 2 to delegate to the unified, ground-truthed
    extraction (``rye2p.thorsync.timestamps.extract_timestamps_from_files``).
    The old in-place implementation dropped frame 0 and volume 0 (known issue
    #22 — every ``stack_times[i]`` was one volume period late), derived
    ``steps_per_frame`` without the ``fast_z``/``averageMode`` guards (#1),
    and assumed analog voltages on the scope pin (#3).

    Args:
        h5_file: path to ``Episode001.h5``.
        meta_file: path to the paired ThorImage ``Experiment.xml``.
        microscope: rig name for the line map (e.g. ``'upstairs-2p'``); None
            falls back to fuzzy line matching with a warning.
        strict: raise on pairing-fingerprint or count mismatches (default)
            instead of warning.

    Returns:
        dict with ``frame_times`` (whole timepoints only, frame 0 included),
        ``stack_times`` (first-plane convention), ``scope_ict``/``scope_fct``,
        ``olf_ict``/``olf_fct`` (all olf pulses, incl. outside scope pulses),
        plus the per-pulse ``split_*`` / ``n_*_per_pulse`` keys.
    """
    from rye2p.thorsync import timestamps as ts

    return ts.extract_timestamps_from_files(h5_file, meta_file,
                                            microscope=microscope,
                                            strict=strict)


def fast_z_enabled(meta_file):
    meta = utils2p.Metadata(meta_file)
    return bool(int(meta.get_metadata_value("Streaming", 'zFastEnable')))


def convert_thorsync_to_timestamps_npy(flacq, raw_dir, proc_dir,
                                       match_stack_times_only=True,
                                       microscope=None, strict=True):
    """ Extracts timing info for frame acquisitions, stimuli, and scope acquisitions.

    Since roadmap step 2 this runs the unified extraction
    (:func:`extract_timestamp_data`), which already returns whole-timepoint
    ``frame_times`` (frame 0 included), first-plane ``stack_times`` with the
    averaging handled via ``steps_per_frame``, and validated counts — the old
    post-hoc repairs (``fix_timestamps0``, the single-plane
    ``[::n_averaging]`` re-decimation) are no longer applied. The final
    ``timestamps.npy`` only differs from ``timestamps0.npy`` by dropping olf
    pulses that fall outside the scope pulses (``fix_olf_times`` — now for
    single-plane recordings too).

    Args:
        proc_dir ():
        raw_dir ():
        flacq (dict): dict from dataset manifesto `flat_linked_thor_acquisitions.json`
        match_stack_times_only (bool): ignored since roadmap step 2 (kept for
            signature compatibility) — the extraction always regroups per
            scope pulse and validates both counts.
        microscope (str): rig name for the line map (e.g. ``'upstairs-2p'``);
            None falls back to fuzzy line matching with a warning.
        strict (bool): raise on pairing-fingerprint or count mismatches
            (default) instead of warning — nothing is saved on a failure.

    Returns:
        timestamps (dict): contains fields 'stack_times', 'frame_times', 'scope_ict', 'scope_fct', 'olf_ict', 'olf_fct'
        """
    from rye2p.thorimage.settings import FunctionalImagingSettings

    # make initial timestamps0.npy
    h5_file, sync_meta_file, meta_file = flacq2rawfiles(flacq, raw_dir)
    SAVE_DIR = flacq2dir(flacq, proc_dir)
    guard_write_dir(SAVE_DIR, what="timestamps npy files")

    print(f'\nConverting ThorSync to timestamps:')
    print(f"\t- SAVE_DIR: {SAVE_DIR}")

    settings = FunctionalImagingSettings.from_xml(meta_file)

    timestamps0 = extract_timestamp_data(h5_file, meta_file,
                                         microscope=microscope, strict=strict)
    np.save(SAVE_DIR.joinpath('timestamps0.npy'), timestamps0)

    # remove olf times outside of scope trigger pulses
    timestamps = fix_frame_times.fix_olf_times(timestamps0)

    expected_frame_times = (settings.n_timepoints * settings.steps_per_frame
                            if settings.fast_z else None)
    print(f"\t- stack_times: {timestamps['stack_times'].size} "
          f"(Experiment.xml n_timepoints: {settings.n_timepoints})")
    if expected_frame_times is not None:
        print(f"\t- frame_times: {timestamps['frame_times'].size} "
              f"(expected {expected_frame_times})")
    else:
        print(f"\t- frame_times: {timestamps['frame_times'].size} raw frames "
              f"(single plane, n_averaging={settings.n_averaging})")

    save_file = SAVE_DIR.joinpath('timestamps.npy')
    np.save(save_file, timestamps)
    print(f"timestamps.npy saved")
    print(f"- {save_file}")

    return save_file, timestamps


def create_proc_dir(flacq, proc_dir):
    """Creates movie folder in processed data directory `PROC_DATA_DIR` for FlatFlyAcquisitions."""

    mov_dir = flacq2dir(flacq, proc_dir)
    guard_write_dir(mov_dir, what="processed movie folder")
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
        guard_write_dir(dest_file.parent, what="Experiment.xml copy")
        shutil.copy(meta_file, dest_file)

        print(f"\nCopied meta_file:"
              f"\n\t- src:{meta_file}"
              f"\n\t- dest: {dest_file}")

    return meta_file, dest_file


def main(flacq, raw_dir, proc_dir, match_stack_times_only=True,
         microscope=None, strict=True):
    print(f'\nProcessing ThorSync files:')
    print(f'---------------------------')
    pp.pprint(flacq)

    mov_dir = create_proc_dir(flacq, proc_dir)

    copy_experiment_xml(flacq, raw_dir, proc_dir)

    timestamps_file, timestamps = convert_thorsync_to_timestamps_npy(flacq, raw_dir, proc_dir,
                                                                     match_stack_times_only=match_stack_times_only,
                                                                     microscope=microscope,
                                                                     strict=strict)
    print('Done processing ThorSync files.')
    return timestamps_file, timestamps
