"""Fixes issues with extracting ttl pulses and timestamp information from ThorSync files"""
import copy
from pathlib import Path
from typing import Union, List

import numpy as np
import utils2p
from scipy.optimize import linear_sum_assignment


def match_pulse_times(ict, fct):
    # too many ict
    dt = scope_time_diff(ict, fct)

    # scope_fct must fall after scope_ict
    cost = copy.deepcopy(dt)
    cost[cost < 0] = np.Inf

    row_idx, col_idx = linear_sum_assignment(cost)

    matched_ict = ict[col_idx]
    matched_fct = fct[row_idx]
    return matched_ict, matched_fct


def match_scope_times(scope_ict, scope_fct):
    # too many ict
    dt = scope_time_diff(scope_ict, scope_fct)

    # scope_fct must fall after scope_ict
    cost = copy.deepcopy(dt)
    cost[cost < 0] = np.Inf

    row_idx, col_idx = linear_sum_assignment(cost)

    matched_scope_ict = scope_ict[col_idx]
    matched_scope_fct = scope_fct[row_idx]
    return matched_scope_ict, matched_scope_fct


def scope_time_diff(scope_ict, scope_fct):
    # return scope_ict[:, np.newaxis] - scope_fct[np.newaxis, :]
    return scope_fct[:, np.newaxis] - scope_ict[np.newaxis, :]


def same_scope_time_sizes(scope_ict, scope_fct):
    return scope_ict.size == scope_fct.size


def drop_frames(timestamps, frames_to_drop):
    timestamps_new = copy.deepcopy(timestamps)
    timestamps_new['stack_times'] = np.delete(timestamps['stack_times'], frames_to_drop)
    timestamps_new['dropped_frames'] = frames_to_drop
    return timestamps_new


def downsample_timestamps(timestamps, tsub, method='mean'):
    stack_times = timestamps['stack_times']
    stack_times_trimmed = stack_times[:(stack_times.size // tsub) * tsub]

    idx = np.arange(0, stack_times_trimmed.size, tsub)
    split_stacks = np.split(stack_times_trimmed, idx[1:])

    stack_times_ds = [item.mean() for item in split_stacks]
    stack_times_ds = np.array(stack_times_ds)

    print(f'downsampled stack_times has {stack_times_ds.size} timepoints')
    timestamps_ds = copy.deepcopy(timestamps)
    timestamps_ds['tsub'] = tsub
    timestamps_ds['stack_times'] = stack_times_ds
    return timestamps_ds


def edit_timestamps(timestamps, frames_to_drop, tsub, method='mean'):
    timestamps1 = drop_frames(timestamps, frames_to_drop)
    timestamps2 = downsample_timestamps(timestamps1, tsub, method=method)
    return timestamps2


def edit_timestamps_file(ts_file, frames_to_drop, tsub, method='mean'):
    timestamps = np.load(ts_file, allow_pickle=True).item()
    timestamps_ds = edit_timestamps(timestamps, frames_to_drop, tsub, method=method)
    ts_ds_file = ts_file.with_name(f'timestamps_tsub{tsub}.npy')
    np.save(ts_ds_file, timestamps_ds)
    return ts_ds_file


def np_in_range(x, x_start, x_end):
    return np.logical_and(x >= x_start, x < x_end)


def frames_by_scope_pulse(frame_times, scope_ict, scope_fct):
    """
    Returns frames that fall within a scope_pulse (start_time, end_time) in a list.

    Args:
        frame_times (np.array): 1D array of frame times
        scope_ict (List, np.array): scope pulse start time (ict stands for initial cross time)
        scope_fct (List, np.array): scope_pulse end time (fct stands for final cross time)

    Returns:
        frames_by_pulse (List[np.array]): frames falling within the first scope pulse are at
        frames_by_pulse[0],
                                            within the second pulse at frames_by_pulse[1], etc.

    """

    if len(scope_ict) != len(scope_fct):
        raise ValueError('scope_ict and scope_fct are not the same length.')

    frames_by_pulse = []

    for ict, fct in zip(scope_ict, scope_fct):
        frames_in_pulse = frame_times[np_in_range(frame_times, ict, fct)]
        frames_by_pulse.append(frames_in_pulse)

    return frames_by_pulse


def correct_frame_times_fastz(frame_times, scope_ict, scope_fct, z_steps):
    """

    Args:
        frame_times (Union[List, np.ndarray]): frame times
        scope_ict (Union[List, tuple, np.ndarray]): interval start time
        scope_fct (Union[List, tuple, np.ndarray]): interval end time
        z_steps (int): # of z-steps + flyback frames, or ```meta.get_n_z() +
        meta.get_n_flyback_frames()```

    Returns:

    """
    split_frame_times = frames_by_scope_pulse(frame_times, scope_ict, scope_fct)
    n_frames_per_pulse = [(item.size // z_steps) * z_steps for item in split_frame_times]

    fixed_frame_times = [item[:n] for item, n in zip(split_frame_times, n_frames_per_pulse)]
    return fixed_frame_times


def fix_timestamp_pulse_times(timestamps):
    """If there isn't an equal # of **_ict and {}_fct elements, they are matched."""
    scope_ict = timestamps['scope_ict']
    scope_fct = timestamps['scope_fct']
    if scope_ict.size != scope_fct.size:
        scope_ict, scope_fct = match_pulse_times(scope_ict, scope_fct)

    olf_ict = timestamps['olf_ict']
    olf_fct = timestamps['olf_fct']
    if olf_ict.size != olf_fct.size:
        olf_ict, olf_fct = match_pulse_times(olf_ict, olf_fct)

    new_timestamps = dict(frame_times=timestamps['frame_times'],
                          stack_times=timestamps['stack_times'],
                          scope_ict=scope_ict,
                          scope_fct=scope_fct,
                          olf_ict=olf_ict,
                          olf_fct=olf_fct)
    return new_timestamps


def fix_olf_times(timestamps0):
    """Removes any olf times that occur outside of a scope pulse (between scope_ict and
    scope_fct)"""
    fixed_olf_ict = frames_by_scope_pulse(timestamps0['olf_ict'],
                                          timestamps0['scope_ict'],
                                          timestamps0['scope_fct'])
    fixed_olf_ict = np.concatenate(fixed_olf_ict)

    fixed_olf_fct = frames_by_scope_pulse(timestamps0['olf_fct'],
                                          timestamps0['scope_ict'],
                                          timestamps0['scope_fct']
                                          )
    fixed_olf_fct = np.concatenate(fixed_olf_fct)

    olf_fixed_timestamps0 = copy.deepcopy(timestamps0)
    olf_fixed_timestamps0['olf_ict'] = fixed_olf_ict
    olf_fixed_timestamps0['olf_fct'] = fixed_olf_fct

    return olf_fixed_timestamps0


def fix_timestamps0(timestamps0_file):
    """

    Args:
        timestamps0_file (Union[str, Path]):
        timestamps_file (str, Path):
        overwrite_ok ():

    Returns:

    """
    if isinstance(timestamps0_file, str):
        timestamps0_file = Path(timestamps0_file)

    timestamps0 = np.load(timestamps0_file, allow_pickle=True).item()

    meta_file = timestamps0_file.with_name("Experiment.xml")
    meta = utils2p.Metadata(meta_file)

    n_frames = int(meta.get_value('Streaming', 'frames'))
    n_timepoints = meta.get_n_time_points()
    steps_per_frame = meta.get_n_z() + meta.get_n_flyback_frames()

    timestamps0 = fix_timestamp_pulse_times(timestamps0)

    split_frame_times = frames_by_scope_pulse(timestamps0['frame_times'],
                                              timestamps0['scope_ict'],
                                              timestamps0['scope_fct'])

    stack_times_per_pulse = [item[steps_per_frame - 1::steps_per_frame] for item in
                             split_frame_times]
    n_stack_times_per_pulse = [item.size for item in stack_times_per_pulse]
    fixed_stack_times = np.concatenate(stack_times_per_pulse)

    frame_times_per_pulse = [item[:n * steps_per_frame] for item, n in
                             zip(split_frame_times, n_stack_times_per_pulse)]
    n_frame_times_per_pulse = [item.size for item in frame_times_per_pulse]
    fixed_frame_times = np.concatenate(frame_times_per_pulse)

    if fixed_stack_times.size != n_timepoints:
        # raise Exception("Could not extract the correct # of stack_times.")
        print("Could not extract the correct # of stack_times.")
    if fixed_frame_times.size != n_frames:
        # raise Exception("Could not extract the correct # of frame_times.")
        print("Could not extract the correct # of frame_times.")

    # timestamps_file = timestamps0_file.with_name('timestamps.npy')

    timestamps = copy.deepcopy(timestamps0)
    timestamps['frame_times'] = fixed_frame_times
    timestamps['stack_times'] = fixed_stack_times
    timestamps['split_frame_times'] = frame_times_per_pulse
    timestamps['split_stack_times'] = stack_times_per_pulse
    timestamps['n_frame_times_per_pulse'] = n_frame_times_per_pulse
    timestamps['n_stack_times_per_pulse'] = n_stack_times_per_pulse

    return timestamps


# %%

if __name__ == '__main__':
    # %%

    # mov_dir = Path("/local/matrix/Remy-Data/projects/odor_space_collab/"
    #                "processed_data/2023-08-29/2/prospective0")
    # mov_dir = Path("/local/matrix/Remy-Data/projects/odor_space_collab/"
    #                "processed_data/2023-09-13/1/validation2c_1")
    # mov_dir = Path("/local/matrix/Remy-Data/projects/odor_space_collab/"
    #                "processed_data/2023-09-13/1/reference")
    # mov_dir = Path("/local/matrix/Remy-Data/projects/odor_space_collab/"
    #                "processed_data/2023-09-19/1/validation2e_single_plane")
    # mov_dir = Path("/local/matrix/Remy-Data/projects/odor_space_collab/"
    #                "processed_data/2023-09-12/1/validation2c_1")
    # mov_dir = Path("/local/matrix/Remy-Data/projects/odor_space_collab/"
    #                "processed_data/2023-09-14/2/validation2d_single_plane_001")
    # mov_dir = Path("/local/matrix/Remy-Data/projects/odor_space_collab/"
    #                "processed_data/2023-09-21/2/validation2f_single_plane")
    # mov_dir = Path("/local/matrix/Remy-Data/projects/odor_space_collab/"
    #                "processed_data/2023-10-03/2/validation2g_single_plane"),
    # mov_dir = Path("/local/matrix/Remy-Data/projects/odor_space_collab/"
    #                "processed_data/2023-10-16/3/validation2i")
    # mov_dir = Path(
    #     "/local/matrix/Remy-Data/projects/odor_space_collab/processed_data/"
    #     "2023-10-17/1/validation2j")
    # mov_dir = Path("/local/matrix/Remy-Data/projects/odor_space_collab/"
    #                "processed_data/2023-10-19/1/validation2k_single_plane")
    # mov_dir = Path("/local/matrix/Remy-Data/projects/odor_space_collab/processed_data/"
    #                "2023-10-25/1/validation2kwm_single_plane")
    # mov_dir = Path("/local/matrix/Remy-Data/projects/odor_space_collab/processed_data/"
    #                "2023-10-25/2/validation2kwm_single_plane")
    # mov_dir = Path("/local/matrix/Remy-Data/projects/odor_space_collab/processed_data/"
    #                "2023-11-02/2/validation2j_single_plane")
    # mov_dir = Path("/local/matrix/Remy-Data/projects/odor_space_collab/processed_data/"
    #                "2023-11-03/1/validation2iwm_single_plane")

    # mov_dir = Path("/local/matrix/Remy-Data/projects/odor_space_collab/processed_data/"
    #                "2023-11-03/3/validation2iwm_single_plane")
    # mov_dir = Path("/local/matrix/Remy-Data/projects/odor_space_collab/processed_data/"
    #                "2023-11-07/1/validation2iwmh_single_plane")
    # mov_dir = Path("/local/matrix/Remy-Data/projects/odor_space_collab/processed_data/"
    #                "2023-11-07/2/validation2iwmh_single_plane")
    # mov_dir = Path("/local/matrix/Remy-Data/projects/odor_space_collab/processed_data/"
    #                "2023-11-21/1/validation2iwmh1_single_plane")

    # mov_dir = Path("/local/matrix/Remy-Data/projects/odor_space_collab/processed_data/"
    #                "2023-11-29/2/validation2iwmh1_single_plane_001")
    # mov_dir = Path("/local/matrix/Remy-Data/projects/odor_space_collab/processed_data/"
    #                "2023-12-06/5/validation2jwmh_single_plane")
    # mov_dir = Path("/local/matrix/Remy-Data/projects/odor_space_collab/processed_data/"
    #                "2023-12-06/5/validation2jwmh_single_plane_001")
    # mov_dir = Path("/local/matrix/Remy-Data/projects/odor_space_collab/"
    #                "processed_data/2023-12-13/1/validation2kwmh")
    # mov_dir = Path("/local/matrix/Lexi/rotation_project/processed_data/"
    #                "2024-04-18/1/megamat06_singleplane")

    # mov_dir = Path("/local/matrix/Lexi/rotation_project/processed_data/"
    #                "2024-04-22/2/megamat06_fast_singleplane")
    # mov_dir = Path("/local/matrix/Lexi/rotation_project/processed_data/"
    #                "2024-05-03/1/lexi10odors_singleplane_001")
    mov_dir = Path("/local/matrix/Lexi/rotation_project/processed_data/"
                   "2024-05-03/2/lexi10odors_singleplane_001")

    meta = utils2p.Metadata(mov_dir.joinpath("Experiment.xml"))

    average_mode = int(meta.get_metadata_value("LSM", "averageMode"))
    n_frames = int(meta.get_value('Streaming', 'frames'))
    n_timepoints = meta.get_n_time_points()
    n_frames_averaged = meta.get_n_averaging()
    fast_z_mode = int(meta.get_metadata_value("Streaming", "zFastEnable"))

    # %%
    # print experiment metadata (expected frame, timepoint #)
    print('\nExperiment.xml')
    print('--------------')
    print(f"n_frames = {n_frames}")
    print(f"n_timepoints = {n_timepoints}")
    print(f"n_frames_averaged = {n_frames_averaged}")
    print(f"average_mode = {average_mode}")
    print(f"fast_z_mode = {fast_z_mode}")

    # print timestamp file info
    timestamps0 = np.load(mov_dir.joinpath("timestamps0.npy"), allow_pickle=True).item()
    print('\ntimestamps0.npy')
    print('--------------')
    print(f"frame_times = {timestamps0['frame_times'].size}")
    print(f"stack_times = {timestamps0['stack_times'].size}")

    timestamps = np.load(mov_dir.joinpath("timestamps.npy"), allow_pickle=True).item()
    print('\ntimestamps.npy')
    print('--------------')
    print(f"frame_times = {timestamps['frame_times'].size}")
    print(f"stack_times = {timestamps['stack_times'].size}")

    # %% if recording stopped early

    # dt = np.diff(timestamps0['stack_times']).mean()
    # stack_times_append = np.arange(n_timepoints - timestamps0['stack_times'].size) + 1
    # stack_times_append = (stack_times_append * dt) + timestamps0['stack_times'][-1]
    #
    # isi = np.diff(timestamps0['olf_ict']).mean()
    # olf_pulse_len = (timestamps0['olf_fct'] - timestamps0['olf_ict']).mean()
    #
    # olf_ict_append = np.arange(51 - timestamps0['olf_ict'].size) + 1
    # olf_ict_append = (olf_ict_append * isi) + timestamps0['olf_ict'][-1]
    # olf_fct_append = olf_ict_append + olf_pulse_len
    #
    # olf_ict_corrected = np.concatenate((timestamps0['olf_ict'], olf_ict_append))
    # olf_fct_corrected = np.concatenate((timestamps0['olf_fct'], olf_fct_append))
    # stack_times_corrected = np.concatenate((timestamps0['stack_times'], stack_times_append))
    #
    # timestamps0['olf_ict'] = olf_ict_corrected
    # timestamps0['olf_fct'] = olf_fct_corrected
    # timestamps0['stack_times'] = stack_times_corrected

    # %% recompute `stack_times` by downsampling timestamps0['frame_times']
    if average_mode == 0:
        if fast_z_mode == 0:
            stack_times = timestamps0['frame_times']
        elif fast_z_mode == 1:
            stack_times = timestamps0['stack_times']
    elif average_mode == 1:
        stack_times = timestamps0['frame_times'][::n_frames_averaged]

    print(f"recomputed stack_time size = {stack_times.size}")

    # %%
    if stack_times.size == n_timepoints:
        timestamps0['stack_times'] = stack_times
    elif stack_times.size == n_timepoints + 1:
        stack_times_corrected = stack_times[:-1]
        timestamps0['stack_times'] = stack_times_corrected
    elif stack_times.size == n_timepoints - 1:
        stack_times_corrected = list(stack_times)
        dt = np.diff(stack_times).mean()
        stack_times_corrected.append(stack_times[-1] + dt)
        stack_times_corrected = np.array(stack_times_corrected)
        timestamps0['stack_times'] = stack_times_corrected

    fps = np.diff(timestamps0['stack_times']).mean()
    fs = 1 / fps

    # %% check timestamps0

    print('\nExperiment.xml')
    print('--------------')
    print(f"n_frames = {n_frames}")
    print(f"n_timepoints = {n_timepoints}")
    print(f"n_frames_averaged = {n_frames_averaged}")
    print(f"average_mode = {average_mode}")
    print(f"fast_z_mode = {fast_z_mode}")

    # print timestamp file info
    print('\ntimestamps0.npy')
    print('--------------')
    print(f"frame_times = {timestamps0['frame_times'].size}")
    print(f"stack_times = {timestamps0['stack_times'].size}")
    print(f"dt = {fps}")
    print(f"fs = {fs}")

    # %% check odor pulse trials

    olf_ict_mask = ((timestamps0['stack_times'][0] < timestamps0['olf_ict']) &
                    (timestamps0['olf_ict'] < timestamps0['stack_times'][-1]))
    olf_fct_mask = ((timestamps0['stack_times'][0] < timestamps0['olf_fct']) &
                    (timestamps0['olf_fct'] < timestamps0['stack_times'][-1]))

    if not olf_ict_mask.all():
        print(f'Trimming odor pulse trials:'
              f'\n\n\t- olf_ict old = {olf_ict_mask.size} trials'
              f'\n\t- olf_ict new = {olf_ict_mask.sum()} trials'
              f'\n\n\t- olf_fct old = {olf_fct_mask.size} trials'
              f'\n\t- olf_fct new = {olf_fct_mask.sum()} trials'
              )
        olf_ict_trimmed = timestamps0['olf_ict'][olf_ict_mask]
        olf_fct_trimmed = timestamps0['olf_fct'][olf_fct_mask]

        timestamps0['olf_ict'] = olf_ict_trimmed
        timestamps0['olf_fct'] = olf_fct_trimmed

    # %%

    # %%
    np.save(mov_dir.joinpath("timestamps.npy"), timestamps0, allow_pickle=True)
    print('timestamps.npy corrected and saved.')
    print(mov_dir.joinpath("timestamps.npy").as_uri())

#     PROC_DATA_DIR = Path("/local/storage/Remy/narrow_odors/processed_data")
#     file_list = sorted(list(PROC_DATA_DIR.rglob('timestamps0.npy')))
#
#     for file in file_list[1:]:
#         print(fix_timestamps0(file, overwrite_ok=True))
