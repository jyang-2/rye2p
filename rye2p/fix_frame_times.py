import numpy as np
from pathlib import Path
import utils2p
import copy
from typing import Union, List


# %%
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
        frames_by_pulse (List[np.array]): frames falling within the first scope pulse are at frames_by_pulse[0],
                                            within the second pulse at frames_by_pulse[1], etc.

    """

    if len(scope_ict) != len(scope_fct):
        raise ValueError('scope_ict and scope_fct are not the same length.')

    frames_by_pulse = []

    for ict, fct in zip(scope_ict, scope_fct):
        frames_in_pulse = frame_times[np_in_range(frame_times, ict, fct)]
        frames_by_pulse.append(frames_in_pulse)

    return frames_by_pulse


# %%

def correct_frame_times_fastz(frame_times, scope_ict, scope_fct, z_steps):
    """

    Args:
        frame_times (Union[List, np.ndarray]): frame times
        scope_ict (Union[List, tuple, np.ndarray]): interval start time
        scope_fct (Union[List, tuple, np.ndarray]): interval end time
        z_steps (int): # of z-steps + flyback frames, or ```meta.get_n_z() + meta.get_n_flyback_frames()```

    Returns:

    """
    split_frame_times = frames_by_scope_pulse(frame_times, scope_ict, scope_fct)
    n_frames_per_pulse = [(item.size // z_steps) * z_steps for item in split_frame_times]

    fixed_frame_times = [item[:n] for item, n in zip(split_frame_times, n_frames_per_pulse)]
    return fixed_frame_times


def fix_timestamps0(timestamps0_file, overwrite_ok=False):
    """

    Args:
        timestamps0_file ():
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

    split_frame_times = frames_by_scope_pulse(timestamps0['frame_times'],
                                              timestamps0['scope_ict'],
                                              timestamps0['scope_fct'])

    stack_times_per_pulse = [item[steps_per_frame - 1::steps_per_frame] for item in split_frame_times]
    n_stack_times_per_pulse = [item.size for item in stack_times_per_pulse]
    fixed_stack_times = np.concatenate(stack_times_per_pulse)

    frame_times_per_pulse = [item[:n * steps_per_frame] for item, n in zip(split_frame_times, n_stack_times_per_pulse)]
    n_frame_times_per_pulse = [item.size for item in frame_times_per_pulse]
    fixed_frame_times = np.concatenate(frame_times_per_pulse)

    if fixed_stack_times.size != n_timepoints:
        raise Exception("Could not extract the correct # of stack_times.")
    if fixed_frame_times.size != n_frames:
        raise Exception("Could not extract the correct # of frame_times.")

    timestamps_file = timestamps0_file.with_name('timestamps.npy')

    if not timestamps_file.is_file() or overwrite_ok:
        timestamps = copy.deepcopy(timestamps0)
        timestamps['frame_times'] = fixed_frame_times
        timestamps['stack_times'] = fixed_stack_times
        timestamps['split_frame_times'] = frame_times_per_pulse
        timestamps['split_stack_times'] = stack_times_per_pulse
        timestamps['n_frame_times_per_pulse'] = n_frame_times_per_pulse
        timestamps['n_stack_times_per_pulse'] = n_stack_times_per_pulse
        np.save(timestamps_file, timestamps)
    return timestamps_file



if __name__ == '__main__':
    PROC_DATA_DIR = Path("/local/storage/Remy/narrow_odors/processed_data")
    file_list = sorted(list(PROC_DATA_DIR.rglob('timestamps0.npy')))

    for file in file_list[1:]:
        print(fix_timestamps0(file, overwrite_ok=True))


