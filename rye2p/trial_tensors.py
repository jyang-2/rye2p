import numpy as np
from scipy.interpolate import interp1d


def split_traces_to_trials(cells_x_time, ts, stim_ict, trial_ts):
    """
    Extracts periods of activity defined by `trial_ts` centered at `stim_ict`.

    Extracted trials should be the same length.

    Args:
        cells_x_time (np.ndarray):
        ts (np.ndarray):
        stim_ict (Union[List, np.ndarray]):
        trial_ts (np.ndarray): timestamps for trial interval, with stim_ict corresponding to 0

    Returns:
        (List[np.ndarray]): cell x trial subarrays, split according to trial_ts.

    Examples:
        >>>> trial_timestamps = np.arange(-5, 20, 0.5)
        >>>> trial_traces = split_traces_to_trials(Fc, ts, olf_ict, trial_timestamps)
        >>>> ttrials =  np.stack(trial_traces, axis=0)
    """

    interp_traces = interp1d(ts, cells_x_time, axis=-1)

    F_trials = []

    for ict in stim_ict:
        F_trials.append(interp_traces(trial_ts+ict))

    return F_trials


def make_trial_tensor(cells_x_time, ts, stim_ict, trial_ts):
    """
    Converts neuron activity array into a 3D tensor, w/ dimensions trials x neurons x time.

    Trial intervals are defined by ts, stim_ict, and trial_ts.

    If you want the time period from 5 sec. before stimulus onset to 20 seconds after stimulus onset,
    trial_ts should be something like np.arange(-5, 20, 0.2).

    This will also result in the data being interpolated to a frame rate of 1/dt, or 1/0.2 = 5


    Args:
        cells_x_time (np.ndarray):
        ts (np.ndarray):
        stim_ict (Union[List, np.ndarray]):
        trial_ts (np.ndarray): timestamps for trial interval, with stim_ict corresponding to 0

    Returns:
        (List[np.ndarray]): cell x trial subarrays, split according to trial_ts.

    Examples:
        >>>> trial_timestamps = np.arange(-5, 20, 0.5)
        >>>> trial_tensors = make_trial_tensors(Fc, ts, olf_ict, trial_timestamps)
        >>>> ttrials.shape
                (57, 1847, 1690)
    """
    F_trials = split_traces_to_trials(cells_x_time, ts, stim_ict, trial_ts)
    trial_tensor = np.stack(F_trials, axis=0)
    return trial_tensor
