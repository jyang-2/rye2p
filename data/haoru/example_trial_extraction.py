import numpy as np
import pandas as pd
from pathlib import Path
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt


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
        F_trials.append(interp_traces(trial_ts + ict))

    F_trials = np.stack(F_trials, axis=0)

    return F_trials


if __name__ == "__main__":
    # %%
    folder = Path('data/haoru')

    # load traces
    file = folder.joinpath('F.tsv')
    df = pd.read_csv(file, sep='\t', index_col=0)

    # load stimuli
    df_stim = pd.read_csv(folder.joinpath('df_stim.csv'), sep='\t')

    # Load timestamps
    timestamps = np.load(folder.joinpath('timestamps.npy'), allow_pickle=True).item()

    # %% Split trace to trial x time

    # get fluorescence trace
    F = df['F'].to_numpy()

    # Resample to 5 seconds before stimulus onset, and 20 seconds after
    # timestamps relative to stimulus onset (odor onset at t = 0.0)
    trial_ts = np.arange(-5.0, 20 + 0.0001, 1.0)

    # Returns a trial x time array
    trial_F = split_traces_to_trials(F,
                                     ts=timestamps['stack_times'],
                                     stim_ict=timestamps['olf_ict'],
                                     trial_ts=trial_ts
                                     )
    # %% Convert into a time x trial DataFrame, with stimuli and trial order in the columns

    # make columns
    mi_columns = pd.MultiIndex.from_frame(df_stim.loc[:, ['stim', 'trial_idx']])

    # make time x trial DataFrame
    df_trial_F = pd.DataFrame(trial_F.T,
                              index=trial_ts,
                              columns=mi_columns)

    # compute stimulus-averaged trial F
    df_stim_F = df_trial_F.groupby(by='stim', axis=1).mean()
    # %% Plot traces
    fig, axarr = plt.subplots(2, 5, figsize=[12, 6],
                              sharex=True, sharey=True)

    for col, ax in zip(df_stim_F.columns, axarr.flat):
        # plot trial traces
        df_trial_F.xs(col, axis=1).plot(ax=ax, legend=True)

        # plot stimulus-averaged total fluorescence
        ax.plot(df_stim_F[col], color='k')
        ax.set_title(col)

    plt.tight_layout()
    plt.show()
    # %%
    fig.savefig(folder.joinpath('example.png'))
# %%
