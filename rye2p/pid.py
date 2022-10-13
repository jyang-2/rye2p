import os
from pathlib import Path
import numpy as np
import h5py
import json
import utils2p
import utils2p.synchronization
import dotenv
import xarray as xr
from rye2p import trial_tensors

# load .env variables
# dotenv.load_dotenv(Path.home().joinpath('dotenv_files/natural_mixtures/.env'))
#
# DB_PRJ_DIR = Path(os.getenv("DB_PRJ_DIR"))
# DB_RAW_DIR = Path(os.getenv("DB_RAW_DIR"))
#
# NAS_PRJ_DIR = Path(os.getenv("NAS_PRJ_DIR"))
# NAS_PROC_DIR = Path(os.getenv("NAS_PROC_DIR"))
# NAS_OLFCONFIG_DIR = Path(os.getenv("NAS_OLFCONFIG_DIR"))

prj = 'odor_space_collab'

if prj == 'odor_space_collab':
    DB_RAW_DIR = Path("/media/remy/remy-storage/Remy's Dropbox Folder/HongLab @ Caltech "
                      "Dropbox/Remy/odor_space_collab/raw_data")
    NAS_PRJ_DIR = Path("/local/matrix/Remy-Data/projects/odor_space_collab")
    NAS_PROC_DIR = Path("/local/matrix/Remy-Data/projects/odor_space_collab/processed_data")


# %%
def flacq2dir(flat_lacq):
    folder = NAS_PROC_DIR.joinpath(flat_lacq['date_imaged'],
                                   str(flat_lacq['fly_num']),
                                   flat_lacq['thorimage'])
    return folder


def flacq2rawfiles(flat_lacq):
    meta_file_ = DB_RAW_DIR.joinpath(flat_lacq['date_imaged'],
                                     str(flat_lacq['fly_num']),
                                     flat_lacq['thorimage'],
                                     'Experiment.xml')

    h5_file_ = DB_RAW_DIR.joinpath(flat_lacq['date_imaged'],
                                   str(flat_lacq['fly_num']),
                                   flat_lacq['thorsync'],
                                   'Episode001.h5')

    sync_meta_file_ = h5_file_.with_name('ThorRealTimeDataSettings.xml')
    return h5_file_, sync_meta_file_, meta_file_


def plot_pid_traces(da_plot):
    """ Plot PID responses in a facet grid


    """
    da_grouped = da_plot.groupby('stim')
    fgrid = da_grouped.mean().plot(x='time',
                                   col='stim',
                                   col_wrap=4,
                                   color='k',
                                   linewidth=0.5, figsize=(8.5, 11))

    for ax, named in zip(fgrid.axes.flatten(), fgrid.name_dicts.flatten()):
        if named is not None:
            da_grouped[named['stim']].plot.line(ax=ax, x='time',
                                                hue='trial',
                                                alpha=0.6,
                                                linewidth=0.2,
                                                zorder=0,
                                                add_legend=False)
            ax.axvline(0, color='0.3', linestyle='--', linewidth=1, zorder=0)
            ax.set_title(named['stim'], fontsize=8)
            ax.set_xlabel('time', fontsize=8)
            ax.set_ylabel('V', fontsize=8)

    fgrid.fig.subplots_adjust(wspace=.2, hspace=.2, top=0.9)
    fgrid.fig.suptitle('pid')
    return fgrid


def load_pid_lines(flacq):
    """Load the lines relevant to odor delivery from thorsync (.h5) recording.

    Args:
        flacq (dict): dictionary containing flat linked acquisition info
                      loaded from natural_mixtures/manifestos/flat_linked_thor_acquisitions.json

    Returns:
        pid (np.ndarray): voltage recorded from aurora photoionization detector
        olf_disp_pin (np.ndarray): olfactometer trigger ttl pulse (0-5V)
        scope_pin (np.ndarray): microscope acquisition ttl pulse (0-5V)
    """
    h5_file, sync_meta_file, meta_file = flacq2rawfiles(flacq)
    meta = utils2p.Metadata(meta_file)
    sync_meta = utils2p.synchronization.SyncMetadata(sync_meta_file)

    line_names = ['olfDispPin', 'scopePin', 'pid']
    olf_disp_pin, scope_pin, pid = utils2p.synchronization.get_lines_from_sync_file(h5_file, line_names)
    olf_disp_pin = utils2p.synchronization.correct_split_edges(olf_disp_pin)
    scope_pin = utils2p.synchronization.correct_split_edges(scope_pin)

    return pid, olf_disp_pin, scope_pin


def pid_lines_to_npz(flacq):
    """
    Loads and saves the lines relevant to odor delivery from thorsync (.h5) recording to a compressed .npz file in
    natural_mixtures/processed_data.

    Args:
        flacq (dict): dictionary containing flat linked acquisition info
                      loaded from natural_mixtures/manifestos/flat_linked_thor_acquisitions.json

    Returns:
        (Path): Path to saved file, ".../pid.npz"
    """
    pid, olf_disp_pin, scope_pin = load_pid_lines(flacq)
    sync_times = utils2p.synchronization.get_times(len(pid), 30e3)

    movie_dir = flacq2dir(flacq)
    np.savez_compressed(movie_dir.joinpath('pid.npz'),
                        sync_times=sync_times,
                        pid=pid,
                        olf_disp_pin=olf_disp_pin,
                        scope_pin=scope_pin)
    return movie_dir.joinpath('pid.npz')


def extract_pid_trials(flacq):
    """make netcdf xr.DataArrays for pid traces (trial x time)"""
    h5_file, sync_meta_file, meta_file = flacq2rawfiles(flacq)
    meta = utils2p.Metadata(meta_file)
    sync_meta = utils2p.synchronization.SyncMetadata(sync_meta_file)

    line_names = ['olfDispPin', 'scopePin', 'pid']
    olf_disp_pin, scope_pin, pid = utils2p.synchronization.get_lines_from_sync_file(h5_file, line_names)
    olf_disp_pin = utils2p.synchronization.correct_split_edges(olf_disp_pin)
    scope_pin = utils2p.synchronization.correct_split_edges(scope_pin)

    sync_times = utils2p.synchronization.get_times(len(pid), sync_meta.get_freq())

    movie_dir = flacq2dir(flacq)

    timestamps = np.load(movie_dir.joinpath('timestamps.npy'), allow_pickle=True).item()

    with open(movie_dir.joinpath('stim_list.json'), 'r') as f:
        stim_list = json.load(f)

    trial_ts = np.arange(-10, 20.0001, 0.001)
    pid_trial_tensor = trial_tensors.make_trial_tensor(pid, sync_times,
                                                       timestamps['olf_ict'],
                                                       trial_ts=trial_ts)
    xrda_pid = xr.DataArray(pid_trial_tensor,
                            coords={'time': trial_ts,
                                    'trial': range(len(timestamps['olf_ict'])),
                                    'stim': ('trial', stim_list['stim_list_flatstr'])
                                    },
                            dims=['trial', 'time'])

    pid_baseline = xrda_pid.loc[:, -4.5:-0.5].mean(axis=1)
    xrda_pid = xrda_pid - pid_baseline
    return xrda_pid


def main(flacq):
    print(f'\nProcessing PID traces')

    MOV_DIR = flacq2dir(flacq)
    print(f'\t- MOV_DIR: {MOV_DIR}')

    pid_file = pid_lines_to_npz(flacq)
    print(f'\t- PID data saved to {pid_file})')

    return pid_file

    # da_pid = extract_pid_trials(flacq)
    # print(f'\t- PID traces extracted successfully.')


# %%

if __name__ == "__main__":
    # load flat linked acquisition list
    with open(NAS_PRJ_DIR.joinpath('manifestos', 'flat_linked_thor_acquisitions.json'), 'r') as f:
        flat_lacq_list = json.load(f)

    filter_by_type = False

    if filter_by_type:
        for flat_acq in filter(lambda x: 'kiwi_components_again' in x['thorimage'], flat_lacq_list):
            print(flat_acq)
            print(pid_lines_to_npz(flat_acq))

    else:
        # for flat_acq in filter(lambda x: 'movie_type' in x.keys(), flat_lacq_list):
        #     print(pid_lines_to_npz(flat_acq))
        for flat_acq in flat_lacq_list:
            print(pid_lines_to_npz(flat_acq))

# %%
# from matplotlib.backends.backend_pdf import PdfPages
#
# pdf = PdfPages('/local/storage/Remy/natural_mixtures/reports/pid.pdf')
#
# pid_netcdf_list = list(NAS_PROC_DIR.rglob('xrda_pid.nc'))
#
# for file in pid_netcdf_list:
#     with xr.open_dataarray(file) as da_pid:
#         fg = plot_pid_traces(da_pid)
#         fg.fig.suptitle(file.relative_to(NAS_PROC_DIR))
#         plt.show()
#         pdf.savefig(fg.fig)
# pdf.close()
