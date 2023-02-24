"""Matches thorsync and thorimage files based on file creation/modification times.

Uses pathlib.stat() to get file metadata.

Here is the list of members of stat structure −

    st_mode − protection bits.

    st_ino − inode number.

    st_dev − device.

    st_nlink − number of hard links.

    st_uid − user id of owner.

    st_gid − group id of owner.

    st_size − size of file, in bytes.

    st_atime − time of most recent access.

    st_mtime − time of most recent content modification.

    st_ctime − time of most recent metadata change.

"""
from pathlib import Path

import numpy as np
import pandas as pd
import pendulum
from scipy.optimize import linear_sum_assignment
from rye2p import datadirs


def get_file_times(files, as_datetime_str=False):
    st_ctimes = []
    st_mtimes = []
    for file in files:
        ctime = pendulum.from_timestamp(file.stat().st_ctime)
        mtime = pendulum.from_timestamp(file.stat().st_mtime)
        print(type(ctime))

        st_ctimes.append(ctime)
        st_mtimes.append(mtime)
    return st_ctimes, st_mtimes


def get_thorsync_acquisition_times(thorsync_dir):
    sync_file = list(thorsync_dir.glob("Episode*.h5"))[0]
    sync_meta_file = list(thorsync_dir.glob("ThorRealTimeDataSettings.xml"))[0]
    start_time = pendulum.from_timestamp(sync_meta_file.stat().st_mtime)
    end_time = pendulum.from_timestamp(sync_file.stat().st_mtime)

    return start_time, end_time


def get_thorimage_acquisition_times(thorimage_dir):
    raw_file = list(thorimage_dir.glob("Image_*_*.raw"))[0]
    preview_tiff_file = list(thorimage_dir.glob("Chan*_Preview.tif"))[0]

    start_time = pendulum.from_timestamp(preview_tiff_file.stat().st_mtime)
    end_time = pendulum.from_timestamp(raw_file.stat().st_mtime)
    return start_time, end_time


def get_thorimage_time_table(thorimage_folders):
    acq_times = [get_thorimage_acquisition_times(folder) for folder in thorimage_folders]
    start_times, end_times = zip(*acq_times)
    df_recording_times = pd.DataFrame.from_dict(
            dict(thorimage_folder=thorimage_folders,
                 thorimage_name=[item.name for item in thorimage_folders],
                 start_time=start_times, end_time=end_times)
            )
    df_recording_times['duration'] = df_recording_times['end_time'] - df_recording_times['start_time']
    return df_recording_times


def get_thorsync_time_table(thorsync_folders):
    acq_times = [get_thorsync_acquisition_times(folder) for folder in thorsync_folders]
    start_times, end_times = zip(*acq_times)
    df_recording_times = pd.DataFrame.from_dict(
            dict(thorsync_folder=thorsync_folders,
                 thorsync_name=[item.name for item in thorsync_folders],
                 start_time=start_times, end_time=end_times)
            )
    df_recording_times['duration'] = df_recording_times['end_time'] - df_recording_times['start_time']

    return df_recording_times


def get_modification_time(file):
    return pendulum.from_timestamp(file.stat().st_mtime)


def match_thorimage_to_thorsync(thorimage_folders, thorsync_folders):
    df_thorimage = get_thorimage_time_table(thorimage_folders)
    df_thorsync = get_thorsync_time_table(thorsync_folders)

    df_fraction_thorimage_contained, df_overlap_duration, df_total_duration = get_thor_acquisition_overlaps(
            df_thorimage, df_thorsync)

    row_idx, col_idx = linear_sum_assignment(df_fraction_thorimage_contained, maximize=True)
    fraction_thorimage_contained = [df_fraction_thorimage_contained.iat[i, j] for i, j in zip(row_idx, col_idx)]

    df_thorimage_ = df_thorimage.set_index(['thorimage_folder', 'thorimage_name']).add_prefix(
            'thorimage_').reset_index()

    df_thorsync_ = df_thorsync.set_index(['thorsync_folder', 'thorsync_name']).add_prefix(
            'thorsync_').reset_index()

    df_thor_matches = pd.concat([df_thorimage_.iloc[row_idx, :],
                                 df_thorsync_.iloc[col_idx, :]
                                 ], axis=1
                                )
    df_thor_matches['fraction_thorimage_contained'] = fraction_thorimage_contained
    return df_thor_matches


def get_thor_acquisition_overlaps(df_thorimage, df_thorsync):
    n_thorimage = df_thorimage.shape[0]
    n_thorsync = df_thorsync.shape[0]

    df_overlap_duration = pd.DataFrame(np.zeros((n_thorimage, n_thorsync)),
                                       index=df_thorimage['thorimage_name'],
                                       columns=df_thorsync['thorsync_name']
                                       )

    df_total_duration = pd.DataFrame(np.zeros((n_thorimage, n_thorsync)),
                                     index=df_thorimage['thorimage_name'],
                                     columns=df_thorsync['thorsync_name']
                                     )

    df_fraction_thorimage_contained = pd.DataFrame(np.zeros((n_thorimage, n_thorsync)),
                                                   index=df_thorimage['thorimage_name'],
                                                   columns=df_thorsync['thorsync_name']
                                                   )
    for row in df_thorimage.itertuples():
        for col in df_thorsync.itertuples():
            latest_start_time = max(row.start_time, col.start_time)
            earliest_end_time = min(row.end_time, col.end_time)

            earliest_start_time = min(row.start_time, col.start_time)
            latest_end_time = max(row.end_time, col.end_time)

            overlap_duration = earliest_end_time - latest_start_time
            total_duration = latest_end_time - earliest_start_time

            df_overlap_duration.loc[row.thorimage_name, col.thorsync_name] = overlap_duration
            df_total_duration.loc[row.thorimage_name, col.thorsync_name] = total_duration

            df_fraction_thorimage_contained.loc[row.thorimage_name, col.thorsync_name] \
                = overlap_duration / row.duration if row.duration != 0 else np.nan

    df_fraction_thorimage_contained = df_fraction_thorimage_contained.clip(lower=0)

    return df_fraction_thorimage_contained, df_overlap_duration, df_total_duration


def main(raw_data_fly_folder, raw_dir, proc_dir, match_files=True):
    # find all thor files
    raw_thorimage_files = sorted(list(raw_data_fly_folder.rglob("Image_*_*.raw")))
    hdf_thorsync_files = sorted(list(raw_data_fly_folder.rglob("Episode*.h5")))

    thorimage_dirs = [item.parent for item in raw_thorimage_files]
    thorsync_dirs = [item.parent for item in hdf_thorsync_files]

    # make timetables for all thorimage and thorsync dirs
    df_sync_times = get_thorsync_time_table(thorsync_dirs)
    print(df_sync_times.loc[:, ['thorsync_name', 'start_time', 'end_time', 'duration']])

    df_thorimage_times = get_thorimage_time_table(thorimage_dirs)
    print(df_thorimage_times.loc[:, ['thorimage_name', 'start_time', 'end_time', 'duration']])

    proc_fly_dir = proc_dir / raw_data_fly_folder.relative_to(raw_dir)
    print(proc_fly_dir)
    df_sync_times.to_csv(proc_fly_dir.joinpath('df_thorsync_fileinfo.csv'))
    df_thorimage_times.to_csv(proc_fly_dir.joinpath('df_thorimage_fileinfo.csv'))

    # match recordings
    if match_files:
        df_fraction_thorimage_contained, df_overlap_duration, df_total_duration \
            = get_thor_acquisition_overlaps(df_thorimage_times, df_sync_times)

        df_fraction_thorimage_contained.to_csv(proc_fly_dir.joinpath('df_fraction_thorimage_contained.csv'))

    print(proc_fly_dir.as_uri())

    return proc_fly_dir
