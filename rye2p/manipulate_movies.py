from pathlib import Path
from prefect import task, Flow, Parameter, case
from rye2p import movie
import numpy as np
import utils2p


#%%

def downsample_acquisitions(acq_dir, dsub):
    acq_files = sorted(list(acq_dir.rglob("acq_*.tif")))

    kept_stack_times = []

    for file in acq_files:
        print(file)
        stack1 = utils2p.load_img(file, memmap=True)
        stack_ds, kst = movie.downsample_time(stack1, dsub)
        kept_stack_times.append(kst)

        save_file = f"{file.stem}__dsub_{dsub:02d}.tif"
        utils2p.save_img(acq_dir.joinpath(save_file), stack_ds)
        print(f"  - {save_file} saved.")

    np.save(acq_dir.joinpath(f"kept_stack_times__dsub_{dsub:02d}.npy"), kept_stack_times)
    return True



@task(name='load-timestamps')
def load_timestamps_from_folder(folder):
    timestamps = np.load(folder.withname('timestamps.npy'), allow_pickle=True).item()
    return timestamps


@task(name="load-movie")
def load_movie_from_folder(folder):
    stack1 = utils2p.load_img(folder.with_name("Image_001_001.tif"), memmap=True)
    return stack1


@task(name='case-run-split')
def run_downsampling(dsub):
    return dsub > 1


@task(name="split-movie-by-acqs")
def split_movie_by_acqs(stack, frames_per_acq):
    split_movies = movie.split_movie(stack, frames_per_acq)
    return split_movies


# def save_split_movies(movie_list, folder):
#     if not folder.joinpath("acqs").is_dir():
#
#
# for file in sorted(list(acq_dir.rglob("acq_*.tif"))):
#     utils2p.load_img(file, memmap=True)
#
#
#
# with Flow("manipulate-movie") as flow:
#     do_split = Parameter("do_split", default=True)
#     dsub = Parameter("dsub", default=3)
#     movie_dir = Parameter("folder")
#
#     timestamps = load_timestamps_from_folder(movie_dir)
#     movie = load_movie_from_folder(movie_dir)
#
#     # divide movies
#     split_movies = split_movie_by_acqs(movie, timestamps['n_stack_times_per_pulse'])
#
#     cond = run_downsampling(dsub)


