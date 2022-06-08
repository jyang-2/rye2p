import numpy as np


def split_movie(stack, frames_per_split):
    """
    Split movie into sub-movies, w/ # of frames per split specified.
      - `frames_per_split` should sum to the # of timepoints in the movie.

    Args:
        stack (np.ndarray): movie to be split, w/ dimensions (time, z, y, x) or (time, y, x)
        frames_per_split (Union[List, Tuple, np.ndarray]): # of timepoints per split

    Returns:
        split_stacks (List[np.ndarray]): List of sub-movies
    """
    n_frames = stack.shape[0]

    if np.sum(frames_per_split) != n_frames:
        raise ValueError('frames_per_split does not sum to the number of timepoints in stack.')

    split_stacks = np.split(stack, np.cumsum(frames_per_split), axis=0)[:-1]
    return split_stacks


def downsample_time(stack, dsub, agg='sum', allow_nondivisor=True):
    """
    Downsamples movie in time by factor `dsub`, using either np.mean or np.sum.

    Args:

        stack (np.ndarray): movie to be split, w/ dimensions (time, z, y, x) or (time, y, x)
        agg (Literal['sum', 'mean']): which agg. function to use in
        dsub (int): temporal downsampling factor
        allow_nondivisor (bool): whether to allow movie downsampling by a nondivisor.
                                 if True, trailing frames (n_frames % dsub) are dropped.

    Returns:
        stack_ds (np.ndarray): downsampled movie
        kept_frames (np.ndarray): 1d vector mapping each frame in `stack` to the corresponding frame in `stack_ds`.
                                    - trailing frames that were dropped have value -1
    """

    n_frames = stack.shape[0]
    dims = stack.shape[1:]

    if not allow_nondivisor and n_frames % dsub != 0:
        raise ValueError('dsub does not divide # of movie frames evenly.')

    n_ds_frames = n_frames // dsub  # number of frames in downsampled movie
    n_frames_incl = n_ds_frames * dsub  # of frames in original movie that are used in the downsampled movie

    if agg == 'sum':
        stack_ds = stack[:n_frames_incl].reshape((-1, dsub, *dims)).sum(axis=1).astype('uint16')
    elif agg == 'mean':
        stack_ds = stack[:n_frames_incl].reshape((-1, dsub, *dims)).mean(axis=1)

    kept_frames = np.arange(n_frames) // dsub
    kept_frames[kept_frames >= n_ds_frames] = -1

    return stack_ds, kept_frames
