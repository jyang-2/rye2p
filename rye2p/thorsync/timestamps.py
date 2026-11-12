"""Per-plane and per-volume timestamp utilities for volumetric (fast-z) recordings.

Contract of the ``timestamps.npy`` dict written by the pipeline (all values are
1-D float64 arrays of seconds since ThorSync acquisition start):

- ``frame_times``: one entry per *frame-counter step*. For fast-z recordings this
  means one timestamp per z-plane, **including flyback planes**, so its length is
  ``n_timepoints * (n_z + n_flyback_frames)``.
- ``stack_times``: one entry per volume (``n_timepoints``), the time of the
  **first plane** of each volume, i.e. ``frame_times[::steps_per_frame]``.
- ``scope_ict`` / ``scope_fct``: rising/falling edge times of the microscope
  acquisition trigger (scope pin).
- ``olf_ict`` / ``olf_fct``: rising/falling edge times of the olfactometer
  dispense pin (stimulus on/off).

The saved movie (``Image_001_001.tif``) does NOT contain flyback planes —
``utils2p.load_raw`` trims them — so ``frame_times`` cannot index the tiff
directly. Use :func:`plane_times` to get a ``(n_timepoints, n_z)`` array whose
``[t, z]`` entry is the acquisition time of plane ``z`` in volume ``t`` of the
saved movie. ``plane_times(...)[:, 0]`` equals ``stack_times``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np

__all__ = ["load_timestamps", "plane_times", "plane_times_from_settings"]


def load_timestamps(timestamps_file: Union[str, Path]) -> dict:
    """Load a ``timestamps.npy`` / ``timestamps0.npy`` file written by the pipeline.

    These files are pickled dicts saved with ``np.save``, so they require
    ``allow_pickle=True`` and ``.item()`` to unwrap — this helper does both.

    Args:
        timestamps_file: path to ``timestamps.npy`` (or ``timestamps0.npy``).

    Returns:
        dict with keys ``frame_times``, ``stack_times``, ``scope_ict``,
        ``scope_fct``, ``olf_ict``, ``olf_fct`` (and, if the file went through
        ``fix_frame_times.fix_timestamps0``, the extra ``split_*`` /
        ``n_*_per_pulse`` keys).
    """
    return np.load(Path(timestamps_file), allow_pickle=True).item()


def plane_times(
    frame_times: np.ndarray,
    n_z: int,
    n_flyback_frames: int = 0,
) -> np.ndarray:
    """Reshape per-plane ``frame_times`` into a ``(n_timepoints, n_z)`` array.

    ``frame_times`` holds one timestamp per frame-counter step — every z-plane of
    every volume, flyback planes included. The saved movie drops flyback planes,
    so this function reshapes to ``(n_timepoints, n_z + n_flyback_frames)`` and
    discards the trailing flyback columns, leaving one timestamp per plane
    actually present in the tiff.

    Args:
        frame_times: 1-D array of per-plane acquisition times (seconds), length
            ``n_timepoints * (n_z + n_flyback_frames)``. Must already be repaired
            (whole volumes only) — e.g. the ``frame_times`` entry of a final
            ``timestamps.npy``, not a raw ``timestamps0.npy`` with partial volumes.
        n_z: number of z-planes per volume in the saved movie
            (``FunctionalImagingSettings.n_z``).
        n_flyback_frames: number of flyback planes per volume that were acquired
            but are not in the saved movie (``FunctionalImagingSettings.n_flyback_frames``).

    Returns:
        ``(n_timepoints, n_z)`` float array: ``[t, z]`` is the time plane ``z`` of
        volume ``t`` was acquired. Column 0 equals ``stack_times`` (first-plane
        convention).

    Raises:
        ValueError: if ``len(frame_times)`` is not a whole number of volumes,
            which usually means the timestamps were not repaired (partial volume
            at the end of a scope pulse) or ``n_z``/``n_flyback_frames`` don't
            match the recording.
    """
    frame_times = np.asarray(frame_times)
    if frame_times.ndim != 1:
        raise ValueError(f"frame_times must be 1-D, got shape {frame_times.shape}")
    steps_per_frame = n_z + n_flyback_frames
    if steps_per_frame < 1:
        raise ValueError(f"n_z + n_flyback_frames must be >= 1, got {steps_per_frame}")
    n_total = frame_times.size
    if n_total % steps_per_frame != 0:
        raise ValueError(
            f"len(frame_times)={n_total} is not divisible by "
            f"steps_per_frame={steps_per_frame} (n_z={n_z} + "
            f"n_flyback_frames={n_flyback_frames}). Repair the timestamps first "
            f"(see fix_frame_times) or check n_z/n_flyback_frames."
        )
    return frame_times.reshape(-1, steps_per_frame)[:, :n_z]


def plane_times_from_settings(frame_times: np.ndarray, settings) -> np.ndarray:
    """:func:`plane_times` using a ``FunctionalImagingSettings`` for the geometry.

    For fast-z recordings, uses ``settings.n_z`` and ``settings.n_flyback_frames``.
    For single-plane recordings (``fast_z`` false), each saved frame is the
    average of ``n_averaging`` raw frames; this returns a ``(n_timepoints, 1)``
    array holding the time of the first raw frame of each averaging block
    (consistent with how ``stack_times`` is computed).

    Args:
        frame_times: 1-D array of per-plane acquisition times (seconds).
        settings (FunctionalImagingSettings): parsed acquisition settings (or any
            object with ``fast_z``, ``n_z``, ``n_flyback_frames``, ``n_averaging``,
            ``average_mode`` attributes).

    Returns:
        ``(n_timepoints, n_z)`` array (``n_z == 1`` for single-plane recordings).
    """
    if settings.fast_z:
        return plane_times(frame_times, settings.n_z, settings.n_flyback_frames)
    n_averaging = settings.n_averaging if settings.average_mode else 1
    return plane_times(frame_times, 1, n_averaging - 1)
