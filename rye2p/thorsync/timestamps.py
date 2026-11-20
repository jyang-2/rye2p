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

import warnings
from pathlib import Path
from typing import List, Union

import numpy as np

__all__ = [
    "load_timestamps",
    "plane_times",
    "plane_times_from_settings",
    "frames_by_scope_pulse",
    "expected_frame_counts",
    "validate_frame_counts",
]


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


def frames_by_scope_pulse(
    frame_times: np.ndarray,
    scope_ict: np.ndarray,
    scope_fct: np.ndarray,
) -> List[np.ndarray]:
    """Group frame times by the scope-trigger pulse they fall in.

    Same semantics as ``fix_frame_times.frames_by_scope_pulse`` (that copy
    can't be imported here without pulling in utils2p/scipy; it retires in
    roadmap step 3): a frame belongs to pulse ``i`` when
    ``scope_ict[i] <= t < scope_fct[i]``.

    Args:
        frame_times: 1-D array of frame times (seconds).
        scope_ict: scope pulse start times (rising edges).
        scope_fct: scope pulse end times (falling edges), same length.

    Returns:
        one array of frame times per pulse, in pulse order.

    Raises:
        ValueError: if ``scope_ict`` and ``scope_fct`` differ in length.
    """
    frame_times = np.asarray(frame_times)
    scope_ict = np.asarray(scope_ict)
    scope_fct = np.asarray(scope_fct)

    if len(scope_ict) != len(scope_fct):
        raise ValueError('scope_ict and scope_fct are not the same length.')

    return [frame_times[(frame_times >= ict) & (frame_times < fct)]
            for ict, fct in zip(scope_ict, scope_fct)]


def expected_frame_counts(
    frame_times: np.ndarray,
    scope_ict: np.ndarray,
    scope_fct: np.ndarray,
    steps_per_frame: int,
) -> "tuple[int, int]":
    """Frame/timepoint counts implied by the scope pulses.

    Per pulse, only whole timepoints count (``n_frames_in_pulse //
    steps_per_frame`` — a partial volume or averaging block at the end of a
    pulse is discarded, matching what the pipeline saves).

    Args:
        frame_times: 1-D array of frame times (seconds), **including** flyback
            planes / all raw frames of averaging blocks (i.e. the raw
            ``frame_times``, not ``stack_times``).
        scope_ict: scope pulse start times.
        scope_fct: scope pulse end times.
        steps_per_frame: raw frames per saved timepoint
            (``FunctionalImagingSettings.steps_per_frame``: ``n_z +
            n_flyback_frames`` for fast-z, ``n_averaging`` for single-plane).

    Returns:
        ``(expected_n_frames, expected_n_timepoints)``: raw-frame and saved
        timepoint counts implied by the pulses.
    """
    per_pulse = frames_by_scope_pulse(frame_times, scope_ict, scope_fct)
    timepoints_per_pulse = [frames.size // steps_per_frame for frames in per_pulse]
    expected_n_timepoints = int(np.sum(timepoints_per_pulse))
    expected_n_frames = expected_n_timepoints * int(steps_per_frame)
    return expected_n_frames, expected_n_timepoints


def _get_setting(settings, name):
    """Read a settings field from a FunctionalImagingSettings or a plain dict."""
    if isinstance(settings, dict):
        return settings[name]
    return getattr(settings, name)


def validate_frame_counts(
    frame_times: np.ndarray,
    scope_ict: np.ndarray,
    scope_fct: np.ndarray,
    settings,
    *,
    strict: bool = False,
    scope_padding: float = 0.0,
) -> dict:
    """Check extracted frame/timepoint counts against ``Experiment.xml``.

    Promoted from the notebook validator (cell 8 of
    ``convert_thorimage_metadata_settings.ipynb``), which — like the legacy
    pipeline check — only printed (known issue #4). Generalized via
    ``steps_per_frame`` so single-plane/averaging recordings are covered too
    (the notebook cell only ran for fast-z).

    Args:
        frame_times: raw per-frame times (flyback planes / all raw frames of
            averaging blocks included).
        scope_ict: scope pulse start times.
        scope_fct: scope pulse end times.
        settings: ``FunctionalImagingSettings`` or its ``to_dict()`` form —
            needs ``steps_per_frame``, ``n_frames``, ``n_timepoints``.
        strict: if True, raise ``ValueError`` on a mismatch instead of
            warning. Default False preserves today's warn-only behavior;
            **roadmap step 2 flips the default to True.**
        scope_padding: seconds added to every ``scope_fct`` before grouping
            (the notebook's knob; 0.0 there and here).

    Returns:
        dict with ``expected_n_frames``, ``expected_n_timepoints``,
        ``n_frames``, ``n_timepoints`` (the ``Experiment.xml`` values),
        ``correct_n_frames``, ``correct_n_timepoints``, and ``ok``.

    Raises:
        ValueError: on count mismatch, when ``strict=True``.
    """
    steps_per_frame = _get_setting(settings, "steps_per_frame")
    n_frames = _get_setting(settings, "n_frames")
    n_timepoints = _get_setting(settings, "n_timepoints")

    expected_n_frames, expected_n_timepoints = expected_frame_counts(
            frame_times, scope_ict, np.asarray(scope_fct) + scope_padding,
            steps_per_frame)

    result = {
        "expected_n_frames": expected_n_frames,
        "expected_n_timepoints": expected_n_timepoints,
        "n_frames": n_frames,
        "n_timepoints": n_timepoints,
        "correct_n_frames": expected_n_frames == n_frames,
        "correct_n_timepoints": expected_n_timepoints == n_timepoints,
    }
    result["ok"] = result["correct_n_frames"] and result["correct_n_timepoints"]

    if not result["ok"]:
        message = (
            f"frame-count mismatch vs Experiment.xml: expected from scope "
            f"pulses n_frames={expected_n_frames}, "
            f"n_timepoints={expected_n_timepoints}; ThorImage says "
            f"n_frames={n_frames}, n_timepoints={n_timepoints} "
            f"(steps_per_frame={steps_per_frame})"
        )
        if strict:
            raise ValueError(message)
        warnings.warn(message)

    return result
