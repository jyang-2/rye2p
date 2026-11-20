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

Since roadmap step 2 this module is also the extraction's home:
:func:`extract_timestamps` (pure numpy) / :func:`extract_timestamps_from_files`
produce the dict above directly from the ThorSync lines, with the counts
validated against ``Experiment.xml`` (:func:`validate_frame_counts`) and the
ThorSync↔ThorImage pairing fingerprint-checked (:func:`validate_pairing`).
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
    "counter_increment_times",
    "extract_timestamps",
    "extract_timestamps_from_files",
    "validate_pairing",
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
    (consistent with how ``stack_times`` is computed). A partial trailing
    block is allowed — ThorImage saves a block that *starts* inside the scope
    pulse even when it completes after the trigger fall (ground-truthed
    2026-11-19), so ``len(frame_times)`` need not be divisible by
    ``n_averaging``.

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
    frame_times = np.asarray(frame_times)
    if frame_times.ndim != 1:
        raise ValueError(f"frame_times must be 1-D, got shape {frame_times.shape}")
    return frame_times[::n_averaging, np.newaxis]


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
    fast_z: bool = True,
) -> "tuple[int, int]":
    """Frame/timepoint counts implied by the scope pulses.

    Encodes ThorImage's save rules, ground-truthed 2026-11-19 on the
    2023-04-11/2 recordings:

    - **fast-z** (``fast_z=True``): a partial volume at the trigger fall is
      discarded — per pulse ``n_frames_in_pulse // steps_per_frame`` whole
      volumes (megamat1: the counter reached 90,752 but the movie holds
      90,740 = 4,537 × 20 frames).
    - **single-plane** (``fast_z=False``): an averaging block that *starts*
      inside the pulse is completed after the trigger fall and saved — per
      pulse ``ceil(n_frames_in_pulse / steps_per_frame)`` saved frames
      (megamat1_calyx: 92,125 in-window raw frames → 23,032 = ceil(92,125/4)
      saved averaged frames).

    ``expected_n_frames`` predicts the ``Streaming/frames`` value in
    ``Experiment.xml``, which counts frames **in the .raw file**: raw planes
    (flyback included) for fast-z, but *averaged* frames for single-plane —
    there it equals ``expected_n_timepoints``.

    Args:
        frame_times: 1-D array of frame times (seconds), **including** flyback
            planes / all raw frames of averaging blocks (i.e. the raw
            ``frame_times``, not ``stack_times``).
        scope_ict: scope pulse start times.
        scope_fct: scope pulse end times.
        steps_per_frame: raw frames per saved timepoint
            (``FunctionalImagingSettings.steps_per_frame``: ``n_z +
            n_flyback_frames`` for fast-z, ``n_averaging`` for single-plane).
        fast_z: whether the recording is volumetric
            (``FunctionalImagingSettings.fast_z``).

    Returns:
        ``(expected_n_frames, expected_n_timepoints)``: the .raw-file frame
        count and saved timepoint count implied by the pulses.
    """
    per_pulse = frames_by_scope_pulse(frame_times, scope_ict, scope_fct)
    steps_per_frame = int(steps_per_frame)
    if fast_z:
        timepoints_per_pulse = [frames.size // steps_per_frame
                                for frames in per_pulse]
        expected_n_timepoints = int(np.sum(timepoints_per_pulse))
        expected_n_frames = expected_n_timepoints * steps_per_frame
    else:
        timepoints_per_pulse = [-(-frames.size // steps_per_frame)
                                for frames in per_pulse]
        expected_n_timepoints = int(np.sum(timepoints_per_pulse))
        expected_n_frames = expected_n_timepoints
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
    strict: bool = True,
    scope_padding: float = 0.0,
) -> dict:
    """Check extracted frame/timepoint counts against ``Experiment.xml``.

    Promoted from the notebook validator (cell 8 of
    ``convert_thorimage_metadata_settings.ipynb``), which — like the legacy
    pipeline check — only printed (known issue #4). Generalized via
    ``steps_per_frame``/``fast_z`` so single-plane/averaging recordings are
    covered too (the notebook cell only ran for fast-z); see
    :func:`expected_frame_counts` for the per-mode save rules.

    Args:
        frame_times: raw per-frame times (flyback planes / all raw frames of
            averaging blocks included).
        scope_ict: scope pulse start times.
        scope_fct: scope pulse end times.
        settings: ``FunctionalImagingSettings`` or its ``to_dict()`` form —
            needs ``steps_per_frame``, ``fast_z``, ``n_frames``,
            ``n_timepoints``.
        strict: if True (default — flipped in roadmap step 2, closing known
            issue #4), raise ``ValueError`` on a mismatch instead of warning.
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
    fast_z = _get_setting(settings, "fast_z")
    n_frames = _get_setting(settings, "n_frames")
    n_timepoints = _get_setting(settings, "n_timepoints")

    expected_n_frames, expected_n_timepoints = expected_frame_counts(
            frame_times, scope_ict, np.asarray(scope_fct) + scope_padding,
            steps_per_frame, fast_z=fast_z)

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


def counter_increment_times(frame_counter: np.ndarray, fs: float) -> np.ndarray:
    """Times of every frame-counter increment — frame 0 included.

    Works on the **raw** counter line, so the very first frame (whose
    increment is the counter's first step off zero) gets a timestamp. This is
    the fix for known issue #22: the legacy path processed and *cropped* the
    counter first, leaving frame/volume 0 already at value 0 with no rising
    edge, which silently dropped the first frame and first volume (every
    legacy ``stack_times[i]`` was the first plane of volume ``i+1``, one
    volume period late).

    Args:
        frame_counter: raw ``FrameCounter`` line (cumulative frame count per
            sample; any integer dtype).
        fs: ThorSync sample rate in Hz.

    Returns:
        1-D float64 array: for each counter increment, the time of the first
        sample at the new count (``index / fs``, seconds since ThorSync
        start) — the same edge convention as :mod:`rye2p.thorsync.lines`.
    """
    frame_counter = np.asarray(frame_counter).astype(np.int64)
    increment_idx = np.flatnonzero(np.diff(frame_counter) > 0) + 1
    return increment_idx / fs


def extract_timestamps(
    frame_counter: np.ndarray,
    scope_pin: np.ndarray,
    olf_disp_pin: np.ndarray,
    fs: float,
    settings,
    *,
    strict: bool = True,
    scope_threshold: float = 0.5,
    olf_threshold: float = 0.5,
    olf_merge_gap: float = 0.001,
) -> dict:
    """Extract all ``timestamps.npy`` arrays from raw ThorSync lines.

    The unified extraction (roadmap step 2), ported from the ThorSync GUI
    after it was verified end-to-end against ground truth on all three
    2023-04-11/2 recording types (fast-z, 4x-averaged, single-plane raw).
    Compared to the legacy ``process_thorsync.extract_timestamp_data`` it
    fixes, in one place:

    - **#22**: frame times come from raw counter increments
      (:func:`counter_increment_times`), so frame 0 and volume 0 are kept.
    - **#1**: ``steps_per_frame`` is read from ``settings``
      (``FunctionalImagingSettings`` guards both the single-plane ``fast_z``
      case and ``averageMode`` off) — no double decimation, no averaging
      applied when averaging is disabled.
    - **#2**: ``stack_times`` uses the **first-plane** convention
      (``kept_frames[::steps_per_frame]`` per pulse).
    - **#3 / #23**: scope/olf edges are found on normalized lines at the full
      sample rate (:mod:`rye2p.thorsync.lines`), so digital 0/1 pins work and
      nothing is 1 ms-quantized.
    - **#4**: counts are validated against ``Experiment.xml`` and **raise**
      by default (``strict=True``) — a mismatched file can't be saved
      silently.

    Per scope pulse, in-window frames are truncated to whole volumes for
    fast-z; for single-plane recordings all in-window frames are kept and a
    partial trailing averaging block still yields a saved timepoint (ThorImage
    completes it after the trigger fall — see :func:`expected_frame_counts`).

    Args:
        frame_counter: raw ``FrameCounter`` line.
        scope_pin: raw scope-trigger line (analog or digital).
        olf_disp_pin: raw olfactometer dispense line.
        fs: ThorSync sample rate in Hz.
        settings: ``FunctionalImagingSettings`` or its ``to_dict()`` form.
        strict: raise on count mismatch (default). ``False`` downgrades to a
            warning, for intentionally inspecting a broken recording.
        scope_threshold: threshold on the normalized scope line.
        olf_threshold: threshold on the normalized olf line.
        olf_merge_gap: olf pulses separated by a gap shorter than this
            (seconds) are merged (ADC chatter on slow edges).

    Returns:
        dict with the ``timestamps.npy`` contract keys — ``frame_times``,
        ``stack_times``, ``scope_ict``, ``scope_fct``, ``olf_ict``,
        ``olf_fct`` — plus the per-pulse keys ``split_frame_times``,
        ``split_stack_times``, ``n_frame_times_per_pulse``,
        ``n_stack_times_per_pulse``. ``olf_ict``/``olf_fct`` contain **all**
        olf pulses, including any outside scope pulses (filter with
        ``fix_frame_times.fix_olf_times`` if needed).

    Raises:
        ValueError: on a frame-count mismatch, when ``strict=True``.
    """
    from .lines import pulse_times

    steps_per_frame = int(_get_setting(settings, "steps_per_frame"))
    fast_z = bool(_get_setting(settings, "fast_z"))

    scope_ict, scope_fct = pulse_times(scope_pin, fs, threshold=scope_threshold)
    olf_ict, olf_fct = pulse_times(olf_disp_pin, fs, threshold=olf_threshold,
                                   merge_gap=olf_merge_gap)

    all_frame_times = counter_increment_times(frame_counter, fs)
    validate_frame_counts(all_frame_times, scope_ict, scope_fct, settings,
                          strict=strict)

    split_frame_times = []
    split_stack_times = []
    for frames in frames_by_scope_pulse(all_frame_times, scope_ict, scope_fct):
        if fast_z:
            kept = frames[:(frames.size // steps_per_frame) * steps_per_frame]
        else:
            kept = frames
        split_frame_times.append(kept)
        split_stack_times.append(kept[::steps_per_frame])

    def _concat(arrays):
        return np.concatenate(arrays) if arrays else np.array([], dtype=np.float64)

    return dict(
            frame_times=_concat(split_frame_times),
            stack_times=_concat(split_stack_times),
            scope_ict=scope_ict,
            scope_fct=scope_fct,
            olf_ict=olf_ict,
            olf_fct=olf_fct,
            split_frame_times=split_frame_times,
            split_stack_times=split_stack_times,
            n_frame_times_per_pulse=[item.size for item in split_frame_times],
            n_stack_times_per_pulse=[item.size for item in split_stack_times],
    )


def validate_pairing(
    frame_counter: np.ndarray,
    fs: float,
    settings,
    *,
    strict: bool = True,
    rate_rtol: float = 0.005,
    counter_slack: int = 8,
) -> dict:
    """Fingerprint check that a ThorSync episode belongs to an ``Experiment.xml``.

    The pairing is recorded in **neither** file, so a wrong guess silently
    yields timestamps from a different movie (found in the wild: 2023-04-11/2
    ``megamat1`` pairs with ``SyncData001``, not SyncData003). Two cheap
    fingerprints catch it (see docs/timestamps.md "Validation"):

    - **counter max**: the final ``FrameCounter`` value vs. the raw frames
      implied by the xml (``n_timepoints × steps_per_frame``). The counter may
      legitimately overrun by a discarded partial volume plus a few frames of
      scanner tail (ground truth: +12 of 90,740 fast-z; +2 single-plane), so
      the window is ``[0, steps_per_frame + counter_slack]``.
    - **realized frame rate**: ``1 / median(diff(increment times))`` vs. the
      xml ``frameRate``. Correct pairings agree to ~0.1 %; the observed wrong
      pairing was ~1.5 % off. Default ``rate_rtol=0.005`` separates cleanly.

    Args:
        frame_counter: raw ``FrameCounter`` line.
        fs: ThorSync sample rate in Hz.
        settings: ``FunctionalImagingSettings`` or its ``to_dict()`` form —
            needs ``steps_per_frame``, ``n_timepoints``, ``raw_frame_rate``.
        strict: raise ``ValueError`` on a failed fingerprint (default);
            ``False`` warns instead.
        rate_rtol: relative tolerance on the realized frame rate.
        counter_slack: extra frames (beyond one partial volume/block) the
            counter may overrun the expectation by.

    Returns:
        dict with ``counter_max``, ``expected_raw_frames``, ``extra_frames``,
        ``realized_frame_rate``, ``raw_frame_rate``, ``ok_counter``,
        ``ok_rate``, and ``ok``.

    Raises:
        ValueError: on a failed fingerprint, when ``strict=True``.
    """
    steps_per_frame = int(_get_setting(settings, "steps_per_frame"))
    n_timepoints = int(_get_setting(settings, "n_timepoints"))
    raw_frame_rate = float(_get_setting(settings, "raw_frame_rate"))

    frame_counter = np.asarray(frame_counter).astype(np.int64)
    counter_max = int(frame_counter.max()) if frame_counter.size else 0
    increment_times = counter_increment_times(frame_counter, fs)
    if increment_times.size >= 2:
        realized_frame_rate = 1.0 / float(np.median(np.diff(increment_times)))
    else:
        realized_frame_rate = float("nan")

    expected_raw_frames = n_timepoints * steps_per_frame
    extra_frames = counter_max - expected_raw_frames

    result = {
        "counter_max": counter_max,
        "expected_raw_frames": expected_raw_frames,
        "extra_frames": extra_frames,
        "realized_frame_rate": realized_frame_rate,
        "raw_frame_rate": raw_frame_rate,
        "ok_counter": 0 <= extra_frames <= steps_per_frame + counter_slack,
        "ok_rate": bool(np.isfinite(realized_frame_rate)
                        and abs(realized_frame_rate - raw_frame_rate)
                        <= rate_rtol * raw_frame_rate),
    }
    result["ok"] = result["ok_counter"] and result["ok_rate"]

    if not result["ok"]:
        message = (
            f"ThorSync/ThorImage pairing fingerprint mismatch — this episode "
            f"likely belongs to a different acquisition: counter max "
            f"{counter_max} vs expected raw frames {expected_raw_frames} "
            f"(extra {extra_frames}, allowed 0..{steps_per_frame + counter_slack}); "
            f"realized frame rate {realized_frame_rate:.3f} Hz vs "
            f"Experiment.xml {raw_frame_rate:.3f} Hz (rtol {rate_rtol})"
        )
        if strict:
            raise ValueError(message)
        warnings.warn(message)

    return result


def extract_timestamps_from_files(
    h5_file: "Union[str, Path]",
    experiment_xml: "Union[str, Path]",
    *,
    microscope: "Union[str, None]" = None,
    strict: bool = True,
    check_pairing: bool = True,
    scope_threshold: float = 0.5,
    olf_threshold: float = 0.5,
    olf_merge_gap: float = 0.001,
) -> dict:
    """:func:`extract_timestamps` straight from the acquisition files.

    Loads the three needed lines through the per-rig YAML maps
    (:func:`rye2p.thorsync.lines.load_sync_lines`), reads the sample rate from
    the ``ThorRealTimeDataSettings.xml`` next to the h5, parses
    ``FunctionalImagingSettings`` from ``experiment_xml``, and — by default —
    checks the pairing fingerprints (:func:`validate_pairing`) before
    extracting.

    Args:
        h5_file: path to ``Episode001.h5``.
        experiment_xml: path to the paired ThorImage ``Experiment.xml``.
        microscope: rig name for the line map (e.g. ``'upstairs-2p'``); None
            falls back to fuzzy line matching with a warning.
        strict: raise on pairing/count mismatches (default) instead of warning.
        check_pairing: run :func:`validate_pairing` first (default).
        scope_threshold: see :func:`extract_timestamps`.
        olf_threshold: see :func:`extract_timestamps`.
        olf_merge_gap: see :func:`extract_timestamps`.

    Returns:
        the :func:`extract_timestamps` dict.

    Raises:
        ValueError: failed pairing fingerprint or count mismatch (``strict``).
        KeyError: a needed line is missing from the h5.
    """
    from rye2p.thorimage.settings import FunctionalImagingSettings

    from .lines import load_sync_lines, sample_rate_from_xml

    h5_file = Path(h5_file)
    settings = FunctionalImagingSettings.from_xml(Path(experiment_xml))

    sync_settings_xml = h5_file.with_name("ThorRealTimeDataSettings.xml")
    if sync_settings_xml.is_file():
        fs = sample_rate_from_xml(sync_settings_xml)
    else:
        warnings.warn(f"no ThorRealTimeDataSettings.xml next to {h5_file}; "
                      f"assuming 30000 Hz")
        fs = 30000.0

    lines = load_sync_lines(
            h5_file,
            line_names=["frame_counter", "scope_pin", "olf_disp_pin"],
            microscope=microscope)

    if check_pairing:
        validate_pairing(lines["frame_counter"], fs, settings, strict=strict)

    return extract_timestamps(
            lines["frame_counter"], lines["scope_pin"], lines["olf_disp_pin"],
            fs, settings, strict=strict, scope_threshold=scope_threshold,
            olf_threshold=olf_threshold, olf_merge_gap=olf_merge_gap)
