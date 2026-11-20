"""Normalize ThorSync lines and find TTL edges at the full sample rate.

New home for edge detection (roadmap step 2). Fixes two catalogued bugs:

- Known issue #3: the legacy extraction thresholded raw voltages at 2.5 V,
  which is all-False on rigs whose ``scopePin`` is a digital 0/1 line. Here
  every line is max-normalized to [0, 1] first (:func:`normalize_line`) and
  thresholded at 0.5 by default, so analog (0-5 V) and digital (0/1) TTLs
  behave identically.
- Known issue #23: the notebook path computed edge times on 1 kHz-decimated
  lines (1 ms quantization). These functions take the full-rate line, so all
  edge times stay on the ThorSync sample clock.

Time convention (identical to the legacy ``utils2p`` path, which the
2026-11-12 ground-truth check reproduced bit-exactly, and to the verified
ThorSync GUI): sample ``k`` of a line is at ``k / fs`` seconds, ``t=0`` is the
first ThorSync sample, and an edge is stamped with the **first sample at the
new level**.

Everything is pure numpy except :func:`load_sync_lines` (h5py) and
:func:`sample_rate_from_xml` (stdlib XML), whose heavy imports are
function-local so this module stays loadable in the numpy-only test
environment.
"""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

__all__ = [
    "normalize_line",
    "edge_times",
    "merge_brief_gaps",
    "match_pulse_edges",
    "pulse_times",
    "sample_rate_from_xml",
    "load_sync_lines",
]


def normalize_line(line: np.ndarray, min_amplitude: float = 0.5) -> np.ndarray:
    """Scale a TTL-ish line to [0, 1] by its maximum.

    Digital 0/1 lines pass through unchanged; analog 0-5 V TTLs scale to
    [0, 1]. A line that never fires (max below ``min_amplitude`` — e.g. a few
    millivolts of ADC noise on an unused pin) returns all zeros instead of
    amplifying its noise past any threshold.

    Args:
        line: 1-D line signal (any float/int dtype).
        min_amplitude: smallest max value treated as a real signal. The
            default 0.5 accepts digital 0/1 lines (max exactly 1) and analog
            TTLs (max ~5 V) while rejecting noise-only lines.

    Returns:
        float64 array in [0, 1] (all zeros if the line never fires).
    """
    line = np.asarray(line, dtype=np.float64)
    peak = np.nanmax(line) if line.size else 0.0
    if not peak >= min_amplitude:
        return np.zeros_like(line)
    return line / peak


def edge_times(
    line: np.ndarray,
    fs: float,
    threshold: float = 0.5,
    normalize: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Rising and falling edge times of a TTL line, at the full sample rate.

    Threshold-crossing detection: an edge is wherever the boolean
    ``line > threshold`` changes. Unlike jump-magnitude detection
    (``utils2p.synchronization.edges(size=...)``) a monotonic rise crosses the
    threshold exactly once, so edges split over several ticks need no
    correction.

    Args:
        line: 1-D line signal.
        fs: sample rate in Hz.
        threshold: level in the **normalized** [0, 1] scale (or in the line's
            own units when ``normalize=False``).
        normalize: max-normalize via :func:`normalize_line` first (default).

    Returns:
        ``(rising_times, falling_times)``: seconds since ThorSync start, each
        stamped with the first sample at the new level (``index / fs``).
    """
    if normalize:
        line = normalize_line(line)
    high = np.asarray(line) > threshold
    step = np.diff(high.astype(np.int8))
    rising = (np.flatnonzero(step == 1) + 1) / fs
    falling = (np.flatnonzero(step == -1) + 1) / fs
    return rising, falling


def merge_brief_gaps(
    ict: np.ndarray,
    fct: np.ndarray,
    max_gap: float = 0.001,
) -> Tuple[np.ndarray, np.ndarray]:
    """Merge pulses separated by a gap shorter than ``max_gap`` seconds.

    A falling edge followed by a rising edge within ``max_gap`` is treated as
    a glitch inside one pulse (e.g. ADC chatter on a slow olfactometer pin)
    and both edges are removed. Same semantics as the verified ThorSync GUI's
    ``correct_split_edges``.

    Args:
        ict: rising edge times (seconds).
        fct: falling edge times (seconds).
        max_gap: gaps up to this long (exclusive bounds) are merged.

    Returns:
        ``(ict, fct)`` with the gap-bounding edges removed.
    """
    ict, fct = list(np.asarray(ict)), list(np.asarray(fct))
    for i in range(len(fct) - 1, -1, -1):
        k = next((j for j, t in enumerate(ict) if fct[i] < t < fct[i] + max_gap),
                 None)
        if k is not None:
            del fct[i]
            del ict[k]
    return np.asarray(ict), np.asarray(fct)


def match_pulse_edges(
    ict: np.ndarray,
    fct: np.ndarray,
    warn: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Pair each rising edge with the next falling edge after it.

    Unpaired edges (a falling edge before the first rise — line high at
    recording start — or a trailing rise with no fall) are dropped, with a
    warning naming how many.

    Args:
        ict: rising edge times.
        fct: falling edge times.
        warn: warn when edges were dropped.

    Returns:
        ``(ict, fct)`` of equal length, with ``ict[i] < fct[i]``.
    """
    ict = np.asarray(ict)
    fct = np.asarray(fct)
    matched_ict, matched_fct = [], []
    j = 0
    for t in ict:
        while j < len(fct) and fct[j] <= t:
            j += 1
        if j < len(fct):
            matched_ict.append(t)
            matched_fct.append(fct[j])
            j += 1
    n_dropped = (len(ict) - len(matched_ict)) + (len(fct) - len(matched_fct))
    if warn and n_dropped:
        warnings.warn(
                f"dropped {n_dropped} unpaired pulse edge(s) "
                f"({len(ict)} rising vs {len(fct)} falling)")
    return np.asarray(matched_ict), np.asarray(matched_fct)


def pulse_times(
    line: np.ndarray,
    fs: float,
    threshold: float = 0.5,
    normalize: bool = True,
    merge_gap: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Matched (rising, falling) pulse times of a TTL line.

    Convenience wrapper: :func:`edge_times` → optional
    :func:`merge_brief_gaps` → :func:`match_pulse_edges`.

    Args:
        line: 1-D line signal.
        fs: sample rate in Hz.
        threshold: see :func:`edge_times`.
        normalize: see :func:`edge_times`.
        merge_gap: if given, merge pulses separated by less than this many
            seconds (use ~0.001 for the olfactometer dispense pin).

    Returns:
        ``(ict, fct)`` of equal length (seconds since ThorSync start).
    """
    ict, fct = edge_times(line, fs, threshold=threshold, normalize=normalize)
    if merge_gap is not None:
        ict, fct = merge_brief_gaps(ict, fct, max_gap=merge_gap)
    return match_pulse_edges(ict, fct)


def sample_rate_from_xml(sync_settings_xml: Union[str, Path]) -> float:
    """ThorSync sample rate from ``ThorRealTimeDataSettings.xml``.

    Reads the enabled ``SampleRate`` of the **active** ``AcquireBoard``
    (mirrors ``utils2p.synchronization.SyncMetadata.get_freq``; ported from
    the verified ThorSync GUI, whose first version read the wrong node and
    got 1 MHz instead of 30 kHz).

    Args:
        sync_settings_xml: path to ``ThorRealTimeDataSettings.xml`` (sits next
            to ``Episode001.h5``).

    Returns:
        sample rate in Hz.

    Raises:
        ValueError: if no enabled SampleRate node exists.
    """
    import xml.etree.ElementTree as ET

    root = ET.parse(Path(sync_settings_xml)).getroot()
    rates = []  # (board_active, rate) for every SampleRate with enable="1"
    for board in root.iter():
        if not board.tag.lower().endswith("acquireboard"):
            continue
        active = board.attrib.get("active", "1") == "1"
        for el in board.iter():
            if el.tag.lower().endswith("samplerate") and el.attrib.get("enable") == "1":
                rate = el.attrib.get("rate") or el.attrib.get("sampleRate")
                if rate is not None:
                    rates.append((active, float(rate)))
    for active, rate in rates:
        if active:
            return rate
    if rates:
        return rates[0][1]
    raise ValueError(f"no enabled SampleRate in {sync_settings_xml}")


def load_sync_lines(
    h5_file: Union[str, Path],
    line_names: Optional[List[str]] = None,
    microscope: Optional[str] = None,
) -> Dict[str, np.ndarray]:
    """Load ThorSync lines from an ``Episode*.h5`` as float64 arrays.

    Canonical names are resolved to the file's dataset names through the
    per-rig YAML map (:func:`rye2p.thorsync.linenames.match_lines`); fuzzy
    matching is the warned fallback. This replaces the legacy per-column
    argmax matcher (known issue #5 — it could silently double-assign).

    Args:
        h5_file: path to ``Episode001.h5``.
        line_names: canonical names to load (default: all of
            ``linenames.default_line_names``). Extraction needs only
            ``['frame_counter', 'scope_pin', 'olf_disp_pin']``.
        microscope: rig name for the YAML map (e.g. ``'upstairs-2p'``);
            None falls back to pure fuzzy matching with a warning.

    Returns:
        ``{canonical_name: 1-D float64 array}``.

    Raises:
        KeyError: if a requested line can't be resolved to a dataset.
    """
    import h5py

    from . import linenames

    wanted = list(line_names) if line_names is not None else list(
            linenames.default_line_names)

    out = {}
    with h5py.File(Path(h5_file), "r") as f:
        paths = {}  # dataset basename -> full h5 path
        def _visit(name, obj):
            if isinstance(obj, h5py.Dataset):
                paths[name.split("/")[-1]] = name
        f.visititems(_visit)
        # GCtr is the 20 MHz board timebase, not a sync line
        observed = [name for name in paths if name != "GCtr"]
        mapping = linenames.match_lines(observed, microscope=microscope,
                                        line_names=wanted)
        for name in wanted:
            observed_name = mapping.get(name)
            if observed_name is None or observed_name not in paths:
                raise KeyError(
                        f"could not resolve line {name!r} in {h5_file} "
                        f"(observed: {sorted(observed)})")
            out[name] = np.asarray(f[paths[observed_name]]).squeeze().astype(
                    np.float64)
    return out
