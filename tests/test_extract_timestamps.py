"""Regression tests for the unified extraction (rye2p/thorsync/timestamps.py
extract_timestamps + validate_pairing) on small synthetic ThorSync traces.

Each test pins one of the roadmap step-2 timing fixes:
#1 (steps_per_frame from settings / no bogus averaging decimation),
#2 (first-plane stack times), #3 (digital scope pin), #22 (frame 0 and
volume 0 kept), #23 (full-rate edge times), #4 (strict count validation),
plus the pairing fingerprints.

Modules are loaded by file path so the tests don't pull in the whole rye2p
package (rye2p/__init__.py imports utils2p, h5py, etc.).
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

_PKG = Path(__file__).parent.parent / "rye2p" / "thorsync"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# extract_timestamps does `from .lines import pulse_times`; register a tiny
# package so the relative import resolves without importing rye2p itself.
import types
_pkg_module = types.ModuleType("ts_pkg")
_pkg_module.__path__ = [str(_PKG)]
sys.modules.setdefault("ts_pkg", _pkg_module)
_spec_lines = importlib.util.spec_from_file_location("ts_pkg.lines", _PKG / "lines.py")
_lines = importlib.util.module_from_spec(_spec_lines)
sys.modules["ts_pkg.lines"] = _lines
_spec_lines.loader.exec_module(_lines)
_spec_ts = importlib.util.spec_from_file_location("ts_pkg.timestamps", _PKG / "timestamps.py")
ts = importlib.util.module_from_spec(_spec_ts)
sys.modules["ts_pkg.timestamps"] = ts
_spec_ts.loader.exec_module(ts)

counter_increment_times = ts.counter_increment_times
extract_timestamps = ts.extract_timestamps
validate_pairing = ts.validate_pairing

FS = 1000.0


class FakeSettings:
    def __init__(self, steps_per_frame, n_frames, n_timepoints, fast_z=True,
                 raw_frame_rate=100.0):
        self.steps_per_frame = steps_per_frame
        self.n_frames = n_frames
        self.n_timepoints = n_timepoints
        self.fast_z = fast_z
        self.raw_frame_rate = raw_frame_rate


def ttl(n_samples, spans, high=5.0):
    line = np.zeros(n_samples)
    for start, stop in spans:
        line[start:stop] = high
    return line


def counter(n_samples, increment_samples):
    impulses = np.zeros(n_samples, dtype=np.int64)
    for s in increment_samples:
        impulses[s] += 1
    return np.cumsum(impulses).astype(np.uint32)


# ------------------------------------------------- counter_increment_times

def test_counter_increment_times_keeps_frame_zero():
    """Known issue #22: the first counter step (frame 0) must get a time."""
    fc = counter(100, [10, 20, 30])
    times = counter_increment_times(fc, FS)
    np.testing.assert_allclose(times, [10 / FS, 20 / FS, 30 / FS])


def test_counter_increment_times_uint_dtype_safe():
    fc = counter(50, [5, 15]).astype(np.uint32)
    np.testing.assert_allclose(counter_increment_times(fc, FS),
                               [5 / FS, 15 / FS])


# -------------------------------------------------------- fast-z extraction

def _fastz_recording():
    """One scope pulse (samples 100-600), frames every 10 samples from 105.

    10 in-window frames, steps_per_frame=4 -> 2 whole volumes (8 frames kept,
    2 excess dropped).
    """
    n = 1000
    scope = ttl(n, [(100, 600)])
    frame_samples = [105 + 10 * k for k in range(10)]
    fc = counter(n, frame_samples)
    olf = ttl(n, [(300, 400)])
    settings = FakeSettings(steps_per_frame=4, n_frames=8, n_timepoints=2)
    return fc, scope, olf, settings, frame_samples


def test_fastz_keeps_first_frame_and_volume():
    """Known issue #22: frame 0 / volume 0 must be stamped (the legacy path
    returned the first plane of volume i+1, one volume period late)."""
    fc, scope, olf, settings, frame_samples = _fastz_recording()
    result = extract_timestamps(fc, scope, olf, FS, settings)
    assert result["frame_times"][0] == pytest.approx(105 / FS)
    assert result["stack_times"][0] == pytest.approx(105 / FS)


def test_fastz_first_plane_stack_times():
    """Known issue #2: stack_times = first plane of each volume."""
    fc, scope, olf, settings, frame_samples = _fastz_recording()
    result = extract_timestamps(fc, scope, olf, FS, settings)
    np.testing.assert_allclose(result["stack_times"], [105 / FS, 145 / FS])
    np.testing.assert_allclose(result["stack_times"],
                               result["frame_times"][::4])


def test_fastz_partial_volume_dropped():
    fc, scope, olf, settings, frame_samples = _fastz_recording()
    result = extract_timestamps(fc, scope, olf, FS, settings)
    assert result["frame_times"].size == 8
    # the 2 excess frames (samples 185, 195) are not in frame_times
    assert 185 / FS not in result["frame_times"]
    assert 195 / FS not in result["frame_times"]
    assert result["n_frame_times_per_pulse"] == [8]
    assert result["n_stack_times_per_pulse"] == [2]


def test_fastz_multiple_pulses_truncate_per_pulse():
    n = 1200
    scope = ttl(n, [(100, 250), (700, 900)])
    pulse_a = [105 + 10 * k for k in range(10)]   # 10 frames -> 2 volumes
    pulse_b = [705 + 10 * k for k in range(12)]   # 12 frames -> 3 volumes
    fc = counter(n, pulse_a + pulse_b)
    olf = ttl(n, [(150, 180)])
    settings = FakeSettings(steps_per_frame=4, n_frames=20, n_timepoints=5)
    result = extract_timestamps(fc, scope, olf, FS, settings)
    np.testing.assert_allclose(
            result["stack_times"],
            np.array([105, 145, 705, 745, 785]) / FS)
    assert result["n_stack_times_per_pulse"] == [2, 3]
    assert result["n_frame_times_per_pulse"] == [8, 12]


def test_digital_scope_pin():
    """Known issue #3: a digital 0/1 scopePin must work (the legacy 2.5 V
    threshold produced an all-False mask and crashed crop_lines)."""
    fc, scope, olf, settings, frame_samples = _fastz_recording()
    digital_scope = (scope > 0).astype(np.uint32)
    result = extract_timestamps(fc, digital_scope, olf, FS, settings)
    np.testing.assert_allclose(result["stack_times"], [105 / FS, 145 / FS])


# -------------------------------------------------- single-plane extraction

def test_single_plane_averaging_first_of_block():
    """Known issue #1: single-plane + averaging decimates exactly once, by
    n_averaging, first-of-block, with the partial tail block kept."""
    n = 1000
    scope = ttl(n, [(100, 200)])
    frame_samples = [105 + 10 * k for k in range(9)]  # 9 raw frames in window
    fc = counter(n, frame_samples)
    olf = ttl(n, [(300, 400)])
    # xml records n_frames == n_timepoints == saved (averaged) frames
    settings = FakeSettings(steps_per_frame=4, n_frames=3, n_timepoints=3,
                            fast_z=False)
    result = extract_timestamps(fc, scope, olf, FS, settings)
    # all raw in-window frames kept in frame_times
    assert result["frame_times"].size == 9
    # stack_times = first raw frame of each averaging block (ceil(9/4) = 3)
    np.testing.assert_allclose(result["stack_times"],
                               np.array([105, 145, 185]) / FS)


def test_single_plane_no_averaging_no_decimation():
    """Known issue #1 (averageMode facet): with averaging off,
    steps_per_frame == 1 and stack_times must equal frame_times."""
    n = 1000
    scope = ttl(n, [(100, 200)])
    frame_samples = [105 + 10 * k for k in range(9)]
    fc = counter(n, frame_samples)
    olf = ttl(n, [(300, 400)])
    settings = FakeSettings(steps_per_frame=1, n_frames=9, n_timepoints=9,
                            fast_z=False)
    result = extract_timestamps(fc, scope, olf, FS, settings)
    np.testing.assert_allclose(result["stack_times"], result["frame_times"])
    assert result["stack_times"].size == 9


# -------------------------------------------------------- edges / olf times

def test_olf_times_full_rate():
    """Known issue #23: edge times on the full sample clock (sample indices
    here are not multiples of 30, so a 1 kHz path could not represent them)."""
    fs = 30000.0
    n = 30000
    scope = ttl(n, [(1000, 25000)])
    frame_samples = [1010 + 300 * k for k in range(8)]
    fc = counter(n, frame_samples)
    olf = ttl(n, [(5407, 8613)])
    settings = FakeSettings(steps_per_frame=4, n_frames=8, n_timepoints=2)
    result = extract_timestamps(fc, scope, olf, fs, settings)
    np.testing.assert_allclose(result["olf_ict"], [5407 / fs])
    np.testing.assert_allclose(result["olf_fct"], [8613 / fs])


def test_olf_times_outside_scope_kept_in_raw_extraction():
    fc, scope, _, settings, _ = _fastz_recording()
    olf = ttl(1000, [(700, 800)])  # after the scope pulse
    result = extract_timestamps(fc, scope, olf, FS, settings)
    np.testing.assert_allclose(result["olf_ict"], [700 / FS])


# ------------------------------------------------------------- validation

def test_extract_raises_on_count_mismatch_by_default():
    """Known issue #4: a wrong extraction must raise, not save silently."""
    fc, scope, olf, settings, _ = _fastz_recording()
    bad = FakeSettings(steps_per_frame=4, n_frames=12, n_timepoints=3)
    with pytest.raises(ValueError, match="frame-count mismatch"):
        extract_timestamps(fc, scope, olf, FS, bad)


def test_extract_warns_when_not_strict():
    fc, scope, olf, settings, _ = _fastz_recording()
    bad = FakeSettings(steps_per_frame=4, n_frames=12, n_timepoints=3)
    with pytest.warns(UserWarning, match="frame-count mismatch"):
        result = extract_timestamps(fc, scope, olf, FS, bad, strict=False)
    assert result["frame_times"].size == 8  # still extracted for inspection


# ---------------------------------------------------------------- pairing

def test_validate_pairing_ok():
    fc, scope, olf, settings, _ = _fastz_recording()
    # frames every 10 samples at 1 kHz -> realized raw rate 100 Hz;
    # counter max 10 vs expected 2 volumes * 4 = 8 -> extra 2, within slack
    result = validate_pairing(fc, FS, settings)
    assert result["ok"]
    assert result["counter_max"] == 10
    assert result["expected_raw_frames"] == 8
    assert result["realized_frame_rate"] == pytest.approx(100.0)


def test_validate_pairing_detects_wrong_episode_by_counter():
    fc, scope, olf, settings, _ = _fastz_recording()
    wrong = FakeSettings(steps_per_frame=4, n_frames=400, n_timepoints=100)
    with pytest.raises(ValueError, match="pairing fingerprint"):
        validate_pairing(fc, FS, wrong)


def test_validate_pairing_detects_wrong_episode_by_rate():
    fc, scope, olf, settings, _ = _fastz_recording()
    # ~1.5 % rate mismatch, like the real mispairing found on 2023-04-11/2
    wrong = FakeSettings(steps_per_frame=4, n_frames=8, n_timepoints=2,
                         raw_frame_rate=98.5)
    with pytest.raises(ValueError, match="pairing fingerprint"):
        validate_pairing(fc, FS, wrong)
    with pytest.warns(UserWarning, match="pairing fingerprint"):
        result = validate_pairing(fc, FS, wrong, strict=False)
    assert not result["ok"] and result["ok_counter"] and not result["ok_rate"]
