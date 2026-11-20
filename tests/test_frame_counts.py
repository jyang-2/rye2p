"""Tests for the frame-count validator in rye2p/thorsync/timestamps.py
(frames_by_scope_pulse, expected_frame_counts, validate_frame_counts).

Import the module directly so the tests don't pull in the whole rye2p package
(rye2p/__init__.py imports utils2p, h5py, etc., which aren't test dependencies).
"""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

_MODULE_PATH = Path(__file__).parent.parent / "rye2p" / "thorsync" / "timestamps.py"
_spec = importlib.util.spec_from_file_location("ts_module_fc", _MODULE_PATH)
ts_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ts_module)

frames_by_scope_pulse = ts_module.frames_by_scope_pulse
expected_frame_counts = ts_module.expected_frame_counts
validate_frame_counts = ts_module.validate_frame_counts

EXAMPLE_TIMESTAMPS = Path(__file__).parent.parent / "data" / "haoru" / "timestamps.npy"


class FakeSettings:
    def __init__(self, steps_per_frame, n_frames, n_timepoints):
        self.steps_per_frame = steps_per_frame
        self.n_frames = n_frames
        self.n_timepoints = n_timepoints


def test_frames_by_scope_pulse_grouping():
    frame_times = np.arange(10, dtype=float)  # 0..9 s
    groups = frames_by_scope_pulse(frame_times, [0.0, 6.5], [3.5, 8.5])
    assert len(groups) == 2
    np.testing.assert_allclose(groups[0], [0.0, 1.0, 2.0, 3.0])
    np.testing.assert_allclose(groups[1], [7.0, 8.0])


def test_frames_by_scope_pulse_length_mismatch_raises():
    with pytest.raises(ValueError, match="not the same length"):
        frames_by_scope_pulse(np.arange(5.0), [0.0, 1.0], [2.0])


def test_expected_frame_counts_floors_partial_volumes():
    # 10 frames in one pulse, 4 steps per timepoint -> 2 whole timepoints
    frame_times = np.arange(10, dtype=float)
    n_frames, n_timepoints = expected_frame_counts(
            frame_times, [0.0], [9.5], steps_per_frame=4)
    assert n_timepoints == 2
    assert n_frames == 8


def test_expected_frame_counts_multiple_pulses():
    # pulse 1: 8 frames (2 timepoints), pulse 2: 5 frames (1 timepoint)
    frame_times = np.concatenate([np.arange(8, dtype=float),
                                  20 + np.arange(5, dtype=float)])
    n_frames, n_timepoints = expected_frame_counts(
            frame_times, [0.0, 20.0], [7.5, 24.5], steps_per_frame=4)
    assert n_timepoints == 3
    assert n_frames == 12


def test_validate_frame_counts_ok():
    frame_times = np.arange(20, dtype=float)
    settings = FakeSettings(steps_per_frame=4, n_frames=20, n_timepoints=5)
    result = validate_frame_counts(frame_times, [0.0], [19.5], settings)
    assert result["ok"]
    assert result["correct_n_frames"] and result["correct_n_timepoints"]
    assert result["expected_n_frames"] == 20
    assert result["expected_n_timepoints"] == 5


def test_validate_frame_counts_accepts_settings_dict():
    frame_times = np.arange(20, dtype=float)
    settings = {"steps_per_frame": 4, "n_frames": 20, "n_timepoints": 5}
    assert validate_frame_counts(frame_times, [0.0], [19.5], settings)["ok"]


def test_validate_frame_counts_warns_on_mismatch():
    frame_times = np.arange(19, dtype=float)  # one frame short of 5 volumes
    settings = FakeSettings(steps_per_frame=4, n_frames=20, n_timepoints=5)
    with pytest.warns(UserWarning, match="frame-count mismatch"):
        result = validate_frame_counts(frame_times, [0.0], [19.5], settings)
    assert not result["ok"]
    assert result["expected_n_timepoints"] == 4


def test_validate_frame_counts_strict_raises():
    frame_times = np.arange(19, dtype=float)
    settings = FakeSettings(steps_per_frame=4, n_frames=20, n_timepoints=5)
    with pytest.raises(ValueError, match="frame-count mismatch"):
        validate_frame_counts(frame_times, [0.0], [19.5], settings, strict=True)


@pytest.mark.skipif(not EXAMPLE_TIMESTAMPS.exists(), reason="example data not present")
def test_example_data_validates():
    """The committed example (fast-z, steps_per_frame=4) is internally
    consistent: the counts implied by its scope pulses match its own arrays."""
    ts = np.load(EXAMPLE_TIMESTAMPS, allow_pickle=True).item()
    settings = FakeSettings(steps_per_frame=4,
                            n_frames=ts["frame_times"].size,
                            n_timepoints=ts["stack_times"].size)
    result = validate_frame_counts(ts["frame_times"], ts["scope_ict"],
                                   ts["scope_fct"], settings)
    assert result["ok"]
    assert result["expected_n_timepoints"] == 1241
    assert result["expected_n_frames"] == 4964
