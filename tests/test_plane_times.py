"""Tests for rye2p.thorsync.timestamps (per-plane timestamp extraction)."""
from pathlib import Path

import numpy as np
import pytest

# Import the module directly so the tests don't pull in the whole rye2p package
# (rye2p/__init__.py imports utils2p, h5py, etc., which aren't test dependencies).
import importlib.util

_MODULE_PATH = Path(__file__).parent.parent / "rye2p" / "thorsync" / "timestamps.py"
_spec = importlib.util.spec_from_file_location("ts_module", _MODULE_PATH)
ts_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ts_module)

plane_times = ts_module.plane_times
plane_times_from_settings = ts_module.plane_times_from_settings
load_timestamps = ts_module.load_timestamps

EXAMPLE_TIMESTAMPS = Path(__file__).parent.parent / "data" / "haoru" / "timestamps.npy"


def test_plane_times_no_flyback():
    # 3 volumes x 4 planes, evenly spaced
    ft = np.arange(12, dtype=float) * 0.1
    pt = plane_times(ft, n_z=4)
    assert pt.shape == (3, 4)
    np.testing.assert_allclose(pt[:, 0], ft[::4])  # first-plane == stack_times
    np.testing.assert_allclose(pt[1], [0.4, 0.5, 0.6, 0.7])


def test_plane_times_drops_flyback():
    # 2 volumes x (3 z + 2 flyback) = 10 frame times
    ft = np.arange(10, dtype=float)
    pt = plane_times(ft, n_z=3, n_flyback_frames=2)
    assert pt.shape == (2, 3)
    np.testing.assert_allclose(pt[0], [0, 1, 2])  # flyback times 3, 4 dropped
    np.testing.assert_allclose(pt[1], [5, 6, 7])  # flyback times 8, 9 dropped


def test_plane_times_rejects_partial_volume():
    ft = np.arange(11, dtype=float)  # not divisible by 4
    with pytest.raises(ValueError, match="not divisible"):
        plane_times(ft, n_z=4)


def test_plane_times_rejects_2d_input():
    with pytest.raises(ValueError, match="1-D"):
        plane_times(np.zeros((3, 4)), n_z=4)


class FakeSettings:
    def __init__(self, fast_z, n_z=1, n_flyback_frames=0, average_mode=False, n_averaging=1):
        self.fast_z = fast_z
        self.n_z = n_z
        self.n_flyback_frames = n_flyback_frames
        self.average_mode = average_mode
        self.n_averaging = n_averaging


def test_plane_times_from_settings_fastz():
    ft = np.arange(10, dtype=float)
    settings = FakeSettings(fast_z=True, n_z=4, n_flyback_frames=1)
    pt = plane_times_from_settings(ft, settings)
    assert pt.shape == (2, 4)
    np.testing.assert_allclose(pt[0], [0, 1, 2, 3])


def test_plane_times_from_settings_single_plane_averaging():
    # single-plane, 4-frame averaging: time of first raw frame per block
    ft = np.arange(8, dtype=float)
    settings = FakeSettings(fast_z=False, average_mode=True, n_averaging=4)
    pt = plane_times_from_settings(ft, settings)
    assert pt.shape == (2, 1)
    np.testing.assert_allclose(pt[:, 0], [0.0, 4.0])


@pytest.mark.skipif(not EXAMPLE_TIMESTAMPS.exists(), reason="example data not present")
def test_real_example_first_plane_equals_stack_times():
    ts = load_timestamps(EXAMPLE_TIMESTAMPS)
    ft = np.asarray(ts["frame_times"])
    st = np.asarray(ts["stack_times"])
    steps = ft.size // st.size
    assert ft.size % st.size == 0
    pt = plane_times(ft, n_z=steps)  # treat all steps as kept planes
    assert pt.shape == (st.size, steps)
    np.testing.assert_allclose(pt[:, 0], st)
