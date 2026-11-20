"""Tests for rye2p/thorsync/lines.py (normalize-first, full-rate edge detection).

Import the module directly so the tests don't pull in the whole rye2p package
(rye2p/__init__.py imports utils2p, h5py, etc., which aren't test dependencies).
"""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

_MODULE_PATH = Path(__file__).parent.parent / "rye2p" / "thorsync" / "lines.py"
_spec = importlib.util.spec_from_file_location("lines_module", _MODULE_PATH)
lines_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(lines_module)

normalize_line = lines_module.normalize_line
edge_times = lines_module.edge_times
merge_brief_gaps = lines_module.merge_brief_gaps
match_pulse_edges = lines_module.match_pulse_edges
pulse_times = lines_module.pulse_times


# ---------------------------------------------------------------- normalize

def test_normalize_analog_ttl():
    line = np.array([0.0, 0.1, 4.9, 4.9, 0.1])
    out = normalize_line(line)
    np.testing.assert_allclose(out.max(), 1.0)
    np.testing.assert_allclose(out, line / 4.9)


def test_normalize_digital_passthrough():
    line = np.array([0, 1, 1, 0], dtype=np.uint8)
    np.testing.assert_array_equal(normalize_line(line), [0.0, 1.0, 1.0, 0.0])


def test_normalize_silent_line_returns_zeros():
    # a never-fired pin with a few mV of ADC noise must not produce edges
    line = np.array([0.001, -0.002, 0.004, 0.0])
    np.testing.assert_array_equal(normalize_line(line), np.zeros(4))
    rising, falling = edge_times(line, fs=1000.0)
    assert rising.size == 0 and falling.size == 0


# ---------------------------------------------------------------- edge_times

def test_edge_times_matches_utils2p_convention():
    # utils2p.synchronization.edges([0,1,1,0,1,1]) -> indices [1, 4]; with
    # get_times (sample k at k/fs) the legacy path stamps t = 1/fs and 4/fs.
    line = np.array([0, 1, 1, 0, 1, 1])
    fs = 20.0
    rising, falling = edge_times(line, fs)
    np.testing.assert_allclose(rising, [1 / fs, 4 / fs])
    np.testing.assert_allclose(falling, [3 / fs])


def test_edge_times_digital_0_1_line():
    """Known issue #3: a digital 0/1 scopePin must produce edges (the legacy
    2.5 V threshold was all-False on it)."""
    line = np.array([0, 0, 1, 1, 1, 0, 0], dtype=np.uint32)
    fs = 1000.0
    rising, falling = edge_times(line, fs)
    np.testing.assert_allclose(rising, [2 / fs])
    np.testing.assert_allclose(falling, [5 / fs])


def test_edge_times_analog_5v_line_same_result():
    digital = np.array([0, 0, 1, 1, 1, 0, 0], dtype=float)
    analog = digital * 4.9
    fs = 1000.0
    for got, want in zip(edge_times(analog, fs), edge_times(digital, fs)):
        np.testing.assert_allclose(got, want)


def test_edge_times_full_rate_not_quantized():
    """Known issue #23: edges must be stamped on the full sample clock, not a
    1 kHz grid. Put an edge at a sample index that is not a multiple of 30."""
    fs = 30000.0
    line = np.zeros(1000)
    line[407:613] = 5.0
    rising, falling = edge_times(line, fs)
    np.testing.assert_allclose(rising, [407 / fs])
    np.testing.assert_allclose(falling, [613 / fs])


def test_edge_times_split_edge_single_crossing():
    # a rise spread over several ticks crosses the threshold exactly once
    line = np.array([0.0, 1.0, 2.0, 3.0, 5.0, 5.0, 3.0, 1.0, 0.0])
    fs = 10.0
    rising, falling = edge_times(line, fs)
    assert rising.size == 1 and falling.size == 1


# ------------------------------------------------------- gap merge / matching

def test_merge_brief_gaps():
    ict = np.array([1.0, 2.0005, 5.0])
    fct = np.array([2.0, 3.0, 6.0])
    merged_ict, merged_fct = merge_brief_gaps(ict, fct, max_gap=0.001)
    np.testing.assert_allclose(merged_ict, [1.0, 5.0])
    np.testing.assert_allclose(merged_fct, [3.0, 6.0])


def test_merge_brief_gaps_keeps_real_gaps():
    ict = np.array([1.0, 3.0])
    fct = np.array([2.0, 4.0])
    merged_ict, merged_fct = merge_brief_gaps(ict, fct, max_gap=0.001)
    np.testing.assert_allclose(merged_ict, ict)
    np.testing.assert_allclose(merged_fct, fct)


def test_match_pulse_edges_drops_leading_fall():
    # line high at recording start: a falling edge precedes the first rise
    with pytest.warns(UserWarning, match="unpaired"):
        ict, fct = match_pulse_edges([2.0, 5.0], [1.0, 3.0, 6.0])
    np.testing.assert_allclose(ict, [2.0, 5.0])
    np.testing.assert_allclose(fct, [3.0, 6.0])


def test_match_pulse_edges_drops_trailing_rise():
    with pytest.warns(UserWarning, match="unpaired"):
        ict, fct = match_pulse_edges([1.0, 4.0], [2.0])
    np.testing.assert_allclose(ict, [1.0])
    np.testing.assert_allclose(fct, [2.0])


def test_pulse_times_end_to_end():
    fs = 1000.0
    line = np.zeros(100)
    line[10:20] = 5.0
    line[50:60] = 5.0
    ict, fct = pulse_times(line, fs)
    np.testing.assert_allclose(ict, [10 / fs, 50 / fs])
    np.testing.assert_allclose(fct, [20 / fs, 60 / fs])


def test_pulse_times_merges_glitch():
    fs = 10000.0
    line = np.zeros(1000)
    line[100:200] = 5.0
    line[203:300] = 5.0  # 0.3 ms glitch gap inside one pulse
    ict, fct = pulse_times(line, fs, merge_gap=0.001)
    np.testing.assert_allclose(ict, [100 / fs])
    np.testing.assert_allclose(fct, [300 / fs])
