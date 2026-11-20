"""Tests for rye2p/pathguard.py (refusal to write into raw-data locations).

Import the module directly so the tests don't pull in the whole rye2p package
(rye2p/__init__.py imports utils2p, h5py, etc., which aren't test
dependencies). pathguard.py is stdlib-only.
"""
import importlib.util
from pathlib import Path

import pytest

_MODULE_PATH = Path(__file__).parent.parent / "rye2p" / "pathguard.py"
_spec = importlib.util.spec_from_file_location("pathguard_module", _MODULE_PATH)
pathguard = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pathguard)

guard_write_dir = pathguard.guard_write_dir
is_raw_data_path = pathguard.is_raw_data_path
RawDataWriteError = pathguard.RawDataWriteError


def test_refuses_tensor_archive():
    with pytest.raises(RawDataWriteError, match="tensor-archive"):
        guard_write_dir("/local/tensor-archive/Remy/thorlabs/some/folder")


def test_refuses_raw_data_marker(tmp_path):
    dest = tmp_path / "raw_data" / "2026-01-13" / "1" / "movie"
    with pytest.raises(RawDataWriteError, match="raw_data"):
        guard_write_dir(dest)


@pytest.mark.parametrize("fingerprint", pathguard.RAW_FINGERPRINT_FILES)
def test_refuses_acquisition_dir(tmp_path, fingerprint):
    acq = tmp_path / "some_movie"
    acq.mkdir()
    acq.joinpath(fingerprint).write_bytes(b"\x00")
    with pytest.raises(RawDataWriteError, match="raw acquisition file"):
        guard_write_dir(acq)


def test_refuses_subdir_of_acquisition_dir(tmp_path):
    """Writing into e.g. <raw acquisition>/stk is refused via the parent check."""
    acq = tmp_path / "some_movie"
    acq.mkdir()
    acq.joinpath("Image_001_001.raw").write_bytes(b"\x00")
    with pytest.raises(RawDataWriteError, match="raw acquisition file"):
        guard_write_dir(acq / "stk")


def test_allows_clean_proc_dir(tmp_path):
    dest = tmp_path / "processed_data" / "2026-01-13" / "1" / "movie"
    assert guard_write_dir(dest) == dest  # nonexistent dirs are fine too
    dest.mkdir(parents=True)
    dest.joinpath("Experiment.xml").write_text("<x/>")  # xml copy is not a fingerprint
    assert guard_write_dir(dest) == dest


def test_env_override_downgrades_to_warning(tmp_path, monkeypatch):
    monkeypatch.setenv("RYE2P_ALLOW_RAW_WRITES", "1")
    dest = tmp_path / "raw_data" / "movie"
    with pytest.warns(UserWarning, match="override active"):
        assert guard_write_dir(dest) == dest


def test_is_raw_data_path(tmp_path):
    assert is_raw_data_path("/local/tensor-archive/anything")
    assert is_raw_data_path(tmp_path / "raw_data" / "x")
    assert not is_raw_data_path(tmp_path / "processed_data" / "x")
