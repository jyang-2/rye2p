"""Tests for rye2p/pipeline/preprocess.py (validate_thorimage_folder and the
metadata-only paths of preprocess_thorimage).

Import the module directly so the tests don't pull in the whole rye2p package
(rye2p/__init__.py imports utils2p, h5py, etc., which aren't test
dependencies). preprocess.py keeps its heavy imports function-local for the
same reason, so the metadata-only paths run under the numpy-only CI env.
"""
import importlib.util
from pathlib import Path

import pytest

_MODULE_PATH = Path(__file__).parent.parent / "rye2p" / "pipeline" / "preprocess.py"
_spec = importlib.util.spec_from_file_location("preprocess_module", _MODULE_PATH)
preprocess_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(preprocess_module)

validate_thorimage_folder = preprocess_module.validate_thorimage_folder
preprocess_thorimage = preprocess_module.preprocess_thorimage

XML_CONTENT = "<ThorImageExperiment></ThorImageExperiment>\n"


def _make_acquisition(folder, raw_name="Image_001_001.raw"):
    folder.mkdir(parents=True, exist_ok=True)
    folder.joinpath("Experiment.xml").write_text(XML_CONTENT)
    folder.joinpath(raw_name).write_bytes(b"\x00\x00")
    return folder


def test_validate_returns_raw_path(tmp_path):
    folder = _make_acquisition(tmp_path / "acq")
    assert validate_thorimage_folder(folder) == folder / "Image_001_001.raw"


def test_validate_accepts_old_raw_name(tmp_path):
    # the notebook original validated this name but then crashed loading it
    folder = _make_acquisition(tmp_path / "acq", raw_name="Image_0001_0001.raw")
    assert validate_thorimage_folder(folder) == folder / "Image_0001_0001.raw"


def test_validate_prefers_new_raw_name(tmp_path):
    folder = _make_acquisition(tmp_path / "acq")
    folder.joinpath("Image_0001_0001.raw").write_bytes(b"\x00\x00")
    assert validate_thorimage_folder(folder).name == "Image_001_001.raw"


def test_validate_missing_xml_raises(tmp_path):
    folder = tmp_path / "acq"
    folder.mkdir()
    folder.joinpath("Image_001_001.raw").write_bytes(b"\x00\x00")
    with pytest.raises(FileNotFoundError, match="Experiment.xml"):
        validate_thorimage_folder(folder)


def test_validate_missing_raw_raises(tmp_path):
    folder = tmp_path / "acq"
    folder.mkdir()
    folder.joinpath("Experiment.xml").write_text(XML_CONTENT)
    with pytest.raises(FileNotFoundError, match=".raw movie"):
        validate_thorimage_folder(folder)


def test_preprocess_copies_xml_only(tmp_path):
    """Metadata-only invocation: copies Experiment.xml, writes nothing else."""
    raw_dir = _make_acquisition(tmp_path / "raw" / "acq")
    proc_dir = tmp_path / "proc" / "acq"

    result = preprocess_thorimage(raw_dir, proc_dir,
                                  save_settings=False,
                                  save_single_tiff=False,
                                  save_tiff_sequence=False)

    assert result == proc_dir
    assert (proc_dir / "Experiment.xml").read_text() == XML_CONTENT
    assert sorted(p.name for p in proc_dir.iterdir()) == ["Experiment.xml"]


def test_preprocess_refuses_proc_dest_in_raw_data_tree(tmp_path):
    """A proc destination under a raw_data tree (e.g. mis-set NAS_PROC_DIR)
    is refused before anything is written."""
    raw_dir = _make_acquisition(tmp_path / "raw" / "acq")
    proc_dir = tmp_path / "raw_data" / "acq"

    with pytest.raises(preprocess_module.RawDataWriteError, match="raw_data"):
        preprocess_thorimage(raw_dir, proc_dir)
    assert not proc_dir.exists()


def test_preprocess_refuses_writing_into_acquisition_folder(tmp_path):
    """proc == raw (swapped/duplicated argument) is refused by the
    acquisition-fingerprint check, regardless of flags."""
    raw_dir = _make_acquisition(tmp_path / "raw" / "acq")

    with pytest.raises(preprocess_module.RawDataWriteError,
                       match="raw acquisition file"):
        preprocess_thorimage(raw_dir, raw_dir, copy_xml=False,
                             save_settings=False, save_single_tiff=False,
                             save_tiff_sequence=False)


def test_preprocess_no_copy_xml(tmp_path):
    raw_dir = _make_acquisition(tmp_path / "raw" / "acq")
    proc_dir = tmp_path / "proc" / "acq"

    preprocess_thorimage(raw_dir, proc_dir,
                         copy_xml=False,
                         save_settings=False,
                         save_single_tiff=False,
                         save_tiff_sequence=False)

    assert proc_dir.is_dir()
    assert list(proc_dir.iterdir()) == []
