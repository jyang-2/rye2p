"""Tests for rye2p/thorsync/linenames.py (YAML-primary line-name maps).

Import the module directly so the tests don't pull in the whole rye2p package
(rye2p/__init__.py imports utils2p, h5py, etc., which aren't test
dependencies). linenames.py keeps its heavy imports (h5py, fuzzywuzzy, scipy)
function-local for the same reason — the YAML paths below must run with only
numpy + pyyaml installed.
"""
import importlib.util
from pathlib import Path

import pytest

_MODULE_PATH = Path(__file__).parent.parent / "rye2p" / "thorsync" / "linenames.py"
_spec = importlib.util.spec_from_file_location("linenames_module", _MODULE_PATH)
linenames_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(linenames_module)

default_line_names = linenames_module.default_line_names
load_line_map = linenames_module.load_line_map
match_lines = linenames_module.match_lines

RIGS = ("upstairs-2p", "downstairs-2p")


@pytest.mark.parametrize("rig", RIGS)
def test_load_line_map_keys(rig):
    """Every shipped rig map covers exactly the canonical line names."""
    line_map = load_line_map(rig)
    assert set(line_map.keys()) == set(default_line_names)
    assert all(isinstance(v, str) for v in line_map.values())


def test_load_line_map_unknown_rig():
    with pytest.raises(FileNotFoundError, match="basement-2p"):
        load_line_map("basement-2p")


@pytest.mark.parametrize("rig", RIGS)
def test_match_lines_yaml_primary(rig):
    """With a complete YAML map, fuzzy matching (and its heavy imports) must
    never run — this test passes in an env without fuzzywuzzy/scipy."""
    line_map = load_line_map(rig)
    observed = list(line_map.values()) + ["GCtr_extra", "Freq"]
    matched = match_lines(observed, microscope=rig)
    assert matched == line_map


def test_match_lines_fuzzy_fallback_warns():
    """Names missing from the map fall back to fuzzy matching with a warning."""
    pytest.importorskip("fuzzywuzzy")
    pytest.importorskip("scipy")

    line_map = load_line_map("upstairs-2p")
    # hide one observed line behind a renamed variant the map can't resolve
    observed = [v for k, v in line_map.items() if k != "scope_pin"]
    observed.append("scope_pin_renamed")

    with pytest.warns(UserWarning, match="scope_pin"):
        matched = match_lines(observed, microscope="upstairs-2p")

    assert matched["scope_pin"] == "scope_pin_renamed"
    for canonical, observed_name in line_map.items():
        if canonical != "scope_pin":
            assert matched[canonical] == observed_name
