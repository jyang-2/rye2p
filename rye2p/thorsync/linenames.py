"""Map canonical ThorSync line names to the dataset names in an Episode001.h5.

The per-rig YAML maps shipped as package data
(``rye2p/data/microscopes/<rig>/thorsync_lines.yaml``) are the primary source
(:func:`load_line_map`); Hungarian fuzzy matching
(:func:`get_matched_thorsync_lines`) is only the fallback for lines the map
doesn't resolve — use :func:`match_lines` for the combined behavior.

Heavy imports (``h5py``, ``fuzzywuzzy``, ``scipy``) are function-local so this
module stays loadable under the numpy-only test environment.
"""

import re
import warnings
from pathlib import Path

import numpy as np
import yaml

# Path-relative rather than importlib.resources: tests load this module by
# file path, and resources.files("rye2p.data") would import rye2p/__init__.py
# (which pulls in utils2p, h5py, ...). The YAMLs live inside the package, so
# this resolves both in-repo and installed.
_MICROSCOPES_DIR = Path(__file__).resolve().parent.parent / "data" / "microscopes"

default_line_names = [
    'piezo_monitor',
    'pockels_monitor',
    'lightpath_shutter',
    'olf_disp_pin',
    'pid',
    'scope_pin',
    'frame_counter',
    'frame_in',
    'frame_out']


def load_line_map(microscope):
    """Load the per-rig line-name map shipped as package data.

    Args:
        microscope (str): rig name, e.g. ``'upstairs-2p'`` or
            ``'downstairs-2p'`` (a folder under ``rye2p/data/microscopes/``).

    Returns:
        dict: ``{canonical_name: observed ThorSync dataset name}`` — keys are
        the names in ``default_line_names``.

    Raises:
        FileNotFoundError: if no map exists for ``microscope`` (the message
            lists the available rigs).
    """
    map_file = _MICROSCOPES_DIR / microscope / "thorsync_lines.yaml"
    if not map_file.is_file():
        available = sorted(p.name for p in _MICROSCOPES_DIR.iterdir()
                           if p.is_dir())
        raise FileNotFoundError(
                f"no line map for microscope {microscope!r} "
                f"(available: {available})")
    with open(map_file, 'r') as f:
        return yaml.safe_load(f)


def match_lines(observed_line_names, microscope=None, ratio_type='simple'):
    """Map canonical line names to observed ThorSync dataset names.

    The per-rig YAML map is primary: every map entry whose observed name is
    actually present in ``observed_line_names`` is used as-is. Fuzzy matching
    (:func:`get_matched_thorsync_lines`) runs only for canonical names the map
    didn't resolve (or for all of them when ``microscope`` is None), with a
    warning naming the lines that fell back.

    Args:
        observed_line_names (List[str]): dataset names found in the .h5 file
            (e.g. from :func:`get_thorsync_line_names`).
        microscope (str): rig name for :func:`load_line_map`; None skips the
            YAML map entirely (pure fuzzy matching).
        ratio_type (str): fuzzy ratio passed to
            :func:`get_matched_thorsync_lines`.

    Returns:
        dict: ``{canonical_name: observed_name}`` for all of
        ``default_line_names``.
    """
    observed_line_names = list(observed_line_names)

    matched = {}
    if microscope is not None:
        line_map = load_line_map(microscope)
        matched = {canonical: observed
                   for canonical, observed in line_map.items()
                   if observed in observed_line_names}

    missing = [name for name in default_line_names if name not in matched]
    if missing:
        warnings.warn(
                f"line map {'(none given)' if microscope is None else repr(microscope)} "
                f"did not resolve {missing}; falling back to fuzzy matching")
        unclaimed = [name for name in observed_line_names
                     if name not in matched.values()]
        fuzzy_matched, _ = get_matched_thorsync_lines(
                missing, unclaimed, ratio_type=ratio_type)
        matched.update(fuzzy_matched)

    return matched


def snake_case(s):
    """ Convert to snake case

    Args:
        s (str): variable name like "linkedThorAcquisitions"

    Returns:
        sc (str): variable name in snake_case (linked_thor_acquisition)

    Examples:

        >>>     snake_case("linkedThorAcquisitions")
        Out[83]: 'linked_thor_acquisitions'

    """
    sc = '_'.join(
            re.sub('([A-Z][a-z]+)', r' \1',
                   re.sub('([A-Z]+)', r' \1',
                          s.replace('-', ' '))).split()).lower()
    return sc


def get_matched_thorsync_lines(target_line_names, observed_line_names, ratio_type='simple'):
    """Match expected ThorSync line names to the closest one in an .h5 ThorSync file.

    Fuzzy fallback — prefer the per-rig YAML maps (:func:`match_lines`).
    Every target gets *some* match (no minimum-ratio threshold), so an absent
    line is silently mis-assigned; the per-rig maps exist to avoid exactly
    that.

    Args:
        target_line_names (list): list of expected snake case ThorSync line names
        observed_line_names (list): list of ThorSync line names
        ratio_type (Literal['simple', 'token', 'partial']): fuzzy string matching method

    Returns:
        tuple: tuple containing:

            matched_thorsync_lines (dict): {target: observed} for all target and observed line names
            match_ratio (np.array): string matching ratio
    """
    from fuzzywuzzy import fuzz
    from scipy.optimize import linear_sum_assignment

    sc_observed_line_names = [snake_case(item) for item in observed_line_names]
    sc_target_line_names = [snake_case(item) for item in target_line_names]

    scc_observed_line_names = [item.replace('_', ' ') for item in sc_observed_line_names]
    scc_target_line_names = [item.replace('_', ' ') for item in sc_target_line_names]

    if ratio_type == 'simple':
        match_ratio = np.array([[fuzz.ratio(x, y)
                                 for x in scc_target_line_names]
                                for y in scc_observed_line_names])
    elif ratio_type == 'token':
        match_ratio = np.array([[fuzz.token_sort_ratio(x, y)
                                 for x in scc_target_line_names]
                                for y in scc_observed_line_names])
    elif ratio_type == 'partial':
        match_ratio = np.array([[fuzz.partial_ratio(x, y)
                                 for x in scc_target_line_names]
                                for y in scc_observed_line_names])
    else:
        raise ValueError('ratio_type must be one of simple, token or partial')

    row_idx, col_idx = linear_sum_assignment(match_ratio, maximize=True)

    matched_target_line_names = np.array(target_line_names)[col_idx]
    matched_observed_line_names = np.array(observed_line_names)[row_idx]

    matched_thorsync_lines = dict(zip(matched_target_line_names.tolist(),
                                      matched_observed_line_names.tolist()))
    return matched_thorsync_lines, match_ratio


def get_thorsync_line_names(h5_file):
    """Get names of all ThorSync lines in h5 file

    Args:
        h5_file (h5py.File): h5 file

    Returns:
        line_names (List[str]): list of names of all ThorSync lines in h5 file

    """
    import h5py

    line_names = []
    with h5py.File(h5_file, 'r') as f:
        for grp in f.keys():
            for item in f[grp]:
                line_names.append(item)
    line_names.remove('GCtr')
    return line_names


def clean_line_name(line_name):
    return line_name.lower().replace(' ', '').replace('_', '')
