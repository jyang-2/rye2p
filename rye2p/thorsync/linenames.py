import h5py
from fuzzywuzzy import fuzz
import re
import numpy as np
from scipy.optimize import linear_sum_assignment

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

    Use this to find the best match, accounting for different cases, spellings, and spacings.

    Args:
        target_line_names (list): list of expected snake case ThorSync line names
        observed_line_names (list): list of ThorSync line names
        ratio_type (Literal['simple', 'token', 'partial']): fuzzy string matching method

    Returns:
        tuple: tuple containing:

            matched_thorsync_lines (dict): {target: observed} for all target and observed line names
            match_ratio (np.array): string matching ratio
    """

    sc_observed_line_names = [snake_case(item) for item in observed_line_names]
    sc_target_line_names = [snake_case(item) for item in target_line_names]

    scc_observed_line_names = [item.replace('_', ' ') for item in sc_observed_line_names]
    scc_target_line_names = [item.replace('_', ' ') for item in sc_target_line_names]

    print(scc_observed_line_names)
    print(scc_target_line_names)

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

    # target_idx, obs_idx = linear_sum_assignment(match_ratio, maximize=True)
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
    thorsync_groups = ['AI', 'CI', 'DI', 'Global']

    line_names = []
    with h5py.File(h5_file, 'r') as f:
        for grp in f.keys():
            for item in f[grp]:
                line_names.append(item)
    line_names.remove('GCtr')
    return line_names


def clean_line_name(line_name):
    return line_name.lower().replace(' ', '').replace('_', '')


if __name__ == '__main__':
    pass
