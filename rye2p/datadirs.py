from pathlib import Path

_MATRIX = Path("/local/matrix")
_DROPBOX = Path("/media/remy/remy-storage/Remy's Dropbox Folder/HongLab @ Caltech Dropbox/Remy")

_default_prj = 'odor_space_collab'

prj = None
RAW_DATA_DIR = None
NAS_PROC_DIR = None
NAS_PRJ_DIR = None
OLF_CONFIG_DIR = None


def get_project_paths(prj=None):
    """Get directory paths for each project."""

    if prj is None:
        prj = _default_prj
    if prj == 'natural_mixtures':
        _RAW_DATA_DIR = _DROPBOX.joinpath('natural_mixtures', 'raw_data')
        _NAS_PRJ_DIR = _MATRIX.joinpath('Remy-Data', 'projects', 'natural_mixtures')
        _NAS_PROC_DIR = _NAS_PRJ_DIR.joinpath('processed_data')
        _OLF_CONFIG_DIR = _NAS_PRJ_DIR.joinpath('olfactometer_configs')
    elif prj == 'narrow_odors':
        _NAS_PRJ_DIR = Path("/local/storage/Remy/narrow_odors")
        _RAW_DATA_DIR = Path("/local/storage/Remy/narrow_odors/raw_data")
        _NAS_PROC_DIR = _NAS_PRJ_DIR.joinpath('processed_data')
        _OLF_CONFIG_DIR = _NAS_PRJ_DIR.joinpath('olfactometer_configs')
    elif prj == 'odor_space_collab':
        _RAW_DATA_DIR = _DROPBOX.joinpath('odor_space_collab', 'raw_data')
        _NAS_PRJ_DIR = _MATRIX.joinpath('Remy-Data', 'projects', 'odor_space_collab')
        _NAS_PROC_DIR = _NAS_PRJ_DIR.joinpath('processed_data')
        _OLF_CONFIG_DIR = _NAS_PRJ_DIR.joinpath('olfactometer_configs')
    elif prj == 'odor_space_collab_tom':
        _RAW_DATA_DIR = Path("/media/remy/remy-storage/Remy's Dropbox Folder/HongLab @ Caltech Dropbox/MB team/Data")
        _NAS_PRJ_DIR = _MATRIX.joinpath('Remy-Data', 'projects', 'odor_space_collab')
        _NAS_PROC_DIR = _NAS_PRJ_DIR.joinpath('processed_data_tom')
        _OLF_CONFIG_DIR = _NAS_PRJ_DIR.joinpath('olfactometer_configs_tom')
    elif prj == 'odor_unpredictability':
        _RAW_DATA_DIR = _DROPBOX.joinpath('odor_unpredictability', 'raw_data')
        _NAS_PRJ_DIR = _MATRIX.joinpath('Remy-Data', 'projects', 'odor_unpredictability')
        _NAS_PROC_DIR = _NAS_PRJ_DIR.joinpath('processed_data')
        _OLF_CONFIG_DIR = _NAS_PRJ_DIR.joinpath('olfactometer_configs')
    elif prj == 'mb_odor_space':
        _RAW_DATA_DIR = Path("/media/remy/remy-storage/Remy's Dropbox Folder/HongLab @ Caltech Dropbox/MB team/Data")
        _NAS_PRJ_DIR = _MATRIX.joinpath('Remy-Data', 'projects', 'odor_space_collab')
        _NAS_PROC_DIR = _NAS_PRJ_DIR.joinpath('processed_data_old')
        _OLF_CONFIG_DIR = _NAS_PRJ_DIR.joinpath('old_olfactometer_configs')
    elif prj == 'for_pratyush':
        _RAW_DATA_DIR = Path("/media/remy/remy-storage/Remy's Dropbox Folder/HongLab @ Caltech Dropbox/Pratyush/2p_res_data")
        _NAS_PRJ_DIR = _MATRIX.joinpath('Remy-Data', 'for_others', 'for_pratyush')
        _NAS_PROC_DIR = _NAS_PRJ_DIR.joinpath('processed_data')
        _OLF_CONFIG_DIR = _NAS_PRJ_DIR.joinpath('olfactometer_configs')
    elif prj == 'for_yang':
        _RAW_DATA_DIR = Path("/media/remy/remy-storage/Remy's Dropbox Folder/HongLab @ Caltech Dropbox/Remy/for_yang/2p_data")
        _NAS_PRJ_DIR = Path("/local/matrix/Remy-Data/for_others/for_yang")
        _NAS_PROC_DIR = _NAS_PRJ_DIR.joinpath('processed_data')
        _OLF_CONFIG_DIR = _NAS_PRJ_DIR.joinpath('olfactometer_configs')

    return _RAW_DATA_DIR, _NAS_PROC_DIR, _NAS_PRJ_DIR, _OLF_CONFIG_DIR


def set_prj(prj):
    global RAW_DATA_DIR, NAS_PRJ_DIR, NAS_PROC_DIR, OLF_CONFIG_DIR

    RAW_DATA_DIR, NAS_PROC_DIR, NAS_PRJ_DIR, OLF_CONFIG_DIR = get_project_paths(prj)
    print(f"\nsetting project paths for {prj}")
    return True


set_prj(None)
