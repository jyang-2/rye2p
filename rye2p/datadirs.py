from pathlib import Path

_MATRIX = Path("/local/matrix")
_DROPBOX = Path("/media/remy/remy-storage/Remy's Dropbox Folder/HongLab @ Caltech Dropbox/Remy")

_default_prj = 'odor_space_collab'

RAW_DATA_DIR = None
NAS_PROC_DIR = None
NAS_PRJ_DIR = None


def get_project_paths(prj=None):
    """Get directory paths for each project."""

    if prj is None:
        prj = _default_prj
    if prj == 'natural_mixtures':
        _RAW_DATA_DIR = _DROPBOX.joinpath('natural_mixtures', 'raw_data')
        _NAS_PRJ_DIR = _MATRIX.joinpath('Remy-Data', 'projects', 'natural_mixtures')
        _NAS_PROC_DIR = _NAS_PRJ_DIR.joinpath('processed_data')
    elif prj == 'narrow_odors':
        _NAS_PRJ_DIR = Path("/local/storage/Remy/narrow_odors")
        _RAW_DATA_DIR = Path("/local/storage/Remy/narrow_odors/raw_data")
        _NAS_PROC_DIR = _NAS_PRJ_DIR.joinpath('processed_data')
    elif prj == 'odor_space_collab':
        _RAW_DATA_DIR = _DROPBOX.joinpath('odor_space_collab', 'raw_data')
        _NAS_PRJ_DIR = _MATRIX.joinpath('Remy-Data', 'projects', 'odor_space_collab')
        _NAS_PROC_DIR = _NAS_PRJ_DIR.joinpath('processed_data')
    elif prj == 'odor_unpredictability':
        _RAW_DATA_DIR = _DROPBOX.joinpath('odor_unpredictability', 'raw_data')
        _NAS_PRJ_DIR = _MATRIX.joinpath('Remy-Data', 'projects', 'odor_unpredictability')
        _NAS_PROC_DIR = _NAS_PRJ_DIR.joinpath('processed_data')
    return _RAW_DATA_DIR, _NAS_PROC_DIR, _NAS_PRJ_DIR


def set_prj(prj):
    global RAW_DATA_DIR, NAS_PRJ_DIR, NAS_PROC_DIR

    RAW_DATA_DIR, NAS_PROC_DIR, NAS_PRJ_DIR = get_project_paths(prj)
    print(f"\nsetting project paths for {prj}")
    return True


set_prj(None)
