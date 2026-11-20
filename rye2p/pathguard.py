"""Refuse to write pipeline outputs into raw-data locations.

Raw 2p acquisitions (ThorImage ``.raw`` + ThorSync ``.h5``) are
irreplaceable; processed outputs must never land in or beside them. Every
rye2p function that chooses an output *root* calls :func:`guard_write_dir`
on the destination directory before writing anything.

Two independent checks:

1. **Path markers** — any path component named ``tensor-archive`` (the raw
   archive mount) or ``raw_data`` (every project's raw tree) refuses.
2. **Acquisition fingerprint** — a destination directory (or its parent)
   that already contains a raw acquisition file (``Image_001_001.raw``,
   ``Image_0001_0001.raw``, or ``Episode001.h5``) refuses, whatever the
   path is named. This catches swapped raw/proc arguments.

Escape hatch for deliberate exceptions (e.g. a hand repair in an unusual
location): set ``RYE2P_ALLOW_RAW_WRITES=1`` in the environment and the
guard warns instead of raising.

stdlib-only on purpose, so it is importable in the numpy-only test
environment and from ``pipeline/preprocess.py``'s fallback loader.
"""

import os
import warnings
from pathlib import Path

#: Directory names that mark a raw-data tree anywhere in a path.
RAW_MARKER_DIRNAMES = frozenset({"tensor-archive", "raw_data"})

#: Files whose presence marks a directory as a raw acquisition folder.
RAW_FINGERPRINT_FILES = ("Image_001_001.raw", "Image_0001_0001.raw", "Episode001.h5")

_OVERRIDE_ENV = "RYE2P_ALLOW_RAW_WRITES"
_TRUTHY = {"1", "true", "yes"}


class RawDataWriteError(RuntimeError):
    """Refused to write pipeline output into a raw-data location."""


def is_raw_data_path(path):
    """True if ``path`` has a raw-data marker component (``tensor-archive``
    or ``raw_data``) anywhere in its resolved parts."""
    return not RAW_MARKER_DIRNAMES.isdisjoint(Path(path).resolve().parts)


def looks_like_acquisition_dir(dirpath):
    """True if ``dirpath`` contains a raw acquisition file
    (``Image_001_001.raw`` / ``Image_0001_0001.raw`` / ``Episode001.h5``)."""
    dirpath = Path(dirpath)
    return any(dirpath.joinpath(name).is_file() for name in RAW_FINGERPRINT_FILES)


def guard_write_dir(dest_dir, what="pipeline output"):
    """Refuse ``dest_dir`` as a write destination if it is a raw-data location.

    Args:
        dest_dir (Union[str, Path]): directory about to be written into
            (it does not need to exist yet).
        what (str): short description of what would be written, used in the
            error message.

    Returns:
        Path: ``dest_dir`` as a ``Path``, when the write is allowed.

    Raises:
        RawDataWriteError: if ``dest_dir`` lies under a raw-data marker
            directory, or it (or its parent) contains a raw acquisition
            file — unless ``RYE2P_ALLOW_RAW_WRITES=1`` is set, which
            downgrades the refusal to a warning.
    """
    dest_dir = Path(dest_dir)

    markers = RAW_MARKER_DIRNAMES.intersection(dest_dir.resolve().parts)
    if markers:
        reason = f"path contains raw-data marker {sorted(markers)}"
    elif looks_like_acquisition_dir(dest_dir) or looks_like_acquisition_dir(dest_dir.parent):
        reason = ("directory (or its parent) contains a raw acquisition file "
                  f"(one of {RAW_FINGERPRINT_FILES})")
    else:
        return dest_dir

    message = (f"refusing to write {what} into {dest_dir}: {reason}. "
               f"Pipeline outputs belong under a processed-data root. "
               f"Set {_OVERRIDE_ENV}=1 to override deliberately.")

    if os.environ.get(_OVERRIDE_ENV, "").strip().lower() in _TRUTHY:
        warnings.warn(f"{_OVERRIDE_ENV} override active — {message}")
        return dest_dir

    raise RawDataWriteError(message)
