import utils2p
from typing import Union
from pathlib import Path
import numpy as np


def load_raw(raw_file):
    """ Convert raw file to single tif stack

    Args:
        raw_file (Union[str, Path]): path to .raw movie file
            Should be named something like 'Image_0001_0001.raw'

    Returns:
        np.ndarray: movie stacks in TZYX order (uint16)

    """
    if isinstance(raw_file, str):
        raw_file_path = Path(raw_file)
    else:
        raw_file_path = raw_file

    meta_file = raw_file_path.with_name('Experiment.xml')
    meta = utils2p.Metadata(meta_file)

    # return tiff stacks in TZYX order
    return utils2p.load_raw(raw_file, meta)



