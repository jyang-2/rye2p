"""Read ThorImage .raw movies and write tiff outputs.

Single home of the raw→tiff conversion functions. ``rye2p.pipeline.raw`` and
``rye2p.convert_raw_fn_thorimage`` re-export from here (roadmap step 1); new
code should import from this module.
"""

import shutil
from pathlib import Path

import utils2p


def load_raw(raw_file):
    """Load a ThorImage .raw movie using the sibling Experiment.xml.

    Args:
        raw_file (Union[str, Path]): path to .raw movie file
            Should be named something like 'Image_0001_0001.raw'

    Returns:
        tuple[np.ndarray, ...]: one uint16 stack per channel, each in TZYX
            order (single-channel recordings return a 1-tuple — unpack with
            ``stack, = load_raw(...)``).

    """
    if isinstance(raw_file, str):
        raw_file = Path(raw_file)

    meta_file = raw_file.with_name('Experiment.xml')
    meta = utils2p.Metadata(meta_file)

    # tiff stacks in TZYX order, one per channel
    return utils2p.load_raw(raw_file, meta)


def copy_raw_dir_2_proc_dir(src_file, raw_dir, proc_dir):
    if isinstance(src_file, str):
        src_file = Path(src_file)

    dest_file = proc_dir.joinpath(*src_file.relative_to(raw_dir).parts)
    shutil.copy(src_file, dest_file)
    return dest_file


def convert_raw_to_tiff(raw_file, raw_dir, proc_dir):
    """ Convert raw file to single tif stack

    Args:
        proc_dir ():
        raw_dir ():
        raw_file ():

    Returns:
        Path: filepath to saved tiff stack

    """
    meta_file = raw_file.with_name('Experiment.xml')
    meta = utils2p.Metadata(meta_file)

    # .raw --> .tif
    stack1, = utils2p.load_raw(raw_file, meta)

    # convert to relative path
    rel_path = raw_file.relative_to(raw_dir)
    print(f"\nRaw image file: {rel_path}")

    # build tiff file name - save full movie as Image_001_001.tif
    save_name = proc_dir.joinpath(rel_path.with_suffix(".tif"))

    save_name.parent.mkdir(parents=True, exist_ok=True)
    print(f"Saving to: {save_name}")

    # save tiff file to project directory on network drive
    utils2p.save_img(save_name, stack1)
    print(f"...saved successfully.\n")
    return save_name


def split_tiff_files(tiff_file, frames_per_batch, save_dir=Path('stk'), overwrite=False):
    """ Split tiff files into smaller stacks (timestamps)

    Args:
        tiff_file (Path): Path to .tif stack<br><br>
        frames_per_batch (int): # of timepoints to include in each batch<br><br>
        save_dir (Path): where split tif stacks are saved.<br>
                        -  If `save_dir` is a relative path, then the save path is relative to tiff_file.parent<br>
                        -  If `save_dir` is an absolute path, then files are saved there.<br><br>
        overwrite (bool): if False (default), silently writes **nothing** and
            returns None when `save_dir` already exists (legacy behavior);
            if True, writes into an existing `save_dir`, overwriting chunks.

    Returns:
        List[Path]: List of saved tiff files

    """
    if isinstance(tiff_file, str):
        tiff_file = Path(tiff_file)

    if isinstance(save_dir, str):
        save_dir = Path(save_dir)

    STK_DIR = tiff_file.parent.joinpath(save_dir)

    if overwrite or not STK_DIR.is_dir():
        saved_files = []
        STK_DIR.mkdir(parents=True, exist_ok=True)

        batch_gen = utils2p.load_stack_batches(tiff_file, frames_per_batch)

        for i, batch in enumerate(batch_gen):
            save_name = STK_DIR.joinpath(f"stk_{i:03d}.tif")
            utils2p.save_img(save_name, batch)
            saved_files.append(save_name)
    else:
        saved_files = None

    return saved_files
