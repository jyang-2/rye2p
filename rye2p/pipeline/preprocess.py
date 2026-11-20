"""Preprocess ThorImage acquisitions: copy metadata and save tiff movies.

Promoted from ``notebooks/pipeline_demo.ipynb`` cell 2 (roadmap step 1); the
notebook is now a thin driver around :func:`preprocess_thorimage`.

Heavy imports (``utils2p`` and the modules that need it) are function-local so
this module stays loadable under the numpy-only test environment — tests load
it by file path, see ``tests/``.
"""

import shutil
from pathlib import Path

try:
    from rye2p.pathguard import guard_write_dir, RawDataWriteError
except ImportError:
    # Loaded by file path outside the package (the numpy-only test env,
    # where rye2p/__init__'s utils2p import fails): load the stdlib-only
    # guard directly from its file instead.
    import importlib.util as _importlib_util

    _pathguard_file = Path(__file__).resolve().parent.parent / "pathguard.py"
    _spec = _importlib_util.spec_from_file_location("rye2p_pathguard", _pathguard_file)
    _pathguard = _importlib_util.module_from_spec(_spec)
    _spec.loader.exec_module(_pathguard)
    guard_write_dir = _pathguard.guard_write_dir
    RawDataWriteError = _pathguard.RawDataWriteError

#: Raw movie filenames ThorImage has used, in lookup order (newer first).
RAW_MOVIE_NAMES = ("Image_001_001.raw", "Image_0001_0001.raw")


def validate_thorimage_folder(thorimage_folder):
    """Check that a folder is a ThorImage raw acquisition; return the .raw path.

    Args:
        thorimage_folder (Union[str, Path]): ThorImage acquisition folder.

    Returns:
        Path: the raw movie file (first of ``RAW_MOVIE_NAMES`` present).

    Raises:
        FileNotFoundError: if ``Experiment.xml`` or the .raw movie is missing.
    """
    thorimage_folder = Path(thorimage_folder)

    if not thorimage_folder.joinpath("Experiment.xml").is_file():
        raise FileNotFoundError("Experiment.xml file not found.")

    for name in RAW_MOVIE_NAMES:
        raw_file = thorimage_folder.joinpath(name)
        if raw_file.is_file():
            return raw_file

    raise FileNotFoundError(".raw movie file not found.")


def preprocess_thorimage(raw_mov_dir, proc_mov_dir, thorsync_name=None,
                         copy_xml=True,
                         save_settings=True,
                         save_single_tiff=True,
                         save_tiff_sequence=True,
                         frames_per_batch=500):
    """Copy metadata and save movies as .tiff files.

    Per ThorImage folder: validate → copy ``Experiment.xml`` → write
    ``thorimage_settings.yaml`` → write the full movie as one ImageJ tiff →
    split it into ``stk/stk_000.tif, ...`` chunks.

    Deviations from the notebook original (all output-identical for data the
    notebook could handle):

    - The tiff is named after the .raw file actually present
      (``Image_001_001.tif``, or ``Image_0001_0001.tif`` on older software);
      the notebook hard-coded ``Image_001_001.raw`` and crashed on the older
      name even though its validator accepted it.
    - Settings are parsed from the **raw** folder's ``Experiment.xml`` (the
      copy is verbatim, so the yaml is identical; the notebook re-read the
      copied file and crashed when ``copy_xml=False``).
    - ``stk/`` chunks overwrite existing files (notebook semantics), via
      ``thorimage.io.split_tiff_files(..., overwrite=True)``.

    Caveats (inherited from the notebook, unchanged):

    - If either tiff flag is set, the full single tiff is always written —
      ``save_single_tiff=False, save_tiff_sequence=True`` still writes it,
      because the chunk splitter reads it back from disk.
    - Single-channel recordings only (``stack, = load_raw(...)``); two-channel
      data raises ``ValueError``.

    Args:
        raw_mov_dir (Union[str, Path]): ThorImage acquisition folder
            (contains the .raw movie and ``Experiment.xml``).
        proc_mov_dir (Union[str, Path]): output folder; created if missing.
        thorsync_name (str): paired ThorSync folder name, recorded in
            ``thorimage_settings.yaml`` (pairing is NOT stored in either
            Thorlabs file — see the validation section of the timestamps docs).
        copy_xml (bool): copy ``Experiment.xml`` into ``proc_mov_dir``.
        save_settings (bool): write ``thorimage_settings.yaml``.
        save_single_tiff (bool): write the full movie as one tiff.
        save_tiff_sequence (bool): split the movie into ``stk/`` chunks.
        frames_per_batch (int): timepoints per ``stk/`` chunk.

    Returns:
        Path: ``proc_mov_dir``.

    Raises:
        RawDataWriteError: if ``proc_mov_dir`` is a raw-data location
            (see :mod:`rye2p.pathguard`) — e.g. swapped raw/proc arguments.
    """
    raw_mov_dir = Path(raw_mov_dir)
    proc_mov_dir = Path(proc_mov_dir)

    raw_file = validate_thorimage_folder(raw_mov_dir)
    guard_write_dir(proc_mov_dir, what="preprocessed ThorImage outputs")
    proc_mov_dir.mkdir(parents=True, exist_ok=True)

    save_tiffs = save_single_tiff or save_tiff_sequence

    if copy_xml:
        shutil.copy(raw_mov_dir.joinpath("Experiment.xml"),
                    proc_mov_dir.joinpath("Experiment.xml"))

    if save_settings:
        from rye2p.thorimage.settings import FunctionalImagingSettings

        thorimage_settings = FunctionalImagingSettings.from_xml(
                raw_mov_dir.joinpath("Experiment.xml")
                )
        if thorsync_name is not None:
            thorimage_settings.thorsync_name = thorsync_name

        thorimage_settings.to_yaml(
                proc_mov_dir.joinpath('thorimage_settings.yaml'))

    if save_tiffs:
        import utils2p

        print('save tiffs')
        meta = utils2p.Metadata(raw_mov_dir.joinpath("Experiment.xml"))

        stack1, = utils2p.load_raw(raw_file, meta)
        tiff_file = proc_mov_dir.joinpath(raw_file.name).with_suffix('.tif')
        utils2p.save_img(tiff_file, stack1)

        if save_tiff_sequence:
            from rye2p.thorimage.io import split_tiff_files

            split_tiff_files(tiff_file, frames_per_batch,
                             save_dir=Path('stk'), overwrite=True)

    return proc_mov_dir
