# Tutorial: convert raw movies to tiff

Converts a ThorImage acquisition folder into the processed outputs:
`Image_001_001.tif`, `thorimage_settings.yaml`, a copied `Experiment.xml`, and
500-frame `stk/` chunks for suite2p/caiman.

!!! note "Current home of this workflow"
    The orchestration function
    [`preprocess_thorimage()`](../api/pipeline-preprocess.md) lives in
    `rye2p.pipeline.preprocess` (promoted from the notebook in
    [roadmap](../roadmap.md) step 1). One call does everything below:

    ```python
    from rye2p.pipeline.preprocess import preprocess_thorimage

    preprocess_thorimage(raw_mov_dir, proc_mov_dir, thorsync_name="SyncData001")
    ```

    The individual library pieces remain available as below.

## Inputs

```
<RAW_ROOT>/<date>/<fly>/<thorimage_name>/
├── Image_001_001.raw      # (or Image_0001_0001.raw on older software)
└── Experiment.xml
```

## Step by step with the library

```python
from pathlib import Path
import shutil

raw_mov_dir  = Path("/raw_root/2025-02-11/1/kiwi_ea_eb_only")
proc_mov_dir = Path("/proc_root/2025-02-11/1/kiwi_ea_eb_only")
proc_mov_dir.mkdir(parents=True, exist_ok=True)

# 1. copy metadata + parse settings
shutil.copy(raw_mov_dir / "Experiment.xml", proc_mov_dir / "Experiment.xml")

from rye2p.thorimage.settings import FunctionalImagingSettings
settings = FunctionalImagingSettings.from_xml(raw_mov_dir / "Experiment.xml")
settings.to_yaml(proc_mov_dir / "thorimage_settings.yaml")

# 2. raw → one big ImageJ tiff (flyback planes are dropped here)
from rye2p.thorimage.io import convert_raw_to_tiff, split_tiff_files
raw_file = raw_mov_dir / "Image_001_001.raw"
tiff_file = convert_raw_to_tiff(raw_file, raw_mov_dir.parents[2], proc_mov_dir.parents[2])

# 3. split into 500-timepoint chunks for suite2p/caiman
split_tiff_files(tiff_file, frames_per_batch=500)   # writes proc_mov_dir/stk/
```

Resulting movie shape: `(T, Z, Y, X)` for fast-z (flyback already removed — see
[Timestamps explained](../timestamps.md#flyback-planes-the-movie-and-the-timestamps-disagree)),
`(T, Y, X)` for single-plane.

## Batch: a whole imaging day

Open `notebooks/pipeline_demo.ipynb`, set in cell 0:

```python
RAW_DATA_DIR = Path("/local/tensor/Remy/<project>/raw_data")
NAS_PROC_DIR = Path("/local/matrix/Remy-Data/projects/<project>/processed_data")
```

then run all cells — it globs every ThorImage folder under the chosen day,
validates it, and runs the steps above.

## Gotchas

- `split_tiff_files` **silently does nothing** if `stk/` already exists —
  pass `overwrite=True` (what `preprocess_thorimage` does) or delete the
  folder to re-split.
- `convert_raw_to_tiff` builds the output path by *mirroring the relative path*
  under the raw root, so pass the roots (not the movie dirs) as `raw_dir` /
  `proc_dir`.
- Memory: the whole movie is loaded into RAM as uint16.
