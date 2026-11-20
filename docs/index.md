# rye2p

**rye2p** converts and parses raw 2-photon imaging acquisitions from Thorlabs
microscopes:

- **ThorImage** folders (`Image_001_001.raw` + `Experiment.xml`) → ImageJ-style
  **tiff movies** plus a parsed settings file.
- **ThorSync** folders (`Episode001.h5` + `ThorRealTimeDataSettings.xml`) →
  **timestamps** for every frame/plane and volume, plus stimulus (odor) on/off
  times and microscope trigger times.
- **Olfactometer config** YAMLs → per-trial stimulus tables.

```
ThorImage folder                    ThorSync folder
  Image_001_001.raw                   Episode001.h5
  Experiment.xml                      ThorRealTimeDataSettings.xml
        │                                   │
        ▼                                   ▼
  Image_001_001.tif  ◄── indexed by ──  timestamps.npy
  thorimage_settings.yaml               (frame/stack times, odor + scope triggers)
  stk/stk_000.tif, ...                  sync_lines.npz
```

## Where to go

| I want to… | Read this |
|---|---|
| Understand the whole flow, end to end | [How the pipeline works](pipeline.md) |
| Understand frame vs. stack vs. per-plane times | [Timestamps explained](timestamps.md) |
| Know exactly what's inside every output file | [Output file reference](outputs.md) |
| Know where files live on disk | [Directory layout](directory-layout.md) |
| Convert a new imaging day | [Tutorial: convert raw movies](tutorials/convert-raw.md) |
| Get odor/stimulus times for trials | [Tutorial: extract timestamps](tutorials/extract-timestamps.md) |
| Use per-plane times in volumetric analysis | [Tutorial: per-plane timestamps](tutorials/per-plane-timestamps.md) |
| Look up a function | API reference (sidebar) |
| Know what's broken or inconsistent | [Known issues](known-issues.md) |
| See the cleanup plan | [Refactor roadmap](roadmap.md) |

!!! warning "Two generations of code coexist"
    The repo currently contains a **legacy path** (`rye2p/process_thorsync.py` +
    `rye2p/master.py`, fuzzy line-name matching) and a **current path**
    (`notebooks/pipeline_demo.ipynb` + `notebooks/thorsync_lines_demo.ipynb`,
    explicit per-microscope line maps in `rye2p/data/microscopes/*/thorsync_lines.yaml`).
    Both write the same `timestamps.npy`; the notebook path is what's actively
    used. The [roadmap](roadmap.md) describes how they get unified. Until then,
    prefer the notebook path for new data.

## Build these docs

```bash
.venv/bin/mkdocs serve   # live-reload at http://localhost:8000
.venv/bin/mkdocs build   # static site in site/
```
