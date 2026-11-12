# Directory layout

## The folder convention

Raw and processed data mirror each other with the same three-level scheme:

```
<RAW_ROOT>/<date_imaged>/<fly_num>/<thorimage_name>/   # ThorImage: Image_001_001.raw, Experiment.xml
<RAW_ROOT>/<date_imaged>/<fly_num>/<thorsync_name>/    # ThorSync: Episode001.h5 (or Episode_0000.h5),
                                                       #           ThorRealTimeDataSettings.xml
<PROC_ROOT>/<date_imaged>/<fly_num>/<thorimage_name>/  # everything the pipeline writes
```

- `date_imaged` like `2025-02-11`
- `fly_num` is a string — usually `1`, `2`, … but sometimes a genotype label
- ThorImage and ThorSync are *sibling* folders under the fly; they're linked
  either by the manifest (`flat_linked_thor_acquisitions.json`), by
  `thormatch` (file-mtime overlap), or by hand in a notebook.

## Storage roots

Historically three machines/mounts (`rye2p/datadirs.py`):

| root | path | role |
|---|---|---|
| matrix (NAS) | `/local/matrix` | processed data (`NAS_PROC_DIR`, `NAS_PRJ_DIR`) |
| tensor | `/local/tensor/Remy` | raw data for most projects |
| Dropbox | `/home/remy/HongLab @ Caltech Dropbox/Remy` | raw data for some projects |

`datadirs.set_prj("<project>")` selects one of nine hard-coded projects and sets
the module globals `RAW_DATA_DIR`, `NAS_PROC_DIR`, `NAS_PRJ_DIR`,
`OLF_CONFIG_DIR`.

!!! warning "datadirs gotchas"
    - Importing `rye2p.datadirs` silently calls `set_prj(None)` → defaults to
      `odor_space_collab` paths. If you forget to `set_prj` your project, you
      read/write **the wrong project's directories** with no error.
    - An unknown project name raises a confusing `UnboundLocalError`.
    - Newer work skips `datadirs` and sets path literals at the top of the
      notebook (`notebooks/pipeline_demo.ipynb` cell 0). The
      [roadmap](roadmap.md) replaces this module with an explicit config object.

## Per-microscope line maps

```
data/microscopes/
├── downstairs-2p/thorsync_lines.yaml
└── upstairs-2p/thorsync_lines.yaml
```

Each YAML maps canonical line names → that rig's actual h5 dataset names, e.g.
downstairs `scope_pin: scopePin` (a digital `DI` line) vs. upstairs analog
lines. These supersede fuzzy name matching. Note the rigs differ electrically:
downstairs' scope pin is 0/1 digital, upstairs' lines are 0–5 V analog —
thresholds must be chosen accordingly (normalize first; see
[Timestamps explained](timestamps.md#stimulus-and-trigger-times)).

## This repository

```
rye2p/                   # the package
├── thorimage/           #   Experiment.xml parsing (settings.py), raw loading (io.py)
├── thorsync/            #   line names, per-plane timestamps, h5 helpers
├── pipeline/            #   raw→tiff conversion steps
├── olfactometer/        #   stimulus config generation (person-specific)
├── process_thorsync.py  #   LEGACY end-to-end ThorSync → timestamps.npy
├── fix_frame_times.py   #   timestamp repairs
├── convert_raw_fn_thorimage.py  # LEGACY raw→tiff (duplicated by pipeline/raw.py)
├── thormatch.py         #   pair ThorImage ↔ ThorSync folders by mtime
├── olf_configs.py       #   olfactometer YAML → stimulus tables
├── datadirs.py          #   LEGACY hard-coded project paths
└── ...                  #   one-off analysis/figure scripts (see roadmap)
notebooks/               # the de-facto current pipeline (see pipeline.md)
scripts/for_*/           # per-person demo/rescue scripts
data/microscopes/        # per-rig thorsync_lines.yaml
data/haoru/              # small committed worked example (timestamps.npy, df_stim.csv, F.tsv)
tests/                   # pytest suite
docs/ + mkdocs.yml       # this documentation site
```
