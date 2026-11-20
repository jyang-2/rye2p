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

### The experiment log: `stimulus_info_<date>.yaml`

Each date folder holds a hand-written experiment log,
`<RAW_ROOT>/<date>/stimulus_info_<date>.yaml` (copied to
`<PROC_ROOT>/<date>/` too). It is **never machine-read** — YAML is used only
because it renders nicely in the rig's text editor and acts as a fill-in
template — but it is the **authoritative record of the
ThorImage ↔ ThorSync ↔ olf_config pairing**, which no data file encodes:
a session header (rig, experimenter, olfactometer, laser, `flies:`
genotype list in fly-number order), then a list of acquisition entries grouped
by `# fly NN` comments, each naming `ThorImage`, `ThorSync` (`null` for
anatomy stacks), `olf_config`, laser power, and free-text notes (frame counts,
single-plane/averaging settings, problems). Flies marked "don't analyze"
simply have no `<PROC_ROOT>/<date>/<fly>/` folder. Because entries are
hand-written, verify the pairing against the data fingerprints described in
[Timestamps explained → Validation](timestamps.md#validation) before trusting
it — the 2026-11 ground-truth check found the log correct where checked
(2023-04-11 fly 2).

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
rye2p/data/microscopes/
├── downstairs-2p/thorsync_lines.yaml
└── upstairs-2p/thorsync_lines.yaml
```

Shipped as package data — load with
`rye2p.thorsync.linenames.load_line_map("<scope>")`.
Each YAML maps canonical line names → that rig's actual h5 dataset names, e.g.
downstairs `scope_pin: scopePin` (a digital `DI` line) vs. upstairs analog
lines. These supersede fuzzy name matching. Note the rigs differ electrically:
downstairs' scope pin is 0/1 digital, upstairs' lines are 0–5 V analog —
thresholds must be chosen accordingly (normalize first; see
[Timestamps explained](timestamps.md#stimulus-and-trigger-times)).

## This repository

```
rye2p/                   # the package
├── thorimage/           #   Experiment.xml parsing (settings.py), raw→tiff (io.py)
├── thorsync/            #   line names, per-plane timestamps, h5 helpers
├── pipeline/            #   preprocess_thorimage orchestration (preprocess.py)
├── data/microscopes/    #   per-rig thorsync_lines.yaml (package data)
├── olfactometer/        #   stimulus config generation (person-specific)
├── process_thorsync.py  #   LEGACY end-to-end ThorSync → timestamps.npy
├── fix_frame_times.py   #   timestamp repairs
├── convert_raw_fn_thorimage.py  # LEGACY raw→tiff (re-exports thorimage/io.py)
├── thormatch.py         #   pair ThorImage ↔ ThorSync folders by mtime
├── olf_configs.py       #   olfactometer YAML → stimulus tables
├── datadirs.py          #   LEGACY hard-coded project paths
└── ...                  #   one-off analysis/figure scripts (see roadmap)
notebooks/               # the de-facto current pipeline (see pipeline.md)
scripts/for_*/           # per-person demo/rescue scripts
data/haoru/              # small committed worked example (timestamps.npy, df_stim.csv, F.tsv)
tests/                   # pytest suite
docs/ + mkdocs.yml       # this documentation site
```
