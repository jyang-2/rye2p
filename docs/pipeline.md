# How the pipeline works

End-to-end, one acquisition goes through five stages:

```
1. match      ThorImage folder ↔ ThorSync folder ↔ olfactometer config
2. convert    Image_001_001.raw → Image_001_001.tif (+ stk/ chunks)
3. metadata   Experiment.xml → copy + thorimage_settings.yaml
4. timing     Episode001.h5 → timestamps.npy (frame/stack times, odor + scope triggers)
5. stimuli    olf config YAML → df_stim.csv, df_pins2odors.csv, stim_list.json
```

Outputs of every stage are documented in the [output file reference](outputs.md).

## The current path (use this)

The actively used pipeline lives in two notebooks:

**1. `notebooks/pipeline_demo.ipynb`** — stages 2–3 for a whole imaging day.
Set `RAW_DATA_DIR` / `NAS_PROC_DIR` literals in cell 0, then run
[`preprocess_thorimage(raw_mov_dir, proc_mov_dir, ...)`](api/pipeline-preprocess.md)
per ThorImage folder: validate folder → copy `Experiment.xml` → write
`thorimage_settings.yaml` → write `Image_001_001.tif` → split `stk/`.

!!! note
    `preprocess_thorimage()` was promoted out of the notebook into
    [`rye2p.pipeline.preprocess`](api/pipeline-preprocess.md) (roadmap step 1);
    the notebook is now a thin batch driver around the library function.

**2. `notebooks/thorsync_lines_demo.ipynb`** — stages 4–5 for one acquisition
(use `scripts/for_ruoyi/thorsync_lines_demo.ipynb` for single-plane recordings):

1. parse olf config → stimulus CSVs
2. load the per-rig line map (`rye2p.thorsync.linenames.load_line_map`, reading
   the package data in `rye2p/data/microscopes/<scope>/thorsync_lines.yaml`)
3. `utils2p.synchronization.get_lines_from_sync_file` → raw lines;
   `SyncMetadata.get_freq()` → sample rate
4. decimate to 1 kHz, save `sync_lines.npz` (QC)
5. frame counter → `frame_times`; normalized TTL edges → scope/olf times
6. group frames by scope pulse, truncate partial volumes, derive `stack_times`
7. sanity-check counts against `thorimage_settings.yaml`, save `timestamps.npy`

**Repair, when counts don't match:** hand-run cells of
`rye2p/fix_frame_times.py`'s `__main__` block, or
`scripts/for_yang/check_frame_number.py`. See
[Timestamps explained](timestamps.md#what-fix_frame_timespy-repairs).

## The legacy path (importable, but has known bugs)

`rye2p/master.py` is the nominal driver — **be careful: it is a cell-script with
no `__main__` guard, so merely importing it runs the pipeline** over hard-coded
paths. Order of operations:

1. `datadirs.set_prj("<project>")`
2. `convert_raw_fn_thorimage.main(day_folder, RAW_DATA_DIR, NAS_PROC_DIR)` —
   rglobs `Image_001_001.raw` (fallback `Image_0001_0001.raw`), converts, splits
3. optional `thormatch.main(...)` — writes the per-fly pairing CSVs
4. loads `manifestos/flat_linked_thor_acquisitions.json`
5. per acquisition: `process_thorsync.main(flacq, raw_dir, proc_dir,
   match_stack_times_only=True)` → proc dir, `Experiment.xml` copy,
   `timestamps0.npy`, `timestamps.npy`
6. `olf_configs.parse_olf_config_file*` → stimulus tables

Fuzzy line-name matching, a 2.5 V analog threshold assumption, and a
single-plane `steps_per_frame` bug make this path unreliable on the downstairs
rig and on single-plane recordings — details in [known issues](known-issues.md).

## Raw-data write guard

Every library function that chooses an output root
(`preprocess_thorimage`, `convert_raw_to_tiff`, `split_tiff_files`,
`copy_raw_dir_2_proc_dir`, and the legacy `process_thorsync` writers) calls
[`pathguard.guard_write_dir`](api/pathguard.md) first and **raises
`RawDataWriteError`** instead of writing when the destination is a raw-data
location: any path containing a `tensor-archive` or `raw_data` component, or
a directory that (itself or its parent) holds a raw acquisition file
(`Image_001_001.raw` / `Image_0001_0001.raw` / `Episode001.h5`). This catches
swapped `raw_dir`/`proc_dir` arguments and mis-set path literals. Deliberate
exceptions: set `RYE2P_ALLOW_RAW_WRITES=1` (downgrades to a warning).

## Downstream contract

What analysis code consumes (see `data/haoru/example_trial_extraction.py` for a
complete worked example):

- `Image_001_001.tif` or `stk/` → suite2p/caiman → ROI traces
- `timestamps.npy` → `stack_times` (volume timebase) and `olf_ict`/`olf_fct`
  (trial windows)
- `df_stim.csv` → trial *i* ↔ `olf_ict[i]` → stimulus identity per trial
- **volumetric analyses**: per-plane timebase via
  [`plane_times`](timestamps.md#per-plane-timestamps-what-volumetric-analyses-should-use)
