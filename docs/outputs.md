# Output file reference

Every file the pipeline writes, exactly where it goes, and exactly what's in it.

All per-movie outputs land in the **processed movie directory**:

```
<PROC_ROOT>/<date_imaged>/<fly_num>/<thorimage_name>/
```

See [Directory layout](directory-layout.md) for how the roots are chosen.

## Per-movie outputs

### `Experiment.xml`
Verbatim copy of the ThorImage metadata XML (`shutil.copy`; skipped if already
present). Written by `process_thorsync.copy_experiment_xml` or
`preprocess_thorimage` (notebook).

### `thorimage_settings.yaml`
Parsed, human-readable acquisition settings — the yaml round-trip of
`FunctionalImagingSettings` (`rye2p/thorimage/settings.py`). 18 scalar fields:

```yaml
raw_frame_rate: ...      # LSM frame rate (per plane), Hz
frame_rate: ...          # effective volume/frame rate = raw_frame_rate / steps_per_frame
n_frames: ...            # total frames in the raw file
n_timepoints: ...        # number of volumes (fast-z) or saved frames (single-plane)
fast_z: true/false
n_z: ...                 # z-planes per volume in the SAVED movie (1 if single-plane)
n_flyback_frames: ...    # acquired but NOT saved in the tiff
step_size_um: ...
average_mode: ...        # single-plane frame averaging on/off
n_averaging: ...
steps_per_frame: ...     # n_z + n_flyback (fast-z) or n_averaging (single-plane)
timelapse_trigger_mode: ...
streaming_trigger_mode: ...
stimulus_triggering: ...
software_version: ...
experiment_notes: ...
thorsync_name: ...       # linked ThorSync folder name
```

Load with `FunctionalImagingSettings.from_yaml(path)`.

### `Image_001_001.tif`
The full movie as one ImageJ hyperstack, `uint16`.

- fast-z: `(T, Z, Y, X)` with **flyback planes already removed**
  (`utils2p.load_raw` trims them; `utils2p.save_img` writes ImageJ axis order).
- single-plane: `(T, Y, X)`.

Written by `convert_raw_fn_thorimage.convert_raw_to_tiff` (= `pipeline.raw`).

### `stk/stk_000.tif`, `stk_001.tif`, …
The same movie re-split into 500-timepoint chunks (`frames_per_batch=500`) for
suite2p/caiman ingestion. ⚠️ `split_tiff_files` silently writes **nothing** if
`stk/` already exists.

### `timestamps0.npy` (legacy path only)
Raw extraction result, **before** repairs. A pickled dict saved with `np.save` —
load with `np.load(f, allow_pickle=True).item()` or
`rye2p.thorsync.timestamps.load_timestamps`. Keys (1-D float64, seconds since
ThorSync start):

| key | length | meaning |
|---|---|---|
| `frame_times` | `n_timepoints × steps_per_frame` | one per z-plane **incl. flyback** |
| `stack_times` | `n_timepoints` | one per volume (first plane) |
| `scope_ict` / `scope_fct` | n scope pulses | acquisition trigger rise/fall |
| `olf_ict` / `olf_fct` | n odor pulses | stimulus on/off |

### `timestamps.npy`
**The file downstream code reads.** Same six keys as `timestamps0.npy`, after
repair (see [Timestamps explained](timestamps.md#what-fix_frame_timespy-repairs)).
If `fix_timestamps0` ran, four extra keys:

| key | type | meaning |
|---|---|---|
| `split_frame_times` | list of arrays | frame times grouped per scope pulse |
| `split_stack_times` | list of arrays | stack times grouped per scope pulse |
| `n_frame_times_per_pulse` | list of int | counts per pulse |
| `n_stack_times_per_pulse` | list of int | counts per pulse; used as tiff split points by `downsample_stack.py` |

### `timestamps_tsub{N}.npy`
Timestamps matched to a temporally downsampled movie (`fix_frame_times.edit_timestamps_file`).
Adds `tsub` (int) and `dropped_frames` keys; `stack_times` is mean-pooled in
bins of `tsub`.

### `sync_lines.npz` (notebook path)
All 9 sync lines + `time`, decimated 30 kHz → 1 kHz for plotting/QC
(`scipy.signal.decimate(q=30)`; `frame_counter` is plain-strided `[::30]` to
avoid filtering a step function). `np.savez_compressed`.

### `pid.npz`
Photoionization-detector trace at full 30 kHz: `sync_times`, `pid`,
`olf_disp_pin`, `scope_pin` (`rye2p/pid.py::pid_lines_to_npz` — currently broken,
see [known issues](known-issues.md)).

### Stimulus tables (from the olfactometer config)

| file | format | contents |
|---|---|---|
| `df_pins2odors.csv` | **tab**-separated despite `.csv` | index `pin`; columns `name`, `log10_conc`, `abbrev`, `stim` (= `"{abbrev} @ {log10_conc}"`) |
| `df_stim.csv` | tab-separated | one row per trial; index `trial_idx`; columns `pin, name, log10_conc, abbrev, stim` |
| `stim_list.json` | json | `{"stim_list_flatstr": [...]}` — the `stim` column as a list |

Pair row *i* of `df_stim` with `olf_ict[i]` / `olf_fct[i]` to get each trial's
stimulus identity and timing (see `data/haoru/example_trial_extraction.py`).

### QC figures
`screenshots/processed_sync_lines.png` — 4-panel line plot (scope_pin,
olf_disp_pin, frame_counter, piezo_monitor), 300 dpi.

### Split/downsampled movies
- `acqs/acq_0000.tif`, … — movie split per scope pulse (split points from
  `n_stack_times_per_pulse`), `uint16`.
- `acqs/acq_XXXX__dsub_NN.tif` + `acqs/kept_stack_times__dsub_NN.npy` —
  temporally downsampled sub-movies plus the kept-frame index vectors.
- `<thorimage_name>_tsub{N}/` — a sibling directory with its own downsampled
  `Image_001_001.tif`, `timestamps.npy`, and `stk/`.

## Per-fly outputs (`<PROC_ROOT>/<date>/<fly>/`)

Written by `thormatch.main` (pairs ThorImage ↔ ThorSync folders by file
modification times):

| file | contents |
|---|---|
| `df_thorimage_fileinfo.csv` | ThorImage folder, name, start/end time (mtime-based), duration |
| `df_thorsync_fileinfo.csv` | same for ThorSync folders |
| `df_fraction_thorimage_contained.csv` | ThorImage × ThorSync time-overlap fraction matrix |

## Project-level inputs (not outputs, but required)

| file | role |
|---|---|
| `<PRJ_DIR>/manifestos/flat_linked_thor_acquisitions.json` | manifest of linked (ThorImage, ThorSync, olf config) triples the legacy driver iterates over |
| `<OLF_CONFIG_DIR>/*.yaml` | olfactometer configs → stimulus tables |
| `data/microscopes/<scope>/thorsync_lines.yaml` | per-rig line-name maps (in this repo) |
