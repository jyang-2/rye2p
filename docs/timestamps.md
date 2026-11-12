# Timestamps explained

This page documents exactly how timing is extracted from the ThorSync `.h5`
file, what the resulting arrays mean, and how to get **per-plane timestamps for
volumetric recordings** — the thing downstream volumetric analyses should use
instead of one time per volume.

## The signals

ThorSync samples a handful of lines at ~30 kHz (the exact rate comes from
`ThorRealTimeDataSettings.xml`, read via `utils2p.synchronization.SyncMetadata.get_freq()`).
Nine lines are loaded, but **only three drive timing**:

| line | h5 group | role |
|---|---|---|
| `frame_counter` | `CI` | hardware counter that increments once per acquired frame (= per z-plane in fast-z). The *sole* source of frame/stack times. |
| `scope_pin` | `DI` (downstairs) / `AI` (upstairs) | microscope acquisition trigger TTL → `scope_ict` / `scope_fct`, and gates which samples count |
| `olf_disp_pin` | `AI` | olfactometer dispense TTL → `olf_ict` / `olf_fct` (stimulus on/off) |

`frame_out` and `frame_in` are loaded but **never used for timing**.
`piezo_monitor`, `pockels_monitor`, and `pid` are QC/plotting only.

The mapping from canonical names to each rig's actual h5 dataset names lives in
[`data/microscopes/<scope>/thorsync_lines.yaml`](directory-layout.md#per-microscope-line-maps)
(current approach). The legacy path fuzzy-matches names instead
(`process_thorsync.get_matched_thorsync_lines`).

## How frame times are computed

In `rye2p/process_thorsync.py::extract_timestamp_data` (legacy) and the
equivalent notebook cells:

1. `sync_times = np.arange(n_samples) / freq` — seconds since ThorSync start.
   **All timestamps in the pipeline are in this clock.**
2. `utils2p.synchronization.process_frame_counter(frame_counter, steps_per_frame=1)`
   labels every sample with the index of the frame being acquired.
3. Samples are masked to where the scope pin is high and the counter is valid,
   then `crop_lines` slices from the first to the last valid sample.
   ⚠️ This is a **contiguous slice, not a boolean mask** — with multiple scope
   pulses in one file, inter-pulse dead time stays in; that's why
   `fix_frame_times.frames_by_scope_pulse` exists to re-group frames per pulse.
4. `get_start_times` takes the time of each counter increment (rising edge).

**Result — `frame_times`:** 1-D float64, one entry per frame-counter step =
**one per z-plane, including flyback planes**. Length is
`n_timepoints × (n_z + n_flyback_frames)` for fast-z.

## How stack (volume) times are computed

Same procedure, but the counter is grouped in blocks of
`steps_per_frame = n_z + n_flyback_frames`, so each volume gets the time of its
**first plane**:

```python
stack_times == frame_times[::steps_per_frame]      # first-plane convention
```

**Result — `stack_times`:** 1-D float64, one entry per volume (`n_timepoints`).

!!! danger "Three conventions coexist for 'which plane stamps the volume'"
    - `process_thorsync.py:217` and the demo notebooks: **first plane** ✅ (and
      the committed example `data/haoru/timestamps.npy` verifies this:
      `frame_times[::4] == stack_times` exactly).
    - `fix_frame_times.fix_timestamps0` (line ~211): takes the **last step** of
      each volume — a systematic offset of `(steps_per_frame − 1) / raw_frame_rate`
      relative to the others. If a movie's timestamps were rescued via
      `fix_timestamps0`, its `stack_times` may be shifted by up to one volume
      period. See [known issues](known-issues.md).
    The roadmap standardizes on the first-plane convention everywhere.

## Flyback planes: the movie and the timestamps disagree

- `frame_times` **contains** flyback-plane timestamps.
- The saved tiff **does not contain flyback planes** — `utils2p.load_raw` trims
  them during raw→tiff conversion.

So for a fast-z movie of shape `(T, n_z, Y, X)`:

- `len(frame_times) == T * (n_z + n_flyback)` → **cannot** index the movie directly.
- `len(stack_times) == T` → indexes volumes 1:1, but throws away within-volume timing.

## Per-plane timestamps (what volumetric analyses should use)

Planes within a volume are *not* simultaneous — in the example data, planes are
~181 ms apart within a ~724 ms volume. A downstream analysis that assigns one
time per volume to every plane carries up to `(n_z−1)/raw_frame_rate` of
systematic timing error on the deepest plane.

`rye2p.thorsync.timestamps.plane_times` gives you a `(T, n_z)` array aligned
with the saved movie:

```python
from rye2p.thorsync.timestamps import load_timestamps, plane_times

ts = load_timestamps(mov_dir / "timestamps.npy")
pt = plane_times(ts["frame_times"], n_z=16, n_flyback_frames=4)

pt.shape         # (n_timepoints, 16) — pt[t, z] = time plane z of volume t was acquired
pt[:, 0]         # == ts["stack_times"]  (first-plane convention)
pt[:, 5]         # the timebase for plane 5 — use this for plane-5 ROI traces
```

`n_z` and `n_flyback_frames` come from `thorimage_settings.yaml`
(`FunctionalImagingSettings`); `plane_times_from_settings(frame_times, settings)`
reads them for you, and also handles single-plane frame-averaged recordings.
Full walkthrough: [Tutorial: per-plane timestamps](tutorials/per-plane-timestamps.md).

## Stimulus and trigger times

Rising/falling edges of the TTL lines, in seconds since ThorSync start
(`ict` = initial crossing time, `fct` = final crossing time):

| keys | line | meaning |
|---|---|---|
| `scope_ict`, `scope_fct` | `scope_pin` | microscope acquisition start/stop (one pair per scope pulse) |
| `olf_ict`, `olf_fct` | `olf_disp_pin` | odor delivery on/off (one pair per trial) |

Edge detection: `utils2p.synchronization.edges` with a threshold. The legacy
path hard-codes a 2.5 V analog threshold; the notebooks normalize each line to
[0, 1] and threshold at 0.5–0.8 — which also works for the downstairs rig's
*digital* scope pin where the 2.5 V assumption breaks (see
[known issues](known-issues.md)).

## Single-plane recordings

For single-plane (no fast-z) acquisitions with frame averaging,
`steps_per_frame = n_averaging` (each saved frame is the average of
`n_averaging` raw frames), and `stack_times` holds the time of the first raw
frame of each averaging block. `FunctionalImagingSettings.from_xml` is the one
place that derives this correctly; the legacy `process_thorsync.py` has a
double-decimation bug here (see [known issues](known-issues.md)).

## What `fix_frame_times.py` repairs

Real recordings misbehave; this module contains the repairs, applied between
`timestamps0.npy` (raw extraction) and `timestamps.npy` (final):

1. **Unequal rising/falling edge counts** — pairs `ict`→`fct` by Hungarian
   assignment, so a falling edge can never precede its rising edge.
2. **Odor pulses outside acquisition** — drops `olf_ict/olf_fct` pairs not
   contained in any scope pulse (`fix_olf_times`; the default repair for fast-z).
3. **Partial volumes** at the end of a scope pulse — `fix_timestamps0`
   regroups `frame_times` per scope pulse, truncates each pulse to whole
   volumes, and adds the keys `split_frame_times`, `split_stack_times`,
   `n_frame_times_per_pulse`, `n_stack_times_per_pulse` (the latter is used as
   tiff split points by `downsample_stack.py`). ⚠️ Uses the last-step stack-time
   convention — see the warning above.
4. **Temporal downsampling** — `downsample_timestamps` / `edit_timestamps_file`
   write `timestamps_tsub{N}.npy` to match a temporally binned movie.
5. **Manual rescue** — the `__main__` block is an interactive, cell-by-cell
   session for hand-fixing individual movies (pads/trims by ±1 timepoint,
   overwrites `timestamps.npy` in place).

## Validation

Counts are checked against `Experiment.xml`:
`len(frame_times) == steps_per_frame × n_timepoints` and
`len(stack_times) == n_timepoints`. ⚠️ In the legacy path this check only
**prints** — a mismatched `timestamps.npy` is still saved. Always eyeball the
printed check, or re-verify downstream (e.g. `scripts/for_yang/check_frame_number.py`,
or cell 8 of `notebooks/convert_thorimage_metadata_settings.ipynb`).
