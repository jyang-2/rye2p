# Timestamps explained

This page documents exactly how timing is extracted from the ThorSync `.h5`
file, what the resulting arrays mean, and how to get **per-plane timestamps for
volumetric recordings** — the thing downstream volumetric analyses should use
instead of one time per volume.

## The signals

ThorSync samples every line at a common rate — nominally 30 kHz; the exact
configured rate comes from `ThorRealTimeDataSettings.xml`, read via
`utils2p.synchronization.SyncMetadata.get_freq()` (the ground-truthed upstairs
acquisition below returns exactly `30000`). Nine canonical lines are loaded,
but **only three drive timing**:

| line | h5 group | role |
|---|---|---|
| `frame_counter` | `CI` | hardware counter that increments once per acquired frame (= per z-plane in fast-z). The *sole* source of frame/stack times. |
| `scope_pin` | `DI` (downstairs, digital 0/1) / `AI` (upstairs, analog ~0–5 V) | microscope acquisition trigger TTL → `scope_ict` / `scope_fct`, and gates which samples count |
| `olf_disp_pin` | `AI` | olfactometer dispense TTL → `olf_ict` / `olf_fct` (stimulus on/off) |

Of the other six: `frame_out` and `frame_in` are loaded but **never used for
timing**; `lightpath_shutter` is loaded and cleaned (`correct_split_edges`) but
also unused for timing; `piezo_monitor`, `pockels_monitor`, and `pid` are
QC/plotting only.

The mapping from canonical names to each rig's actual h5 dataset names lives in
[`data/microscopes/<scope>/thorsync_lines.yaml`](directory-layout.md#per-microscope-line-maps)
(current approach). The legacy path fuzzy-matches names instead
(`process_thorsync.get_matched_thorsync_lines`).

### Line inventory (ground-truthed)

Everything actually inside a ThorSync `.h5`, per rig. Upstairs was verified
2026-11-12 against a real acquisition
(`odor_space_collab/raw_data/2023-04-11/2/SyncData003/Episode001.h5`,
51,519,000 samples @ 30 kHz — a single-plane recording; fast-z–specific
behaviors were verified on `SyncData001` of the same fly). The downstairs
listing is the dump recorded in
`data/microscopes/downstairs-2p/thorsync_lines.yaml`.

**Upstairs** (11 datasets — exactly the 9 canonical lines plus `Freq/` and `Global/`):

| dataset | group | dtype | observed (verified acquisition) |
|---|---|---|---|
| `PiezoMonitor` | `AI` | f8 | analog readout of the piezo z position. On the fast-z episode (`SyncData001`): a clean per-volume staircase — 16 ascending steps of ~34 mV (one per imaging plane) + a 4-frame flyback descent, per-plane values repeatable to ~1–2 mV → each plane's piezo level is distinguishable, usable as an independent check of plane identity / volume phase. Near-constant on single-plane recordings (as in `SyncData003`) |
| `Pockels1Monitor` | `AI` | f8 | 0.03–0.38 V, analog |
| `lightpathshutter` | `AI` | f8 | 0–4.0 V TTL; 2 pulses bracketing the acquisition |
| `olfDispPin` | `AI` | f8 | 0–4.9 V TTL; one pulse per stimulus (51 pulses = 51 stimuli in `stim_list.json`) |
| `pid` | `AI` | f8 | 0.07–4.1 V analog |
| `scopePin` | `AI` | f8 | 0–4.9 V **analog** TTL; one pulse spanning the acquisition — the 2.5 V legacy threshold is valid upstairs |
| `FrameCounter` | `CI` | u4 | monotonic, one increment per physical frame |
| `FrameIn` | `DI` | u4 | **constant 0** (unconnected) |
| `FrameOut` | `DI` | u4 | toggles **0↔2** (not 0/1 — the legacy `frame_out > 0` binarization handles this); one pulse per physical frame, count equals `FrameCounter` max |
| `FitHz`, `Hz` | `Freq` | f8 | 3,000 samples, all zero (unused ThorSync feature; not a line) |
| `GCtr` | `Global` | u8 | the **20 MHz board timebase latched at every sample** (~666.7 ticks/sample), *not* a frame counter; nothing in the pipeline reads it (the one reader, `thorsync/utils.read_gctr`, is broken — see [known issues](known-issues.md)) |

!!! note "Clock accuracy"
    In the verified acquisition, `GCtr` advanced 666.60 ticks/sample, i.e. the
    20 MHz timebase and the nominal 30 kHz sample clock disagree by ~1×10⁻⁴
    (equivalently: if `GCtr` is exactly 20 MHz, the real sample rate was
    ≈30,003 Hz). All pipeline timestamps trust the nominal rate from
    `ThorRealTimeDataSettings.xml`; the discrepancy is ≤ ~0.2 s drift over a
    ~29 min recording and currently ignored.

**Downstairs** (18 datasets) adds rig-specific extras, none read by the
pipeline: `ai2`, `ai5`, `scopePin_OLD` (the analog survivor of the digital-pin
rewire) in `AI`; `Bleach Complete`, `Bleach Out`, `Digital 01`, `Line 4`,
`Line 7` in `DI`. There `scopePin` is **digital u4** (hence the normalized
thresholds in the notebooks), `lightpath_shutter` maps to `flipperMirror`, and
there is no `Freq/` group. Full dump in
`data/microscopes/downstairs-2p/thorsync_lines.yaml`.

## How frame times are computed

In `rye2p/process_thorsync.py::extract_timestamp_data` (legacy):

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

!!! danger "The legacy path drops the first frame (and first volume)"
    `get_start_times` is called with the default `zero_based_counter=False` on
    the **cropped** counter, which already sits at frame 0 at the crop
    boundary — so frame 0 has no rising edge and is silently dropped:
    `frame_times[j]` is really the time of physical frame **j+1**, one frame
    (~17 ms) late everywhere, and the same mechanism shifts `stack_times` by a
    whole volume (see the convention warning below). Ground-truthed 2026-11-12
    by bit-exact reproduction of a real `timestamps.npy`
    (`odor_space_collab` 2023-04-11 fly 2 `megamat1`): saved length was 90,749
    for an acquisition whose counter recorded 90,752 physical frames (first
    frame dropped at the crop boundary, 2 dropped after the scope pin fell),
    vs. 90,740 frames in the saved movie. Known issue #22.

    The notebook path does **not** have this bug: it runs `get_start_times`
    on the raw (uncropped) counter, where frame 0 produces a genuine 0→1
    edge.

The notebook path (`notebooks/thorsync_lines_demo.ipynb`) differs from the
algorithm above in more ways than the fix: it skips `process_frame_counter`,
the scope-pin mask, and `crop_lines` entirely (it re-groups frames per scope
pulse afterwards instead), and — after cell 17 reloads `sync_lines.npz` — it
computes everything on lines **decimated to 1 kHz**, so its edge times
(`scope/olf ict/fct`) are quantized to 1 ms rather than the 1/30 ms of the
legacy path. Frame times still come from counter values, which survive
decimation.

**Result — `frame_times`:** 1-D float64, one entry per frame-counter step =
**one per z-plane, including flyback planes**. Length is
`n_timepoints × (n_z + n_flyback_frames)` for fast-z — in practice the legacy
length is off by the dropped/partial frames above and the saved movie length,
so **validate, don't assume** (see [Validation](#validation)).

## How stack (volume) times are computed

Same procedure, but the counter is grouped in blocks of
`steps_per_frame = n_z + n_flyback_frames`, so each volume gets the time of its
**first plane**:

```python
stack_times == frame_times[::steps_per_frame]      # first-plane convention
```

**Result — `stack_times`:** 1-D float64, one entry per volume (`n_timepoints`).

!!! danger "Three conventions coexist for 'which plane stamps the volume'"
    Ground-truthed 2026-11-12 against the raw ThorSync h5 + saved outputs of
    `odor_space_collab` 2023-04-11 fly 2 `megamat1`:

    - **Demo notebooks: true first plane** ✅ — `stack_times = frame_times[::steps_per_frame]`
      by construction, on an unshifted `frame_times`. The committed example
      `data/haoru/timestamps.npy` verifies this (`frame_times[::4] == stack_times`
      exactly). This is the convention the roadmap standardizes on.
    - **Legacy `process_thorsync.py:217`: first plane of the *next* volume** —
      because of the dropped first frame/volume above, `stack_times[i]` is the
      first-plane time of volume **i+1**: one full volume period late
      (~337 ms in the verified acquisition). Numerically,
      `stack_times == frame_times[19::20]` within the file's own (shifted)
      arrays — which masquerades as a "last step" convention. Previously this
      path was wrongly documented here as first-plane.
    - **`fix_frame_times.fix_timestamps0` (line ~211): last step** of each
      volume — `(steps_per_frame − 1) / raw_frame_rate` after the true first
      plane (one frame *earlier* than what the legacy extractor produces).

    Consequence: any `timestamps.npy` produced by the legacy path (most
    `odor_space_collab`-era files) has every `stack_times[t]` ≈ one volume
    period later than movie volume `t`'s true first-plane time. See
    [known issues](known-issues.md) #2 and #22.

## Flyback planes: the movie and the timestamps disagree

- `frame_times` **contains** flyback-plane timestamps.
- The saved tiff **does not contain flyback planes** — `utils2p.load_raw` trims
  them during raw→tiff conversion.

So for a fast-z movie of shape `(T, n_z, Y, X)`:

- `len(frame_times) == T * (n_z + n_flyback)` → **cannot** index the movie directly.
- `len(stack_times) == T` → indexes volumes 1:1, but throws away within-volume timing.

### Which acquired frames reach the movie

A frame being acquired (counted by `FrameCounter`) does **not** mean it is in
the saved `.raw`/tiff. ThorImage records while the `scopePin` trigger is high;
the piezo steps through its programmed z-sweep regardless of where the trigger
falls, and the scanner keeps running for a few frames (~2–3 observed) after
the trigger drops. The save rules, as verified on the 2023-04-11 fly 2
acquisitions (raw file sizes match `Experiment.xml` `frames` **byte-exactly**
in all three recordings — `bytes / (pixelX × pixelY × 2)`):

- **Saving starts at the very first acquired frame** (no frames dropped at the
  start).
- **Fast-z: frames from an incomplete volumetric sweep are not written.**
  `megamat1`: 90,752 frames counted, ~90,750 within the scope window, but the
  movie holds exactly 90,740 = 4,537 complete volumes × 20 steps — the
  12-frame partial 4,538th sweep was discarded entirely, including its ~9
  frames that fell *inside* the scope window.
- **Single-plane: every frame whose acquisition started inside the scope
  window is saved.** `megamat1_calyx_001`: 92,127 counted → 92,125 saved =
  exactly the frames that started before the trigger fell.

**Resolved for averaging blocks (2026-11-19)**: a block that *starts* inside
the window and *completes* after the trigger falls **is kept**. `megamat1_calyx`
(4× averaging): only 92,125 raw frames started inside the scope window, but the
movie holds 23,032 × 4 = 92,128 — the final block (f92124–f92127) finished
~3 frames after the trigger fell and was saved anyway. So completion is the
criterion, not full in-window containment. Still unverified for fast-z
z-sweeps (no acquisition where the trigger fell inside a sweep's last frames),
but the same rule is likely.

The robust recipe (what the notebook path implements): crop frame times to the
scope-pin window **with a few frame-periods of padding at the end**, then keep
the first `n_timepoints × steps_per_frame` of them, taking `n_timepoints` from
`Experiment.xml` / `thorimage_settings.yaml` — which is ground truth for the
file's actual content (byte-exact, above). Count from the *true first frame*
(the legacy path loses it — known issue #22). The `PiezoMonitor` staircase
(see the line inventory) gives an independent empirical check that frame 0 of
each kept block really is the top/first plane.

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
pt[:, 0]         # == ts["stack_times"] for notebook-produced files (first-plane
                 # convention — legacy/repaired files differ, see warning above)
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

Edge detection: `utils2p.synchronization.edges(line, size=...)` — and `size`
is the magnitude of the **sample-to-sample jump**, *not* an absolute level
threshold. The legacy path uses `size=2.5`, which works on the upstairs rig's
analog ~0–5 V TTLs (verified: a 0→4.9 V step is one 4.9-unit jump) but finds
**zero edges** on a digital 0/1 line, where every jump has magnitude 1 (see
[known issues](known-issues.md) #3). The notebooks instead normalize each line
by its max — making every real transition a jump of ~1.0 — and use
`size=0.5–0.8`, which works on both rigs precisely *because* of the jump
semantics. Two further notebook quirks to know about: edges are computed on
the **1 kHz-decimated** lines (1 ms quantization, see above), and the scope
falling-edge cell drops the first falling edge (`scope_fct = ...[1:]`) as an
undocumented rig-specific hack.

## Single-plane recordings

For single-plane (no fast-z) acquisitions with frame averaging,
`steps_per_frame = n_averaging` (each saved frame is the average of
`n_averaging` raw frames), and `stack_times` holds the time of the first raw
frame of each averaging block. `FunctionalImagingSettings.from_xml` is the one
place that derives this correctly; the legacy `process_thorsync.py` has a
double-decimation bug here (see [known issues](known-issues.md)).

!!! danger "What single-plane files in the wild actually contain (ground-truthed 2026-11-12)"
    Verified on `odor_space_collab` 2023-04-11 fly 2's two calyx recordings
    (`megamat1_calyx`, 4× averaging, 23,032 timepoints; `megamat1_calyx_001`,
    no averaging, 92,125 timepoints):

    - **`timestamps0.npy`** carries both extraction bugs: `frame_times` is
      missing the first physical frame (issue #22), and `stack_times` is
      garbage — 4,606 entries from `steps_per_frame = n_z + n_flyback = 20`,
      because `ZStage/steps` reads 16 even when fast-z is off (issue #1).
      (`thorimage_settings.yaml` gets it right: `fast_z: false`, `n_z: 1`,
      `steps_per_frame = n_averaging`.)
    - **`timestamps.npy`** was repaired by hand via the
      `fix_frame_times.py` `__main__` rescue scratchpad (≈ lines 383–414):
      `stack_times = frame_times[::n_averaging]`, and because that comes up
      one short (the dropped first frame), **one extrapolated entry
      (`last + mean dt`) is appended** to reach `n_timepoints`. Verified:
      `stack_times[:-1] == frame_times[::n_averaging]` exactly in both files.
    - Net effect: every saved frame's timestamp is **one raw frame
      (~16.6 ms) late** (with averaging: the *second* raw frame of each block,
      not the documented first), the final entry is synthetic, and
      `frame_times` was left unrepaired, so
      `len(frame_times) != len(stack_times) × n_averaging` — the file is
      internally inconsistent (issue #24).

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
**prints** — a mismatched `timestamps.npy` is still saved. This is not
hypothetical: the verified `odor_space_collab` 2023-04-11/2/megamat1 file was
saved with `len(frame_times) == 90749` against an expected 90,740 (not even
divisible by `steps_per_frame=20` — `plane_times` refuses it outright). Always
eyeball the printed check, or re-verify downstream (e.g.
`scripts/for_yang/check_frame_number.py`, or cell 8 of
`notebooks/convert_thorimage_metadata_settings.ipynb`).

Also validate the **pairing** of ThorSync episode ↔ ThorImage folder before
trusting any timestamps: nothing in either file names the other. Three cheap
fingerprints identify the right episode: acquisition order/time of day,
`FrameCounter`'s final value vs. the movie's frame count, and the realized
frame rate (median counter-increment spacing) vs. `frameRate` in
`Experiment.xml`. During the ground-truth check, these fingerprints showed the
2023-04-11 fly 2 pairing is `megamat1`↔`SyncData001` (not `SyncData003`,
which belongs to `megamat1_calyx_001`).
