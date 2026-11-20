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
[`rye2p/data/microscopes/<scope>/thorsync_lines.yaml`](directory-layout.md#per-microscope-line-maps)
(current approach). The legacy path fuzzy-matches names instead
(`process_thorsync.get_matched_thorsync_lines`).

### Line inventory (ground-truthed)

Everything actually inside a ThorSync `.h5`, per rig. Upstairs was verified
2026-11-12 against a real acquisition
(`odor_space_collab/raw_data/2023-04-11/2/SyncData003/Episode001.h5`,
51,519,000 samples @ 30 kHz — a single-plane recording; fast-z–specific
behaviors were verified on `SyncData001` of the same fly). The downstairs
listing is the dump recorded in
`rye2p/data/microscopes/downstairs-2p/thorsync_lines.yaml`.

**Upstairs** (11 datasets — exactly the 9 canonical lines plus `Freq/` and `Global/`):

| dataset | group | dtype | observed (verified acquisition) |
|---|---|---|---|
| `PiezoMonitor` | `AI` | f8 | analog readout of the piezo z position. On the fast-z episode (`SyncData001`), verified with counter-aligned volume phase: planes 0–15 ascend ~34 mV/step from the bottom; the 4 flyback frames are **one further step up to the max, then a 3-frame descent** back to the bottom. The monitor settles ~5–8 ms after each step (about a third of a frame period) — sample mid-frame, not at a fixed early offset. Per-step values repeatable to ~1–2 mV → each plane's piezo level is distinguishable, usable as an independent check of plane identity / volume phase (`FrameOut` cannot do this — it fires identically for every frame, flyback included). Near-constant on single-plane recordings (as in `SyncData003`) |
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
`rye2p/data/microscopes/downstairs-2p/thorsync_lines.yaml`.

## How frame times are computed

Since roadmap step 2, all paths run the **unified extraction**
(`rye2p.thorsync.timestamps.extract_timestamps`, ported from the ThorSync GUI
after end-to-end verification on all three 2023-04-11/2 recording types):

1. Sample `k` of a line is at `k / fs` seconds — seconds since ThorSync start.
   **All timestamps in the pipeline are in this clock.**
2. `counter_increment_times` stamps every increment of the **raw** counter
   with the time of the first sample at the new count — so frame 0 (the
   counter's first step off zero) gets a timestamp.
3. Scope/olf pulses come from `rye2p.thorsync.lines.pulse_times`
   (max-normalized lines, 0.5 threshold, full sample rate), so analog
   (0–5 V) and digital (0/1) pins behave identically.
4. Frames are grouped per scope pulse (`frames_by_scope_pulse`); fast-z
   truncates each pulse to whole volumes, single-plane keeps all in-window
   frames (ThorImage saves a started averaging block — see
   [Validation](#validation)); `stack_times` is `kept[::steps_per_frame]`
   per pulse, with `steps_per_frame` read from `FunctionalImagingSettings`.
5. Counts are validated against `Experiment.xml` (raises on mismatch) and
   the ThorSync↔ThorImage pairing is fingerprint-checked.

!!! danger "The pre-step-2 legacy path dropped the first frame (and first volume)"
    `extract_timestamp_data` used to call `get_start_times` with the default
    `zero_based_counter=False` on a **cropped** counter, which already sat at
    frame 0 at the crop boundary — so frame 0 had no rising edge and was
    silently dropped: `frame_times[j]` was really the time of physical frame
    **j+1**, one frame (~17 ms) late everywhere, and the same mechanism
    shifted `stack_times` by a whole volume (see the convention warning
    below). Ground-truthed 2026-11-12 by bit-exact reproduction of a real
    `timestamps.npy` (`odor_space_collab` 2023-04-11 fly 2 `megamat1`): saved
    length was 90,749 for an acquisition whose counter recorded 90,752
    physical frames, vs. 90,740 frames in the saved movie. Known issue #22 —
    **fixed in the extraction, but every `timestamps.npy` on disk written by
    the legacy path still carries the shift** (re-extract, or shift on load).

The demo notebook (`notebooks/thorsync_lines_demo.ipynb`) now calls the same
library functions. Before step 2 it computed edge times on lines **decimated
to 1 kHz** (1 ms quantization, known issue #23) and needed an unexplained
`scope_fct[1:]` hack; both went away with full-rate edge detection.

**Result — `frame_times`:** 1-D float64, one entry per frame-counter step =
**one per z-plane, including flyback planes**. Length is
`n_timepoints × (n_z + n_flyback_frames)` for fast-z — for pre-step-2 files
the length is off by the dropped/partial frames above,
so **validate, don't assume** (see [Validation](#validation)).

## How stack (volume) times are computed

Same procedure, but the counter is grouped in blocks of
`steps_per_frame = n_z + n_flyback_frames`, so each volume gets the time of its
**first plane**:

```python
stack_times == frame_times[::steps_per_frame]      # first-plane convention
```

**Result — `stack_times`:** 1-D float64, one entry per volume (`n_timepoints`).

!!! danger "Files on disk: three conventions for 'which plane stamps the volume'"
    Since roadmap step 2 **all code paths produce the true first-plane
    convention** (extraction and `fix_timestamps0` alike). But files written
    earlier differ by which code produced them — ground-truthed 2026-11-12
    against the raw ThorSync h5 + saved outputs of `odor_space_collab`
    2023-04-11 fly 2 `megamat1`:

    - **Demo notebooks: true first plane** ✅ — `stack_times = frame_times[::steps_per_frame]`
      by construction, on an unshifted `frame_times`. The committed example
      `data/haoru/timestamps.npy` verifies this (`frame_times[::4] == stack_times`
      exactly). This is the convention everything now standardizes on.
    - **Pre-step-2 legacy extraction: first plane of the *next* volume** —
      because of the dropped first frame/volume above, `stack_times[i]` is the
      first-plane time of volume **i+1**: one full volume period late
      (~337 ms in the verified acquisition). Numerically,
      `stack_times == frame_times[19::20]` within the file's own (shifted)
      arrays — which masquerades as a "last step" convention.
    - **Pre-step-2 `fix_frame_times.fix_timestamps0`: last step** of each
      volume — `(steps_per_frame − 1) / raw_frame_rate` after the true first
      plane (one frame *earlier* than what the legacy extractor produces).

    Consequence: any `timestamps.npy` produced by the pre-step-2 legacy path
    (most `odor_space_collab`-era files) has every `stack_times[t]` ≈ one
    volume period later than movie volume `t`'s true first-plane time. See
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

Edge detection now lives in
[`rye2p.thorsync.lines`](api/thorsync-timestamps.md): every line is
max-normalized to [0, 1] first, then thresholded at 0.5
(`pulse_times` / `edge_times`, full sample rate), so analog ~0–5 V and
digital 0/1 pins behave identically.

Historical context for old code and files: the pre-step-2 legacy path used
`utils2p.synchronization.edges(line, size=2.5)` — `size` is the magnitude of
the **sample-to-sample jump**, not a level threshold — which works on the
upstairs rig's analog ~0–5 V TTLs but finds **zero edges** on a digital 0/1
line (known issue #3, fixed). The pre-step-2 notebook computed edges on
**1 kHz-decimated** lines (1 ms quantization) and dropped the first scope
falling edge (`scope_fct = ...[1:]`) as an undocumented hack (known issue
#23, fixed — the notebook now calls the library at full rate).

## Single-plane recordings

For single-plane (no fast-z) acquisitions with frame averaging,
`steps_per_frame = n_averaging` (each saved frame is the average of
`n_averaging` raw frames), and `stack_times` holds the time of the first raw
frame of each averaging block — including a trailing block that starts inside
the scope pulse but completes after the trigger fall, which ThorImage saves
(ground-truthed 2026-11-19 on `megamat1_calyx`: 92,125 in-window raw frames →
23,032 = ceil(92,125/4) saved frames). `FunctionalImagingSettings.from_xml`
derives `steps_per_frame` with both guards (`fast_z`, `averageMode`), and
since roadmap step 2 the extraction reads it from there — the legacy path's
double-decimation bug (known issue #1) is fixed; files it wrote earlier still
carry it.

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
   volumes (fast-z; single-plane keeps all in-window frames), and adds the
   keys `split_frame_times`, `split_stack_times`, `n_frame_times_per_pulse`,
   `n_stack_times_per_pulse` (the latter is used as tiff split points by
   `downsample_stack.py`). Since roadmap step 2 it stamps the **first plane**
   (it used the last step before — see the warning above) — but it is only
   needed for pre-step-2 `timestamps0.npy` files; the unified extraction
   already returns regrouped output, and old inputs still carry the dropped
   first frame (#22), so prefer re-extracting from the h5.
4. **Temporal downsampling** — `downsample_timestamps` / `edit_timestamps_file`
   write `timestamps_tsub{N}.npy` to match a temporally binned movie.
5. **Manual rescue** — the `__main__` block is an interactive, cell-by-cell
   session for hand-fixing individual movies (pads/trims by ±1 timepoint,
   overwrites `timestamps.npy` in place).

## Validation

Counts are checked against `Experiment.xml`:
`len(frame_times) == steps_per_frame × n_timepoints` and
`len(stack_times) == n_timepoints`. Use
[`rye2p.thorsync.timestamps.validate_frame_counts`](api/thorsync-timestamps.md)
(promoted from the old notebook cell in roadmap step 1; works for fast-z *and*
single-plane/averaging recordings):

```python
from rye2p.thorsync.timestamps import validate_frame_counts

validate_frame_counts(frame_times, scope_ict, scope_fct, settings)  # raises on mismatch
validate_frame_counts(..., strict=False)                            # downgrade to warning
```

Since roadmap step 2 the check **raises by default** and the unified
extraction (`thorsync.timestamps.extract_timestamps`) calls it before
returning, so a mismatched `timestamps.npy` can no longer be saved silently.
The validator also encodes ThorImage's per-mode save rules (ground-truthed
2026-11-19): fast-z discards a partial volume at the trigger fall (floor);
single-plane **saves** an averaging block that starts in-window even when it
completes after the fall (ceil), and `Streaming/frames` counts frames in the
.raw file — raw planes for fast-z, *averaged* frames for single-plane.

⚠️ Files written before the fix were never validated. This is not
hypothetical: the verified `odor_space_collab` 2023-04-11/2/megamat1 file was
saved with `len(frame_times) == 90749` against an expected 90,740 (not even
divisible by `steps_per_frame=20` — `plane_times` refuses it outright).

Also validate the **pairing** of ThorSync episode ↔ ThorImage folder before
trusting any timestamps: nothing in either file names the other. Three cheap
fingerprints identify the right episode: acquisition order/time of day,
`FrameCounter`'s final value vs. the movie's frame count, and the realized
frame rate (median counter-increment spacing) vs. `frameRate` in
`Experiment.xml`. During the ground-truth check, these fingerprints showed the
2023-04-11 fly 2 pairing is `megamat1`↔`SyncData001` (not `SyncData003`,
which belongs to `megamat1_calyx_001`). The last two fingerprints are
automated in `thorsync.timestamps.validate_pairing` (raises by default;
calibrated on the three ground-truthed recordings: correct pairings agree to
~0.1 % in rate and overrun the counter by ≤ one partial volume + scanner
tail, the observed wrong pairing was ~1.5 % off in rate), and
`extract_timestamps_from_files` runs it before extracting.
