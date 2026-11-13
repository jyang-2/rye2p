# Tutorial: per-plane timestamps for volumetric analysis

**Problem:** in a fast-z (volumetric) recording, the planes of one volume are
acquired sequentially — in the worked example below, ~181 ms apart within a
~724 ms volume. Downstream analyses that use `stack_times` assign the *first*
plane's time to *every* plane, so an ROI on the deepest plane carries a
systematic timing error of `(n_z − 1) / raw_frame_rate`. For trial-aligned
averaging or response-latency estimates, that error can be a large fraction of
a volume period.

**Solution:** `frame_times` already contains one timestamp per acquired plane.
`plane_times` reshapes it into a `(n_timepoints, n_z)` array aligned with the
saved movie.

## 1. Load timestamps and settings

```python
from pathlib import Path
from rye2p.thorsync.timestamps import load_timestamps, plane_times
from rye2p.thorimage.settings import FunctionalImagingSettings

mov_dir = Path("/path/to/processed_data/2025-02-11/1/kiwi_ea_eb_only")

ts = load_timestamps(mov_dir / "timestamps.npy")
settings = FunctionalImagingSettings.from_yaml(mov_dir / "thorimage_settings.yaml")
```

## 2. Build the per-plane time array

```python
pt = plane_times(ts["frame_times"], settings.n_z, settings.n_flyback_frames)
# or equivalently: plane_times_from_settings(ts["frame_times"], settings)

pt.shape          # (n_timepoints, n_z) — matches the tiff's (T, Z) axes
```

Guarantees:

- `pt[t, z]` = acquisition time (seconds, ThorSync clock) of plane `z` in
  volume `t` of `Image_001_001.tif` — flyback planes are dropped, matching the
  saved movie exactly.
- `pt[:, 0] == ts["stack_times"]` (first-plane convention — holds for
  notebook-produced files; see the warning below for legacy/repaired files).
- Raises `ValueError` if `frame_times` isn't a whole number of volumes — that
  means the timestamps weren't repaired; see
  [Timestamps explained](../timestamps.md#what-fix_frame_timespy-repairs).

## 3. Use the right timebase per plane

Suite2p/caiman traces from plane `z` should be interpolated/aligned on
`pt[:, z]`, not `stack_times`:

```python
import numpy as np

# F: (n_rois, n_timepoints) traces from plane z
t_plane = pt[:, z]

# trial-align on a common window around each odor onset
pre, post = 5.0, 20.0                      # seconds
trial_ts = np.arange(-pre, post, 1 / settings.frame_rate)

from scipy.interpolate import interp1d
trials = np.stack([
    interp1d(t_plane, F, axis=-1, bounds_error=False)(ict + trial_ts)
    for ict in ts["olf_ict"]
])   # (n_trials, n_rois, n_timepoints_trial)
```

Because every plane is interpolated onto the same `trial_ts` grid *using its own
acquisition times*, ROIs from different planes are now directly comparable —
this is the adaptation your volumetric analysis needs. (Compare
`rye2p.trial_tensors.split_traces_to_trials`, which does the same interpolation
with a single timebase.)

## 4. Sanity checks worth keeping in your analysis

```python
assert pt.shape[0] == movie.shape[0] and pt.shape[1] == movie.shape[1]  # (T, Z) match
np.testing.assert_allclose(pt[:, 0], ts["stack_times"])                 # convention check
dz = np.diff(pt, axis=1)
assert (dz > 0).all()                       # planes strictly ordered within a volume
```

!!! warning
    If `pt[:, 0]` does **not** equal `stack_times`, the file came from one of
    the two non-first-plane producers
    ([known issues #2 and #22](../known-issues.md)):

    - **Legacy `process_thorsync.py`** (most `odor_space_collab`-era files):
      `stack_times[i]` is the first plane of volume **i+1** (~one volume
      period late), and `frame_times` itself is one frame (~17 ms) late —
      these files usually also fail the divisibility check in `plane_times`
      and need repair first.
    - **`fix_timestamps0`-repaired files**: volumes are stamped with the
      *last* step instead of the first.

    In either case trust `frame_times` + `plane_times` over `stack_times`
    (worst case ~one frame of residual error instead of ~one volume).

## Worked example data

`data/haoru/timestamps.npy` in this repo: `frame_times` (4964,), `stack_times`
(1241,) → `steps_per_frame = 4`; `plane_times(ft, 4)` gives `(1241, 4)` with
column 0 exactly equal to `stack_times`. The test
`tests/test_plane_times.py::test_real_example_first_plane_equals_stack_times`
verifies this.
