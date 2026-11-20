# Tutorial: extract timestamps from ThorSync

Produces `timestamps.npy` — frame/stack times, odor on/off times, scope trigger
times — from a ThorSync `.h5`. Concepts behind every step:
[Timestamps explained](../timestamps.md).

!!! note "Current home of this workflow"
    The up-to-date walkthrough is **`notebooks/thorsync_lines_demo.ipynb`**
    (fast-z) and **`scripts/for_ruoyi/thorsync_lines_demo.ipynb`** (single
    plane). The importable `rye2p.process_thorsync` works but has the legacy
    bugs listed in [known issues](../known-issues.md) (fine for upstairs-rig
    fast-z data, which is what it was written for). Unifying them is
    [roadmap](../roadmap.md) step 2.

## Inputs

```
<RAW_ROOT>/<date>/<fly>/<thorsync_name>/
├── Episode001.h5                  # (or Episode_0000.h5)
└── ThorRealTimeDataSettings.xml   # sample rate lives here
<PROC_ROOT>/<date>/<fly>/<thorimage_name>/thorimage_settings.yaml   # from the convert step
rye2p/data/microscopes/<scope>/thorsync_lines.yaml                  # per-rig line names (package data)
```

## The notebook path, step by step

```python
import numpy as np
import utils2p.synchronization as sync

# 1. per-rig line map (canonical name -> h5 dataset name), shipped with the package
from rye2p.thorsync.linenames import load_line_map
line_map = load_line_map("upstairs-2p")

# 2. load lines + sample rate
lines = sync.get_lines_from_sync_file(h5_file, list(line_map.values()))
lines = dict(zip(line_map.keys(), lines))
freq = sync.SyncMetadata(sync_xml).get_freq()          # typically 30000 Hz
sync_times = np.arange(len(lines["frame_counter"])) / freq

# 3. frame times: one per z-plane (incl. flyback), from the hardware frame counter
counter = sync.process_frame_counter(lines["frame_counter"], steps_per_frame=1)
scope   = lines["scope_pin"] / lines["scope_pin"].max()      # normalize: works on analog AND digital rigs
mask = (scope > 0.5) & (counter >= 0)
cropped_times, cropped_counter = sync.crop_lines(mask, [sync_times, counter])
frame_times = sync.get_start_times(cropped_counter, cropped_times)

# 4. trigger/stimulus edges on normalized lines
olf = lines["olf_disp_pin"] / lines["olf_disp_pin"].max()
scope_ict = sync_times[sync.edges(scope, size=0.5)[0]]
scope_fct = sync_times[sync.edges(scope, size=(-np.inf, -0.5))[0]]
olf_ict   = sync_times[sync.edges(olf,   size=0.5)[0]]
olf_fct   = sync_times[sync.edges(olf,   size=(-np.inf, -0.5))[0]]

# 5. group frames per scope pulse, truncate partial volumes, derive stack_times
from rye2p.fix_frame_times import frames_by_scope_pulse
from rye2p.thorimage.settings import FunctionalImagingSettings
settings = FunctionalImagingSettings.from_yaml(proc_mov_dir / "thorimage_settings.yaml")
spf = settings.steps_per_frame

pulses = frames_by_scope_pulse(frame_times, scope_ict, scope_fct)
pulses = [p[: (len(p) // spf) * spf] for p in pulses]        # whole volumes only
frame_times = np.concatenate(pulses)
stack_times = frame_times[::spf]                             # first-plane convention

# 6. validate against ThorImage metadata, then save
assert len(stack_times) == settings.n_timepoints, (len(stack_times), settings.n_timepoints)
timestamps = dict(frame_times=frame_times, stack_times=stack_times,
                  scope_ict=scope_ict, scope_fct=scope_fct,
                  olf_ict=olf_ict, olf_fct=olf_fct)
np.save(proc_mov_dir / "timestamps.npy", timestamps)
```

File contents are specified in the [output file reference](../outputs.md#timestampsnpy).

## Stimulus tables

Parse the olfactometer config so trial *i* ↔ `olf_ict[i]`:

```python
from rye2p import olf_configs
df_pins2odors, df_stim = olf_configs.parse_olf_config_file(olf_config_yaml)
df_stim.to_csv(proc_mov_dir / "df_stim.csv", sep="\t")
df_pins2odors.to_csv(proc_mov_dir / "df_pins2odors.csv", sep="\t")
```

Sanity check: `len(df_stim) == len(olf_ict)`. If there are extra odor pulses
(delivered before/after acquisition), drop the ones outside the scope pulses
with `rye2p.fix_frame_times.fix_olf_times`.

## When counts don't match

Symptoms → repairs (details:
[Timestamps explained](../timestamps.md#what-fix_frame_timespy-repairs)):

| symptom | repair |
|---|---|
| more `olf_ict` than trials | `fix_olf_times` — drop pulses outside acquisition |
| `len(scope_ict) != len(scope_fct)` | `fix_timestamp_pulse_times` — Hungarian edge pairing |
| partial volume at pulse end | truncate per pulse (step 5 above / `fix_timestamps0`) |
| one-off timepoint mismatch | hand-fix: `fix_frame_times.py` `__main__` cells |
