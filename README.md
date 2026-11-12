# rye2p

Convert and parse **ThorImage** / **ThorSync** data from 2-photon imaging
experiments: raw movies → tiffs, sync `.h5` → frame/stack/plane timestamps,
stimulus on/off times, and microscope trigger times.

> 🧭 **Lost? Start here.** This README is the landing page — every important
> file and doc is linked below.

## What do you need right now?

| I want to… | Go to |
|---|---|
| **Process a new imaging day** (raw → tiff) | [`notebooks/pipeline_demo.ipynb`](notebooks/pipeline_demo.ipynb) · [tutorial](docs/tutorials/convert-raw.md) |
| **Extract timestamps** from ThorSync | [`notebooks/thorsync_lines_demo.ipynb`](notebooks/thorsync_lines_demo.ipynb) · [tutorial](docs/tutorials/extract-timestamps.md) |
| **Per-plane timestamps for volumetric analysis** | [`rye2p/thorsync/timestamps.py`](rye2p/thorsync/timestamps.py) · [tutorial](docs/tutorials/per-plane-timestamps.md) |
| Understand frame vs. stack vs. plane times | [docs/timestamps.md](docs/timestamps.md) |
| Know what's in every output file & where it goes | [docs/outputs.md](docs/outputs.md) · [docs/directory-layout.md](docs/directory-layout.md) |
| See a complete worked example (traces → trials) | [`data/haoru/example_trial_extraction.py`](data/haoru/example_trial_extraction.py) |
| Fix broken timestamps for one movie | [`rye2p/fix_frame_times.py`](rye2p/fix_frame_times.py) · [repair guide](docs/timestamps.md#what-fix_frame_timespy-repairs) |
| Know what's buggy/inconsistent before trusting it | [docs/known-issues.md](docs/known-issues.md) |
| See the cleanup plan & current status | [docs/roadmap.md](docs/roadmap.md) |

## The 30-second overview

```
ThorImage folder                      ThorSync folder
  Image_001_001.raw ─┐                  Episode001.h5 ─┐
  Experiment.xml ────┤                                 │
                     ▼                                 ▼
          Image_001_001.tif (T,Z,Y,X)        timestamps.npy
          thorimage_settings.yaml              frame_times  (one per z-plane, incl. flyback)
          stk/stk_000.tif, ...                 stack_times  (one per volume)
                                               olf_ict/fct  (stimulus on/off)
                                               scope_ict/fct (acquisition trigger)
```

Everything lands in `<PROC_ROOT>/<date>/<fly>/<thorimage_name>/`.

⚠️ **Two generations of code coexist** (legacy `process_thorsync.py`/`master.py`
vs. the current notebook-driven path). The notebook path is the one in active
use; unification is in progress on the `cleanup` branch — see the
[roadmap](docs/roadmap.md).

## Setup

```bash
python -m venv .venv
.venv/bin/pip install -e ".[dev,docs]"   # needs utils2p (editable checkout at ~/git/utils2p)
```

Run tests:

```bash
.venv/bin/python -m pytest
```

Browse the documentation site (with live reload):

```bash
.venv/bin/mkdocs serve    # → http://localhost:8000
```

## Repo map

```
rye2p/               the package — see docs/directory-layout.md for the per-module map
notebooks/           the de-facto current pipeline (pipeline_demo, thorsync_lines_demo)
scripts/for_*/       per-person demos & rescue scripts
data/microscopes/    per-rig ThorSync line-name maps (thorsync_lines.yaml)
data/haoru/          small committed worked example (timestamps.npy, df_stim.csv, F.tsv)
tests/               pytest suite
docs/ + mkdocs.yml   documentation site sources
```
