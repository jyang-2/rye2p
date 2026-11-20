# Refactor roadmap

Goal: one obvious, importable path per task; notebooks become thin
demonstrations of library functions; the legacy and notebook pipelines unify.
Work happens on the `cleanup` branch in small, reviewable commits — each step
leaves the repo working.

## Target package shape

```
rye2p/
├── config.py            # explicit ProjectPaths (attrs) — replaces datadirs globals
├── thorimage/
│   ├── settings.py      # FunctionalImagingSettings (exists, keep)
│   └── io.py            # load_raw, save_tiff, split_tiff (merge convert_raw_fn_thorimage + pipeline/raw)
├── thorsync/
│   ├── linenames.py     # YAML per-rig maps (primary) + Hungarian fuzzy match (fallback)
│   ├── lines.py         # load + normalize sync lines, edge detection (normalize-first)
│   ├── timestamps.py    # frame/stack/plane times (plane_times exists, add extraction)
│   └── repair.py        # fix_frame_times contents, minus the __main__ scratchpad
├── stimuli.py           # olf_configs → df_stim / df_pins2odors / stim_list
├── matching.py          # thormatch
├── movie.py             # pure-numpy movie ops (exists, keep)
└── pipeline/
    └── preprocess.py    # preprocess_thorimage + process_thorsync orchestration, CLI entry
```

One-off/personal code moves to `scripts/` (kept, not packaged); dead code is
deleted (it stays recoverable in git history).

## Steps

### 1. Promote the notebook pipeline into the library  *(unblocks everything)*
- Move `preprocess_thorimage()` from `notebooks/pipeline_demo.ipynb` into
  `rye2p/pipeline/preprocess.py`; make `pipeline/raw.py` re-export from
  `thorimage/io.py` instead of duplicating it.
- Make `thorsync/linenames.py` read `data/microscopes/*/thorsync_lines.yaml`
  (move under `rye2p/data/` as package data); fuzzy match becomes fallback only.
- Promote the frame-count validator from
  `notebooks/convert_thorimage_metadata_settings.ipynb` cell 8.

### 2. Fix timing correctness  *(the science-critical step)*
- All `steps_per_frame` derivations read `FunctionalImagingSettings` (fixes the
  single-plane double decimation).
- Fix the legacy dropped-first-frame/volume shift (known issue #22:
  `zero_based_counter=False` on cropped counters) — the largest systematic
  error found (one volume period, ~337 ms).
- Standardize **first-plane** stack-time convention everywhere; fix
  `fix_timestamps0`'s last-step indexing.
- Normalize lines to [0,1] before edge detection (fixes digital scope pin);
  compute edges at full rate, not on 1 kHz-decimated lines (known issue #23).
- Count validation **raises** by default (`strict=True`); validate the
  ThorSync↔ThorImage pairing by fingerprint (counter max, realized frame
  rate) — see [Validation](timestamps.md#validation).
- Add regression tests using small synthetic counter/TTL traces + the
  `data/haoru` example.

### 3. Deduplicate and guard
- Delete the verbatim duplicates (§21 of [known issues](known-issues.md)).
- `__main__`-guard or strip all import-time execution (`master.py` → retire;
  `datadirs` → `config.py`; `pid.py` repairs; `scrape_chem.py` → scripts/).
- Fix the small catalogued bugs (`read_gctr`, `tsub,` tuples, broken script
  imports).

### 4. Environment modernization  *(separate, explicit step)*
- Drop pydantic (dead code only); drop `fuzzywuzzy` if YAML maps suffice;
  `pendulum` → stdlib.
- numpy 2 compatibility (`np.Inf` → `np.inf`), refresh pins, test against a
  fresh env.
- Decide the `utils2p` strategy: keep pinned, vendor, or wrap narrowly.

### 5. Deprecation shims + downstream migration
- Old import paths (`rye2p.process_thorsync`, `rye2p.convert_raw_fn_thorimage`,
  `rye2p.datadirs`, `rye2p.fix_frame_times`) become thin re-export shims with
  `DeprecationWarning`s so existing analysis code keeps running.
- Migrate downstream volumetric analyses from `stack_times` to
  [`plane_times`](timestamps.md#per-plane-timestamps-what-volumetric-analyses-should-use).

### 6. Docs & tutorials finish
- Convert the three tutorial pages into executed notebooks (or mkdocs-jupyter)
  against the `data/haoru` example.
- API docstrings pass over the surviving modules.

### 7. Release for other people  *(after 1–5)*
- Quickstart for outsiders: install → tiny bundled example → first
  `timestamps.npy` in ten minutes, without access to the lab's data roots.
- `pip install rye2p` from PyPI (pyproject is ready; blockers are the fixes
  above plus the `utils2p` dependency decision from step 4).
- Versioning + changelog; publish the docs site (e.g. GitHub Pages via the
  existing CI).

## Done so far (cleanup branch)

- ✅ Checkpoint commit of all WIP (`67ee170`); `master` preserved.
- ✅ `pyproject.toml` replaces `setup.py` (now packages the subpackages); junk
  removed (`None` PDF, `.run/`, empty Sphinx skeleton); `.gitignore` fixed so
  `docs/` is tracked.
- ✅ `rye2p/thorsync/timestamps.py` — `load_timestamps`, `plane_times`,
  `plane_times_from_settings` + 7 passing tests (`tests/test_plane_times.py`).
- ✅ This documentation site (mkdocs-material + mkdocstrings), README landing
  page, and a local (untracked) AI-assistant orientation file.
- ✅ Step 1 (partial): raw→tiff functions consolidated into
  `rye2p/thorimage/io.py` (`load_raw`, `copy_raw_dir_2_proc_dir`,
  `convert_raw_to_tiff`, `split_tiff_files` + `overwrite` flag);
  `pipeline/raw.py` and `convert_raw_fn_thorimage.py` are now re-export shims
  (first part of known issue #21 closed).
- ✅ Step 1 (partial): `preprocess_thorimage` + `validate_thorimage_folder`
  promoted into `rye2p/pipeline/preprocess.py` (fixes the raw-filename and
  `copy_xml=False` traps — known issue #25); `pipeline_demo.ipynb` is now a
  thin driver importing from the library.
- ✅ Step 1 (partial): per-rig line maps moved to
  `rye2p/data/microscopes/*/thorsync_lines.yaml` (package data);
  `thorsync/linenames.py` gains `load_line_map` + `match_lines` (YAML primary,
  Hungarian fuzzy matching fallback-only, heavy imports now lazy) + tests.
- ✅ Raw-data write guard (`rye2p/pathguard.py`): every output-root write
  path (`preprocess_thorimage`, `thorimage.io.*`, legacy `process_thorsync`
  writers) refuses destinations under `tensor-archive`/`raw_data` or inside
  a folder holding a raw acquisition file; `RYE2P_ALLOW_RAW_WRITES=1`
  downgrades to a warning. Prompted by the 2026-11-19 write-path audit
  (tensor-archive is mounted rw with open permissions — no OS backstop).
- ✅ **Step 1 complete**: frame-count validator promoted into
  `thorsync/timestamps.py` (`frames_by_scope_pulse`, `expected_frame_counts`,
  `validate_frame_counts(..., strict=False)` — warns now, step 2 flips the
  default to raise; covers single-plane/averaging, not just fast-z) + tests;
  both notebook copies of the validator cell now call the library.
- ✅ ThorSync ground-truth check against a real upstairs acquisition
  (`odor_space_collab` 2023-04-11 fly 2): line inventory added to
  [timestamps.md](timestamps.md) and both `thorsync_lines.yaml` files; legacy
  extraction reproduced bit-exactly; found known issues #22 (legacy drops
  first frame/volume → stack times one volume late) and #23 (notebook edge
  times 1 ms-quantized); conventions re-attributed in issue #2.
