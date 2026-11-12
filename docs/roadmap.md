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
- Standardize **first-plane** stack-time convention everywhere; fix
  `fix_timestamps0`'s last-step indexing.
- Normalize lines to [0,1] before edge detection (fixes digital scope pin).
- Count validation **raises** by default (`strict=True`).
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

## Done so far (cleanup branch)

- ✅ Checkpoint commit of all WIP (`67ee170`); `master` preserved.
- ✅ `pyproject.toml` replaces `setup.py` (now packages the subpackages); junk
  removed (`None` PDF, `.run/`, empty Sphinx skeleton); `.gitignore` fixed so
  `docs/` is tracked.
- ✅ `rye2p/thorsync/timestamps.py` — `load_timestamps`, `plane_times`,
  `plane_times_from_settings` + 7 passing tests (`tests/test_plane_times.py`).
- ✅ This documentation site (mkdocs-material + mkdocstrings), README landing
  page, and a local (untracked) AI-assistant orientation file.
