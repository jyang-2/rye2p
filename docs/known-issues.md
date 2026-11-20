# Known issues

Catalogued during the 2026-11 cleanup audit. File:line references are to the
checkpoint commit (`67ee170`). Roadmap steps that fix each are noted.

## Timing correctness (the ones that can silently corrupt science)

1. **Single-plane `stack_times` double decimation** —
   `process_thorsync.py:211` computes `steps_per_frame = n_z + n_flyback`
   with no `fast_z` guard (`get_n_z()` reads `ZStage/steps`, which isn't forced
   to 1 for single-plane recordings), then `:307` decimates *again* by
   `n_averaging`. `FunctionalImagingSettings.from_xml` has the correct logic;
   the notebook path reads it and is trustworthy. Confirmed in the wild
   (2026-11-12): single-plane `timestamps0.npy` files on 2023-04-11/2 carry
   the `steps_per_frame=20` garbage (`stack_times` = 4,606 = frames/20); the
   final files were later hand-rescued (see #24). A second facet (2026-11-19):
   `utils2p.Metadata.get_n_averaging()` returns `LSM/averageNum` without
   checking `averageMode`, so the `:307` decimation fires even with averaging
   **off** — `megamat1_calyx_001` (averageMode=0, averageNum=4) would get
   `stack_times` = 4,606[::4] = 1,152 vs the correct 92,125; re-derivation
   confirms the current end-to-end code produces 1,152 for *both* calyx
   recordings. *Fix: roadmap step 2.*

2. **Stack-time convention conflict** (re-attributed by the 2026-11-12
   ground-truth check, see #22) — the notebooks stamp each volume with its
   true **first** plane's time; `process_thorsync.py:217` produces the first
   plane of the **next** volume (one volume period late — a consequence of
   #22, not of the grouping); `fix_frame_times.py:211` (`fix_timestamps0`)
   uses the **last** step. Three different stamps depending on which code
   produced/repaired the file. *Fix: standardize first-plane, roadmap step 2.*

3. **Analog-voltage assumptions on a digital rig** — `scope_pin > 1` masks and
   hard-coded 2.5 V thresholds (`process_thorsync.py:204, 219, 225, 233`) are
   all-False on the downstairs rig's digital (0/1) `scopePin`, crashing
   `crop_lines` on an empty index. Notebooks normalize to [0,1] and threshold at
   0.5–0.8 instead. *Fix: normalize-first edge detection in the library, roadmap step 2.*

4. **Validation only prints** — count mismatches vs `Experiment.xml` are
   printed, not raised (`process_thorsync.py:297-310`; the `raise`s in
   `fix_timestamps0` are commented out). A wrong `timestamps.npy` still gets
   saved — verified in the wild: `odor_space_collab` 2023-04-11/2/megamat1 was
   saved with 90,749 frame times (expected 90,740; not divisible by
   `steps_per_frame`). *Fix: raise by default, roadmap step 2.*

5. **Fuzzy line matching can double-assign** — the legacy matcher
   (`process_thorsync.py:109`) does per-column argmax with no 1:1 constraint;
   two canonical names can map to the same h5 line. `thorsync/linenames.py:41`
   (Hungarian) fixes this but nothing calls it; the per-rig YAML maps are the
   real successor. *Fix: roadmap step 2.*

## Import-time side effects (hazardous to tooling and humans)

6. **`master.py` runs the pipeline on import** — no `__main__` guard; top-level
   conversion loop over hard-coded paths. Never `import rye2p.master`.
7. **`datadirs.py` selects a default project on import** (`set_prj(None)` →
   `odor_space_collab`) — wrong-project reads/writes if you forget `set_prj`.
8. **`pid.py` plots at import** and its `flacq2dir`/`flacq2rawfiles` reference
   undefined globals (`NAS_PROC_DIR`, `DB_RAW_DIR`) → `NameError` when called.
9. **`scrape_chem.py` makes a PubChem network call at import.**

## Broken / dead code

10. `thorsync/utils.py::read_gctr` returns an h5py dataset handle **after the
    file is closed** — unusable; needs `[:]` before close.
11. `scripts/for_ruoyi/thorsync_lines_demo.py` doesn't compile
    (IndentationError at :192, :212) and `StackTimeCorrector` references
    nonexistent attributes. The notebook version works.
12. `scripts/for_yang/check_frame_number.py` imports `linenames` but calls
    `lines.…` → `NameError`.
13. `olfactometer/shuffle_config.py` crashes at import (`balance_2_pins`
    undefined at :166); `haoru_shuffle.py` is the working rewrite.
14. `manipulate_movies.py` imports `prefect` (not in requirements);
    `rastermodel.py` has a broken import path; `pydantic_models.py` is imported
    by nothing. `thorsync/scope.py` is empty (0 bytes).
15. `downsample_stack.py` — `timestamps_ds['tsub'] = tsub,` trailing commas
    store 1-tuples; `edit_timestamps` (:64) is a stub returning None; ~40 lines
    duplicated across four functions.

## Packaging / environment

16. `requirements.txt` pins `pydantic~=0.18.2` (typo for 1.8.2) and a bogus
    `yaml~=0.2.5` next to `pyyaml`. Pydantic is only used by dead/one-off code —
    the living code uses attrs/cattrs. *Fix: drop pydantic, roadmap step 4.*
17. `np.Inf` (`fix_frame_times.py:17`) was removed in numpy 2.0; current pins
    (`numpy~=1.23`) predate numpy 2. *Roadmap step 4.*
18. Old `setup.py` only packaged the top-level `rye2p` module, so installed
    copies were missing `thorsync/`, `thorimage/`, `pipeline/`, `olfactometer/`.
    **Fixed** in the new `pyproject.toml` (cleanup branch).
19. Two tiff backends in play: writes via `utils2p.save_img`, reads/memmaps via
    `tifffile` directly.
20. `utils2p` is pinned at `~=0.1.0` and every ThorImage/ThorSync read goes
    through it — a single point of failure worth vendoring or wrapping
    narrowly. Local install reality (checked 2026-11-12): the env that
    actually runs data code (`conda env twoppp311`) has a plain site-packages
    *copy*, not an editable install; the git checkout at
    `/home/remy/git/utils2p` differs from it only in functions the pipeline
    doesn't use (`beh_idx_to_2p_idx`, `event_based_frame_indices`); the one
    editable install (`twoppp37`) has a broken egg-link.

## Duplication (drift risk)

21. ~~`pipeline/raw.py` is a verbatim copy of two functions from
    `convert_raw_fn_thorimage.py`~~ **fixed (step 1)**: the implementations now
    live in `rye2p/thorimage/io.py`; both old modules re-export from it.
    Still open: `get_thorsync_line_names`, `snake_case`,
    `clean_line_name`, `frames_by_scope_pulse`, `np_in_range` each exist in two
    modules; `match_pulse_times`/`match_scope_times` are byte-identical twins;
    `steps_per_frame` logic is re-derived inline in ≥4 places instead of reading
    `FunctionalImagingSettings`. *Fix: roadmap steps 1–3.*

## Found by the ThorSync ground-truth check (2026-11-12)

Verified against `odor_space_collab/raw_data/2023-04-11/2` (upstairs rig) and
its saved outputs; the legacy extraction was reproduced **bit-exactly** from
`SyncData001`, so these are confirmed behaviors, not hypotheses. Details in
[Timestamps explained](timestamps.md).

22. **Legacy extraction drops the first frame and first volume** —
    `extract_timestamp_data` calls `get_start_times` with the default
    `zero_based_counter=False` on the **cropped** counter
    (`process_thorsync.py:207, 217`); the cropped line already sits at
    frame/volume 0, which therefore has no rising edge and is dropped. Every
    `frame_times[j]` is physical frame j+1 (~17 ms late) and every
    `stack_times[i]` is the first plane of volume i+1 (**one volume period
    late**, ~337 ms in the verified acquisition — larger than the per-plane
    error `plane_times` was built to remove). Affects every legacy-produced
    `timestamps.npy`. *Fix: `zero_based_counter=True` (or extract before
    cropping), roadmap step 2; downstream files need re-extraction or a
    one-volume shift on load.*

23. **Notebook edge times are 1 ms-quantized and drop a falling edge** —
    `notebooks/thorsync_lines_demo.ipynb` cell 17 reloads the 1 kHz
    `sync_lines.npz`, so all `scope/olf ict/fct` computed after it carry 1 ms
    quantization (the doc's "all timestamps on the 30 kHz clock" holds only
    for the legacy path and for counter-derived frame times); cell 29 also
    discards the first scope falling edge (`scope_fct = ...[1:]`) as an
    unexplained rig-specific hack. *Fix: compute edges at full rate in
    `thorsync/lines.py`, roadmap steps 1–2.*

24. **Hand-rescued single-plane files are internally inconsistent** —
    single-plane `timestamps.npy` files repaired via the
    `fix_frame_times.py::__main__` scratchpad (≈ lines 383–414) have
    `stack_times = frame_times[::n_averaging]` **plus one extrapolated final
    entry** (appended when the count came up one short — the real cause was
    the dropped first frame, #22). Verified on 2023-04-11/2
    `megamat1_calyx{,_001}`: `stack_times[:-1] == frame_times[::n_averaging]`
    exactly; every timestamp one raw frame (~16.6 ms) late; last entry
    synthetic; `frame_times` left unrepaired (92,124 ≠ 23,032×4 resp.
    92,125×1), and `timestamps0.npy` still holds the #1-bug garbage
    `stack_times` (4,606 = frames/20). *Fix: re-extract after the #22 fix,
    roadmap step 2; the `__main__` scratchpad retires into `thorsync/repair.py`.*
