# Known issues

Catalogued during the 2026-11 cleanup audit. File:line references are to the
checkpoint commit (`67ee170`). Roadmap steps that fix each are noted.

## Timing correctness (the ones that can silently corrupt science)

1. ~~**Single-plane `stack_times` double decimation**~~ **fixed (step 2)** —
   the legacy extraction computed `steps_per_frame = n_z + n_flyback`
   with no `fast_z` guard (`get_n_z()` reads `ZStage/steps`, which isn't forced
   to 1 for single-plane recordings), then decimated *again* by
   `n_averaging`. Confirmed in the wild (2026-11-12): single-plane
   `timestamps0.npy` files on 2023-04-11/2 carry the `steps_per_frame=20`
   garbage (`stack_times` = 4,606 = frames/20); the final files were later
   hand-rescued (see #24). A second facet (2026-11-19):
   `utils2p.Metadata.get_n_averaging()` returns `LSM/averageNum` without
   checking `averageMode`, so the decimation fired even with averaging
   **off** — re-derivation confirmed the old end-to-end code produced 1,152
   stack times for *both* calyx recordings. Now every path reads
   `steps_per_frame` from `FunctionalImagingSettings` (which guards both
   `fast_z` and `averageMode`) and decimates exactly once, inside the unified
   extraction. Files written before the fix still carry the garbage.

2. ~~**Stack-time convention conflict**~~ **fixed (step 2)** (re-attributed
   by the 2026-11-12 ground-truth check, see #22) — the notebooks stamped
   each volume with its true **first** plane's time; the legacy extraction
   produced the first plane of the **next** volume (one volume period late —
   a consequence of #22); `fix_timestamps0` used the **last** step. All code
   paths now produce the true first-plane convention (`fix_timestamps0`'s
   last-step indexing fixed too). Files on disk keep whichever convention
   their producer had — see the convention box in
   [timestamps.md](timestamps.md).

3. ~~**Analog-voltage assumptions on a digital rig**~~ **fixed (step 2)** —
   `scope_pin > 1` masks and hard-coded 2.5 V thresholds in the legacy
   extraction were all-False on the downstairs rig's digital (0/1)
   `scopePin`, crashing `crop_lines` on an empty index. Edge detection now
   lives in `thorsync/lines.py`: lines are max-normalized to [0,1] and
   thresholded at 0.5, so analog and digital pins behave identically.
   (Not yet re-verified against a real downstairs recording — see WORKLOG.)

4. ~~**Validation only prints**~~ **fixed (step 2)** — count mismatches vs
   `Experiment.xml` used to be printed, not raised (legacy
   `process_thorsync.py`; the `raise`s in `fix_timestamps0` commented out),
   so a wrong `timestamps.npy` still got saved — verified in the wild:
   `odor_space_collab` 2023-04-11/2/megamat1 was saved with 90,749 frame
   times (expected 90,740; not divisible by `steps_per_frame`). Now
   `thorsync.timestamps.validate_frame_counts` **raises by default**
   (`strict=False` downgrades to a warning), the unified extraction
   (`thorsync.timestamps.extract_timestamps`) calls it before returning, and
   the validator knows the single-plane save rule (a partial averaging block
   that starts in-window is saved — ceil, not floor). The
   ThorSync↔ThorImage pairing is fingerprint-checked too
   (`validate_pairing`: counter max + realized frame rate vs
   `Experiment.xml`). Old files saved before the fix remain unvalidated.

5. **Fuzzy line matching can double-assign** — *mostly fixed (step 2)*: the
   unified extraction resolves lines through the per-rig YAML maps
   (`lines.load_sync_lines` → `linenames.match_lines`), with Hungarian fuzzy
   matching as a warned fallback (pass `microscope=` to get the map). The
   old per-column argmax matcher (`process_thorsync.py::get_matched_thorsync_lines`)
   still exists but nothing on the extraction path calls it — it is deleted
   in roadmap step 3.

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
    modules (step 1 made `thorsync/timestamps.py::frames_by_scope_pulse` the
    canonical, numpy-only copy — `fix_frame_times`' retires in step 3);
    `match_pulse_times`/`match_scope_times` are byte-identical twins;
    `steps_per_frame` logic is re-derived inline in ≥4 places instead of reading
    `FunctionalImagingSettings`. *Fix: roadmap steps 2–3.*

## Found by the ThorSync ground-truth check (2026-11-12)

Verified against `odor_space_collab/raw_data/2023-04-11/2` (upstairs rig) and
its saved outputs; the legacy extraction was reproduced **bit-exactly** from
`SyncData001`, so these are confirmed behaviors, not hypotheses. Details in
[Timestamps explained](timestamps.md).

22. ~~**Legacy extraction drops the first frame and first volume**~~
    **fixed in code (step 2); files on disk still affected** — the old
    `extract_timestamp_data` called `get_start_times` with the default
    `zero_based_counter=False` on the **cropped** counter; the cropped line
    already sat at frame/volume 0, which therefore had no rising edge and was
    dropped. Every `frame_times[j]` was physical frame j+1 (~17 ms late) and
    every `stack_times[i]` the first plane of volume i+1 (**one volume period
    late**, ~337 ms in the verified acquisition — larger than the per-plane
    error `plane_times` was built to remove). The unified extraction stamps
    raw counter increments (`counter_increment_times`), so frame 0 / volume 0
    are kept. **Every `timestamps.npy` written by the pre-step-2 legacy path
    still carries the shift** — re-extract, or shift one volume on load.

23. ~~**Notebook edge times are 1 ms-quantized and drop a falling edge**~~
    **fixed (step 2)** — `notebooks/thorsync_lines_demo.ipynb` reloaded the
    1 kHz `sync_lines.npz` and computed all `scope/olf ict/fct` from it (1 ms
    quantization), and discarded the first scope falling edge
    (`scope_fct = ...[1:]`) as an unexplained hack. The notebook now computes
    frame times and edges from the full-rate lines via
    `thorsync.lines.pulse_times` / `timestamps.counter_increment_times`
    (the 1 kHz copy is loaded as `sync_lines_1khz`, for plotting only), and
    the `[1:]` hack is gone.

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
    `stack_times` (4,606 = frames/20). *The step-2 extraction now produces
    these files correctly from the h5 (the GUI verified both calyx recordings
    against ground truth) — what remains is re-extracting the affected files;
    the `__main__` scratchpad retires into `thorsync/repair.py` (roadmap
    step 3).*

## Found during the step-1 promotion (2026-11-19)

25. **Notebook `preprocess_thorimage` traps (mostly resolved at promotion)** —
    the `pipeline_demo.ipynb` original (a) validated both raw filenames but
    hard-coded `Image_001_001.raw` in the loader, crashing on
    `Image_0001_0001.raw` folders; (b) re-read the *copied* `Experiment.xml`,
    crashing when `copy_xml=False` + `save_settings=True`; (c) still wrote the
    full tiff with `save_single_tiff=False` (the splitter reads it back from
    disk). The promoted `rye2p/pipeline/preprocess.py` fixes (a) and (b); (c)
    is kept but documented in the docstring. Single-channel-only remains an
    open limitation (`stack, = load_raw(...)`).
