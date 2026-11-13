# Worklog

One entry per work session, **newest first**. The NOW block is the single
source of truth for what's in flight — update it before stopping work.
(Claude sessions: maintenance rules are in CLAUDE.md.)

## Session ritual (context-window management)

**Starting:** open a session *in this folder* → say what you want to work on
(e.g. "pick up roadmap step 1") — Claude reads CLAUDE.md + this file and
orients itself. One session = one task.

**Ending (or stepping away):** say **"update the worklog and stop."**
That's the whole ritual.

**Start a FRESH session when:** switching to a different task; the current
session gets slow, repeats itself, or forgets earlier decisions; or a long
debugging dig ended in a small finding (have it write the finding down first).
Fresh sessions are cheap — this file is the handoff.

**Working in parallel:** second session goes in its own worktree
(`git worktree add ../rye2p-<topic> -b <branch>`) and claims a line in NOW.

## NOW

- **Active work:** `cleanup` branch — ground-truth check done; ready for
  roadmap step 1.
- **Next action:** roadmap step 1 — promote `preprocess_thorimage()` from
  `notebooks/pipeline_demo.ipynb` into `rye2p/pipeline/preprocess.py`
  (see [docs/roadmap.md](docs/roadmap.md)). Architecture changes and bug fixes
  proceed together in roadmap order (steps 1–3), not fixes-first. Step 2 now
  includes the newly found issue #22 (legacy drops first frame/volume —
  stack_times one volume late in all legacy-produced files).
- **Heads-up for downstream work:** legacy `timestamps.npy` files (e.g. all
  odor_space_collab-era) have `stack_times[i]` = first plane of volume i+1
  (~337 ms late) and `frame_times` one frame late — see known issues #22 and
  docs/timestamps.md. Also: ThorSync↔ThorImage pairings need fingerprint
  verification (2023-04-11/2: megamat1↔SyncData001, NOT SyncData003).
- **Parallel sessions:** before starting a second simultaneous session, claim a
  line here (branch + worktree path), e.g.
  `- session B: branch fix-timing, worktree ../rye2p-timing`.

## 2026-11-12 — ThorSync ground-truth check (upstairs rig)

**Done**
- Enumerated every dataset in a real upstairs `Episode001.h5`
  (odor_space_collab 2023-04-11 fly 2, read-only): 9 canonical lines + `Freq/`
  (zeros, unused) + `Global/GCtr` (20 MHz board timebase, not a frame
  counter). Inventory tables added to `docs/timestamps.md` ("Line inventory")
  and as comments in `data/microscopes/upstairs-2p/thorsync_lines.yaml`.
- Verified: upstairs `scopePin` is analog AI 0–4.9 V (2.5 V thresholds valid);
  `FrameIn` constant 0; `FrameOut` toggles 0↔2; `PiezoMonitor` near-constant
  (useless for z-QC); `edges(size=...)` is jump-magnitude, not level-threshold
  (docs wording fixed); sample rate exactly 30000 from XML; GCtr vs sample
  clock disagree ~1e-4 (noted, ignored).
- **Reproduced the saved `timestamps.npy`/`timestamps0.npy` bit-exactly**
  (max diff 0 on all 6 keys) from SyncData001 + current legacy code — after
  discovering the real pairing is megamat1↔SyncData001 (fingerprints:
  counter max 90,752 vs 90,740 movie frames; realized 59.29 Hz vs xml 59.273;
  SyncData003 belongs to megamat1_calyx_001 at 60.17 Hz).
- **New known issue #22 (the big one):** legacy `extract_timestamp_data` drops
  the first frame and first volume (`get_start_times` default
  `zero_based_counter=False` on a cropped counter): `frame_times[j]` =
  physical frame j+1, `stack_times[i]` = first plane of volume i+1 (~337 ms
  late). Known issue #2 re-attributed (notebooks = true first-plane; legacy =
  next-volume first plane; fix_timestamps0 = last step). Issue #23: notebook
  edge times are 1 kHz-quantized + undocumented `scope_fct[1:]` hack.
- Saved megamat1 file demonstrates issue #4 live: 90,749 frame times saved
  (expected 90,740, not divisible by 20; `plane_times` refuses it).
- Docs patched: timestamps.md (inventory, legacy-drop danger box, convention
  box rewrite, edges semantics, notebook 1 kHz, validation + pairing
  fingerprints), known-issues (#2/#4/#20 updated, #22/#23 added), roadmap
  (step 2 scope + done-list), per-plane tutorial warning box.
- Env recorded in CLAUDE.md: data inspection runs in conda `twoppp311`;
  `~/git/utils2p` checkout is NOT the installed copy.

**Why**
- NOW action queued this before any code changes; the user supplied the
  acquisition. The per-plane downstream goal is directly affected: legacy
  stack_times error (one volume) is ~20× the per-plane error plane_times
  corrects.

**Decisions**
- No code changes yet (roadmap order holds); fixes for #22/#23 added to
  roadmap step 2 scope.
- Inventory lives in timestamps.md + yaml comments (prose-with-evidence, not
  generated — revisit when extraction code is unified in step 2).

**Open questions**
- Were other flies' timestamps extracted from mispaired SyncData episodes?
  Worth a fingerprint sweep over processed_data when pipeline code is unified
  (cheap: counter max + frame-rate check per movie).
- Should legacy-produced timestamps.npy files be mass re-extracted after the
  step-2 fix, or shimmed on load (one-volume shift)?

## 2026-11-11/12 — Cleanup kickoff: audit, docs site, per-plane timestamps

**Done**
- Full codebase audit (module map + line-by-line timestamp/output trace).
- Checkpoint commit `67ee170` of all pre-existing WIP; all work since is on the
  `cleanup` branch (`master` = checkpoint).
- `pyproject.toml` replaces `setup.py` (which was silently omitting the
  `thorsync/`, `thorimage/`, `pipeline/`, `olfactometer/` subpackages).
- **`rye2p/thorsync/timestamps.py`** — `plane_times()`: per-plane timestamps
  for volumetric stacks, `(n_timepoints, n_z)`, aligned with the saved tiff.
  7 passing tests (`tests/test_plane_times.py`), verified against
  `data/haoru/timestamps.npy`.
- Documentation site (MkDocs Material): timestamps deep-dive, output-file
  reference, pipeline overview, 3 tutorials, API pages, 21-item known-issues
  catalog, refactor roadmap. `mkdocs serve` → localhost:8000.
- README landing page; local-only `CLAUDE.md` (gitignored, never in git history).
- Junk removed: stray `None` PDF, `.run/`, empty Sphinx skeleton.
- Docs-drift enforcement: CI workflow (pytest + mkdocs build, fails on broken
  links) and warn-only pre-commit hook (`.githooks/`, enabled via
  `git config core.hooksPath .githooks`). Principle adopted: prefer generated
  docs (docstrings, tests pinning invariants) over prose.
- Session workflow: ritual at top of this file; stop phrase = "update the
  worklog and stop" (then Claude offers to push); tripwire rule — sessions
  name conflicts and require fresh confirmation before risky git/workflow
  actions.
- `CLAUDE.md` + `.claude/` version-controlled in a local-only side repo
  (`.claude-local.git`, `git local` alias), auto-snapshotted by the pre-commit
  hook; never pushed.
- Pushed `master` + `cleanup` to GitHub (first backup of all cleanup work);
  roadmap gained step 7 (PyPI release / outsider quickstart).
- Found a stash from Oct 2022 (master.py / fix_frame_times.py / pid.py);
  left untouched — likely superseded, diff before ever dropping.

**Why**
- Repo had three coexisting pipeline generations (flat legacy modules, half-built
  subpackages, newest orchestration living only in notebooks) and untracked
  timing bugs. Goal: intuitive modules, real docs, handoff-ready for other
  Claude Code projects.
- Driving need: downstream volumetric analyses must use **per-plane** timestamps
  instead of one time per volume (planes are ~181 ms apart in the example data —
  up to ~543 ms systematic error on the deepest plane).

**Key decisions**
- `stack_times` = **first-plane** convention (matches notebooks + committed
  example; `fix_timestamps0`'s last-step indexing is a bug to fix — known issue #2).
- attrs/cattrs for serialization; pydantic is legacy, being removed.
- MkDocs Material + mkdocstrings (not Sphinx); `pymdown-extensions<12`.
- Code behavior unchanged so far — bugs documented first, fixes staged in the
  roadmap (steps 1–3 next).

**Open questions**
- `utils2p` strategy (pinned editable checkout at `~/git/utils2p`): keep /
  vendor / wrap narrowly — decide in roadmap step 4.
- Which downstream repo(s) consume `rye2p` imports and need deprecation shims
  (roadmap step 5) — list them before moving modules.
