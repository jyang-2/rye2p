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

- **Active work:** `cleanup` branch — documentation/scaffolding phase complete.
- **Next action (before any code changes):** ThorSync ground-truth check —
  open a real ThorSync `.h5`, enumerate every line, verify the docs' account of
  each line's physical meaning and pipeline role against the actual data, and
  patch `docs/timestamps.md` with anything missing or wrong.
- **Then:** roadmap step 1 — promote `preprocess_thorimage()` from
  `notebooks/pipeline_demo.ipynb` into `rye2p/pipeline/preprocess.py`
  (see [docs/roadmap.md](docs/roadmap.md)). Architecture changes and bug fixes
  proceed together in roadmap order (steps 1–3), not fixes-first.
- **Parallel sessions:** before starting a second simultaneous session, claim a
  line here (branch + worktree path), e.g.
  `- session B: branch fix-timing, worktree ../rye2p-timing`.

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
