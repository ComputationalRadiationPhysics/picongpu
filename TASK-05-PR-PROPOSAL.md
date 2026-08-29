# Task 05 — PR Proposal

## Proposed PR title

`Fix CWL workflow outputs referencing the internal cwltool job cache`

## Branch

`task-05-cwl-cache-ref-purge` (based on `dev` @ `b4e4ca5b2`)

## What

After a CWL workflow run, the user-facing final outputs in `run_dir`
(`tbg/submit.start`, `link_results.sh`) contained references into
cwltool's internal per-step job cache `run_dir/.cwl_cache/<md5>`:

- `tbg/submit.start` → `TBG_dstPath=<run_dir>/.cwl_cache/<md5>` (and
  `--chdir=<run_dir>/.cwl_cache/<md5>` for slurm templates)
- `link_results.sh` → `ln -s <run_dir>/.cwl_cache/<md5>/simOutput $1`

Both are fixed: final outputs now reference the stable run directory.

## Why

`.cwl_cache/<md5>` is cwltool's internal step-reuse cache, keyed by a hash
of the step inputs — not stable or meaningful. If the cache is purged or the
key changes, references dangle; on a cache hit stale files are re-served; and
if the user (re-)submits via `tbg/submit.start`, all results (`simOutput`)
land inside the cache directory instead of the run directory.

## Mechanism

Two complementary changes (the second makes the first a safety net):

1. **Root cause** — `pypicongpu/runner.py` `generate_submission_command()`:
   the submit step's working directory is cwltool's job cache dir, so the
   generated `submit.sh` no longer bakes `$(pwd -P)` into
   `TBG_dstPath`/`--chdir`/`link_results.sh`. A new optional
   `destination_path` workflow input (the run directory, added in
   `workflow.cwl`/`submit.cwl` and written to `input.yaml` by
   `generate_workflow_input()`) is passed to the submit script as `$2` and
   used instead. `$(pwd -P)` remains only as `${2:-$(pwd -P)}` fallback for
   standalone runs of `submit.sh`.
2. **Safety net** — `templates/workflow/scripts/organize_output.sh`: after
   copying `tbg/` and `link_results.sh`, a precise
   `sed -E 's|/\.cwl_cache/[0-9a-fA-F]+||g'` strips the
   `/.cwl_cache/<hash>` path component from `tbg/submit.start` and
   `link_results.sh`, restoring the run directory they are derived from.
   The script stays `set -euxo pipefail` safe (no-op when the pattern is
   absent, existence-checked before `sed -i`).

cwltool cache behaviour itself (`cachedir`) is unchanged — it is what
enables step reuse on re-runs.

## Verification

- **Local cwltool repro** (minimal 2-step workflow, same RuntimeContext as
  `runner.py` — `cachedir=run_dir/.cwl_cache`, `rm_tmpdir=False`,
  `move_outputs="copy"` — real generated `submit.sh`, real
  `organize_output.cwl`/`.sh`, fake tbg batch file):
  - *before fix*: `tbg/submit.start` →
    `TBG_dstPath=/tmp/repro-task05-_r0t18fh/run/.cwl_cache/79c1166a1abedf768c358bf7d959ef0d`,
    `link_results.sh` → `ln -s .../run/.cwl_cache/79c1166a1abedf768c358bf7d959ef0d/simOutput $1`
    (bug reproduced)
  - *after fix*: all three configurations clean —
    (a) legacy `submit.sh` + new `organize_output.sh` (safety net rewrites
    the cache path), (b) new `submit.sh` without `destination_path`
    (fallback + safety net), (c) new `submit.sh` + `destination_path`
    (correct from the start):
    `TBG_dstPath=/tmp/repro-task05-.../run`,
    `ln -s /tmp/repro-task05-.../run/simOutput $1`, no `.cwl_cache` left.
  (repro script kept outside the repo: `/tmp/opencode/repro-task05/repro.py`;
  not committed as a slow test)
- **Regression tests** (new, `quick/picmi/test_workflow.py`):
  `organize_output.sh` run against a fake `tbg/submit.start` containing a
  `.cwl_cache` path (asserts rewrite of `TBG_dstPath`/`--chdir`/
  `link_results.sh`), a no-op pass-through case (pattern absent,
  `set -euxo pipefail` safe), and a check that the generated
  `input.yaml`/`submit.sh` use the stable destination.
- `cd lib/python/test/picongpu && python -m pytest quick/ -q` →
  **177 passed, 2 xfailed, 1 xpassed** (baseline before change:
  174 passed, 2 xfailed, 1 xpassed; +3 = the new tests).
- `pre-commit run --all-files` → all hooks pass.

## Note for task 09 (partial workflow execution)

Task 09's branch should be **based on / merged after this branch**: it also
modifies `pypicongpu/runner.py` (stage adapter). The runner.py changes here
are intentionally minimal and self-contained (command list in
`generate_submission_command()` and one new key in
`generate_workflow_input()`), so the rebase should be clean; the new
`destination_path` input is orthogonal to stage selection.
