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

`tbg` resolves every `!TBG_dstPath` placeholder at *prepare* time to its
per-step cwltool job cache dir (`<run_dir>/.cwl_cache/<md5>/run_dir`), so the
resolved batch file carries that ephemeral path in several places (the
`TBG_dstPath=` line, `--chdir=`/`--workdir=` directives, `cd`, and inline
executable references). Three complementary changes fix the references and
keep the in-workflow job runnable:

1. **Stable destination** — a new optional `destination_path` workflow input
   (the run directory, added in `workflow.cwl`/`submit.cwl` and written to
   `input.yaml` by `generate_workflow_input()`) is passed to the generated
   `submit.sh` as `$2` (`pypicongpu/runner.py`
   `generate_submission_command()`), which no longer bakes `$(pwd -P)` (the
   per-step cache dir) into `link_results.sh`. `$(pwd -P)` remains only as a
   `${2:-$(pwd -P)}` fallback for standalone runs.
2. **Rewrite every reference + keep the job runnable** (submit step):
   - *All occurrences* of the prepare-time resolved value are rewritten to the
     stable destination in one pass (regex-escaping the old path,
     replacement-escaping the new one) — not just the `TBG_dstPath=` and
     `--chdir=` lines — so `--workdir=`, `cd` and inline executable references
     all point at the run directory too.
   - The job inputs already staged in the submit step's workdir are copied into
     the destination *before* the job is launched: `organize_output_step`
     stages the full `run_dir/input` only *after* the submit step, so without
     this the default local (bash/zsh) in-workflow job would start from the
     destination before `input/bin/picongpu` exists and fail to launch the
     simulation. No-op for standalone runs (destination == workdir).
3. **Safety net** — `templates/workflow/scripts/organize_output.sh`: after
   copying `tbg/` and `link_results.sh`, a precise
   `sed -E 's#/\.cwl_cache/[0-9a-fA-F]{32}(/|$)#\1#g'` strips a leftover
   `/.cwl_cache/<md5>` path component (cwltool keys are md5, exactly 32 hex
   chars; a legitimate path that merely contains the string is left
   untouched). The script stays `set -euxo pipefail` safe (no-op when the
   pattern is absent, existence-checked before `sed -i`).

cwltool cache behaviour itself (`cachedir`) is unchanged — it is what
enables step reuse on re-runs.

## Verification

- **Committed end-to-end regression**
  (`quick/picmi/test_workflow.py::test_in_workflow_job_runs_from_stable_destination`,
  no GPU / no PIConGPU build): runs the *actual* generated `submit.sh` and the
  *real* `organize_output` step inside a dummy 2-step cwltool workflow (same
  workdir layout and `RuntimeContext` as `runner.run()`) whose fake batch file
  has the structure of the real templates (`cd $TBG_dstPath` + execute
  `$TBG_dstPath/input/bin/picongpu`). It asserts (a) no `.cwl_cache` reference
  remains anywhere in the final `run_dir`, and (b) the in-workflow job actually
  ran — it `cd`'d to the stable destination, found `input/bin/picongpu` there
  at run time, and wrote `simOutput` exactly where `link_results.sh` points.
  This is the guard that catches both "the job runs from a directory whose
  input is not staged yet" and "leftover resolved references".
- **Unit regression tests** (`quick/picmi/test_workflow.py`): `organize_output.sh`
  run against a fake `tbg/submit.start` containing a `.cwl_cache` path (asserts
  the rewrite), a no-op pass-through case (pattern absent, `set -euxo pipefail`
  safe), and a check that the generated `input.yaml`/`submit.sh` use the stable
  destination.
- `cd lib/python/test/picongpu && python -m pytest quick/ -q` →
  **178 passed, 2 xfailed, 1 xpassed** (baseline before the original branch:
  174 passed; +3 original unit tests, +1 end-to-end regression).
- `pre-commit run --all-files` → all hooks pass.

## Note for task 09 (partial workflow execution)

Task 09's branch should be **based on / merged after this branch**: it also
modifies `pypicongpu/runner.py` (stage adapter). The runner.py changes here
are intentionally minimal and self-contained (command list in
`generate_submission_command()` and one new key in
`generate_workflow_input()`), so the rebase should be clean; the new
`destination_path` input is orthogonal to stage selection.
