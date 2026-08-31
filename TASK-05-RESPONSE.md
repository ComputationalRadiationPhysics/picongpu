# Task 05 — Rework Response

All findings verified against the real flow (generated `submit.sh`, the
`bash/` template structure, `tbg` resolution, and the step order
build → prepare → submit → organize_output). No findings rejected; two
suggested fixes were corrected for correctness (noted below).

## Findings

- **C1 — accepted** (`dc43c6b70`). Pre-stage the job inputs (already in the
  submit step's workdir) into the stable destination *before* the job is
  launched, so the default local (bash) in-workflow job runs from the stable
  `run_dir` where `input/bin/picongpu` exists at run time. Chose the review's
  sketch 2 (pre-stage) over sketch 1 (two contexts): sketch 1 leaves the local
  `simOutput` in the ephemeral cache dir with a dangling `link_results.sh`,
  which breaks the real e2e acceptance (`test_minimal.py` requires
  `run_dir/simOutput/output`) and the DoD ("link_results.sh links simOutput
  accordingly"); pre-staging makes `simOutput` land in `run_dir` for both the
  in-workflow and re-submission contexts. Proven by the committed e2e
  regression (job executes `run_dir/input/bin/picongpu` from `run_dir`).
- **M1 — accepted** (`54f819cc3`). Rewrite *every* occurrence of the prepare-time
  resolved value (not just `TBG_dstPath=`/`--chdir=`) to the stable
  destination, regex-escaping the old path and replacement-escaping the new
  one. Verified against the `--workdir=` (jureca/aris) and inline-executable
  (hemera/pmacc) template shapes; the `<run_dir>/run_dir` mangling no longer
  occurs.
- **m1 — accepted, suggested pattern corrected** (`2d450e90a`). Anchored to a
  full 32-hex md5 component. The review's literal pattern
  `s|/\.cwl_cache/[0-9a-fA-F]{32}(/|$)||g` is broken: the `|` in `(/|$)`
  collides with the `|` sed delimiter (sed: "unknown option to `s`"), and any
  parsing variant consumes the trailing separator, joining components
  (`/run/.cwl_cache/<32hex>/run_dir` → `/runrun_dir`). Used `#` as the
  delimiter and a capture group to re-emit the separator:
  `s#/\.cwl_cache/[0-9a-fA-F]{32}(/|$)#\1#g`. Negative tests confirm a
  legitimate `.cwl_cache`-named dir / hex-prefix component is untouched.
- **m2 — accepted** (`6f2160538`). Committed the faithful end-to-end
  regression (`test_in_workflow_job_runs_from_stable_destination`): real
  generated `submit.sh` + real `organize_output` step in a dummy 2-step
  cwltool workflow (same workdir layout and `RuntimeContext` as `runner.run()`),
  fake batch file with the real template structure (`cd $TBG_dstPath` +
  `$TBG_dstPath/input/bin/picongpu`), asserting (a) no `.cwl_cache` in the
  final `run_dir` and (b) the job actually ran and wrote `simOutput` where
  `link_results.sh` points. No GPU / no PIConGPU build.
- **n1 — accepted** (`6f2160538`). Replaced the
  `$(pwd -P)` count assertion with semantic checks (stable destination used for
  the results link; `$(pwd -P)` only as the standalone fallback).

## E2E proof (dummy picongpu, no compile/GPU)

- **(a)** no `.cwl_cache` references in the final outputs — PASS
  (whole-`run_dir` scan beyond the two files, excluding `.cwl_cache/` itself).
- **(b)** the job's paths resolve at run time and the simulation starts —
  PASS (job `cd`s to `run_dir`, finds `run_dir/input/bin/picongpu` there,
  executes it, and writes `run_dir/simOutput/output` exactly where
  `link_results.sh` points: `ln -s <run_dir>/simOutput $1`).

## Gates

- `pytest quick/ -q` → **178 passed, 2 xfailed, 1 xpassed**
  (baseline at the review tip was 177; +1 = the new e2e regression).
- `pre-commit run --all-files` → all hooks pass (exit 0). `TASK-*.md`
  task artifacts (incl. the unmodified `TASK-05-REVIEW.md`) are excluded from
  `require-ascii` with a documented reason (non-ASCII review document; must not
  be rewritten).
- cwltool cache behaviour (`cachedir`) unchanged; `organize_output.sh` remains
  `set -euxo pipefail` safe (no-match no-op).
