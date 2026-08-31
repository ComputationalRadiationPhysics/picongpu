# Review — Task 05: Purge `.cwl_cache` references from `organize_output` / submission outputs

- **Branch:** `task-05-cwl-cache-ref-purge` (tip `33d89313f`, base `dev` @ `b4e4ca5b2`)
- **Reviewed:** 2026-08-31 · **Scope:** 5 commits, 7 files, +229/−3
- **Verdict:** REQUEST CHANGES
  (the reference purge works and is well tested at file-content level, but the "root-cause" `destination_path` fix regresses the default local in-workflow run — the submitted job now starts before `run_dir/input` exists and fails silently.)

## 1. Summary

The branch fixes the documented leak in two layers: (1) a new optional `destination_path` workflow input (the run dir) is passed into the submit step and baked into `tbg/submit.start`/`link_results.sh` instead of `$(pwd -P)` (the per-step cwltool job-cache dir); (2) `organize_output.sh` gains a targeted `sed` safety net that strips `/.cwl_cache/<hex>` components from `tbg/submit.start` and `link_results.sh`. The file-content goal is met: I re-ran a faithful cwltool reproduction (same `RuntimeContext` as `runner.py`) in 4 modes — `dev` still leaks, all branch modes come out clean, and a whole-`run_dir` scan (beyond the two files the author checked) finds no `.cwl_cache` references. Quick gate re-run: 177 passed / 2 xfailed / 1 xpassed, matching the claim.

The most important problems:

- **C1:** the root-cause fix changes *where the in-workflow submitted job runs*: it now `cd`s to the final `run_dir` and executes `$TBG_dstPath/input/bin/picongpu`, but `run_dir/input` is only staged by `organize_output_step`, which runs *after* the `submit_step` that backgrounds the job. The local (default `bash`) in-workflow job therefore fails with "executable missing" and the simulation never runs, while the workflow still reports success. Reproduced deterministically; it is a race that only a multi-second job preamble wins.
- **M1:** the submit-side `sed` only rewrites the `TBG_dstPath=` and `--chdir=` lines. Templates that resolve `!TBG_dstPath` inline (e.g. `#SBATCH --workdir=!TBG_dstPath` in `etc/picongpu/jureca-jsc/batch.tpl`) keep the prepare-time cache path, which the safety net then mangles into the *stable-but-nonexistent* `<run_dir>/run_dir`.
- **m1:** the safety-net pattern is not component-anchored and can corrupt legitimate paths containing a real `.cwl_cache` directory.
- **m2:** the author's verification never mimics the real resolved template structure (`cd !TBG_dstPath` + `$TBG_dstPath/input/bin/picongpu`), which is exactly why C1 and M1 were missed.

## 2. Findings

### 2.1 Critical

**C1** — **`lib/python/picongpu/pypicongpu/runner.py:337-339` (with `workflow.cwl:143`, `submit.cwl:42-48`)** — the `destination_path` fix makes the in-workflow submitted job run from the final `run_dir`, where its inputs do not exist yet; the default local run no longer executes the simulation.
- Real resolved batch files do `cd $TBG_dstPath` and run the binary via the variable, e.g. `etc/picongpu/bash/mpiexec.tpl:35-36,56` (`TBG_dstPath="!TBG_dstPath"`, `cd $TBG_dstPath`, `mpiexec ... $TBG_dstPath/input/bin/picongpu ... | tee output`). With this branch, `TBG_dstPath` = final `run_dir`, so the job needs `run_dir/input/bin/picongpu`.
- `run_dir/input` is only produced by `organize_output_step` (`cp -r "$PROJECT_PATH" input`, `cp -r "$BIN_DIRECTORY" input/bin`), and the workflow data dependencies (`workflow.cwl:146-155`) make it run *after* `submit_step`, which launches the job in the background (`$submission_cmd $submission_script &`, `runner.py:342`). On `dev`, the job ran from the submit step's workdir, where cwltool had *already* staged `input/bin` via `InitialWorkDirRequirement` (`submit.cwl:11-13`) before `submit.sh` executed — deterministic. On this branch the job and the organize step race; with the default local profile (trivial, see generated `picongpu.profile`: just a `PATH` export) the job reaches the executable reference in milliseconds and loses.
- *Evidence:* faithful 2-step cwltool repro (real generated `submit.sh`, real `organize_output.cwl`/`.sh`, `RuntimeContext` identical to `runner.py:463-471`, submit step with the same `InitialWorkDir` layout as `submit.cwl`, fake batch file with the real template's structure, fake `picongpu` executable) — `/tmp/opencode/review-05/repro/repro2.py`:
  - `legacy` (dev submit.sh + dev organize_output.sh): job **succeeds** (`exec=…/run/.cwl_cache/<md5>/input/bin/picongpu`), final outputs **leak** `.cwl_cache` (`RESULT-DOCD: BUG PRESENT`) — baseline bug confirmed.
  - `branch` (+`destination_path`): final outputs clean, but `JOB-FAILED: executable missing at …/run/input/bin/picongpu (cwd=…/run)` and no `simOutput` — **the simulation did not run**, workflow exit 0.
  - `branch` + 3 s job preamble delay (simulating HPC module/profile loading): job **succeeds**, writes `run/simOutput` — confirms it is a race, won only when the job's preamble is slower than the organize step.
  - `branch` without `destination_path` (standalone fallback): job succeeds from the step workdir and writes `simOutput` into the **cache dir**, while the final `link_results.sh` points to `run_dir/simOutput` — dangling link for that job's actual results.
- Impact: `submit_system` defaults to `"bash"` (`runner.py:142`), so local/dev runs and the `end_to_end` suite (`test_minimal.py::test_has_finished_run` requires `run_dir/simOutput/output`) are affected: they now fail or become machine-speed-dependent instead of passing deterministically. Slurm/cluster runs are unaffected (the job is queued and starts after the workflow, when `run_dir/input` exists) — and for re-submission from `run_dir` the fix is correct.
- *Suggested fix:* decouple "where the in-workflow job runs" from "what the final file references". Two viable sketches:
  1. In `generate_submission_command()`, for the `bash`/`zsh` (in-workflow) branch, launch the job from a *copy* of `submit.start` rewritten with `TBG_dstPath=$(pwd -P)` (the step workdir, where `input/` is staged — dev execution semantics), while the committed `tbg/submit.start` keeps the `$destination_path` rewrite for post-run re-submission; for real batch systems (`else` branch) submit the `$destination_path`-rewritten file as today. Do **not** sed the same file after launching the background job (bash re-reads scripts; that would be a new race).
  2. Or make the destination usable at job start: also stage `input/` into `$destination_path` before the submit step (e.g. have `prepare_submission.sh`/a pre-step copy project+bin into `$destination_path/input`), so a single `TBG_dstPath=run_dir` is valid for both contexts. Heavier, and steps writing outside their CWL workdir is non-hermetic — weigh carefully.
  Whichever is chosen, add a regression that actually runs the in-workflow job and asserts it found and executed the binary (see m2).
- *Alternative:* the task's own "primary fix" (organize_output safety net alone, keeping dev's `$(pwd -P)` execution semantics) has no such race, but leaves local-run results inside the ephemeral cache dir with a `link_results.sh` pointing elsewhere — i.e. it fixes the letter of the DoD, not the local-run behavior. The two-context split (fix sketch 1) is the minimal way to get both.

### 2.2 Major

**M1** — **`lib/python/picongpu/pypicongpu/runner.py:338-339` + `templates/workflow/scripts/organize_output.sh:23-27`** — the submit-side `sed` rewrites only the `TBG_dstPath=` and `--chdir=` *lines*; every other `!TBG_dstPath` occurrence resolved by `tbg` at prepare time survives as a prepare-cache path, and the safety net then mangles it into a stable-but-nonexistent `<run_dir>/run_dir`.
- `tbg` resolves `!TBG_dstPath` to its `job_outDir` = `<prepare-step-workdir>/run_dir` (the literal `run_dir` subdirectory exists because `prepare_submission.cwl:65` globs `run_dir/tbg`). Verified with the real `bin/tbg` (`tbg -t mytpl.tpl -c etc/picongpu/N.cfg . run_dir`): the resolved `submit.start` contains **three** workdir references — `#SBATCH --workdir=<workdir>/run_dir`, `TBG_dstPath="<workdir>/run_dir"`, `cd <workdir>/run_dir` — of which only the second is rewritten by the branch's `sed`.
- Consequences for in-repo templates:
  - `#SBATCH --workdir=!TBG_dstPath` — `etc/picongpu/jureca-jsc/batch.tpl:35`, `jureca-jsc/booster.tpl:35`, `aris-grnet/gpu.tpl:36-37`: final line `#SBATCH --workdir=<run_dir>/run_dir` → slurm rejects the (re-)submitted job (invalid workdir). (≈26 other templates use `--chdir=!TBG_dstPath` and *are* covered by the `--chdir=` sed.)
  - Inline executable paths — `etc/picongpu/hemera-hzdr/{defq,gpu,fwkt_v100}.tpl` (`… !TBG_dstPath/input/bin/picongpu …`), `share/pmacc/examples/gameOfLife2D/submit/bash/*.tpl:37,49` (`cd !TBG_dstPath`, `!TBG_dstPath/bin/!TBG_PROGRAM`): final files reference `<run_dir>/run_dir/…` → job fails at `cd`/exec.
- *Evidence:* applied the branch's exact `submit.sh` seds, then the real `organize_output.sh`, to a jureca-shaped `submit.start`: final `#SBATCH --workdir=/tmp/proberun/run_dir` (directory does not exist). The `.cwl_cache` substring is gone (DoD letter met), but the reference now dangles at a *stable* wrong path — worse than a transparently-ephemeral one, because it looks final.
- *Suggested fix:* rewrite **all occurrences of the old resolved value**, not two line patterns. Sketch in `submit.sh` (before the current seds):
  ```bash
  old_dst=$(sed -n 's/^TBG_dstPath="*\([^"]*\)"*$/\1/p' "$submission_script" | head -n1)
  if [ -n "$old_dst" ] && [ "$old_dst" != "$destination_path" ]; then
      sed -i "s|${old_dst//|/\\|}|$destination_path|g" "$submission_script"
  fi
  ```
  (escape the `|` delimiter in the extracted value; it is a filesystem path). This fixes `--workdir=`, `cd`, and inline binary references in one pass; keep the organize_output safety net as the backstop.

### 2.3 Minor

**m1** — **`lib/python/picongpu/templates/workflow/scripts/organize_output.sh:25`** — the safety-net pattern `s|/\.cwl_cache/[0-9a-fA-F]+||g` is not anchored to a path component: `[0-9a-fA-F]+` matches the hex *prefix* of a longer component and can corrupt legitimate paths that genuinely contain a `.cwl_cache` directory (the exact failure mode the task's Notes warned about).
- *Evidence:*
  - `TBG_dstPath=/data/.cwl_cache/deadbeefrun/run` → `TBG_dstPath=/datarun/run` (component prefix `deadbeef` stripped mid-name).
  - `TBG_dstPath=/data/.cwl_cache/deadbeef/run` → `TBG_dstPath=/data/run` (legit dir named `.cwl_cache` destroyed).
  - `TBG_dstPath=/data/.cwl_cache_backup/…` → untouched (good).
- *Suggested fix:* cwltool cache keys are md5 — exactly 32 hex chars — and the stripped component must end the path or be followed by `/`:
  ```bash
  sed -E -i 's|/\.cwl_cache/[0-9a-fA-F]{32}(/|$)||g' "$file"
  ```
  Add a test case with a legitimate `.cwl_cache`-named directory to pin this down.

**m2** — **`lib/python/test/picongpu/quick/picmi/test_workflow.py:101-151` + author repro** — the verification never exercises the real batch-file structure, which is why C1 and M1 were missed.
- The author's repro fake `submit.start` only does `mkdir -p simOutput; echo done > simOutput/done.txt` — no `cd $TBG_dstPath`, no `$TBG_dstPath/input/bin/picongpu`, no slurm directive. Any fix that keeps the file "clean" passes it, including ones that break the actual job. The new unit tests likewise only check generated *text* (`input.yaml`, `submit.sh` substrings) and `organize_output.sh` in isolation; none run `submit.sh`'s runtime behavior or the two fixes in combination.
- *Suggested fix:* commit a faithful repro as a slow test (no GPU needed): the 2-step dummy workflow used for review (`/tmp/opencode/review-05/repro/repro2.py`) with a fake `picongpu` executable and a fake template that does `cd $TBG_dstPath` + executes `$TBG_dstPath/input/bin/picongpu`; assert (a) no `.cwl_cache` in final outputs, (b) the in-workflow job actually ran and `simOutput` is where `link_results.sh` points. This single test would have caught C1.

### 2.4 Nits

**n1** — **`lib/python/test/picongpu/quick/picmi/test_workflow.py:151`** — `assert submission_script.count("$(pwd -P)") == 1` asserts an implementation-detail count that will break on any benign refactor. Prefer asserting the semantics: `'"${2:-$(pwd -P)}"' in submission_script` and that the `sed`/`ln` lines use `$destination_path` (already partially asserted on 148-149).

## 3. Requirement traceability

| # | Requirement (from task file) | Status | Where / note |
|---|---|---|---|
| 1 | Adjust `organize_output_step` to purge `.cwl_cache` refs from `submit.start` (title) | met | `organize_output.sh:23-27`; verified in 4 repro modes |
| 2 | DoD: no final output in `run_dir` (`tbg/submit.start`, `link_results.sh`, `submission_information.txt`, `input/`) contains `.cwl_cache` refs | met (letter) | whole-`run_dir` byte scan of final outputs: clean. Caveat: for inline-`!TBG_dstPath` templates the file instead carries dangling `<run_dir>/run_dir` refs (M1) — no `.cwl_cache` substring, but not a "stable, meaningful location" either |
| 3 | DoD: `TBG_dstPath` points to a stable, meaningful location; `link_results.sh` links accordingly | partial | true for the `TBG_dstPath` variable and the link in the common (`--chdir`/bash-var) templates; other resolved refs in the same file are wrong (M1); and for the no-`destination_path` fallback the link dangles relative to where the in-workflow job actually wrote (C1 evidence, mode 4) |
| 4 | A regression test exists | met | 3 new tests in `quick/picmi/test_workflow.py`; coverage gaps noted in m2 |
| 5 | Verification: local repro, same `RuntimeContext` as `runner.py`, final outputs clean after fix | met | author's manual repro; re-verified independently (4 modes incl. whole-run_dir scan) |
| 6 | Verification: `pytest quick/` green | met | re-run: 177 passed, 2 xfailed, 1 xpassed (baseline 174 + 3 new tests) |
| 7 | Verification: pre-commit green | claimed | not fully re-runnable read-only (hooks include autofixers); spot-checked the changed files read-only: `ruff check` + `ruff format --check` pass on both changed `.py` files; no tabs/trailing-ws/EOF issues in changed non-markdown files — consistent with the claim |
| 8 | Notes: do not change cwltool cache behavior (`cachedir`) | met | `runner.py:463-471` untouched; cache dirs still created and used |
| 9 | Notes: `organize_output.sh` stays `set -euxo pipefail`-safe (no-match must not fail) | met | `sed` no-match exits 0; `[ -f ]` guard; no-op test passes and cwltool runs succeed |
| 10 | Suggested approach: precise pattern, not blanket `.cwl_cache` replace | partial | precise (`/.cwl_cache/<hex>` component) but not component-anchored (m1) |
| 11 | Suggested approach: root-cause fix in `generate_submission_command()` + organize as safety net, "do both if clean" | partial | both implemented; the root-cause side is not behaviorally clean (C1) — the in-workflow job's execution location changed as a side effect |

## 4. Claim verification (author artifact)

| Claim (from TASK-05-PR-PROPOSAL.md) | Re-verified? | Result / delta |
|---|---|---|
| Quick suite: 177 passed, 2 xfailed, 1 xpassed (baseline 174/2/1, +3 new tests) | yes | **177 passed, 2 xfailed, 1 xpassed** — matches exactly; +3 = the 3 new tests |
| Before fix: `submit.start`/`link_results.sh` leak `.cwl_cache/<md5>` | yes | reproduced with dev's `submit.sh` + dev's `organize_output.sh` (repro2 `legacy` mode): both files contain `…/run/.cwl_cache/<md5>` |
| After fix: all three configurations clean, no `.cwl_cache` left | yes (with caveat) | all 3 (+ a 4th, no-`destination_path`) come out clean, including a whole-`run_dir` scan beyond the two checked files; **caveat:** "clean" was only ever checked for the `.cwl_cache` substring — the repro's fake batch file never ran the real job structure, so the in-workflow job failure (C1) and the `<run_dir>/run_dir` mangling (M1) are invisible to it |
| `pre-commit run --all-files` → all hooks pass | partially | read-only spot checks on changed files pass (ruff lint+format, tabs/ws/EOF); full run skipped (autofix hooks would modify the read-only worktree) |
| cwltool cache behaviour unchanged | yes | `RuntimeContext`/`cachedir` diff: none |
| "Root cause" labeling (mechanism §1) | — | the fix addresses the *symptom* of `$(pwd -P)` being the cache dir; the actual root cause is that `tbg` resolves `!TBG_dstPath` at prepare time to a step-workdir path (see §5) — and only 2 of N occurrences are rewritten (M1) |
| Note for task 09 (clean rebase) | not verifiable | task 09 branch not available in this review |

## 5. Design discussion

The two-layer structure (root-cause input + organize_output safety net) follows the task's suggested approach, and the safety net itself is well done: targeted, `set -e`-safe, tested for the no-op case, and verified to leave `input/`, `submission_information.txt`, `tbg/submit.tpl` and `tbg/submit.cfg` clean (checked: the only `.cwl_cache`-bearing file produced by `tbg` is `submit.start`).

The conceptual problem is that **`TBG_dstPath` plays two roles that have incompatible timing**:
1. *Runtime workdir of the job launched in-workflow* — must be a place where `input/bin` already exists (during the submit step: only the step workdir qualifies).
2. *Stable location the user (re-)submits from after the run* — must be the final `run_dir` (only true after the workflow finishes).

The legacy code satisfied (1) with an ephemeral value (the documented bug); this branch satisfies (2) with a value that breaks (1) for fast local jobs. No single static value in one file satisfies both — which is why the branch "works" for slurm (role 2 only matters; the job starts after the workflow) and races for local `bash` (role 1 is exercised immediately). A maintainer should decide which of the §2.1 sketches to adopt: (a) two execution contexts (in-workflow copy rewritten to the step workdir; committed file keeps the destination), which is minimal and restores dev's deterministic local behavior while keeping the post-run semantics; or (b) pre-stage `input/` into `run_dir` before the submit step, which is more uniform but has the job's inputs written by an earlier step outside its CWL workdir (non-hermetic; on multi-node clusters the prepare/submit steps may run on different nodes, so this requires `run_dir` to be shared storage — usually true, but a new assumption).

The "root cause" label is also imprecise in a second way: the real root cause is that `tbg` resolves `!TBG_dstPath` during the *prepare* step to `<prepare-workdir>/run_dir` (the `run_dir/tbg` glob trick in `prepare_submission.cwl:65` forces the literal `run_dir` suffix). The branch rewrites two line patterns (`TBG_dstPath=`, `--chdir=`) of the resulting file; everything else resolved inline is left to the safety net, whose component-stripping then produces `<run_dir>/run_dir` (M1). Rewriting *all occurrences of the old value* (sketch in M1) is a smaller, more robust root-cause patch than adding more line patterns.

Finally, the safety net is acceptable as a backstop but should be exact (m1): cwltool keys are 32-hex md5, so anchor to that.

## 6. Prioritized next steps

1. **Fix C1:** make the in-workflow (bash/zsh) job's execution location independent of `destination_path` (launch it from a copy rewritten to the step workdir, or pre-stage `input/` into `run_dir` before the submit step). Add a repro/test that runs the in-workflow job with the real template structure and asserts it executed the binary (would have caught this).
2. **Fix M1:** in `submit.sh`, replace all occurrences of the old resolved `TBG_dstPath` value with `$destination_path` (sketch in the finding) instead of two line patterns; validate against the `--workdir` (jureca/aris) and inline-path (hemera, pmacc) templates.
3. **Fix m1:** anchor the safety-net pattern to a full md5 component: `sed -E -i 's|/\.cwl_cache/[0-9a-fA-F]{32}(/|$)||g'`; add the legitimate-path negative test.
4. Commit the faithful cwltool repro as a slow test (fake `picongpu`, no GPU) covering: leak-free final outputs, in-workflow job executed, `simOutput` located where `link_results.sh` points.
5. Update `TASK-05-PR-PROPOSAL.md` (drop/qualify the "root cause" claim) and the CHANGELOG line if the design changes in steps 1–2.

## FYI (inherited from base, not scored here)

- `submit.cwl`/`prepare_submission.cwl` share the copy-pasted label "Run PIConGPU Simulation"; `submit_system` sits at `position: 2` with an empty slot 1 (works, but odd).
- The generated `submit.sh`/`link_results.sh` use unquoted `$destination_path`/`$(pwd -P)` in `sed` replacements and `ln -s`: paths containing spaces, `&`, or `|` would misbehave (same class of issue as the legacy `$(pwd -P)` code).
- The `run_dir/tbg` glob trick (`prepare_submission.cwl:65`) — a literal `run_dir` subdirectory inside the step workdir — is what injects the `/run_dir` suffix into every resolved `!TBG_dstPath` value and thus the M1 mangling.
- `link_results.sh` carries a 24-space-indented `ln -s` line (multi-line `echo` in `script_content_with`) — cosmetic, pre-existing.
- Test logs are flooded with rocrate SHACL validator warnings (`ConjunctiveGraph is deprecated`) — pre-existing noise.
