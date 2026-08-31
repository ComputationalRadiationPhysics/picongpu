# Documentation snippets

The code blocks shown in the PIConGPU Python package documentation
(`docs/source/python_package/`) are **not written in the `.rst` files**.
They are rendered from the real scripts in this directory via
`.. literalinclude::` directives, so that **the code the reader sees is
exactly the code that is executed in CI**.

## How it works

- Each snippet is a real, minimal, faithful script in one of the topic
  subdirectories (mirroring the documentation chapter it belongs to):
   - `configuring_environment/`: `rc_params` usage
   - `defining_simulation/`: PICMI input scripts
   - `running_simulation/`: bash commands and workflow invocations
- Python snippets are executed by the pytest suite in `test_snippets.py`
  (one test per snippet): each script runs in a subprocess with a fresh
  working directory and an isolated environment (`HOME`, `PIC_RC`) and must
  exit with code 0.
  Per-snippet expected artifacts (generated files, stdout/stderr content)
  are checked afterwards.
- Snippets that call `simulation.run()` are executed with the workflow run
  step replaced by a no-op (see `run_snippet.py`), so that no compilation
  or job submission is required.
  Where a snippet reads simulation results (post-processing, optimization),
  the harness emulates the corresponding output files deterministically.
- Bash snippets are syntax-checked with `bash -n` in the same pytest suite.
   The `docs-snippets` CI job (see `.gitlab-ci.yml`) additionally executes
   one bash flow for real: setup generation with the `bash` preset and
   sourcing of the generated profile, as
   `running_simulation/legacy_workflow.sh` performs it
   (re-implemented in `share/ci/docs_snippets_profile_check.sh`, not the
   snippet file itself).
   No other bash snippet - in particular the `cwltool`, `pic-build` and
   `tbg` invocations - is executed in CI.
   The CI job also builds the Sphinx documentation; a Sphinx build fails
   on unresolvable `literalinclude` paths, keeping the rendered docs and
   the tested scripts in sync.

## Which snippets are executed where

- **Executed by the pytest suite (one test per file):** all Python snippets
  in `configuring_environment/` and `defining_simulation/` (the ones that
  call `simulation.run()` are run with the run step emulated, see above).
- **Syntax-checked with `bash -n` only:** all bash snippets in
  `running_simulation/`.
- **Executed for real by the CI job:** the legacy-workflow flow (setup
  generation + profile sourcing), re-implemented in
  `share/ci/docs_snippets_profile_check.sh`.

## Inclusion conventions

- **Default: one file = one snippet, whole-file `literalinclude`, no
  markers.** The shown code is exactly the tested code, imports and all.
- **`:start-after:` / `:end-before:` markers are only used when a single
  tested file feeds multiple doc sections** (e.g. the staged tutorial in
  `defining_simulation/lwfa_example.py` or the wrapped/scan parts of
  `defining_simulation/multiple_simulations.py`).
   Marker names are semantic (`BEGIN-<NAME>` / `END-<NAME>`), never line
   numbers.
   A section that starts at the top of the file (e.g. the first tutorial
   stage) only has an `END-<NAME>` marker, so that no marker lines are
   shown in the rendered docs.
- Non-executable config/parameter excerpts (e.g. `.picongpurc.toml`
  examples) may stay inline in the `.rst` files; they are not tested
  because they cannot be executed on their own.

## Adding or changing a snippet

1. Create/modify the script in the appropriate subdirectory.
   Python snippets are self-contained and must exit with code 0 when run
   standalone (see the PEP 723 metadata block for dependencies).
2. Reference it from the `.rst` file with `.. literalinclude::`
   (relative path from the `.rst` file, plus `:language:`).
3. If the script produces artifacts or prints something worth asserting,
   add per-snippet expectations to `EXPECTED_FILES` in `test_snippets.py`.
4. Run the suite:

   ```
   python -m pytest docs/source/python_package/snippets/ -q
   ```

5. Build the docs (`docs/`: `doxygen && make html`, see
   `docs/source/dev/sphinx.rst`) and check the rendered pages.

`run_snippet.py` and `test_snippets.py` are test infrastructure only:
they are never rendered into the documentation.
