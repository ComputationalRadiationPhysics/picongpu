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
   - `quickstart/`: the commands of the quick start guide
   - `selected_topics/`: the feature deep dives
 - Python snippets are executed by the pytest suite in `test_snippets.py`
   (one test per snippet): each script runs in a subprocess with a fresh
   working directory and an isolated environment (`HOME`, `PIC_RC`) and must
   exit with code 0.
   Per-snippet expected artifacts (generated files, stdout/stderr content)
   are checked afterwards.
 - TOML snippets (`.picongpurc.toml` examples) are parsed with `tomllib`
   and then applied for real in the same pytest suite:
   a subprocess with an isolated `HOME` and `PIC_RC` pointed at the snippet
   file imports the PIConGPU Python package,
   and the resulting `rc_params` content is checked.
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

> **Limitation (not covered by the harness).** A few short, illustrative
> fragments are written as inline `.. code-block:: python` blocks directly
> in the `.rst` files instead of being rendered from a checked-in script
> (e.g. the `UnitDimension` examples in `units.rst` and the `Simulation(...)`
> one-liners in `simulation_settings.rst`, `interactions.rst`, `lasers.rst` and
> `species_distributions_layouts.rst`). Those blocks are **not** executed
> or syntax-checked. They are kept short and are covered by the executed
> pages around them; do not rely on them as the source of truth.

- **Executed by the pytest suite (one test per file):** all Python snippets
  in `configuring_environment/` and `defining_simulation/` (the ones that
  call `simulation.run()` are run with the run step emulated, see above).
- **Applied via `PIC_RC` by the pytest suite (one test per file):** all
  TOML snippets in `configuring_environment/` (parsed with `tomllib`, then
  loaded by a subprocess importing the package; the resulting `rc_params`
  content is checked).
- **Syntax-checked with `bash -n` only:** all bash snippets in
  `running_simulation/` and `quickstart/`.
- **Executed for real by the CI job:** the legacy-workflow flow (setup
  generation + profile sourcing), re-implemented in
  `share/ci/docs_snippets_profile_check.sh`.

## Inclusion conventions

- **One file = one snippet.** The `literalinclude` directives render the
  browser-visible code from the real, executed scripts in this directory.
  The preferred way to focus a rendered block on the relevant lines of a
  file is a semantic `BEGIN-<NAME>` / `END-<NAME>` marker pair used as
  `:start-after:` / `:end-before:`, which keeps file boilerplate (the
  shebang, the PEP 723 `/// script` metadata block, the license header)
  out of the rendered docs while the harness still executes the whole
  file.
- Marker names are semantic (`BEGIN-<NAME>` / `END-<NAME>`), never line
  numbers.
   A file feeding multiple doc sections (e.g. the staged tutorial in
   `defining_simulation/lwfa_example.py` or the wrapped/scan parts of
   `defining_simulation/multiple_simulations.py`) uses one marker pair per
   doc section.
   A section that starts at the top of the file (e.g. the first tutorial
   stage) only has an `END-<NAME>` marker, so that no marker lines are
   shown in the rendered docs.
- **Not every include uses a marker pair.** Where a stable, semantic marker
  would be noisy or would have to be inserted into a compact fragment, the
  directive instead selects by a distinctive code line via
  `:start-after:` / `:end-before:` (and occasionally `:start-at:`, e.g. a
  function `def gamma` or a `sim.run(` call). Examples:
  `selected_topics/binning.py`, `selected_topics/openpmd.py`,
  `selected_topics/energy_histogram.py`, `selected_topics/phase_space.py`,
  `selected_topics/radiation.py`, `selected_topics/checkpoint.py`,
  `selected_topics/macro_particle_count.py` and
  `selected_topics/time_steps.py`.
  A few files are short enough that the whole file is intended, so the
  directive carries no focus option at all (e.g. the TOML configuration
  examples).
- `.picongpurc.toml` (runtime configuration) examples are checked-in
  snippet files (`configuring_environment/*.toml`),
  rendered via `literalinclude` like any other snippet
  and tested by the suite (see "Which snippets are executed where").

## Adding or changing a snippet

1. Create/modify the script in the appropriate subdirectory.
   Python snippets are self-contained and must exit with code 0 when run
   standalone (see the PEP 723 metadata block for dependencies).
2. Reference it from the `.rst` file with `.. literalinclude::`
   (relative path from the `.rst` file, `:language:`, and - where
   boilerplate should not be shown - a `:start-after:` /
   `:end-before:` marker pair focusing the rendered block on the relevant
   lines; see "Inclusion conventions" for the cases that select by a code
   line instead).
   For TOML configuration snippets, also add the expected `rc_params`
   content to `TOML_EXPECTED` in `test_snippets.py`.
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
