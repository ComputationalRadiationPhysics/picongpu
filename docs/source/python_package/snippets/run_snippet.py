"""
Test harness runner for the documentation snippets.

This file is part of PIConGPU.
Copyright 2026 PIConGPU contributors
Authors: opencode
License: GPLv3+

Runs a Python snippet in this process with ``runpy`` so that it behaves
like a standalone script (``__name__ == "__main__"``).

With ``--no-run``, the workflow run step (``picongpu.pypicongpu.runner.Runner.run``)
is replaced by a no-op that emulates a finished simulation: it writes a synthetic
EnergyHistogram output into the run directory so that snippets reading
diagnostic results work without compiling or submitting anything.
This harness is test infrastructure only, not part of the documented interface.
"""

import math
import re
import runpy
import sys
from pathlib import Path

# The synthetic landscape of the emulated runs (single source of truth;
# imported by test_snippets.py).
PEAK_FOCAL = 4.6e-5
PEAK_SIGMA = 1e-6
PEAK_COUNT = 1000
SCAN_FOCALS = (4.4e-5, 4.6e-5, 4.8e-5)


def emulated_electron_count(run_dir):
    """Deterministic stand-in for a simulation result.

    The number of electrons in the energy histogram is a Gaussian around
    the focal position PEAK_FOCAL (sigma PEAK_SIGMA, peak PEAK_COUNT);
    other run directories get a fixed, arbitrary count.
    """
    match = re.search(r"focal_([0-9]+(?:\.[0-9]+)?(?:[eE][-+]?[0-9]+)?)", str(run_dir))
    if match is None:
        return 42
    focal_position = float(match.group(1))
    return int(PEAK_COUNT * math.exp(-(((focal_position - PEAK_FOCAL) / PEAK_SIGMA) ** 2)))


def write_synthetic_energy_histogram(
    run_dir,
    total_count,
    species="electrons",
    max_energy_kev=1000.0,
    bin_count=100,
    iteration=99,
    dt_si=1.39e-16,
):
    """Write the files a real run would have produced for the EnergyHistogram plugin."""
    run_dir = Path(run_dir)
    (run_dir / "simOutput").mkdir(parents=True, exist_ok=True)
    (run_dir / "simOutput" / "output").write_text(f"\tsim.unit.time() {dt_si}\n")
    edges = [max_energy_kev * (i + 1) / bin_count for i in range(bin_count)]
    header = ["iteration", "underflow"] + [f"{edge:.6f}" for edge in edges] + ["overflow", "sum"]
    counts = [0] * bin_count
    counts[bin_count // 2] = total_count
    row = [str(iteration), "0"] + [str(count) for count in counts] + ["0", str(total_count)]
    (run_dir / "simOutput" / f"{species}_energyHistogram_all.dat").write_text(
        " ".join(header) + "\n" + " ".join(row) + "\n"
    )


def _fake_runner_run(self):
    write_synthetic_energy_histogram(self.run_dir, emulated_electron_count(self.run_dir))


def main(argv):
    no_run = False
    if argv and argv[0] == "--no-run":
        no_run = True
        argv = argv[1:]
    if not argv:
        print(f"usage: {Path(sys.argv[0]).name} [--no-run] <snippet.py> [snippet args...]", file=sys.stderr)
        return 2
    snippet = Path(argv[0])
    snippet_argv = argv[1:]

    if no_run:
        from picongpu.pypicongpu import runner as runner_module

        runner_module.Runner.run = _fake_runner_run

    sys.argv = [str(snippet), *snippet_argv]
    runpy.run_path(str(snippet), run_name="__main__")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
