#!/usr/bin/env python
# /// script
# requires-python = ">=3.11,<3.14"
# dependencies = [
#   "matplotlib",
#   "numpy",
#   "picongpu @ git+https://github.com/ComputationalRadiationPhysics/picongpu@dev#subdirectory=lib/python"
# ]
# ///
"""
This file is part of PIConGPU.
Copyright 2026 PIConGPU contributors
Authors: opencode
License: GPLv3+

Usage: python postprocess_histogram.py [run_dir ...]

Post-processes the output of the EnergyHistogram diagnostic
found in one or more PIConGPU run directories.
"""

# BEGIN-POSTPROCESS-HISTOGRAM
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from picongpu.extra.plugins.data import EnergyHistogramData

SPECIES = "electrons"
MIN_ENERGY_KEV = 100.0
MAX_ENERGY_KEV = 1000.0


def count_electrons_in_energy_range(run_dir, min_energy_kev=MIN_ENERGY_KEV, max_energy_kev=MAX_ENERGY_KEV):
    """Count electrons with energies in [min_energy_kev, max_energy_kev) keV in a run directory."""
    data = EnergyHistogramData(str(run_dir))
    iterations = data.get_iterations(SPECIES)
    counts, bins = data.get(iteration=[int(iterations[-1])], species=SPECIES)[:2]
    return int(np.sum(counts[(bins >= min_energy_kev) * (bins < max_energy_kev)]))


def main(argv):
    run_dirs = [Path(arg) for arg in argv] or [Path("scan") / "focal_4.6e-05"]
    counts = [count_electrons_in_energy_range(run_dir) for run_dir in run_dirs]
    for run_dir, count in zip(run_dirs, counts):
        print(f"{run_dir}: {count} electrons in [{MIN_ENERGY_KEV}, {MAX_ENERGY_KEV}) keV")

    plt.plot(range(len(run_dirs)), counts, marker="o")
    plt.xlabel("simulation")
    plt.ylabel(f"electron count in [{MIN_ENERGY_KEV}, {MAX_ENERGY_KEV}) keV")
    plt.savefig("electron_count.png")
    plt.show()


if __name__ == "__main__":
    main(sys.argv[1:])
# END-POSTPROCESS-HISTOGRAM
