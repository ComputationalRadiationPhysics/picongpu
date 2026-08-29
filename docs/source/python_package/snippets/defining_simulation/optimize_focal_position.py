#!/usr/bin/env python
# /// script
# requires-python = ">=3.11,<3.14"
# dependencies = [
#   "numpy",
#   "picongpu @ git+https://github.com/ComputationalRadiationPhysics/picongpu@dev#subdirectory=lib/python"
# ]
# ///
"""
This file is part of PIConGPU.
Copyright 2026 PIConGPU contributors
Authors: opencode
License: GPLv3+

Optimizes the laser focal position of an LWFA simulation
by maximizing the number of ejected electrons in a given energy range.
"""

import shutil
from pathlib import Path

import numpy as np
from picongpu import picmi
from scipy.constants import c
from scipy.optimize import minimize

NUM_CELLS = [192, 2048, 192]
CELL_SIZE = [0.1772e-6, 0.4430e-7, 0.1772e-6]

FIXED_KWARGS = dict(
    max_steps=100,
)
FIXED_LASER_KWARGS = dict(
    wavelength=0.8e-6,
    waist=5.0e-6 / 1.17741,
    duration=5.0e-15,
    propagation_direction=[0.0, 1.0, 0.0],
    polarization_direction=[1.0, 0.0, 0.0],
    a0=8.0,
    phi0=0.0,
)
PULSE_INIT = 15.0  # in units of the laser pulse duration

TRANSVERSE_FOCUS = NUM_CELLS[0] * CELL_SIZE[0] / 2.0


def make_laser(focal_position):
    return picmi.GaussianLaser(
        **FIXED_LASER_KWARGS,
        focal_position=[TRANSVERSE_FOCUS, focal_position, TRANSVERSE_FOCUS],
        centroid_position=[TRANSVERSE_FOCUS, -0.5 * PULSE_INIT * FIXED_LASER_KWARGS["duration"] * c, TRANSVERSE_FOCUS],
    )


def make_simulation(focal_position):
    grid = picmi.Cartesian3DGrid(
        number_of_cells=NUM_CELLS,
        lower_bound=[0, 0, 0],
        upper_bound=[n * s for n, s in zip(NUM_CELLS, CELL_SIZE)],
        lower_boundary_conditions=["open", "open", "open"],
        upper_boundary_conditions=["open", "open", "open"],
    )
    electrons = picmi.Species(name="electrons", particle_type="electron")
    energy_histogram = picmi.diagnostics.EnergyHistogram(
        species=electrons,
        period=picmi.diagnostics.TimeStepSpec[-1],
        bin_count=100,
        min_energy=0.0,
        max_energy=1000.0,
    )
    return picmi.Simulation(
        **FIXED_KWARGS,
        solver=picmi.ElectromagneticSolver(method="Yee", cfl=0.95, grid=grid),
        picongpu_lasers=[make_laser(focal_position)],
        picongpu_diagnostics=[energy_histogram],
    )


def count_electrons_in_energy_range(run_dir, min_energy_kev=100.0, max_energy_kev=1000.0):
    """Count electrons with energies in [min_energy_kev, max_energy_kev) keV in a run directory."""
    from picongpu.extra.plugins.data import EnergyHistogramData

    data = EnergyHistogramData(str(run_dir))
    iterations = data.get_iterations("electrons")
    counts, bins = data.get(iteration=[int(iterations[-1])], species="electrons")[:2]
    return int(np.sum(counts[(bins >= min_energy_kev) * (bins < max_energy_kev)]))


def electron_count_of(focal_position):
    run_dir = Path("scan") / f"focal_{focal_position:.1e}"
    # the optimizer may evaluate the same focal position more than once,
    # so start from a clean slate:
    if run_dir.exists():
        shutil.rmtree(run_dir)
    simulation = make_simulation(focal_position)
    simulation.run(setup_dir=run_dir / "setup", run_dir=run_dir)
    # Note: waiting for the submitted job to finish before reading its results
    # is system specific (see "Immediate post-processing");
    # in a real workflow, you would insert that wait here.
    return count_electrons_in_energy_range(run_dir)


def maximize(objective, x0):
    result = minimize(
        lambda x: -objective(float(x[0])), [x0], method="Nelder-Mead", options={"xatol": 1e-8, "maxiter": 100}
    )
    result.fun = -result.fun
    return result


ESTIMATED_FOCAL_POSITION = 4.4e-5

if __name__ == "__main__":
    result = maximize(electron_count_of, ESTIMATED_FOCAL_POSITION)
    optimal_focal_position = float(result.x[0])
    maximal_electron_count = result.fun
    print(f"optimal focal position: {optimal_focal_position:.3e}")
    print(f"maximal electron count: {maximal_electron_count}")
