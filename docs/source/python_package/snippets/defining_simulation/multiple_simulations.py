#!/usr/bin/env python
# /// script
# requires-python = ">=3.11,<3.14"
# dependencies = [
#   "picongpu @ git+https://github.com/ComputationalRadiationPhysics/picongpu@dev#subdirectory=lib/python"
# ]
# ///
"""
This file is part of PIConGPU.
Copyright 2026 PIConGPU contributors
Authors: opencode
License: GPLv3+
"""

from pathlib import Path

from picongpu import picmi
from scipy.constants import c

NUM_CELLS = [192, 2048, 192]
CELL_SIZE = [0.1772e-6, 0.4430e-7, 0.1772e-6]

# BEGIN-MS-WRAP
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
# where to initialize the laser pulse, in units of the pulse duration
PULSE_INIT = 15.0

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


# END-MS-WRAP

# BEGIN-MS-SCAN
FOCAL_POSITIONS = [4.4e-5, 4.6e-5, 4.8e-5]


def run_simulation(focal_position):
    run_dir = Path("scan") / f"focal_{focal_position:.1e}"
    simulation = make_simulation(focal_position)
    simulation.run(setup_dir=run_dir / "setup", run_dir=run_dir)
    return run_dir


if __name__ == "__main__":
    for focal_position in FOCAL_POSITIONS:
        run_simulation(focal_position)
# END-MS-SCAN
