#!/usr/bin/env python
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "numpy",
#   "picongpu @ git+https://github.com/ComputationalRadiationPhysics/picongpu@dev#subdirectory=lib/python"
# ]
# ///
"""
This file is part of PIConGPU.
Copyright 2026 PIConGPU contributors
Authors: Julian Lenz
License: GPLv3+
"""

import datetime
from pathlib import Path
from typing import Literal

import numpy as np
from picongpu import picmi
from picongpu.picmi.diagnostics import Checkpoint, TimeStepSpec

"""
@file PICMI user script reproducing a minimal 2D laser-injection benchmark

A 2D3V port of the PIConGPU ``IncidentField`` example: a focused Gaussian laser
propagating along +y into a uniform plasma. Only the essential pieces (a
2-component grid and a 2D GaussianLaser) are kept.
"""

# generation modifiers
OUTPUT_DIRECTORY_PATH = Path("2d_incident_field")
MODE: Literal["run", "write"] = "run"

NUM_CELLS = np.array([128, 128])
CELL_SIZE = np.array([0.5e-3, 0.5e-3])  # unit: meter


grid = picmi.Cartesian2DGrid(
    number_of_cells=NUM_CELLS.tolist(),
    lower_bound=[0, 0],
    upper_bound=(NUM_CELLS * CELL_SIZE).tolist(),
    lower_boundary_conditions=["open", "open"],
    upper_boundary_conditions=["open", "open"],
)

solver = picmi.ElectromagneticSolver(grid=grid, method="Yee", cfl=0.999)

LASER_DURATION = 26.0e-15
PULSE_INIT = 15.0

laser = picmi.GaussianLaser(
    wavelength=800e-9,
    waist=16e-3,
    duration=LASER_DURATION,
    propagation_direction=[0.0, 1.0, 0.0],
    polarization_direction=[1.0, 0.0, 0.0],
    focal_position=[
        float(NUM_CELLS[0] * CELL_SIZE[0] / 2.0),
        float(NUM_CELLS[1] * CELL_SIZE[1] / 2.0),
        0.0,
    ],
    centroid_position=[
        float(NUM_CELLS[0] * CELL_SIZE[0] / 2.0),
        -0.5 * PULSE_INIT * LASER_DURATION * picmi.constants.c,
        0.0,
    ],
    picongpu_polarization_type=picmi.lasers.PolarizationType.LINEAR,
    a0=8.0,
    phi0=0.0,
)

electrons = picmi.Species(
    particle_type="electron",
    name="electrons",
    initial_distribution=picmi.UniformDistribution(density=1.0e24),
)
random_layout = picmi.PseudoRandomLayout(n_macroparticles_per_cell=2)

sim = picmi.Simulation(
    max_steps=1000,
    solver=solver,
    picongpu_walltime=datetime.timedelta(hours=1.0),
)

sim.add_species(electrons, layout=random_layout)
sim.add_laser(laser, None)
sim.diagnostics = [Checkpoint(period=TimeStepSpec[::100])]


if __name__ == "__main__":
    match MODE:
        case "run":
            sim.run(setup_dir=OUTPUT_DIRECTORY_PATH / "setup", run_dir=OUTPUT_DIRECTORY_PATH / "run")
        case "write":
            sim.write_input_file(OUTPUT_DIRECTORY_PATH / "setup")
        case _:
            raise ValueError(f"Unknown {MODE=}.")
