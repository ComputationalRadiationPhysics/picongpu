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
Authors: Julian Lenz
License: GPLv3+

Defines a Gaussian laser, specified by its normalized vector potential a0,
propagating into positive y direction.
"""

from pathlib import Path

from picongpu import picmi
from scipy.constants import c

NUM_CELLS = [192, 2048, 192]
CELL_SIZE = [0.1772e-6, 0.4430e-7, 0.1772e-6]
LASER_DURATION = 5.0e-15

laser = picmi.GaussianLaser(
    wavelength=0.8e-6,
    waist=5.0e-6 / 1.17741,
    # the pulse duration is the 1-sigma width of the intensity profile:
    duration=LASER_DURATION,
    # the propagation direction must point into the box (positive y):
    propagation_direction=[0.0, 1.0, 0.0],
    polarization_direction=[1.0, 0.0, 0.0],
    focal_position=[NUM_CELLS[0] * CELL_SIZE[0] / 2.0, 4.62e-5, NUM_CELLS[2] * CELL_SIZE[2] / 2.0],
    # the pulse centroid at time zero must be outside of the box:
    centroid_position=[
        NUM_CELLS[0] * CELL_SIZE[0] / 2.0,
        -10.0 * LASER_DURATION * c,
        NUM_CELLS[2] * CELL_SIZE[2] / 2.0,
    ],
    # give exactly one of a0 (normalized vector potential) or E0 (peak field):
    a0=8.0,
    phi0=0.0,
)

grid = picmi.Cartesian3DGrid(
    number_of_cells=NUM_CELLS,
    lower_bound=[0, 0, 0],
    upper_bound=[n * s for n, s in zip(NUM_CELLS, CELL_SIZE)],
    lower_boundary_conditions=["open", "open", "open"],
    upper_boundary_conditions=["open", "open", "open"],
)
solver = picmi.ElectromagneticSolver(method="Yee", cfl=0.95, grid=grid)

simulation = picmi.Simulation(max_steps=100, solver=solver, lasers=[laser])

simulation.write_input_file(Path("lasers_setup"))
