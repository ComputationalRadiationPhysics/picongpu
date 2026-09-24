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

Defines a 3D grid distributed over two GPUs in y direction
and an electromagnetic solver on it.
"""

from pathlib import Path

from picongpu import picmi

grid = picmi.Cartesian3DGrid(
    number_of_cells=[128, 128, 128],
    lower_bound=[0.0, 0.0, 0.0],
    upper_bound=[1e-6, 1e-6, 1e-6],
    # open (absorbing) or periodic, per axis, lower == upper:
    lower_boundary_conditions=["periodic", "periodic", "periodic"],
    upper_boundary_conditions=["periodic", "periodic", "periodic"],
    # distribute the grid over 2 GPUs in y direction:
    picongpu_n_gpus=[1, 2, 1],
    # the grid must be divisible by the GPU count and the super-cell size:
    picongpu_super_cell_size=[8, 8, 4],
)

solver = picmi.ElectromagneticSolver(
    method="Yee",
    # exactly one of cfl / simulation.time_step_size; the other is derived:
    cfl=0.95,
    grid=grid,
)

simulation = picmi.Simulation(max_steps=100, solver=solver)

simulation.write_input_file(Path("grids_and_solvers_setup"))
