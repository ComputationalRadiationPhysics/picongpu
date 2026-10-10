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

Configures the grid absorbing field on a 3D grid:
the PML depth via the standard ``pml_cells`` and the absorber profile
via the PIConGPU extension.
"""

from pathlib import Path

from picongpu import picmi

grid = picmi.Cartesian3DGrid(
    number_of_cells=[128, 128, 128],
    lower_bound=[0.0, 0.0, 0.0],
    upper_bound=[1e-6, 1e-6, 1e-6],
    # an absorbing layer is only applied on "open" boundaries:
    lower_boundary_conditions=["open", "open", "open"],
    upper_boundary_conditions=["open", "open", "open"],
    # standard PICMI: per-axis symmetric PML depth in cells:
    pml_cells=[16, 16, 12],
    # PIConGPU extension: choose the absorber profile (pml or exponential):
    picongpu_absorber_kind="exponential",
    # PIConGPU extension: exponential::STRENGTH per axis and direction:
    picongpu_exponential_strength=[[1e-3, 1e-3], [1e-3, 1e-3], [1e-3, 1e-3]],
)

solver = picmi.ElectromagneticSolver(method="Yee", cfl=0.95, grid=grid)

simulation = picmi.Simulation(max_steps=100, solver=solver)

simulation.write_input_file(Path("grid_absorber_setup"))
