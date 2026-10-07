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

Loads a species from an external openPMD particle file
(the PICMI-standard ``FromFileDistribution``).
"""

from pathlib import Path

from picongpu import picmi

grid = picmi.Cartesian3DGrid(
    number_of_cells=[128, 128, 128],
    lower_bound=[0.0, 0.0, 0.0],
    upper_bound=[5e-5, 5e-5, 5e-5],
    lower_boundary_conditions=["periodic", "periodic", "periodic"],
    upper_boundary_conditions=["periodic", "periodic", "periodic"],
)
solver = picmi.ElectromagneticSolver(method="Yee", cfl=0.7, grid=grid)

# BEGIN-FROM-FILE
# particles are read from the openPMD file instead of being generated
driver = picmi.FromFileDistribution(
    file_path="/path/to/bunch.bp5",
    iteration=0,
)

electrons = picmi.Species(
    name="bunch",
    particle_type="electron",
    initial_distribution=driver,
)
# END-FROM-FILE

# a from-file distribution determines its own positions, so it takes no layout
simulation = picmi.Simulation(
    max_steps=500,
    solver=solver,
    species=[electrons],
    layouts=[None],
)

simulation.write_input_file(Path("from_file_setup"))
