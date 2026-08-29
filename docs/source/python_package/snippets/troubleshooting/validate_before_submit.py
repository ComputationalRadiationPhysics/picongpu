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

Generates the input files without compiling or submitting:
the recommended first step after any change to a simulation
definition, because most configuration errors already show up here.
"""

from pathlib import Path

from picongpu import picmi

sim = picmi.Simulation(
    max_steps=10,
    solver=picmi.ElectromagneticSolver(
        method="Yee",
        cfl=0.5,
        grid=picmi.Cartesian3DGrid(
            number_of_cells=[32, 32, 32],
            lower_bound=[0, 0, 0],
            upper_bound=[1e-6, 1e-6, 1e-6],
            lower_boundary_conditions=["periodic", "periodic", "periodic"],
            upper_boundary_conditions=["periodic", "periodic", "periodic"],
        ),
    ),
)

setup_dir = Path("validated_setup")
sim.write_input_file(setup_dir)
print("Input files generated in", setup_dir)
