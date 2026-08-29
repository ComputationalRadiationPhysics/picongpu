#!/usr/bin/env -S uv run
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

A minimal PIConGPU simulation.

Run it with ``uv run my_first_simulation.py``
(or with a Python environment that contains the picongpu package).
"""

from pathlib import Path

from picongpu import picmi

simulation = picmi.Simulation(
    max_steps=100,
    solver=picmi.ElectromagneticSolver(
        method="Yee",
        cfl=0.95,
        grid=picmi.Cartesian3DGrid(
            number_of_cells=[128, 128, 128],
            lower_bound=[0.0, 0.0, 0.0],
            upper_bound=[1e-6, 1e-6, 1e-6],
            lower_boundary_conditions=["periodic", "periodic", "periodic"],
            upper_boundary_conditions=["periodic", "periodic", "periodic"],
        ),
    ),
)

simulation.run(setup_dir=Path("my_first_simulation_setup"), run_dir=Path("my_first_simulation_run"))
