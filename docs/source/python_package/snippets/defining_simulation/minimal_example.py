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
"""

from pathlib import Path

from picongpu import picmi

sim = picmi.Simulation(
    max_steps=100,
    solver=picmi.ElectromagneticSolver(
        method="Yee",
        cfl=0.95,
        grid=picmi.Cartesian3DGrid(
            number_of_cells=[192, 2048, 192],
            lower_bound=[0, 0, 0],
            upper_bound=[0.1772e-6, 0.4430e-7, 0.1772e-6],
            lower_boundary_conditions=["periodic", "periodic", "periodic"],
            upper_boundary_conditions=["periodic", "periodic", "periodic"],
        ),
    ),
)

sim.run(setup_dir=Path("minimal_example_setup"), run_dir=Path("minimal_example_run"))
