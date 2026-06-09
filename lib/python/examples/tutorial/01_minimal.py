#!/usr/bin/env -S uv run

# SPDX-FileCopyrightText: PIConGPU contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

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
"""

from pathlib import Path

from picongpu.picmi import Cartesian3DGrid, ElectromagneticSolver, Simulation

sim = Simulation(
    max_steps=100,
    solver=ElectromagneticSolver(
        method="Yee",
        cfl=0.95,
        grid=Cartesian3DGrid(
            number_of_cells=[192, 2048, 192],
            lower_bound=[0, 0, 0],
            upper_bound=[0.1772e-6, 0.4430e-7, 0.1772e-6],
            lower_boundary_conditions=["periodic", "periodic", "periodic"],
            upper_boundary_conditions=["periodic", "periodic", "periodic"],
        ),
    ),
)

OUTPUT_PATH = Path(__file__[: -len(".py")])
sim.run(setup_dir=OUTPUT_PATH / "setup", run_dir=OUTPUT_PATH / "run")
