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

Shows how to use a step-dependent (composite) particle pusher whose active
pusher changes over the course of the simulation.
"""

from pathlib import Path

from picongpu import picmi

grid = picmi.Cartesian3DGrid(
    number_of_cells=[32, 32, 32],
    lower_bound=[0.0, 0.0, 0.0],
    upper_bound=[1e-6, 1e-6, 1e-6],
    lower_boundary_conditions=["periodic", "periodic", "periodic"],
    upper_boundary_conditions=["periodic", "periodic", "periodic"],
)
solver = picmi.ElectromagneticSolver(method="Yee", cfl=0.5, grid=grid)

plasma = picmi.UniformDistribution(
    density=1.0e24,
    rms_velocity=[0.01 * picmi.constants.c] * 3,
)

# BEGIN-COMPOSITE-PUSHER
# A CompositePusher maps pre-called TimeStepSpec instances to pusher names.
# The first matching specification wins, so the order matters:
electrons = picmi.Species(
    name="electrons",
    particle_type="electron",
    initial_distribution=plasma,
    method=picmi.CompositePusher(
        {
            # free-streaming for the first 100 steps (steps 0..99) ...
            picmi.diagnostics.TimeStepSpec[:99]("steps"): "free-streaming",
            # ... then Boris for every remaining step
            picmi.diagnostics.TimeStepSpec[100:]("steps"): "Boris",
        }
    ),
)

simulation = picmi.Simulation(
    max_steps=200,
    solver=solver,
    species=[electrons],
    layouts=[picmi.PseudoRandomLayout(n_macroparticles_per_cell=2)],
)
# END-COMPOSITE-PUSHER

simulation.write_input_file(Path("composite_pusher_setup"))
