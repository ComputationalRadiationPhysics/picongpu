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

Defines a simulation with a checkpoint diagnostic:
the full simulation state is written every 20th step
to the "checkpoints" directory of the output,
so that the run can be resumed from the latest checkpoint.
"""

from pathlib import Path

from picongpu import picmi
from picongpu.picmi.diagnostics import Checkpoint, TimeStepSpec

grid = picmi.Cartesian3DGrid(
    number_of_cells=[32, 32, 32],
    lower_bound=[0, 0, 0],
    upper_bound=[1e-6, 1e-6, 1e-6],
    lower_boundary_conditions=["periodic", "periodic", "periodic"],
    upper_boundary_conditions=["periodic", "periodic", "periodic"],
)
solver = picmi.ElectromagneticSolver(method="Yee", cfl=0.5, grid=grid)
distribution = picmi.UniformDistribution(density=1e23)
layout = picmi.PseudoRandomLayout(n_macroparticles_per_cell=1)
electrons = picmi.Species(name="electrons", particle_type="electron", initial_distribution=distribution)

sim = picmi.Simulation(max_steps=100, solver=solver)
sim.add_species(electrons, layout)

checkpoint = Checkpoint(
    period=TimeStepSpec[::20],
    directory="checkpoints",
    file="checkpoint",
)
sim.add_diagnostic(checkpoint)

# A follow-up run that resumes from the latest checkpoint
# (or starts from scratch if none exists) uses:
#
#     Checkpoint(tryRestart=True)

sim.run(setup_dir=Path("checkpoint_setup"), run_dir=Path("checkpoint_run"))
