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

Defines a simulation with a phase-space diagnostic for the electron species:
the y-position is plotted against the y-momentum, sampled every 10th step.
"""

from pathlib import Path

from picongpu import picmi
from picongpu.picmi.diagnostics import PhaseSpace, TimeStepSpec

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

phase_space = PhaseSpace(
    species=electrons,
    period=TimeStepSpec[::10],
    spatial_coordinate="y",
    momentum_coordinate="py",
    min_momentum=-2e-26,
    max_momentum=2e-26,
)
sim.add_diagnostic(phase_space)

sim.run(setup_dir=Path("phase_space_setup"), run_dir=Path("phase_space_run"))
