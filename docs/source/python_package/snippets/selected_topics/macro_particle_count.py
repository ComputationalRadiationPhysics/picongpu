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

Defines a simulation with a macro-particle count diagnostic for the
electron species, written every 10th step.
A useful tool for debugging the particle content of your simulation.
"""

from pathlib import Path

from picongpu import picmi
from picongpu.picmi.diagnostics import MacroParticleCount, TimeStepSpec

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

counter = MacroParticleCount(species=electrons, period=TimeStepSpec[::10])
sim.add_diagnostic(counter)

sim.run(setup_dir=Path("macro_particle_count_setup"), run_dir=Path("macro_particle_count_run"))
