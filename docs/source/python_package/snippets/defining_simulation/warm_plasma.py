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

A warm, quasi-neutral plasma:
ions and electrons share the same uniform density profile.
"""

from pathlib import Path

from picongpu import picmi

grid = picmi.Cartesian3DGrid(
    number_of_cells=[64, 64, 64],
    lower_bound=[0.0, 0.0, 0.0],
    upper_bound=[2e-6, 2e-6, 2e-6],
    lower_boundary_conditions=["periodic", "periodic", "periodic"],
    upper_boundary_conditions=["periodic", "periodic", "periodic"],
)
solver = picmi.ElectromagneticSolver(method="Yee", cfl=0.7, grid=grid)

# a uniform plasma with a thermal velocity spread
# and a small collective drift in x direction
plasma = picmi.UniformDistribution(
    density=1.0e24,
    rms_velocity=[0.01 * picmi.constants.c] * 3,
    directed_velocity=[0.001 * picmi.constants.c, 0.0, 0.0],
)

ions = picmi.Species(
    name="ions",
    particle_type="H",
    charge_state=1,
    initial_distribution=plasma,
)
# the electrons share the same density profile
# and are placed at the same positions as the ions;
# a density_scale of 1.0 keeps the plasma charge-neutral
electrons = picmi.Species(
    name="electrons",
    particle_type="electron",
    initial_distribution=plasma,
    density_scale=1.0,
)

# place 8 macroparticles per cell on a 2x2x2 sub-grid
layout = picmi.GriddedLayout(n_macroparticles_per_cell=[2, 2, 2])

simulation = picmi.Simulation(max_steps=100, solver=solver)
simulation.add_species(ions, layout)
simulation.add_species(electrons, layout)

simulation.run(setup_dir=Path("warm_plasma_setup"), run_dir=Path("warm_plasma_run"))
