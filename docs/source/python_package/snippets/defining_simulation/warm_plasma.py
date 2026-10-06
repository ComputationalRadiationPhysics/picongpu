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

A warm, quasi-neutral plasma:
ions and electrons share the same uniform density profile.
"""

# BEGIN-WARM-PLASMA
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

# collective initialisation: electrons are placed at the same in-cell
# positions as the ions; equal proportions keep the plasma charge-neutral
multispecies = picmi.MultiSpecies(
    particle_types=["H", "electron"],
    names=["ions", "electrons"],
    charge_states=[1, None],
    proportions=[1.0, 1.0],
    initial_distribution=plasma,
)

# place 8 macroparticles per cell on a 2x2x2 sub-grid
layout = picmi.GriddedLayout(n_macroparticles_per_cell=[2, 2, 2])

# pass the whole group as one species together with one layout for all members
simulation = picmi.Simulation(
    max_steps=100,
    solver=solver,
    species=[multispecies],
    layouts=[layout],
)

simulation.run(setup_dir=Path("warm_plasma_setup"), run_dir=Path("warm_plasma_run"))
# END-WARM-PLASMA
