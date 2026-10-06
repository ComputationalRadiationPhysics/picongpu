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

Collective (coordinated) initialisation of ions and electrons with a
``MultiSpecies``, so both are placed on identical in-cell positions.
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

# the one density profile shared by all members of the group
plasma = picmi.UniformDistribution(
    density=1.0e24,
    rms_velocity=[0.01 * picmi.constants.c] * 3,
)

# BEGIN-MULTI-SPECIES
# explicit collective initialisation: all members share the distribution and
# their proportion becomes their density scale (electron/ion number ratio 1)
multispecies = picmi.MultiSpecies(
    particle_types=["H", "electron"],
    names=["ions", "electrons"],
    proportions=[1.0, 1.0],
    initial_distribution=plasma,
)

# the MultiSpecies is passed as one species together with one layout for the
# entire group (fully declarative)
layout = picmi.PseudoRandomLayout(n_macroparticles_per_cell=4)

simulation = picmi.Simulation(
    max_steps=100,
    solver=solver,
    species=[multispecies],
    layouts=[layout],
)
# END-MULTI-SPECIES

simulation.write_input_file(Path("multi_species_setup"))
