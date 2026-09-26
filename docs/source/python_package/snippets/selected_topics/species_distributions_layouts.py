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

Defines a simulation whose species share one density profile but use
different layouts, showing the species, distribution and layout blocks.
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

# a thermal, quasi-neutral plasma shared by both species:
plasma = picmi.UniformDistribution(
    density=1.0e24,
    rms_velocity=[0.01 * picmi.constants.c] * 3,
)

ions = picmi.Species(
    name="ions",
    particle_type="H",
    charge_state=1,
    initial_distribution=plasma,
)
electrons = picmi.Species(
    name="electrons",
    particle_type="electron",
    initial_distribution=plasma,
    density_scale=1.0,  # keep the plasma charge-neutral
)

# layouts determine the particle positions within a cell
ion_layout = picmi.GriddedLayout(n_macroparticle_per_cell=[2, 2, 2])
electron_layout = picmi.PseudoRandomLayout(n_macroparticles_per_cell=2)

simulation = picmi.Simulation(
    max_steps=100,
    solver=solver,
    species=[ions, electrons],
    layouts=[ion_layout, electron_layout],
)

simulation.write_input_file(Path("species_distributions_layouts_setup"))
