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

Shows how the particle shape and the pusher method default from the
Simulation and can be overridden per species.
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

# BEGIN-SPECIES_SHAPE
# species without their own shape/method inherit the Simulation defaults:
ions = picmi.Species(
    name="ions",
    particle_type="H",
    charge_state=1,
    initial_distribution=plasma,
)
# an explicit shape overrides the Simulation-level default:
electrons = picmi.Species(
    name="electrons",
    particle_type="electron",
    initial_distribution=plasma,
    density_scale=1.0,
    particle_shape="cubic",
)

simulation = picmi.Simulation(
    max_steps=100,
    solver=solver,
    particle_shape="linear",  # inherited by species that do not set their own
    species=[ions, electrons],
    layouts=[
        picmi.GriddedLayout(n_macroparticle_per_cell=[2, 2, 2]),
        picmi.PseudoRandomLayout(n_macroparticles_per_cell=2),
    ],
)
# END-SPECIES_SHAPE

simulation.write_input_file(Path("species_shape_and_method_setup"))
