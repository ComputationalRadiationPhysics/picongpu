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

Defines a finite Gaussian particle bunch (the PICMI-standard
``GaussianBunchDistribution``) and attaches it to a species.
"""

from pathlib import Path

from picongpu import picmi

grid = picmi.Cartesian3DGrid(
    number_of_cells=[128, 128, 128],
    lower_bound=[0.0, 0.0, 0.0],
    upper_bound=[5e-5, 5e-5, 5e-5],
    lower_boundary_conditions=["periodic", "periodic", "periodic"],
    upper_boundary_conditions=["periodic", "periodic", "periodic"],
)
solver = picmi.ElectromagneticSolver(method="Yee", cfl=0.7, grid=grid)

# a finite tri-Gaussian electron bunch, moving in +y
bunch = picmi.GaussianBunchDistribution(
    n_physical_particles=1.0e10,
    rms_bunch_size=[1.5e-6, 3.0e-6, 1.5e-6],
    centroid_position=[2.5e-5, 2.5e-5, 2.5e-5],
    centroid_velocity=[0.0, 100.0e6, 0.0],
    rms_velocity=[1.0e5, 1.0e5, 1.0e5],
)

electrons = picmi.Species(
    name="bunch",
    particle_type="electron",
    initial_distribution=bunch,
)
layout = picmi.PseudoRandomLayout(n_macroparticles_per_cell=2)

simulation = picmi.Simulation(
    max_steps=500,
    solver=solver,
    species=[electrons],
    layouts=[layout],
)

simulation.write_input_file(Path("gaussian_bunch_setup"))
