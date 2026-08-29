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

Several laser types in a single simulation:
a standard Gaussian pulse specified via its peak electric field
and a dispersive Gaussian pulse with group delay dispersion.
"""

from pathlib import Path

from picongpu import picmi

grid = picmi.Cartesian3DGrid(
    number_of_cells=[64, 64, 64],
    lower_bound=[0.0, 0.0, 0.0],
    upper_bound=[2e-6, 2e-6, 2e-6],
    lower_boundary_conditions=["open", "open", "open"],
    upper_boundary_conditions=["open", "open", "open"],
)
solver = picmi.ElectromagneticSolver(method="Yee", cfl=0.7, grid=grid)

gaussian = picmi.GaussianLaser(
    wavelength=0.8e-6,
    waist=5.0e-6,
    duration=5.0e-15,
    propagation_direction=[0.0, 1.0, 0.0],
    polarization_direction=[1.0, 0.0, 0.0],
    focal_position=[1e-6, 1.5e-6, 1e-6],
    # the pulse centroid at time zero must be outside of the box
    centroid_position=[1e-6, -1.5e-5, 1e-6],
    # exactly one of a0 or E0 must be given, the other is derived
    E0=2.0e11,
    phi0=0.5,
)
dispersive = picmi.DispersivePulseLaser(
    wavelength=1.0e-6,
    waist=10.0e-6,
    duration=10.0e-15,
    propagation_direction=[0.0, 1.0, 0.0],
    polarization_direction=[1.0, 0.0, 0.0],
    focal_position=[1e-6, 1.5e-6, 1e-6],
    centroid_position=[1e-6, -1.5e-5, 1e-6],
    a0=1.0,
    picongpu_gdd_si=1.0e-25,
)

simulation = picmi.Simulation(
    max_steps=10,
    solver=solver,
    picongpu_lasers=[gaussian, dispersive],
)

simulation.run(setup_dir=Path("laser_variants_setup"), run_dir=Path("laser_variants_run"))
