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

Defines a simulation with an energy histogram diagnostic for the electron
species: 50 bins between 0 and 500 keV, written every 10th step.
"""

from pathlib import Path

from picongpu import picmi
from picongpu.picmi.diagnostics import EnergyHistogram, TimeStepSpec

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

histogram = EnergyHistogram(
    species=electrons,
    period=TimeStepSpec[::10],
    bin_count=50,
    min_energy=0.0,
    max_energy=500.0,
)
sim.add_diagnostic(histogram)

sim.run(setup_dir=Path("energy_histogram_setup"), run_dir=Path("energy_histogram_run"))
