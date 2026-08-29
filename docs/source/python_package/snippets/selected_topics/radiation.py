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

Defines a simulation with a radiation diagnostic for the electron species:
the radiation field is observed in 32 directions distributed over the
unit sphere and accumulated, with a total-spectrum dump
every 5th step (starting at step 2,
as the plugin needs a few steps of particle history).
"""

from pathlib import Path
from sympy import cos, pi, sin

from picongpu import picmi
from picongpu.picmi.diagnostics import Radiation, TimeStepSpec
from picongpu.pypicongpu.output.radiation import RadiationObserverConfiguration

N_OBSERVER = 32


def observation_direction(index):
    return (
        sin(pi * index / N_OBSERVER) * cos(2 * pi * index / N_OBSERVER),
        sin(pi * index / N_OBSERVER) * sin(2 * pi * index / N_OBSERVER),
        cos(pi * index / N_OBSERVER),
    )


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

radiation = Radiation(
    species=electrons,
    period=TimeStepSpec[2:-1:5],
    observer=RadiationObserverConfiguration(N_observer=N_OBSERVER, index_to_direction=observation_direction),
    num_accumulation_steps=5,
    total_radiation=True,
)
sim.add_diagnostic(radiation)

sim.run(setup_dir=Path("radiation_setup"), run_dir=Path("radiation_run"))
