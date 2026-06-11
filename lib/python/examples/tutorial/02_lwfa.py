#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11,<3.14"
# dependencies = [
#   "numpy",
#   "picongpu @ git+https://github.com/ComputationalRadiationPhysics/picongpu@dev#subdirectory=lib/python"
# ]
# ///
"""
# SPDX-FileCopyrightText: Julian Lenz
#
# SPDX-License-Identifier: GPL-3.0-or-later
"""

from pathlib import Path

import numpy as np
from picongpu.picmi import (
    ADK,
    ADKVariant,
    Cartesian3DGrid,
    ElectromagneticSolver,
    PseudoRandomLayout,
    Simulation,
    Species,
)
from picongpu.picmi.constants import c
from picongpu.picmi.diagnostics import Checkpoint, MacroParticleCount, TimeStepSpec
from picongpu.picmi.distribution import GaussianDistribution
from picongpu.picmi.lasers import GaussianLaser, PolarizationType

NUM_CELLS = np.array([192, 2048, 192])
CELL_SIZE = np.array([0.1772e-6, 0.4430e-7, 0.1772e-6])

grid = Cartesian3DGrid(
    number_of_cells=NUM_CELLS,
    lower_bound=[0, 0, 0],
    upper_bound=NUM_CELLS * CELL_SIZE,
    lower_boundary_conditions=["periodic", "periodic", "periodic"],
    upper_boundary_conditions=["periodic", "periodic", "periodic"],
)
solver = ElectromagneticSolver(method="Yee", grid=grid, cfl=0.95)

LASER_DURATION = 5.0e-15
PULSE_INIT = 15.0

laser = GaussianLaser(
    wavelength=0.8e-6,
    waist=5.0e-6 / 1.17741,
    duration=LASER_DURATION,
    propagation_direction=[0.0, 1.0, 0.0],
    polarization_direction=[1.0, 0.0, 0.0],
    focal_position=[
        float(NUM_CELLS[0] * CELL_SIZE[0] / 2.0),
        4.62e-5,
        float(NUM_CELLS[2] * CELL_SIZE[2] / 2.0),
    ],
    centroid_position=[
        float(NUM_CELLS[0] * CELL_SIZE[0] / 2.0),
        -0.5 * PULSE_INIT * LASER_DURATION * c,
        float(NUM_CELLS[2] * CELL_SIZE[2] / 2.0),
    ],
    picongpu_polarization_type=PolarizationType.LINEAR,
    a0=8.0,
    phi0=0.0,
)

particle_distribution = GaussianDistribution(
    density=1.0e25,
    center_front=8.0e-5,
    sigma_front=8.0e-5,
    center_rear=10.0e-5,
    sigma_rear=8.0e-5,
    factor=-1.0,
    power=4.0,
    vacuum_front=50 * CELL_SIZE[1],
)
particle_layout = PseudoRandomLayout(n_macroparticles_per_cell=2)

electrons = Species(particle_type="electron", name="electrons", initial_distribution=particle_distribution)
hydrogen_ions = Species(
    particle_type="H",
    name="hydrogen",
    charge_state=0,
    initial_distribution=particle_distribution,
)

adk_ionization = ADK(
    ADK_variant=ADKVariant.LinearPolarization,
    ion_species=hydrogen_ions,
    ionization_electron_species=electrons,
    ionization_current=None,
)

checkpoint = Checkpoint(period=TimeStepSpec[::100])
macro_particle_count = MacroParticleCount(
    species=electrons,
    # Resulting values for period:
    # 0, 17, 50, 57, 64, 71, 100, 200, ...
    period=TimeStepSpec[::100, 50:72:7, 17],
)

sim = Simulation(
    max_steps=1000,
    solver=solver,
    picongpu_lasers=laser,
    picongpu_species=[electrons, hydrogen_ions],
    picongpu_particle_layout=particle_layout,
    picongpu_interaction=[adk_ionization],
    picongpu_diagnostics=[checkpoint, macro_particle_count],
)

OUTPUT_PATH = Path(__file__[: -len(".py")])
sim.run(setup_dir=OUTPUT_PATH / "setup", run_dir=OUTPUT_PATH / "run")
