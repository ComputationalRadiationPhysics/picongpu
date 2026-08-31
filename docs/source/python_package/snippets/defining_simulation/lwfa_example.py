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
"""

from pathlib import Path

from picongpu import picmi
from scipy.constants import c

NUM_CELLS = [192, 2048, 192]
CELL_SIZE = [0.1772e-6, 0.4430e-7, 0.1772e-6]

grid = picmi.Cartesian3DGrid(
    number_of_cells=NUM_CELLS,
    lower_bound=[0, 0, 0],
    upper_bound=[n * s for n, s in zip(NUM_CELLS, CELL_SIZE)],
    lower_boundary_conditions=["open", "open", "open"],
    upper_boundary_conditions=["open", "open", "open"],
)
solver = picmi.ElectromagneticSolver(method="Yee", cfl=0.95, grid=grid)
# END-LWFA-CONSTANTS

# BEGIN-LWFA-LASER
LASER_DURATION = 5.0e-15
PULSE_INIT = 15.0  # in units of the laser pulse duration

laser = picmi.GaussianLaser(
    wavelength=0.8e-6,
    waist=5.0e-6 / 1.17741,
    duration=LASER_DURATION,
    propagation_direction=[0.0, 1.0, 0.0],
    polarization_direction=[1.0, 0.0, 0.0],
    focal_position=[NUM_CELLS[0] * CELL_SIZE[0] / 2.0, 4.62e-5, NUM_CELLS[2] * CELL_SIZE[2] / 2.0],
    centroid_position=[
        NUM_CELLS[0] * CELL_SIZE[0] / 2.0,
        -0.5 * PULSE_INIT * LASER_DURATION * c,
        NUM_CELLS[2] * CELL_SIZE[2] / 2.0,
    ],
    a0=8.0,
    phi0=0.0,
)
# END-LWFA-LASER

# BEGIN-LWFA-SPECIES
distribution = picmi.GaussianDistribution(
    density=1.0e25,
    center_front=8.0e-5,
    sigma_front=8.0e-5,
    center_rear=10.0e-5,
    sigma_rear=8.0e-5,
    factor=-1.0,
    power=4.0,
    vacuum_front=50 * CELL_SIZE[1],
)
layout = picmi.PseudoRandomLayout(n_macroparticles_per_cell=2)

hydrogen = picmi.Species(
    name="hydrogen",
    particle_type="H",
    charge_state=0,
    initial_distribution=distribution,
)
electrons = picmi.Species(
    name="electrons",
    particle_type="electron",
    initial_distribution=None,
)
# END-LWFA-SPECIES

# BEGIN-LWFA-ADK
adk = picmi.ADK(
    ADK_variant=picmi.ADKVariant.LinearPolarization,
    ion_species=hydrogen,
    ionization_electron_species=electrons,
    ionization_current=None,
)
# END-LWFA-ADK

# BEGIN-LWFA-SIMULATION
sim = picmi.Simulation(
    max_steps=100,
    solver=solver,
    picongpu_lasers=[laser],
    picongpu_interaction=[adk],
)
sim.add_species(hydrogen, layout)
sim.add_species(electrons, None)
# END-LWFA-SIMULATION

# BEGIN-LWFA-DIAGNOSTICS
checkpoint = picmi.diagnostics.Checkpoint(period=picmi.diagnostics.TimeStepSpec[::50])
macro_particle_count = picmi.diagnostics.MacroParticleCount(
    species=electrons, period=picmi.diagnostics.TimeStepSpec[::10]
)
sim.diagnostics = [checkpoint, macro_particle_count]
# END-LWFA-DIAGNOSTICS

# BEGIN-LWFA-RUN
sim.run(setup_dir=Path("lwfa_example_setup"), run_dir=Path("lwfa_example_run"))
# END-LWFA-RUN
