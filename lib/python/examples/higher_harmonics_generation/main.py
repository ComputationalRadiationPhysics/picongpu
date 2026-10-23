#!/usr/bin/env python
# /// script
# requires-python = ">=3.11,<3.14"
# dependencies = [
#   "numpy",
#   "sympy",
#   "picongpu @ git+https://github.com/ComputationalRadiationPhysics/picongpu@dev#subdirectory=lib/python"
# ]
# ///
"""
This file is part of PIConGPU.
Copyright 2026 PIConGPU contributors
Authors: Julian Lenz, Ben Lehnhardt
License: GPLv3+
"""

import numpy as np
from pathlib import Path

from picongpu import picmi
from scipy.constants import c
from picongpu.picmi.distribution import GaussianDistribution, AnalyticDistribution
from sympy import Piecewise, Abs, exp
from picongpu.picmi.diagnostics.field_dump import (NativeFieldDump, DerivedFieldDump)


NUM_CELLS = np.array([1536, 4096])
CELL_SIZE = np.array([1.6e-8, 1.6e-8]) 
# dx=dy, because plasma waves propagate in both x and y, meaning you need equal resolution on both axes.
# dx (and dy) should be a fifth of the plasma wavelenth (using a lower resolution is not recommended). 

grid = picmi.Cartesian2DGrid(
    picongpu_n_gpus=[2, 2], 
    # this simulation also runs a single gpu
    # for larger simulations 4 gpus (picongpu_n_gpus=[2, 2]) is strongly advised.
    number_of_cells=NUM_CELLS,
    lower_bound=[0, 0],
    upper_bound=NUM_CELLS * CELL_SIZE,
    lower_boundary_conditions=["open", "open"],
    upper_boundary_conditions=["open", "open"],
)
solver = picmi.ElectromagneticSolver(method="Yee", cfl=0.95, grid=grid)
# END-HHG-CONSTANTS

# BEGIN-HHG-LASER
laser_duration = 2.0 * 30.0e-15 / 2.354820045
laser_pulse_init = 10.0
laser = picmi.DispersivePulseLaser(
    wavelength = 800.e-9,
    waist = 2.0e-6, #W0
    duration = laser_duration,
    propagation_direction = [0.0, 1.0, 0.0],
    polarization_direction = [1.0, 0.0, 0.0],
    focal_position=[
        float(NUM_CELLS[0] * CELL_SIZE[0] / 2.0),
        30.0e-6,
        float(0.0),
    ],
    centroid_position=[
        float(NUM_CELLS[0] * CELL_SIZE[0] / 2.0),
        -0.25 * laser_duration * laser_pulse_init * c,
        float(0.0),
    ],
    picongpu_polarization_type = picmi.lasers.PolarizationType.LINEAR,
    a0 = 40.0,
    phi0 = 0.0,
    picongpu_gdd_si = 0.0e-30,
    picongpu_spectral_support = 4.0
)
# END-HHG-LASER

# BEGIN-HHG-SPECIES
distribution = picmi.GaussianDistribution( 
    density=1.742e29, 
    #1.742e27 is the critical density at which the laser gets reflected. I use 100 times that. 
    #In this case, the plasma wavelength is 8.0e-8 (80 nanometers, i.e. one thenth of the laser wavelength). 
    center_front=3e-5,
    sigma_front=1.0e-10,
    center_rear=5e-5,
    sigma_rear=1.0e-10,
    factor=-1.0,
    power=4.0,
    vacuum_front=50 * CELL_SIZE[1],
)
# END-HHG-SPECIES

# FIXME(2026-10-23, @Kn0pf42): The following funcion is a workaround for ISSUE-5847. 
# The gaussian distribution currently doesnt work in 2D simulations.
# Please remove the workaround once that issue is resolved.

# BEGIN-WORKAROUND
@AnalyticDistribution
def custom_particle_distribution(x, y, z):
    # PIConGPU's predefined GaussianDistribution is evaluated at the center of the cell
    y += -0.5 * CELL_SIZE[1]
    # The last term undoes the shift to the cell origin.
    vacuum_y = int(distribution.vacuum_front / CELL_SIZE[1]) * CELL_SIZE[1] - 0.5 * CELL_SIZE[1]

    exponent = Piecewise(
        (
            Abs((y - distribution.center_front) / distribution.sigma_front),
            y < distribution.center_front,
        ),
        (
            Abs((y - distribution.center_rear) / distribution.sigma_rear),
            y >= distribution.center_rear,
        ),
        (0.0, True),
    )
    return Piecewise(
        (0.0, y < vacuum_y),
        (
            distribution.density * exp(distribution.factor * exponent**distribution.power),
            True,
        ),
    )

layout = picmi.PseudoRandomLayout(n_macroparticles_per_cell=20)

electrons = picmi.Species(
    name="electrons",
    particle_type="electron",
    initial_distribution=custom_particle_distribution,
)
# END-WORKAROUND

# BEGIN-HHG-DIAGNOSTICS
checkpoint = picmi.diagnostics.Checkpoint(period=picmi.diagnostics.TS[::1000])
electric_field_dump = NativeFieldDump(fieldname="E", period=picmi.diagnostics.TS[::1000])
# END-HHG-DIAGNOSTICS

# BEGIN-HHG-SIMULATION
sim = picmi.Simulation(
    max_steps=25000,
    solver=solver,
    lasers=[laser],
    species=[electrons],
    layouts=[layout],
    #interactions=[adk],
    diagnostics=[checkpoint, electric_field_dump],
)
# END-HHG-SIMULATION

# BEGIN-HHG-RUN
OUTPUT_PATH = "HHG"
sim.run(setup_dir=Path(f"{OUTPUT_PATH}/setup"), run_dir=Path(f"{OUTPUT_PATH}/run"))
# END-HHG-RUN

#I something doesnt work, blame the 11th grader who wrote this setup. Have fun :)
