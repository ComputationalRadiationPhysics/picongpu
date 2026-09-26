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

Configures simulation-wide settings: a moving window, the normalization
reference density and typical particles-per-cell, and a wall-clock limit.
"""

from datetime import timedelta
from pathlib import Path

from picongpu import picmi

grid = picmi.Cartesian3DGrid(
    number_of_cells=[64, 128, 64],
    lower_bound=[0.0, 0.0, 0.0],
    upper_bound=[1e-6, 2e-6, 1e-6],
    lower_boundary_conditions=["open", "open", "open"],
    upper_boundary_conditions=["open", "open", "open"],
)
solver = picmi.ElectromagneticSolver(method="Yee", cfl=0.95, grid=grid)

electrons = picmi.Species(
    name="electrons",
    particle_type="electron",
    initial_distribution=picmi.UniformDistribution(density=1e24),
)

simulation = picmi.Simulation(
    max_steps=1000,
    solver=solver,
    species=[electrons],
    layouts=[picmi.PseudoRandomLayout(n_macroparticles_per_cell=4)],
    # start moving the window once a light ray has crossed 90% of the box:
    picongpu_moving_window_move_point=0.9,
    # stop moving it after this iteration:
    picongpu_moving_window_stop_iteration=800,
    # normalization reference density (default 1.0e25 m^-3) and typical ppc:
    picongpu_base_density=1.0e25,
    picongpu_typical_ppc=4,
    # floating-point precision of the simulation core: 32 (single) or 64 (double)
    picongpu_precision=64,
    # per-namespace precision overrides ("core", 32 or 64; "core" = follow the core):
    picongpu_precision_config=picmi.PrecisionConfig(sqrt=64, exp="core", trig="core"),
    # memory / exchange-buffer knobs (sizes are byte counts; use the constants for readability):
    picongpu_memory_config=picmi.MemoryConfig(reserved_gpu_memory_size=350 * picmi.constants.MiB),
    # ask the scheduler for at most one hour of wall-clock time:
    picongpu_walltime=timedelta(hours=1),
)

simulation.write_input_file(Path("simulation_settings_setup"))
