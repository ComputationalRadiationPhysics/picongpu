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

Defines a simulation with a generic ``OpticalImaging`` diagnostic.

Unlike the ``Shadowgraphy`` preset, ``OpticalImaging`` has no defaults: the
three mask functions are user Python callables that are rendered into the C++
plugin. Here they are constant, which is the simplest possible imaging setup
(no windowing, no Fourier filtering).
"""

from pathlib import Path

from picongpu import picmi
from picongpu.picmi.diagnostics import OpticalImaging

grid = picmi.Cartesian3DGrid(
    number_of_cells=[64, 64, 128],
    lower_bound=[0, 0, 0],
    upper_bound=[6.4e-6, 6.4e-6, 12.8e-6],
    lower_boundary_conditions=["open", "open", "open"],
    upper_boundary_conditions=["open", "open", "open"],
)
solver = picmi.ElectromagneticSolver(method="Yee", cfl=1.0, grid=grid)
distribution = picmi.UniformDistribution(density=1e23)
layout = picmi.PseudoRandomLayout(n_macroparticles_per_cell=1)
electrons = picmi.Species(name="electrons", particle_type="electron", initial_distribution=distribution)


# BEGIN-OPTICAL-IMAGING-CUSTOM
# The three mask functions receive the plugin's coordinates and must return a
# sympy expression. They may reference the compile-time `params::*` constants
# and `sim.*` quantities by C++ name, e.g. `sympy.Symbol("params::posWfSizeX")`.
def constant_mask(*_):
    return 1.0


optical_imaging = OpticalImaging(
    start=200,
    duration=400,
    slice_point=0.75,
    focus_pos=1e-3,
    fourier_output=True,
    position_wf=constant_mask,
    time_wf=constant_mask,
    mask_fourier=constant_mask,
)
# END-OPTICAL-IMAGING-CUSTOM

sim = picmi.Simulation(
    max_steps=1000,
    solver=solver,
    species=[electrons],
    layouts=[layout],
    diagnostics=[optical_imaging],
)

sim.run(setup_dir=Path("optical_imaging_custom_setup"), run_dir=Path("optical_imaging_custom_run"))
