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

Defines a simulation with two shadowgraphy diagnostics.

``Shadowgraphy`` is the ready-made optical-imaging preset: it fills in the Tukey
windows and the numerical-aperture band-pass mask and enables the final
shadowgram output. The plugin is a multi-instance plugin, so two ``Shadowgraphy``
diagnostics may run side by side (e.g. two slices at different depths).
"""

from pathlib import Path

from picongpu import picmi
from picongpu.picmi.diagnostics import Shadowgraphy

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

# BEGIN-OPTICAL-IMAGING-SHADOWGRAPHY
shadowgraphy = Shadowgraphy(
    start=0,
    duration=600,
    slice_point=0.5,
    file="shadowgram",
    ext="bp5",
)

# the plugin is a multi-instance plugin: a second image at another slice depth
shadowgraphy_back = Shadowgraphy(
    start=200,
    duration=400,
    slice_point=0.75,
    file="shadowgram_back",
    fourier_output=True,
)
# END-OPTICAL-IMAGING-SHADOWGRAPHY

sim = picmi.Simulation(
    max_steps=1000,
    solver=solver,
    species=[electrons],
    layouts=[layout],
    diagnostics=[shadowgraphy, shadowgraphy_back],
)

sim.run(setup_dir=Path("optical_imaging_setup"), run_dir=Path("optical_imaging_run"))
