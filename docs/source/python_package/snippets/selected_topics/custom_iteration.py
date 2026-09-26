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

Shows a custom template that iterates over all species in the
rendering context and combines it with custom user input.
"""

from pathlib import Path

from picongpu import picmi
from picongpu.pypicongpu.customuserinput import CustomUserInput

template_dir = Path("templates")
(template_dir / "include" / "picongpu").mkdir(parents=True, exist_ok=True)
(template_dir / "include" / "picongpu" / "species_report.mustache").write_text(
    "// one line per species, with a custom number:\n"
    "{{#species}}\n"
    "// {{{name}}}: {{{customuserinput.report_tag}}}\n"
    "{{/species}}\n"
)

grid = picmi.Cartesian3DGrid(
    number_of_cells=[32, 32, 32],
    lower_bound=[0, 0, 0],
    upper_bound=[1e-6, 1e-6, 1e-6],
    lower_boundary_conditions=["periodic", "periodic", "periodic"],
    upper_boundary_conditions=["periodic", "periodic", "periodic"],
)
solver = picmi.ElectromagneticSolver(method="Yee", cfl=0.5, grid=grid)

electrons = picmi.Species(
    name="electrons",
    particle_type="electron",
    initial_distribution=picmi.UniformDistribution(density=1e23),
)
ions = picmi.Species(
    name="ions",
    particle_type="H",
    charge_state=1,
    initial_distribution=picmi.UniformDistribution(density=1e23),
)

simulation = picmi.Simulation(
    max_steps=100,
    solver=solver,
    species=[electrons, ions],
    layouts=[picmi.PseudoRandomLayout(n_macroparticles_per_cell=1)] * 2,
    picongpu_template_dir=template_dir,
)

custom = CustomUserInput()
custom.addToCustomInput({"report_tag": "mine"}, tag="report")
simulation.picongpu_add_custom_user_input(custom)

simulation.write_input_file(Path("custom_iteration_setup"))
print((Path("custom_iteration_setup") / "include" / "picongpu" / "species_report").read_text())
