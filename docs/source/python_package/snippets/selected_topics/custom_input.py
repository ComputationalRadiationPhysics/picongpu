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

Registers custom user input and a custom template directory on a simulation.
The custom values become available to the templates as
``{{{customuserinput.<key>}}}``.
"""

from pathlib import Path

from picongpu import picmi
from picongpu.pypicongpu.customuserinput import CustomUserInput

# a template directory that shadows the packaged templates;
# here it only adds one extra file
template_dir = Path("templates")
(template_dir / "include" / "picongpu").mkdir(parents=True, exist_ok=True)
(template_dir / "include" / "picongpu" / "my_param.mustache").write_text(
    "// rendered from custom input\nconstexpr int MY_NUMBER = {{{customuserinput.my_number}}};\n"
)

grid = picmi.Cartesian3DGrid(
    number_of_cells=[32, 32, 32],
    lower_bound=[0, 0, 0],
    upper_bound=[1e-6, 1e-6, 1e-6],
    lower_boundary_conditions=["periodic", "periodic", "periodic"],
    upper_boundary_conditions=["periodic", "periodic", "periodic"],
)
solver = picmi.ElectromagneticSolver(method="Yee", cfl=0.5, grid=grid)

simulation = picmi.Simulation(
    max_steps=100,
    solver=solver,
    picongpu_template_dir=template_dir,
)

custom = CustomUserInput()
custom.addToCustomInput({"my_number": 42}, tag="my_parameters")
simulation.picongpu_add_custom_user_input(custom)

simulation.write_input_file(Path("custom_input_setup"))
print((Path("custom_input_setup") / "include" / "picongpu" / "my_param").read_text())
print("It worked!")
