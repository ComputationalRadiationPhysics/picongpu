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

Defines an analytic density profile in two equivalent ways:
as a sympy function handed over as a decorator, and as a
sympy-parseable ``density_expression`` string.
"""

from pathlib import Path

from sympy import exp

from picongpu import picmi


# BEGIN-DENSITY-FUNCTION
@picmi.AnalyticDistribution
def density(x, y, z):
    return 1e25 * exp(-(((x - 1e-6) / 1e-7) ** 2))


# END-DENSITY-FUNCTION

# BEGIN-DENSITY-EXPRESSION
# the same profile as a sympy-parseable string of x, y and z:
density_string = picmi.AnalyticDistribution(density_expression="1e25 * exp(-((x - 1e-6) / 1e-7) ** 2)")
# END-DENSITY-EXPRESSION

grid = picmi.Cartesian3DGrid(
    number_of_cells=[32, 32, 32],
    lower_bound=[0.0, 0.0, 0.0],
    upper_bound=[2e-6, 2e-6, 2e-6],
    lower_boundary_conditions=["periodic", "periodic", "periodic"],
    upper_boundary_conditions=["periodic", "periodic", "periodic"],
)
solver = picmi.ElectromagneticSolver(method="Yee", cfl=0.7, grid=grid)

electrons = picmi.Species(
    name="electrons",
    particle_type="electron",
    initial_distribution=density_string,
)
layout = picmi.PseudoRandomLayout(n_macroparticles_per_cell=2)

simulation = picmi.Simulation(
    max_steps=100,
    solver=solver,
    species=[electrons],
    layouts=[layout],
)

simulation.write_input_file(Path("analytic_distribution_setup"))
