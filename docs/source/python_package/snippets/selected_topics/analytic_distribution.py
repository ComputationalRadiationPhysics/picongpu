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

Defines an analytic density profile together with a constant drift and thermal
spread. The three field families -- the density, the per-axis ``momentum``
(``gamma * velocity``) and the per-axis ``momentum_spread`` (Gaussian sigma) --
are treated on the same footing: each is given either as a sympy-parseable
string, as a sympy callable with the same keyword substitution, or read back as
a parsed sympy expression.
"""

from pathlib import Path

from sympy import exp

from picongpu import picmi


# BEGIN-FIELD-EXPRESSIONS
# density, per-axis momentum (gamma * velocity [m/s]) and per-axis momentum
# spread (Gaussian sigma [m/s]) as sympy-parseable strings; `None` marks an axis
# that is not supplied. `n0`, `k`, `vz` and `vth` are constants that are
# collected automatically into `user_defined_kw`:
from_expression = picmi.AnalyticDistribution(
    density_expression="n0 * exp(-(x**2 + y**2) * k)",
    momentum_expressions=[None, None, "vz"],
    momentum_spread_expressions=[None, None, "vth"],
    n0=1e25,
    k=1e14,
    vz=1.0e6,
    vth=1.0e5,
)
# END-FIELD-EXPRESSIONS

# BEGIN-FIELD-FUNCTIONS
# each family may instead be given as a sympy callable, with the same extra
# (beyond x, y and z, which stay positional) keyword arguments consumed the
# same way:
from_function = picmi.AnalyticDistribution(
    density_function=lambda x, y, z, n0, k: n0 * exp(-(x**2 + y**2) * k),
    momentum_functions=[None, None, lambda x, y, z, vz: vz],
    momentum_spread_functions=[None, None, lambda x, y, z, vth: vth],
    n0=1e25,
    k=1e14,
    vz=1.0e6,
    vth=1.0e5,
)

# the two spellings are interchangeable and describe the very same distribution:
assert from_function == from_expression
# END-FIELD-FUNCTIONS

# BEGIN-FIELD-SYMPY
# whichever spelling was given, both are available after construction and the
# parsed sympy expressions are exposed as public properties:
assert from_function.density_sympy == from_expression.density_sympy
assert from_function.momentum_sympy == from_expression.momentum_sympy
assert from_function.momentum_spread_sympy == from_expression.momentum_spread_sympy
# END-FIELD-SYMPY

# BEGIN-FIELD-KWARGS
# constants referenced in any of the three families are collected uniformly from
# the extra keyword arguments into `user_defined_kw` and substituted before the
# C++ is rendered:
assert from_expression.user_defined_kw == {"n0": 1e25, "k": 1e14, "vz": 1.0e6, "vth": 1.0e5}
# END-FIELD-KWARGS

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
    initial_distribution=from_expression,
)
layout = picmi.PseudoRandomLayout(n_macroparticles_per_cell=2)

simulation = picmi.Simulation(
    max_steps=100,
    solver=solver,
    species=[electrons],
    layouts=[layout],
)

simulation.write_input_file(Path("analytic_distribution_setup"))
