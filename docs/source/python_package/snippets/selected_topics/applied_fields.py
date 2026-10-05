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

Defines a constant and an analytic applied (background) field and attaches them
declaratively to a simulation.
"""

from pathlib import Path

import numpy as np
from sympy import pi, sin

from picongpu import picmi

grid = picmi.Cartesian3DGrid(
    number_of_cells=[128, 128, 128],
    lower_bound=[0.0, 0.0, 0.0],
    upper_bound=[1.0e-6, 1.0e-6, 1.0e-6],
    lower_boundary_conditions=["open", "open", "open"],
    upper_boundary_conditions=["open", "open", "open"],
)
solver = picmi.ElectromagneticSolver(method="Yee", cfl=0.95, grid=grid)

# BEGIN-APPLIED-FIELD-CONSTANT
# a spatially and temporally constant field: Ex in V/m, Bz in T
constant_field = picmi.ConstantAppliedField(
    Ex=1.0e6,
    Bz=0.5,
)
# END-APPLIED-FIELD-CONSTANT

# BEGIN-APPLIED-FIELD-ANALYTIC
# x, y, z (position in m) and t (time in s) are the free variables.
# Each component is backed by the same _FieldFunctor and accepts any of the
# interchangeable spellings: a sympy-parseable ``*_expression`` string, a
# ``*_function`` callable, or the read-back ``*_sympy`` expression. Named
# parameters are passed as additional keyword arguments.
analytic_field = picmi.AnalyticAppliedField(
    Ex_expression="E0 * sin(2 * pi * y / wavelength) * cos(2 * pi * t / period)",
    Ey_function=lambda x, y, z, t, E1, period: E1 * sin(2 * pi * t / period),
    E0=1.0e5,
    E1=2.0e5,
    wavelength=0.8e-6,
    period=50.0e-15,
)
# END-APPLIED-FIELD-ANALYTIC

# BEGIN-APPLIED-FIELD-SYMPY
# whichever spelling was given, all three are available and consistent after
# construction; ``*_sympy`` is the resolved sympy expression of the component:
assert analytic_field.Ex_sympy is not None
assert analytic_field.Ey_sympy is not None
assert analytic_field.Ex_expression is not None
assert analytic_field.Ey_function is not None
# END-APPLIED-FIELD-SYMPY

# BEGIN-APPLIED-FIELD-CALL
# every applied field is callable: it returns the six components (in SI units)
# at the given coordinates, exactly as the C++ functor evaluates them
values = analytic_field(x=0.3e-6, y=0.4e-6, z=0.0, t=10.0e-15)
assert set(values) == {"Ex", "Ey", "Ez", "Bx", "By", "Bz"}
assert values["Ez"] is None  # no field was set for that component
# the coordinates may be numpy arrays; the components are then arrays too
positions = np.array([0.1e-6, 0.2e-6, 0.3e-6])
assert np.shape(analytic_field(x=positions, y=positions, z=positions, t=0.0)["Ex"]) == positions.shape
# END-APPLIED-FIELD-CALL

# BEGIN-APPLIED-FIELD-INFLUENCE
# PIConGPU-specific visibility knobs (the picongpu_ prefix marks code-specific
# PICMI inputs): whether plugins and dumps see the background. The background is
# always applied around the particle push; these knobs configure the single
# background functor pair, so all applied fields of one simulation must agree.
influence_field = picmi.AnalyticAppliedField(
    Ex_expression="1.0e5 * y",
    picongpu_influences_plugins=False,
    picongpu_influences_dumps=True,
)
# END-APPLIED-FIELD-INFLUENCE

# BEGIN-APPLIED-FIELD-ADD
# applied fields are attached declaratively at construction; their contributions
# are summed per component into the single background the C++ core evaluates
simulation = picmi.Simulation(
    max_steps=100,
    solver=solver,
    applied_fields=[constant_field, analytic_field],
)
# END-APPLIED-FIELD-ADD

simulation.write_input_file(Path("applied_fields_setup"))
