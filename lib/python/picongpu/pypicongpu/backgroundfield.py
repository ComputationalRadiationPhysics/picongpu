"""
This file is part of PIConGPU.
Copyright 2026 PIConGPU contributors
Authors: Julian Lenz
License: GPLv3+
"""

from typing import Annotated, Literal

from pydantic import BaseModel, BeforeValidator, Field, model_validator

from ._field_functor import check_parameter_names, render_identifier
from ._field_functor import render as _render_field_expression


class _Parameter(BaseModel):
    """A named parameter used inside a field expression."""

    name: Annotated[str, BeforeValidator(render_identifier)]
    """
    name of the parameter as used inside the expressions, rendered through the
    PMAccPrinter so that C++ keywords are escaped (``float`` -> ``float_``),
    identical to how the printer spells the symbol inside the expressions.
    """
    value: float
    """value assigned to the parameter (SI units)"""


class BackgroundField(BaseModel):
    """
    Background field applied to the grid E and B fields.

    The background is added to the existing field values around the particle
    push, i.e. particles feel it but the solver itself does not evolve it
    (see the C++ ``fieldBackground.param`` + ``FieldBackground.hpp``).

    The field expressions denote the SI value of the respective component
    (V/m for E, T for B) as a function of position ``x``, ``y``, ``z`` (m) and
    time ``t`` (s). Expressions are compiled to device functors via the
    PMAccPrinter, i.e. they must be expressions sympy can parse and print.
    The rendering itself lives in ``_field_functor`` so that it is shared with
    :class:`~picongpu.picmi.distribution.AnalyticDistribution.AnalyticDistribution`.

    This is the minimal, whole-domain variant of the applied-field feature.
    The design deliberately mirrors the PICMI applied-field surface. Field
    arithmetic, ``as_initial`` or ``as_injected`` map onto *separate* C++
    mechanisms (initial field assignment, incident-field planes) that would
    need their own models and templates; they are not implemented here.

    Several PICMI applied fields may be combined into one ``BackgroundField``:
    the translation sums the individual contributions per component (see
    :meth:`~picongpu.picmi.simulation.Simulation._get_background_field`).
    """

    type_backgroundfield: Literal[True] = True
    """discriminator for the renderer (always True)"""

    ex: Annotated[str, BeforeValidator(_render_field_expression)] = "0"
    """E_x component of the background field in V/m"""
    ey: Annotated[str, BeforeValidator(_render_field_expression)] = "0"
    """E_y component of the background field in V/m"""
    ez: Annotated[str, BeforeValidator(_render_field_expression)] = "0"
    """E_z component of the background field in V/m"""
    bx: Annotated[str, BeforeValidator(_render_field_expression)] = "0"
    """B_x component of the background field in T"""
    by: Annotated[str, BeforeValidator(_render_field_expression)] = "0"
    """B_y component of the background field in T"""
    bz: Annotated[str, BeforeValidator(_render_field_expression)] = "0"
    """B_z component of the background field in T"""

    influences_plugins: bool = True
    """
    Whether plugins see the background (the C++
    ``fieldBackground.influencesPlugins`` runtime option, default ``True``).
    """

    influences_dumps: bool = True
    """
    Whether dumps (incl. checkpoints) include the background (the C++
    ``fieldBackground.influencesDumps`` runtime option, default ``True``).
    """

    user_defined_kw: list[_Parameter] = Field(default_factory=list)
    """
    Named parameters used inside the field expressions.

    They are rendered as compile-time constants inside the generated functors
    so that symbolic parameters of an ``AnalyticAppliedField`` resolve
    correctly (e.g. ``{"name": "wl", "value": 8.0e-7}``).
    """

    @model_validator(mode="after")
    def _check_parameter_names(self):
        check_parameter_names(parameter.name for parameter in self.user_defined_kw)
        return self
