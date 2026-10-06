"""
This file is part of PIConGPU.
Copyright 2026 PIConGPU contributors
Authors: Julian Lenz
License: GPLv3+

Rendering primitives for the sympy-backed field functors.

Every field that is rendered into a generated C++ functor (the applied-field
components of :class:`~picongpu.picmi.applied_field.AnalyticAppliedField`, the
density / momentum / spread fields of
:class:`~picongpu.picmi.distribution.AnalyticDistribution.AnalyticDistribution`,
...) follows the same rules:

* the free variables must be exactly the supported coordinates/time,
* parameters must not shadow a live identifier of the generated functor,
* the expressions (and the parameter names) are rendered to PMAcc C++ with the
  :class:`PMAccPrinter`; that printer is the single source of truth for how an
  identifier is spelled (including escaping C++ keywords).

The PICMI-level expression/function/sympy triple that uses these primitives
lives in :mod:`picongpu.picmi._FieldFunctor`; the pypicongpu models consume the
rendered strings and the ``{"name": ..., "value": ...}`` parameter lists.
"""

import re
from collections.abc import Mapping

import sympy

from picongpu.pypicongpu.rendering.pmaccprinter import PMAccPrinter

_RENDERER = PMAccPrinter()

_RENDERED_CODE_MARKER = re.compile(r"pmacc::|::")

#: Identifiers that are always live inside the generated C++ functors. A
#: user-defined parameter must not shadow one of these: unlike a C++ keyword
#: (which the ``PMAccPrinter`` escapes), this would silently bind a different
#: quantity inside the generated expression.
GENERATED_IDENTIFIERS = frozenset(
    {
        # mathtools free variables + locals inside the generated functors
        "x",
        "y",
        "z",
        "t",
        "cellIdx",
        "currentStep",
        "m_unitField",
        "sim",
    }
)


def render(value) -> str:
    """
    Render a sympy expression, a number or an already-rendered C++ string.

    ``None`` denotes a zero component. Plain strings are parsed by sympy, while
    strings that already carry rendered PMAcc code (detected by a ``::``) are
    passed through verbatim, so a model round trip does not re-render.
    """
    if value is None:
        return "0"
    if isinstance(value, str) and _RENDERED_CODE_MARKER.search(value):
        return value
    return _RENDERER.doprint(value)


def render_identifier(name: str) -> str:
    """
    Render a single identifier through the PMAccPrinter.

    The printer escapes language keywords (e.g. ``float`` -> ``float_``) using
    its own reserved-word data, so user-provided parameter names cannot silently
    emit invalid C++. The same function is used for the parameter declaration in
    the generated functor, keeping the declaration and the expression in sync.
    """
    return _RENDERER.doprint(sympy.Symbol(name))


def sympify_expression(expression, locals: Mapping[str, sympy.Symbol] | None = None) -> sympy.Expr:
    """
    Parse a PICMI expression string.

    Mirrors the PICMI-standard normalisation (newlines are removed) and coerces
    non-string inputs to their string form, so a bare number becomes a constant
    expression. ``locals`` names the user-defined parameters as symbols, so a
    parameter whose name also exists in sympy's namespace (e.g. ``E1``) is still
    parsed as that symbol rather than as the sympy object.
    """
    return sympy.sympify(f"{expression}".replace("\n", ""), locals=locals or {})


def check_parameter_names(names) -> None:
    """
    Reject parameter names that would shadow a live generated identifier.

    C++ keywords are *not* checked here: the :class:`PMAccPrinter` escapes them
    when rendering, so a parameter named ``float`` simply becomes ``float_`` in
    the generated code (both in the declaration and in the expression).
    """
    for name in names:
        if name in GENERATED_IDENTIFIERS:
            raise ValueError(
                f"Parameter name {name!r} collides with a coordinate/time variable or a generated "
                "identifier in the C++ field functors (x, y, z, t, cellIdx, currentStep, "
                "m_unitField, sim); choose a different name."
            )


def check_allowed_symbols(expressions, allowed: set[str], context: str) -> None:
    """
    Reject expressions that reference symbols we cannot resolve.

    The generated C++ functors only define the supported free variables plus the
    user-defined parameters, so any other symbol would be rendered as undefined
    C++ and only fail (cryptically) at device-compile time. Fail in Python.
    """
    undefined: set[str] = set()
    for expression in expressions.values():
        undefined |= {str(symbol) for symbol in sympy.sympify(expression).free_symbols} - allowed
    if undefined:
        raise ValueError(
            f"{context} expression(s) reference undefined symbol(s) {sorted(undefined)}; the "
            "generated C++ functors only know the position (x/y/z), the time (t) and the "
            "parameters passed as additional keyword arguments."
        )
