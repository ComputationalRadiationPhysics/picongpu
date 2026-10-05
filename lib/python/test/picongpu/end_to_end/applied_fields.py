"""
This file is part of PIConGPU.
Copyright 2026 PIConGPU contributors
Authors: Julian Lenz
License: GPLv3+

Predefined background (applied) field setups for the end-to-end test.

The fields are attached to the simulation declaratively (via
``Simulation(..., applied_fields=[...])``). Several of them are used at once so
that the test also covers the summation of multiple applied fields into the
single C++ background functor pair.

The expected total field is not written out separately: it is obtained from the
call operator of the very same :class:`~picongpu.picmi.applied_field.AnalyticAppliedField`
/ :class:`~picongpu.picmi.applied_field.ConstantAppliedField` instances through
:func:`combined_field_values`, so the test compares the dumped field against the
PICMI input itself rather than against a hand-maintained copy.
"""

from sympy import cos, exp, sin

from picongpu import picmi

#: The E/B components in the order used by the PICMI classes.
COMPONENTS = ("Ex", "Ey", "Ez", "Bx", "By", "Bz")

# A spatial-temporal scale in SI units that keeps the field smooth over the
# small test box (positions are between 0 and 64 m; see arbitrary_parameters).
WAVENUMBER = 2.0 * 3.141592653589793 / 64.0
FREQUENCY = 2.0 * 3.141592653589793 / 50.0


def build_applied_fields():
    """The applied fields of the test, in the order they are summed.

    Covers a constant field, an analytic field given by expression strings, and
    an analytic field given by callables, so all three input spellings are
    exercised end to end.
    """
    return [
        # constant field: one E and one B component
        picmi.ConstantAppliedField(Ex=1.0e6, Bz=0.5),
        # analytic strings, including additional named parameters
        picmi.AnalyticAppliedField(
            Ey_expression="E0 * sin(k * x) * cos(w * t)",
            Bx_expression="B0 * exp(-t / tau)",
            E0=2.0e6,
            B0=0.25,
            k=WAVENUMBER,
            w=FREQUENCY,
            tau=1.0e-3,
        ),
        # analytic callables, mirroring the AnalyticDistribution interface;
        # the named parameters are passed as additional keyword arguments
        picmi.AnalyticAppliedField(
            Ez_function=lambda x, y, z, t, E1, k, w: E1 * sin(k * y) * cos(w * t),
            By_function=lambda x, y, z, t, B1, tau: B1 * exp(-t / tau),
            E1=1.5e6,
            B1=0.125,
            k=WAVENUMBER,
            w=FREQUENCY,
            tau=1.0e-3,
        ),
    ]


APPLIED_FIELDS = build_applied_fields()


def combined_field_values(x, y, z, t):
    """Sum the call operator of every applied field, per component.

    This is the Python-side reference for the field the C++ core renders from
    the same inputs: the ``Simulation`` sums the applied fields per component,
    exactly as this helper does. Components that no field sets are ``None``.
    """
    totals = dict.fromkeys(COMPONENTS)
    for applied_field in APPLIED_FIELDS:
        for component, value in applied_field(x, y, z, t).items():
            if value is None:
                continue
            totals[component] = value if totals[component] is None else totals[component] + value
    return totals
