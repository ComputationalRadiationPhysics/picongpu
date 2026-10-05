"""
This file is part of PIConGPU.
Copyright 2025 PIConGPU contributors
Authors: Julian Lenz
License: GPLv3+
"""

from unittest import TestCase
import math
import numpy as np

from scipy.constants import c
from sympy import Piecewise, exp, sin, symbols, sympify

from picongpu.picmi import AnalyticDistribution
from picongpu.picmi.species import Species
from picongpu.picmi.species_requirements import SimpleMomentumOperation, run_construction
from picongpu.pypicongpu.util import UnsupportedFeatureError
import pytest

x, y, z = symbols("x, y, z")

# allow numpy broadcasting (see https://numpy.org/doc/stable/user/basics.broadcasting.html)
# some examples to check:
VALID_CALLS = [
    # scalar arguments produce scalar results
    ((1, 2, 3), 6),
    # broadcasting in the first argument, function is evaluated for (1,2,3) and (2,2,3)
    (([1, 2], 2, 3), [6, 12]),
    # broadcasting in the last argument, (1,2,3) and (1,2,4)
    ((1, 2, [3, 4]), [6, 8]),
    # broadcasting in all arguments, shapes must match, scalar arguments are (1,3,5) and (2,4,6)
    (([1, 2], [3, 4], [5, 6]), [15, 48]),
]

INVALID_DENSITIES = [
    # wrong number of arguments
    (lambda x, y: x + y, TypeError),
    (lambda x, y, z, too_much: x + y + z + too_much, TypeError),
    # bad return type
    (lambda x, y, z: "string", TypeError),
    # constructs not understood by sympy
    (lambda x, y, z: x if x > 0 else y * z, TypeError),
]


def velocity(gamma):
    return np.sqrt(c**2 * (1.0 - 1.0 / gamma**2))


class TestAnalyticDistribution(TestCase):
    def setUp(self):
        self.valid_density = lambda x, y, z: x * y * z
        self.dist = AnalyticDistribution(self.valid_density, directed_velocity=(1.0, 2.0, 3.0))

    def test_density_expression_invalid(self):
        for density, err in INVALID_DENSITIES:
            with self.subTest(density=density, err=err):
                with pytest.raises(err):
                    AnalyticDistribution(density).get_as_pypicongpu()

    def test_drift_input_types(self):
        types = [list, tuple, np.array]
        # this needs to be large, so that gamma != 1
        drift = 1.0e7 * np.array([3.0, 4.0, 5.0])
        for t in types:
            dist = AnalyticDistribution(lambda x, y, z: x + y + z, directed_velocity=t(drift))
            result = dist.get_picongpu_drift()
            np.testing.assert_allclose(velocity(result.gamma) * np.asarray(result.direction_normalized), drift)

    def test_drift_is_none_for_vanishing_vector(self):
        assert AnalyticDistribution(lambda *x: sum(x), directed_velocity=[0, 0, 0]).get_picongpu_drift() is None

    def test_drift_wrong_dimensionality(self):
        from pydantic_core import ValidationError

        # Test drift with wrong dimensionality
        with pytest.raises(ValidationError):
            AnalyticDistribution(
                lambda x, y, z: x + y + z,
                # Only 2 elements
                directed_velocity=[1.0, 2.0],
            ).get_picongpu_drift()

    def test_call(self):
        for args, result in VALID_CALLS:
            with self.subTest(args=args, result=result):
                np.testing.assert_allclose(np.asarray(self.dist(*args)), np.asarray(result))


class TestAnalyticDistributionFromExpression(TestCase):
    """
    A density_expression string must sympify into exactly the same
    density_function as the equivalent hand-written callable, so the
    rendered C++ is identical.
    """

    CASES = [
        ("x*y*z", lambda x, y, z: x * y * z),
        ("x**2 + y**2 + 2*z*sin(x)", lambda x, y, z: x**2 + y**2 + 2 * z * sin(x)),
        ("Piecewise((1, x < 1), (2, True))", lambda x, y, z: Piecewise((1, x < 1), (2, True))),
        ("10**20 * exp(-x**2 / 2)", lambda x, y, z: 10**20 * exp(-(x**2) / 2)),
    ]

    def test_density_function_matches_equivalent_callable(self):
        for expression, callable_density in self.CASES:
            with self.subTest(expression=expression):
                string_dist = AnalyticDistribution(density_expression=expression)
                callable_dist = AnalyticDistribution(callable_density)
                x, y, z = symbols("x, y, z")
                self.assertEqual(string_dist.density_function(x, y, z), callable_dist.density_function(x, y, z))
                # the standard string field round-trips to the equivalent sympy expression
                self.assertEqual(sympify(string_dist.density_expression), callable_dist.density_function(x, y, z))

    def test_rendered_cpp_matches_equivalent_callable(self):
        for expression, callable_density in self.CASES:
            with self.subTest(expression=expression):
                string_dist = AnalyticDistribution(density_expression=expression)
                callable_dist = AnalyticDistribution(callable_density)
                self.assertEqual(
                    string_dist.get_as_pypicongpu(None).function_body,
                    callable_dist.get_as_pypicongpu(None).function_body,
                )

    def test_both_density_fields_are_always_available(self):
        x, y, z = symbols("x, y, z")
        for expression, callable_density in self.CASES:
            with self.subTest(expression=expression):
                from_expression = AnalyticDistribution(density_expression=expression)
                from_callable = AnalyticDistribution(callable_density)
                # whichever input was given, both the string and the callable are populated
                for dist in (from_expression, from_callable):
                    self.assertEqual(dist.density_function(x, y, z), sympify(expression))
                    self.assertEqual(sympify(dist.density_expression), sympify(expression))
                    self.assertEqual(dist.density_sympy, sympify(expression))

    def test_density_sympy_is_public_counterpart_of_density_function(self):
        # density_sympy needs no symbols/arguments and matches the rendered density
        distribution = AnalyticDistribution(density_expression="x**2 + y**2")
        self.assertEqual(distribution.density_sympy, symbols("x") ** 2 + symbols("y") ** 2)
        self.assertEqual(
            distribution.get_as_pypicongpu(None).function_body,
            AnalyticDistribution(distribution.density_function).get_as_pypicongpu(None).function_body,
        )

    def test_expression_is_normalised_before_sympify(self):
        # the standard string normalisation removes newlines, so an
        # indented / line-broken expression parses to the same density.
        indented = "x*y\n + z"
        flat = "x*y + z"
        self.assertEqual(
            AnalyticDistribution(density_expression=indented).density_expression,
            AnalyticDistribution(density_expression=flat).density_expression,
        )

    def test_string_density_is_callable(self):
        string_dist = AnalyticDistribution(density_expression="x*y*z")
        np.testing.assert_allclose(np.asarray(string_dist(1, 2, 3)), np.asarray(6))

    def test_field_assignment_is_validated_and_supported(self):
        # validate_assignment=True (inherited from the standard base) must not make
        # standard fields immutable: re-validating an already-constructed instance
        # has both density fields set, so the "exactly one input" rule must not re-fire.
        d = AnalyticDistribution(density_expression="x+y")
        d.rms_velocity = [1.0, 2.0, 3.0]
        self.assertEqual(d.rms_velocity, [1.0, 2.0, 3.0])
        d.rms_velocity = d.rms_velocity
        self.assertEqual(d.rms_velocity, [1.0, 2.0, 3.0])
        # an invalid assignment is still rejected (values are not floats)
        with pytest.raises(Exception):
            d.rms_velocity = ["a", "b", "c"]

    def test_assigning_a_density_field_keeps_the_other_in_sync(self):
        x, y, z = symbols("x, y, z")
        d = AnalyticDistribution(density_expression="x+y")
        d.density_expression = "x*y"
        self.assertEqual(d.density_sympy, sympify("x*y"))
        self.assertEqual(d.density_function(x, y, z), x * y)
        d.density_function = lambda x, y, z: x + y + z
        self.assertEqual(d.density_sympy, sympify("x+y+z"))
        self.assertEqual(sympify(d.density_expression), x + y + z)


def _momentum_of(distribution):
    """translate the drift/temperature of an AnalyticDistribution through a real species"""
    species = Species(name="e", particle_type="electron", initial_distribution=distribution)
    return run_construction(SimpleMomentumOperation(species))


class TestAnalyticDistributionFullSurface(TestCase):
    """
    The standard surface of PICMI_AnalyticDistribution: constant momentum /
    momentum_spread expressions rendered to the pypicongpu Drift/Temperature ops,
    the automatic user_defined_kw parameter substitution, and the PIConGPU
    density_function callable extension (kept alongside density_expression).
    """

    def test_constant_momentum_expressions_render_to_drift(self):
        # a constant gamma*velocity per axis becomes a Drift via from_gamma_velocity
        distribution = AnalyticDistribution(density_expression="1", momentum_expressions=[None, "vx", None], vx=3.0e7)
        drift = distribution.get_picongpu_drift()
        self.assertIsNotNone(drift)
        self.assertLess(math.sqrt(1 + (3.0e7 / c) ** 2) - drift.gamma, 1e-9)
        self.assertEqual(drift.direction_normalized, (0.0, 1.0, 0.0))
        # and through a full species, the drift op is attached
        self.assertEqual(_momentum_of(distribution).drift, drift)

    def test_constant_momentum_expressions_literal(self):
        distribution = AnalyticDistribution(density_expression="1", momentum_expressions=[None, None, "2e7"])
        drift = distribution.get_picongpu_drift()
        self.assertLess(math.sqrt(1 + (2.0e7 / c) ** 2) - drift.gamma, 1e-9)
        self.assertEqual(drift.direction_normalized, (0.0, 0.0, 1.0))

    def test_no_momentum_expressions_keeps_directed_velocity_semantics(self):
        # the legacy directed_velocity is a plain velocity (from_velocity), not gamma*velocity
        drift = 1.0e7 * np.array([3.0, 4.0, 5.0])
        distribution = AnalyticDistribution(lambda x, y, z: x + y + z, directed_velocity=drift)
        result = distribution.get_picongpu_drift()
        np.testing.assert_allclose(
            np.sqrt(c**2 * (1.0 - 1.0 / result.gamma**2)) * np.asarray(result.direction_normalized), drift
        )

    def test_zero_momentum_expressions_give_no_drift(self):
        self.assertIsNone(
            AnalyticDistribution(density_expression="1", momentum_expressions=[0, 0, 0]).get_picongpu_drift()
        )

    def test_constant_momentum_spread_render_to_temperature(self):
        # a constant Gaussian sigma per axis becomes a (directional) Temperature
        distribution = AnalyticDistribution(density_expression="1", momentum_spread_expressions=[None, None, 1.0e5])
        self.assertEqual(distribution.picongpu_get_rms_velocity_si(), (0.0, 0.0, 1.0e5))
        temperature = _momentum_of(distribution).temperature
        self.assertIsNotNone(temperature)
        self.assertEqual(temperature.temperature_kev, None)
        self.assertEqual(temperature.temperature_kev_directional[2], 5.685630111285689e-05)

    def test_constant_momentum_spread_isotropic(self):
        distribution = AnalyticDistribution(density_expression="1", momentum_spread_expressions=[1.0e5, 1.0e5, 1.0e5])
        temperature = _momentum_of(distribution).temperature
        self.assertIsNotNone(temperature)
        self.assertEqual(temperature.temperature_kev_directional, None)
        self.assertEqual(temperature.temperature_kev, 5.685630111285689e-05)

    def test_position_dependent_momentum_rejected(self):
        with self.assertRaises(UnsupportedFeatureError):
            AnalyticDistribution(density_expression="1", momentum_expressions=["x", None, None]).get_picongpu_drift()
        with self.assertRaises(UnsupportedFeatureError):
            AnalyticDistribution(
                density_expression="1", momentum_spread_expressions=[None, None, "z"]
            ).picongpu_get_rms_velocity_si()

    def test_unresolved_momentum_parameter_rejected(self):
        # a momentum expression referencing a kwarg that was never supplied raises
        with self.assertRaises(ValueError):
            AnalyticDistribution(density_expression="1", momentum_expressions=["v", None, None]).get_picongpu_drift()

    def test_user_defined_kw_substituted_in_rendered_density(self):
        distribution = AnalyticDistribution(density_expression="n0 * x", n0=2.0)
        self.assertEqual(distribution.user_defined_kw, {"n0": 2.0})
        # the collected constant is substituted into the rendered density
        self.assertIn("2.0*x", distribution.get_as_pypicongpu(None).function_body)
        # and the callable evaluates the substituted constant
        np.testing.assert_allclose(np.asarray(distribution(3, 0, 0)), np.asarray(6.0))

    def test_user_defined_kw_in_momentum_expression(self):
        distribution = AnalyticDistribution(density_expression="1", momentum_expressions=[None, "vx", None], vx=3.0e7)
        self.assertEqual(distribution.user_defined_kw, {"vx": 3.0e7})
        self.assertLess(math.sqrt(1 + (3.0e7 / c) ** 2) - distribution.get_picongpu_drift().gamma, 1e-9)

    def test_momentum_and_spread_reach_species_operations(self):
        # the standard momentum/spread expressions (with their user_defined_kw
        # substituted) must reach the pypicongpu species operations: a constant
        # gamma*velocity along y and a Gaussian sigma along z.
        distribution = AnalyticDistribution(
            density_expression="n0",
            n0=1.0e25,
            momentum_expressions=[None, "vx", None],
            momentum_spread_expressions=[None, None, "vth"],
            vx=3.0e7,
            vth=1.0e5,
        )
        momentum = _momentum_of(distribution)
        self.assertIsNotNone(momentum.drift)
        self.assertEqual(momentum.drift.direction_normalized, (0.0, 1.0, 0.0))
        self.assertLess(math.sqrt(1 + (3.0e7 / c) ** 2) - momentum.drift.gamma, 1e-9)
        self.assertIsNotNone(momentum.temperature)
        self.assertIsNone(momentum.temperature.temperature_kev)
        self.assertEqual(momentum.temperature.temperature_kev_directional[0], 0.0)
        self.assertEqual(momentum.temperature.temperature_kev_directional[1], 0.0)
        self.assertEqual(momentum.temperature.temperature_kev_directional[2], 5.685630111285689e-05)

    def test_momentum_function_matches_expression_and_is_available(self):
        # the per-axis momentum callable is the sympy equivalent of the string
        # expression: both spellings are accepted and both fields are populated
        from_expression = AnalyticDistribution(
            density_expression="1", momentum_expressions=[None, "vx", None], vx=3.0e7
        )
        from_function = AnalyticDistribution(
            density_expression="1", momentum_functions=[None, lambda x, y, z: 3.0e7, None]
        )
        # both reference forms and both sympy views agree
        self.assertEqual(from_function.momentum_sympy, from_expression.momentum_sympy)
        self.assertEqual(from_function.momentum_functions[1](x, y, z), 3.0e7)
        self.assertEqual(sympify(from_function.momentum_expressions[1]), from_expression.momentum_sympy[1])
        self.assertEqual(
            from_function.get_picongpu_drift().direction_normalized,
            from_expression.get_picongpu_drift().direction_normalized,
        )
        self.assertLess(
            abs(from_function.get_picongpu_drift().gamma - from_expression.get_picongpu_drift().gamma), 1e-9
        )

    def test_momentum_spread_function_matches_expression_and_is_available(self):
        from_expression = AnalyticDistribution(
            density_expression="1", momentum_spread_expressions=[None, None, "vth"], vth=1.0e5
        )
        from_function = AnalyticDistribution(
            density_expression="1", momentum_spread_functions=[None, None, lambda x, y, z: 1.0e5]
        )
        self.assertEqual(from_function.momentum_spread_sympy, from_expression.momentum_spread_sympy)
        self.assertEqual(from_function.momentum_spread_functions[2](x, y, z), 1.0e5)
        self.assertEqual(
            sympify(from_function.momentum_spread_expressions[2]), from_expression.momentum_spread_sympy[2]
        )
        self.assertEqual(from_function.picongpu_get_rms_velocity_si(), from_expression.picongpu_get_rms_velocity_si())
        temperature = _momentum_of(from_function).temperature
        self.assertEqual(temperature.temperature_kev_directional[2], 5.685630111285689e-05)

    def test_momentum_function_kwargs_substituted(self):
        # a momentum/spread callable takes the same extra keyword arguments as density_function
        distribution = AnalyticDistribution(
            density_expression="1",
            momentum_functions=[None, None, lambda x, y, z, vz: vz],
            momentum_spread_functions=[None, None, lambda x, y, z, vth: vth],
            vz=2.0e7,
            vth=1.0e5,
        )
        self.assertEqual(distribution.user_defined_kw, {"vz": 2.0e7, "vth": 1.0e5})
        self.assertEqual(distribution.momentum_sympy, [None, None, sympify("2.0e7")])
        self.assertEqual(distribution.momentum_spread_sympy, [None, None, sympify("1.0e5")])
        self.assertEqual(distribution.picongpu_get_rms_velocity_si(), (0.0, 0.0, 1.0e5))
        self.assertLess(abs(distribution.get_picongpu_drift().gamma - math.sqrt(1 + (2.0e7 / c) ** 2)), 1e-9)

    def test_momentum_function_and_expression_must_agree(self):
        # supplying both spellings with different values for one axis is rejected
        with pytest.raises(ValueError, match="disagree"):
            AnalyticDistribution(
                density_expression="1",
                momentum_expressions=[None, None, "2e7"],
                momentum_functions=[None, None, lambda x, y, z: 3.0e7],
            )
        # equal spellings are accepted and collapse to the same surface
        distribution = AnalyticDistribution(
            density_expression="1",
            momentum_spread_expressions=[None, None, "1e5"],
            momentum_spread_functions=[None, None, lambda x, y, z: 1.0e5],
        )
        self.assertEqual(distribution.picongpu_get_rms_velocity_si(), (0.0, 0.0, 1.0e5))

    def test_position_dependent_momentum_function_rejected(self):
        with self.assertRaises(UnsupportedFeatureError):
            AnalyticDistribution(
                density_expression="1", momentum_functions=[lambda x, y, z: x, None, None]
            ).get_picongpu_drift()
        with self.assertRaises(UnsupportedFeatureError):
            AnalyticDistribution(
                density_expression="1", momentum_spread_functions=[None, None, lambda x, y, z: z]
            ).picongpu_get_rms_velocity_si()

    def test_momentum_sympy_public_properties(self):
        distribution = AnalyticDistribution(
            density_expression="1",
            momentum_expressions=[None, "vx", None],
            momentum_spread_expressions=[None, None, "vth"],
            vx=3.0e7,
            vth=1.0e5,
        )
        self.assertEqual(distribution.momentum_sympy, [None, sympify("3.0e7"), None])
        self.assertEqual(distribution.momentum_spread_sympy, [None, None, sympify("1.0e5")])

    def test_assigning_axis_function_keeps_the_expression_in_sync(self):
        distribution = AnalyticDistribution(density_expression="1")
        distribution.momentum_functions = [None, None, lambda x, y, z: 3.0e7]
        self.assertEqual(distribution.momentum_expressions[0:2], [None, None])
        self.assertEqual(sympify(distribution.momentum_expressions[2]), sympify("3.0e7"))
        self.assertEqual(distribution.momentum_sympy, [None, None, sympify("3.0e7")])
        distribution.momentum_spread_expressions = [None, None, "1e5"]
        self.assertEqual(distribution.momentum_spread_sympy, [None, None, sympify("1.0e5")])
        self.assertEqual(distribution.picongpu_get_rms_velocity_si(), (0.0, 0.0, 1.0e5))

    def test_user_defined_kw_in_momentum_spread_expression(self):
        # a constant referenced *only* in a momentum_spread_expression is collected and substituted
        distribution = AnalyticDistribution(
            density_expression="1", momentum_spread_expressions=[None, None, "sigma"], sigma=1.0e5
        )
        self.assertEqual(distribution.user_defined_kw, {"sigma": 1.0e5})
        self.assertEqual(distribution.picongpu_get_rms_velocity_si(), (0.0, 0.0, 1.0e5))
        temperature = _momentum_of(distribution).temperature
        self.assertEqual(temperature.temperature_kev_directional[2], 5.685630111285689e-05)

    def test_user_defined_kw_shared_between_density_and_spread(self):
        # a single constant used in both the density and a spread expression is collected once
        distribution = AnalyticDistribution(
            density_expression="a", momentum_spread_expressions=[None, "a", None], a=2.0e5
        )
        self.assertEqual(distribution.user_defined_kw, {"a": 2.0e5})
        self.assertEqual(distribution.picongpu_get_rms_velocity_si(), (0.0, 2.0e5, 0.0))

    def test_directed_velocity_and_momentum_expressions_conflict_rejected(self):
        # the two mutually-exclusive drift setters must not silently override each other
        with pytest.raises(ValueError, match="mutually exclusive"):
            AnalyticDistribution(
                density_expression="1", directed_velocity=(1.0e7, 0.0, 0.0), momentum_expressions=[None, 3.0e7, None]
            )
        # each on its own (with the other at its default) remains valid
        self.assertIsNotNone(
            AnalyticDistribution(density_expression="1", directed_velocity=(1.0e7, 0.0, 0.0)).get_picongpu_drift()
        )
        self.assertIsNotNone(
            AnalyticDistribution(density_expression="1", momentum_expressions=[None, 3.0e7, None]).get_picongpu_drift()
        )

    def test_exactly_one_density_input_rule(self):
        with self.assertRaises(Exception):
            AnalyticDistribution()
        with self.assertRaises(Exception):
            AnalyticDistribution(density_expression="x", density_function=lambda x, y, z: x)

    def test_callable_path_still_works(self):
        # keyword construction
        keyword = AnalyticDistribution(density_function=lambda x, y, z: x * y * z)
        np.testing.assert_allclose(np.asarray(keyword(2, 3, 4)), np.asarray(24))

        # the decorator form (function passed positionally)
        @AnalyticDistribution
        def density(x, y, z):
            return x + y + z

        np.testing.assert_allclose(np.asarray(density(1, 2, 3)), np.asarray(6))
        # positional function + keyword drift
        positional = AnalyticDistribution(lambda x, y, z: x + y + z, directed_velocity=(1.0, 2.0, 3.0))
        self.assertEqual(positional.directed_velocity, [1.0, 2.0, 3.0])
        self.assertIsNotNone(positional.get_picongpu_drift())

    def test_decorator_kwargs_match_expression_and_callable(self):
        # the same constants can be supplied to the decorator, to density_expression
        # and to density_function; all three render the identical density
        @AnalyticDistribution(a=1, b=2)
        def decorated(x, y, z, a, b):
            return x + y + z + a + b

        expression = AnalyticDistribution(density_expression="x+y+z+a+b", a=1, b=2)
        callable_ = AnalyticDistribution(density_function=lambda x, y, z, a, b: x + y + z + a + b, a=1, b=2)

        # the three spellings are semantically equal (same rendered density + surface)
        self.assertEqual(decorated, expression)
        self.assertEqual(decorated, callable_)
        self.assertEqual(expression, callable_)

        x, y, z = symbols("x, y, z")
        self.assertEqual(decorated.density_sympy, expression.density_sympy)
        self.assertEqual(decorated.density_sympy, callable_.density_sympy)
        self.assertEqual(decorated.user_defined_kw, {"a": 1, "b": 2})
        self.assertEqual(expression.user_defined_kw, {"a": 1, "b": 2})
        # the rendered C++ is identical for all three spellings
        self.assertEqual(
            decorated.get_as_pypicongpu(None).function_body,
            expression.get_as_pypicongpu(None).function_body,
        )
        self.assertEqual(
            decorated.get_as_pypicongpu(None).function_body,
            callable_.get_as_pypicongpu(None).function_body,
        )
        self.assertEqual(decorated.density_function(x, y, z, a=1, b=2), x + y + z + 3)

    def test_decorator_kwargs_missing_value_rejected(self):
        # a parameter without a supplied value is not substituted, so rendering the
        # density fails with the usual arity error rather than silently dropping it
        with pytest.raises(TypeError):
            AnalyticDistribution(density_function=lambda x, y, z, a: x + y + z + a)

    def test_dim_property(self):
        # the dimensionality is derived from the density expression itself
        self.assertEqual(AnalyticDistribution(lambda x, y, z: x + y).dim, 2)
        self.assertEqual(AnalyticDistribution(lambda x, y, z: x + y + z).dim, 3)
        self.assertEqual(AnalyticDistribution(density_expression="x+y+z").dim, 3)
        self.assertEqual(AnalyticDistribution(density_expression="x+y").dim, 2)
        # a constant density is 2D (no z dependence)
        self.assertEqual(AnalyticDistribution(density_expression="1").dim, 2)
