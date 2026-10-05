"""
This file is part of PIConGPU.
Copyright 2025 PIConGPU contributors
Authors: Julian Lenz
License: GPLv3+
"""

from unittest import TestCase
import numpy as np

from scipy.constants import c
from sympy import Piecewise, exp, sin, symbols

from picongpu.picmi import AnalyticDistribution
import pytest

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
                self.assertEqual(string_dist.density_expression, callable_dist.density_expression)

    def test_rendered_cpp_matches_equivalent_callable(self):
        for expression, callable_density in self.CASES:
            with self.subTest(expression=expression):
                string_dist = AnalyticDistribution(density_expression=expression)
                callable_dist = AnalyticDistribution(callable_density)
                self.assertEqual(
                    string_dist.get_as_pypicongpu(None).function_body,
                    callable_dist.get_as_pypicongpu(None).function_body,
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
