import unittest
import numpy as np

from scipy.constants import c

from picongpu.picmi import AnalyticDistribution
import typeguard

VALID_CALLS = [
    ((1, 2, 3), 6),
    (([1, 2], 2, 3), [6, 12]),
    ((1, 2, [3, 4]), [6, 8]),
    (([1, 2], [3, 4], [5, 6]), [15, 48]),
]

INVALID_DENSITIES = [
    # not a function
    ("string", TypeError),
    (1, TypeError),
    # wrong number of arguments
    (lambda x, y: x + y, TypeError),
    (lambda x, y, z, too_much: x + y + z + too_much, TypeError),
    # bad return type
    (lambda x, y, z: "string", typeguard.TypeCheckError),
    # constructs not understood by sympy
    (lambda x, y, z: x if x > 0 else y * z, TypeError),
]


def velocity(gamma):
    return np.sqrt(c**2 * (1.0 - 1.0 / gamma**2))


class TestAnalyticDistribution(unittest.TestCase):
    def setUp(self):
        self.valid_density = lambda x, y, z: x * y * z
        self.dist = AnalyticDistribution(self.valid_density, directed_velocity=(1.0, 2.0, 3.0))

    def test_density_expression_invalid(self):
        for density, err in INVALID_DENSITIES:
            with self.subTest(density=density, err=err):
                with self.assertRaises(err):
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
        self.assertIsNone(AnalyticDistribution(lambda *x: sum(x), directed_velocity=[0, 0, 0]).get_picongpu_drift())

    def test_drift_wrong_dimensionality(self):
        # Test drift with wrong dimensionality
        with self.assertRaises(typeguard.TypeCheckError):
            AnalyticDistribution(
                lambda x, y, z: x + y + z,
                # Only 2 elements
                directed_velocity=[1.0, 2.0],
            ).get_picongpu_drift()

    def test_call(self):
        for args, result in VALID_CALLS:
            with self.subTest(args=args, result=result):
                np.testing.assert_allclose(np.asarray(self.dist(*args)), np.asarray(result))


if __name__ == "__main__":
    unittest.main()
