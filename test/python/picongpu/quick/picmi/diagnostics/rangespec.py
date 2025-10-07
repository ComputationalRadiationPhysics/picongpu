"""
This file is part of PIConGPU.
Copyright 2025 PIConGPU contributors
Authors: Masoud Afshari
License: GPLv3+
"""

from picongpu.picmi.diagnostics import RangeSpec
import unittest

TESTCASES_VALID = [
    (RangeSpec[:], [slice(None, None, None)]),
    (RangeSpec[0:10], [slice(0, 10, None)]),
    (RangeSpec[10:5], [slice(10, 5, None)]),
    (RangeSpec[-5:15], [slice(-5, 15, None)]),
    (RangeSpec[0:10, 5:15], [slice(0, 10, None), slice(5, 15, None)]),
    (RangeSpec[0:10, 5:15, 2:8], [slice(0, 10, None), slice(5, 15, None), slice(2, 8, None)]),
]

TESTCASES_INVALID = [
    ((), "RangeSpec must have at least one range"),
    (
        (slice(0, 10, None), slice(5, 15, None), slice(2, 8, None), slice(1, 2, None)),
        "RangeSpec must have at most 3 ranges",
    ),
    ((slice(0, 10, 2),), "Step must be None in dimension 1"),
    ((slice("0", 10, None),), "Begin in dimension 1 must be int or None"),
    ((slice(0, "10", None),), "End in dimension 1 must be int or None"),
]

TESTCASES_WARNING = [
    (RangeSpec[10:5], "RangeSpec has begin > end in dimension 1, resulting in an empty range"),
    (RangeSpec[-5:10], "RangeSpec has an empty range in dimension 1, disabling output"),
]


class PICMI_TestRangeSpec(unittest.TestCase):
    def test_rangespec(self):
        """Test RangeSpec instantiation with valid inputs."""
        for rs, ranges in TESTCASES_VALID:
            with self.subTest(rs=rs):
                self.assertEqual(rs.ranges, ranges)
                # Pass simulation_box based on number of dimensions
                simulation_box = tuple([128] * len(rs.ranges))
                rs.check(simulation_box)

    def test_rangespec_invalid(self):
        """Test invalid RangeSpec inputs."""
        for args, error in TESTCASES_INVALID:
            with self.subTest(args=args, error=error):
                with self.assertRaisesRegex((ValueError, TypeError), error):
                    RangeSpec(*args)

    def test_rangespec_warning(self):
        """Test warnings for empty or invalid ranges."""
        for rs, warning in TESTCASES_WARNING:
            with self.subTest(rs=rs, warning=warning):
                # Pass simulation_box for 1D
                simulation_box = (128,)
                with self.assertWarnsRegex(UserWarning, warning):
                    rs.check(simulation_box)


if __name__ == "__main__":
    unittest.main()
