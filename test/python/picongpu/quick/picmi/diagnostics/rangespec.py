"""
This file is part of PIConGPU.
Copyright 2025 PIConGPU contributors
Authors: Masoud Afshari
License: GPLv3+
"""

from picongpu.picmi.diagnostics import RangeSpec
from picongpu.pypicongpu.output.rangespec import RangeSpec as PyPIConGPURangeSpec
import unittest

TESTCASES_VALID = [
    (RangeSpec[:], [slice(None, None, None)], (20,), [slice(0, 19, None)], [{"begin": 0, "end": 19}]),
    (RangeSpec[0:10], [slice(0, 10, None)], (20,), [slice(0, 10, None)], [{"begin": 0, "end": 10}]),
    (RangeSpec[10:5], [slice(10, 5, None)], (20,), [slice(10, 10, None)], [{"begin": 10, "end": 10}]),
    (
        RangeSpec[0:10, 5:15],
        [slice(0, 10, None), slice(5, 15, None)],
        (20, 30),
        [slice(0, 10, None), slice(5, 15, None)],
        [{"begin": 0, "end": 10}, {"begin": 5, "end": 15}],
    ),
    (
        RangeSpec[0:10, 5:15, 2:8],
        [slice(0, 10, None), slice(5, 15, None), slice(2, 8, None)],
        (20, 30, 40),
        [slice(0, 10, None), slice(5, 15, None), slice(2, 8, None)],
        [{"begin": 0, "end": 10}, {"begin": 5, "end": 15}, {"begin": 2, "end": 8}],
    ),
]

TESTCASES_INVALID = [
    ((), "RangeSpec must have at least one range"),
    (
        (slice(0, 10, None), slice(5, 15, None), slice(2, 8, None), slice(1, 2, None)),
        "RangeSpec must have at most 3 ranges",
    ),
    ((slice(0, 10, 2),), "Step must be None in dimension 1"),
    ((slice("0", 10, None),), "Begin in dimension 1 must be int or None"),
]

TESTCASES_WARNING = [
    (RangeSpec[10:5], "RangeSpec has begin > end in dimension 1, resulting in an empty range"),
    (RangeSpec[-5:10], "RangeSpec has an empty range in dimension 1, disabling output"),
]


class PICMI_TestRangeSpec(unittest.TestCase):
    def test_rangespec(self):
        """Test RangeSpec instantiation, serialization, and clipping."""
        for rs, ranges, sim_box, pypicongpu_ranges, serialized in TESTCASES_VALID:
            with self.subTest(rs=rs, sim_box=sim_box):
                self.assertEqual(rs.ranges, ranges)
                rs.check()
                pypicongpu_rs = rs.get_as_pypicongpu(sim_box)
                self.assertIsInstance(pypicongpu_rs, PyPIConGPURangeSpec)
                self.assertEqual(pypicongpu_rs.ranges, pypicongpu_ranges)
                self.assertEqual(pypicongpu_rs.get_rendering_context()["ranges"], serialized)

    def test_rangespec_invalid(self):
        """Test invalid RangeSpec inputs and simulation box."""
        for args, error in TESTCASES_INVALID:
            with self.subTest(args=args, error=error):
                with self.assertRaisesRegex((ValueError, TypeError), error):
                    RangeSpec(*args)
        rs = RangeSpec[0:10, 5:15]
        with self.assertRaisesRegex(ValueError, "Number of range specifications"):
            rs.get_as_pypicongpu((20,))
        with self.assertRaisesRegex(ValueError, "Dimension size must be positive"):
            rs.get_as_pypicongpu((20, 0))

    def test_rangespec_warning(self):
        """Test warnings for empty ranges."""
        for rs, warning in TESTCASES_WARNING:
            with self.subTest(rs=rs, warning=warning):
                with self.assertWarnsRegex(UserWarning, warning):
                    rs.check()


if __name__ == "__main__":
    unittest.main()
