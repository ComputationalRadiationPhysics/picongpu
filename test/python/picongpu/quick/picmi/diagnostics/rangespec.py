"""
This file is part of PIConGPU.
Copyright 2025 PIConGPU contributors
Authors: Masoud Afshari
License: GPLv3+
"""

from picongpu.picmi.diagnostics import RangeSpec
from picongpu.pypicongpu.output.rangespec import RangeSpec as PyPIConGPURangeSpec

import unittest


# Test cases for RangeSpec in 1D, 2D, 3D (slice-based)
TESTCASES_VALID = [
    (RangeSpec[0:10], [slice(0, 10, None)], (20,), [slice(0, 10, None)], [{"begin": 0, "end": 10}]),
    (RangeSpec[:], [slice(None, None, None)], (20,), [slice(0, 19, None)], [{"begin": 0, "end": 19}]),
    (RangeSpec[-5:10], [slice(-5, 10, None)], (20,), [slice(10, 10, None)], [{"begin": 10, "end": 10}]),
    (RangeSpec[5:-2], [slice(5, -2, None)], (20,), [slice(5, 18, None)], [{"begin": 5, "end": 18}]),
    (RangeSpec[10:5], [slice(10, 5, None)], (20,), [slice(5, 10, None)], [{"begin": 5, "end": 10}]),
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
    (
        RangeSpec[-5:-2, :, 0:3],
        [slice(-5, -2, None), slice(None, None, None), slice(0, 3, None)],
        (20, 30, 40),
        [slice(15, 18, None), slice(0, 29, None), slice(0, 3, None)],
        [{"begin": 15, "end": 18}, {"begin": 0, "end": 29}, {"begin": 0, "end": 3}],
    ),
]

# Invalid test cases for instantiation
TESTCASES_INVALID = [
    (RangeSpec(), "RangeSpec must have at least one range"),
    (RangeSpec[0:10, 5:15, 2:8, 1:2], "RangeSpec must have at most 3 ranges"),
    (RangeSpec[0:10:2], "Step must be None in dimension 1"),
    (RangeSpec["0:10"], "All elements must be slice objects"),
    (RangeSpec[0.5:10], "Begin in dimension 1 must be int or None"),
    (RangeSpec[0:10.5], "End in dimension 1 must be int or None"),
]


class TestRangeSpec(unittest.TestCase):
    def test_rangespec_instantiation(self):
        """Test RangeSpec instantiation and validation."""
        for rs, expected_ranges, _, _, _ in TESTCASES_VALID:
            with self.subTest(rs=rs, expected_ranges=expected_ranges):
                self.assertEqual(rs.ranges, expected_ranges)
                rs.check()  # Should not raise

        for rs_args, expected_error in TESTCASES_INVALID:
            with self.subTest(rs_args=rs_args, expected_error=expected_error):
                with self.assertRaisesRegex((ValueError, TypeError), expected_error):
                    rs_args.check() if isinstance(rs_args, RangeSpec) else RangeSpec(rs_args)

    def test_rangespec_serialization(self):
        """Test RangeSpec serialization to PyPIConGPURangeSpec."""
        for rs, _, sim_box, expected_pypicongpu_ranges, expected_serialized in TESTCASES_VALID:
            with self.subTest(rs=rs, sim_box=sim_box, expected_pypicongpu_ranges=expected_pypicongpu_ranges):
                pypicongpu_rs = rs.get_as_pypicongpu(sim_box)
                self.assertIsInstance(pypicongpu_rs, PyPIConGPURangeSpec)
                self.assertEqual(pypicongpu_rs.ranges, expected_pypicongpu_ranges)
                serialized = pypicongpu_rs.get_rendering_context()
                self.assertEqual(serialized["ranges"], expected_serialized)

    def test_rangespec_invalid_simulation_box(self):
        """Test invalid simulation box dimensions."""
        rs = RangeSpec[0:10, 5:15]
        with self.assertRaisesRegex(ValueError, "Number of range specifications"):
            rs.get_as_pypicongpu((20,))  # Too few dimensions
        with self.assertRaisesRegex(ValueError, "Number of range specifications"):
            rs.get_as_pypicongpu((20, 30, 40))  # Too many dimensions
        with self.assertRaisesRegex(ValueError, "Dimension size must be positive"):
            rs.get_as_pypicongpu((20, 0))  # Non-positive dimension

    def test_rangespec_clipping(self):
        """Test clipping of ranges to simulation box."""
        rs = RangeSpec[0:30, -10:40]  # Beyond sim box (20, 30)
        pypicongpu_rs = rs.get_as_pypicongpu((20, 30))
        self.assertEqual(pypicongpu_rs.ranges, [slice(0, 19, None), slice(20, 29, None)])
        serialized = pypicongpu_rs.get_rendering_context()
        self.assertEqual(serialized["ranges"], [{"begin": 0, "end": 19}, {"begin": 20, "end": 29}])
