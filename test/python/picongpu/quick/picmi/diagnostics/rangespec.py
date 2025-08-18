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
    (
        RangeSpec[-5:10],
        [slice(-5, 10, None)],
        (20,),
        [slice(15, 15, None)],  # Fixed: empty range due to begin > end
        [{"begin": 15, "end": 15}],  # Fixed: serialized empty range
    ),
    (RangeSpec[5:-2], [slice(5, -2, None)], (20,), [slice(5, 18, None)], [{"begin": 5, "end": 18}]),
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
    (
        RangeSpec[-5:-2, :, 0:3],
        [slice(-5, -2, None), slice(None, None, None), slice(0, 3, None)],
        (20, 30, 40),
        [slice(15, 18, None), slice(0, 29, None), slice(0, 3, None)],
        [{"begin": 15, "end": 18}, {"begin": 0, "end": 29}, {"begin": 0, "end": 3}],
    ),
    (
        RangeSpec[10:5, 15:5],
        [slice(10, 5, None), slice(15, 5, None)],
        (20, 30),
        [slice(10, 10, None), slice(15, 15, None)],
        [{"begin": 10, "end": 10}, {"begin": 15, "end": 15}],
    ),
]

# Invalid test cases for instantiation
TESTCASES_INVALID = [
    (
        (),  # Empty args for RangeSpec()
        "RangeSpec must have at least one range",
    ),
    (
        (slice(0, 10, None), slice(5, 15, None), slice(2, 8, None), slice(1, 2, None)),  # Too many ranges
        "RangeSpec must have at most 3 ranges",
    ),
    (
        (slice(0, 10, 2),),  # Non-None step
        "Step must be None in dimension 1",
    ),
    (
        (slice("0", 10, None),),  # Invalid slice start type
        "Begin in dimension 1 must be int or None",
    ),
    (
        (slice(0, 10.5, None),),  # Invalid slice stop type
        "End in dimension 1 must be int or None",
    ),
]

# Test cases for warning when range is empty
TESTCASES_WARNING = [
    (RangeSpec[10:5], "RangeSpec has begin > end in dimension 1, resulting in an empty range after processing"),
    (
        RangeSpec[10:5, 15:5],
        "RangeSpec has begin > end in dimension [1-2], resulting in an empty range after processing",
    ),
    (RangeSpec[-5:10], "RangeSpec has an empty range in dimension 1, disabling output for this dimension"),
]


class PICMI_TestRangeSpec(unittest.TestCase):
    def test_rangespec_instantiation(self):
        """Test RangeSpec instantiation and validation."""
        for rs, expected_ranges, _, _, _ in TESTCASES_VALID:
            with self.subTest(rs=rs, expected_ranges=expected_ranges):
                self.assertEqual(rs.ranges, expected_ranges)
                rs.check()  # Warnings tested separately

        for rs_args, expected_error in TESTCASES_INVALID:
            with self.subTest(rs_args=rs_args, expected_error=expected_error):
                with self.assertRaisesRegex((ValueError, TypeError), expected_error):
                    RangeSpec(*rs_args)  # Test instantiation failure

    def test_rangespec_serialization(self):
        """Test RangeSpec serialization to PyPIConGPURangeSpec."""
        for rs, _, sim_box, expected_pypicongpu_ranges, expected_serialized in TESTCASES_VALID:
            with self.subTest(rs=rs, sim_box=sim_box, expected_pypicongpu_ranges=expected_pypicongpu_ranges):
                pypicongpu_rs = rs.get_as_pypicongpu(sim_box)
                self.assertIsInstance(pypicongpu_rs, PyPIConGPURangeSpec)
                self.assertEqual(pypicongpu_rs.ranges, expected_pypicongpu_ranges)
                serialized = pypicongpu_rs.get_rendering_context()
                self.assertEqual(serialized["ranges"], expected_serialized)

    def test_rangespec_warning(self):
        """Test warnings for empty ranges or begin > end."""
        for rs, expected_warning in TESTCASES_WARNING:
            with self.subTest(rs=rs, expected_warning=expected_warning):
                with self.assertWarnsRegex(UserWarning, expected_warning):
                    rs.check()

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


if __name__ == "__main__":
    unittest.main()
