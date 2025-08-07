"""
This file is part of PIConGPU.
Copyright 2025 PIConGPU contributors
Authors: Masoud Afshari
License: GPLv3+
"""

from picongpu.pypicongpu.output.rangespec import RangeSpec

import unittest


# Mock PyPIConGPURangeSpec for testing
class MockPyPIConGPURangeSpec:
    def __init__(self, slices):
        self.slices = slices


class TestRangeSpec(unittest.TestCase):
    def test_empty(self):
        """Minimal and default range strings are handled correctly."""
        # Default ":" (1D, full range)
        rs = RangeSpec(":")
        self.assertEqual(rs.slices, (slice(None, None, 1),))

        # Default ":,:" (2D, full range)
        rs = RangeSpec(":,:")
        self.assertEqual(rs.slices, (slice(None, None, 1), slice(None, None, 1)))

        # Default ":,:,:," (3D, full range)
        rs = RangeSpec(":,:,:")
        self.assertEqual(rs.slices, (slice(None, None, 1), slice(None, None, 1), slice(None, None, 1)))

    def test_types(self):
        """Type safety is ensured for range_str."""
        # Invalid range_str types
        invalid_inputs = [1, [], {}, None]
        for invalid in invalid_inputs:
            with self.assertRaises(TypeError):
                RangeSpec(invalid)

        # Valid range_str
        rs = RangeSpec("0:10")
        self.assertEqual(rs.slices, (slice(0, 10, 1),))

    def test_validation(self):
        """Constraints on range format and slices are enforced."""
        # More than 3 dimensions
        with self.assertRaises(ValueError, match="Range must specify at most three dimensions"):
            RangeSpec("0:10,0:10,0:10,0:10")

        # Invalid format
        invalid_formats = ["a:b", "0:10:2", "0-10", "0:", ":0"]
        for fmt in invalid_formats:
            with self.assertRaises(ValueError, match="Invalid range format"):
                RangeSpec(fmt)

        # Invalid step (step != 1)
        rs = RangeSpec("0:10")
        rs.slices = (slice(0, 10, 2),)
        with self.assertRaises(ValueError, match="Step size must be 1"):
            rs._validate()

        # Invalid dimension size
        with self.assertRaises(ValueError, match="Dimension size must be positive"):
            rs.get_as_pypicongpu((0,))

        # Valid configurations
        rs = RangeSpec("0:10")
        rs._validate()  # Should succeed
        rs = RangeSpec("0:10,5:15,-2:8")
        rs._validate()  # Should succeed

    def test_rendering(self):
        """Conversion to PyPIConGPURangeSpec is correct."""
        # 1D: Positive indices
        rs = RangeSpec("2:10")
        result = rs.get_as_pypicongpu((20,))
        self.assertEqual(result.slices, (slice(2, 9, 1),))

        # 1D: Negative indices
        rs = RangeSpec("-5:-1")
        result = rs.get_as_pypicongpu((20,))
        self.assertEqual(result.slices, (slice(15, 19, 1),))

        # 2D: Mixed indices
        rs = RangeSpec("0:10,-5:15")
        result = rs.get_as_pypicongpu((20, 30))
        self.assertEqual(result.slices, (slice(0, 9, 1), slice(25, 15, 1)))

        # 3D: Full range with None
        rs = RangeSpec(":,:,:")
        result = rs.get_as_pypicongpu((20, 30, 40))
        self.assertEqual(result.slices, (slice(0, 19, 1), slice(0, 29, 1), slice(0, 39, 1)))

        # Mismatched dimensions
        rs = RangeSpec("0:10,0:10")
        with self.assertRaises(ValueError, match="Number of range specifications"):
            rs.get_as_pypicongpu((20,))


if __name__ == "__main__":
    unittest.main()
