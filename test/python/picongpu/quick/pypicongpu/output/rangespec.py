"""
This file is part of PIConGPU.
Copyright 2025 PIConGPU contributors
Authors: Masoud Afshari
License: GPLv3+
"""

from picongpu.pypicongpu.output.rangespec import RangeSpec as PyPIConGPURangeSpec

import unittest


class TestRangeSpec(unittest.TestCase):
    def test_empty(self):
        """Minimal and default range slices are handled correctly."""
        # Default 1D full range
        rs = PyPIConGPURangeSpec[:]
        self.assertEqual(rs.ranges, [slice(None, None, None)])

        # Default 2D full range
        rs = PyPIConGPURangeSpec[:, :]
        self.assertEqual(rs.ranges, [slice(None, None, None), slice(None, None, None)])

        # Default 3D full range
        rs = PyPIConGPURangeSpec[:, :, :]
        self.assertEqual(rs.ranges, [slice(None, None, None), slice(None, None, None), slice(None, None, None)])

    def test_types(self):
        """Type safety is ensured for ranges."""
        # Invalid inputs (non-slice)
        invalid_inputs = [1, "0:10", {}, None, [0, 10]]
        for invalid in invalid_inputs:
            with self.assertRaises(TypeError):
                PyPIConGPURangeSpec[invalid]

        # Invalid endpoint types (non-int, non-None)
        invalid_endpoints = [slice(0.0, 10), slice(0, 10.0), slice("a", 10), slice(0, "b"), slice(None, [])]
        for invalid in invalid_endpoints:
            with self.assertRaises(TypeError):
                PyPIConGPURangeSpec[invalid]

        # Invalid step
        with self.assertRaisesRegex(ValueError, "Step must be None"):
            PyPIConGPURangeSpec[0:10:2]

        # Valid ranges
        rs = PyPIConGPURangeSpec[0:10]
        self.assertEqual(rs.ranges, [slice(0, 10, None)])
        rs = PyPIConGPURangeSpec[0:10, 5:15]
        self.assertEqual(rs.ranges, [slice(0, 10, None), slice(5, 15, None)])

    def test_validation(self):
        """Constraints on range format are enforced."""
        # More than 3 dimensions
        with self.assertRaisesRegex(ValueError, "RangeSpec must have at most 3 ranges"):
            PyPIConGPURangeSpec[0:10, 0:10, 0:10, 0:10]

        # Empty ranges
        with self.assertRaisesRegex(ValueError, "RangeSpec must have at least one range"):
            PyPIConGPURangeSpec()

        # Valid configurations
        PyPIConGPURangeSpec[0:10]
        PyPIConGPURangeSpec[0:10, 5:15, -2:8]

    def test_rendering(self):
        """Serialized data is correctly formatted for template consumption."""
        # 1D: Positive indices
        rs = PyPIConGPURangeSpec[2:10]
        serialized = rs._get_serialized()
        self.assertEqual(serialized["ranges"], [{"begin": 2, "end": 10}])

        # 1D: Negative indices
        rs = PyPIConGPURangeSpec[-5:-1]
        serialized = rs._get_serialized()
        self.assertEqual(serialized["ranges"], [{"begin": -5, "end": -1}])

        # 2D: Mixed indices
        rs = PyPIConGPURangeSpec[0:10, -5:15]
        serialized = rs._get_serialized()
        self.assertEqual(serialized["ranges"], [{"begin": 0, "end": 10}, {"begin": -5, "end": 15}])

        # 3D: Full range
        rs = PyPIConGPURangeSpec[:, :, :]
        serialized = rs._get_serialized()
        self.assertEqual(
            serialized["ranges"], [{"begin": 0, "end": -1}, {"begin": 0, "end": -1}, {"begin": 0, "end": -1}]
        )


if __name__ == "__main__":
    unittest.main()
