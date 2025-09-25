"""
This file is part of PIConGPU.
Copyright 2025 PIConGPU contributors
Authors: Masoud Afshari
License: GPLv3+
"""

from picongpu.pypicongpu.output.rangespec import RangeSpec as PyPIConGPURangeSpec
import unittest


class TestRangeSpec(unittest.TestCase):
    def test_instantiation_and_types(self):
        """Test instantiation, type safety, and valid serialization."""
        # Valid configurations
        rs = PyPIConGPURangeSpec[0:10]
        self.assertEqual(rs.ranges, [slice(0, 10, None)])
        context = rs.get_rendering_context()
        self.assertEqual(context["ranges"], [{"begin": 0, "end": 10}])

        rs = PyPIConGPURangeSpec[0:10, 5:15]
        self.assertEqual(rs.ranges, [slice(0, 10, None), slice(5, 15, None)])
        context = rs.get_rendering_context()
        self.assertEqual(context["ranges"], [{"begin": 0, "end": 10}, {"begin": 5, "end": 15}])

        rs = PyPIConGPURangeSpec[:, :, :]
        self.assertEqual(rs.ranges, [slice(None, None, None), slice(None, None, None), slice(None, None, None)])
        context = rs.get_rendering_context()
        self.assertEqual(context["ranges"], [{"begin": 0, "end": -1}, {"begin": 0, "end": -1}, {"begin": 0, "end": -1}])

        # Type safety
        invalid_inputs = ["string", 1]
        for invalid in invalid_inputs:
            with self.subTest(invalid=invalid):
                with self.assertRaises(TypeError):
                    PyPIConGPURangeSpec[invalid]

        invalid_endpoints = [slice(0.0, 10), slice(0, "b")]
        for invalid in invalid_endpoints:
            with self.subTest(invalid=invalid):
                with self.assertRaises(TypeError):
                    PyPIConGPURangeSpec[invalid]

    def test_rendering_and_validation(self):
        """Test serialization output and validation errors."""
        # Valid serialization
        rs = PyPIConGPURangeSpec[0:10, 5:15, 2:8]
        context = rs.get_rendering_context()
        self.assertEqual(context["ranges"], [{"begin": 0, "end": 10}, {"begin": 5, "end": 15}, {"begin": 2, "end": 8}])

        # Validation errors
        with self.assertRaisesRegex(ValueError, "RangeSpec must have at most 3 ranges"):
            PyPIConGPURangeSpec[0:10, 0:10, 0:10, 0:10]

        with self.assertRaisesRegex(ValueError, "RangeSpec must have at least one range"):
            PyPIConGPURangeSpec()

        with self.assertRaisesRegex(ValueError, "Step must be None"):
            PyPIConGPURangeSpec[0:10:2]

        with self.assertRaises(TypeError):
            PyPIConGPURangeSpec[slice(0, 10.0)]


if __name__ == "__main__":
    unittest.main()
