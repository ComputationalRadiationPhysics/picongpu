"""
This file is part of PIConGPU.
Copyright 2025 PIConGPU contributors
Authors: Masoud Afshari
License: GPLv3+
"""

from picongpu.picmi.diagnostics.openpmd_sources import Auto
from picongpu.pypicongpu.output.openpmd_sources import Auto as PyPIConGPUAuto
import unittest


# Test cases for valid Auto inputs
TESTCASES_VALID = [
    (
        {"filter": None},
        {"filter": None},
    ),
    (
        {"filter": "electrons"},
        {"filter": "electrons"},
    ),
    (
        {"filter": "ions"},
        {"filter": "ions"},
    ),
]

# Invalid test cases for instantiation
TESTCASES_INVALID = [
    (
        {"filter": 123},
        "Filter must be a string or None",
    ),
    (
        {"filter": ""},
        "Filter must be a non-empty string or None",
    ),
]


class PICMI_TestAuto(unittest.TestCase):
    def test_auto_instantiation(self):
        """Test Auto instantiation and validation."""
        for params, _ in TESTCASES_VALID:
            with self.subTest(params=params):
                auto = Auto(**params)
                for key, value in params.items():
                    self.assertEqual(getattr(auto, key), value)
                auto.check()  # Should not raise

        for params, expected_error in TESTCASES_INVALID:
            with self.subTest(params=params, expected_error=expected_error):
                with self.assertRaisesRegex(ValueError, expected_error):
                    auto = Auto(**params)
                    auto.check()

    def test_auto_serialization(self):
        """Test Auto serialization to PyPIConGPUAuto."""
        for params, expected_serialized in TESTCASES_VALID:
            with self.subTest(params=params, expected_serialized=expected_serialized):
                auto = Auto(**params)
                pypicongpu_auto = auto.get_as_pypicongpu()
                self.assertIsInstance(pypicongpu_auto, PyPIConGPUAuto)
                serialized = pypicongpu_auto._get_serialized()
                self.assertEqual(serialized["typeID"], {"auto": True})
                serialized_data = serialized["data"]
                for key, value in expected_serialized.items():
                    self.assertEqual(serialized_data[key], value)


if __name__ == "__main__":
    unittest.main()
