"""
This file is part of PIConGPU.
Copyright 2025 PIConGPU contributors
Authors: Masoud Afshari
License: GPLv3+
"""

from picongpu.picmi.diagnostics.openpmd_sources import DerivedAttributes
from picongpu.pypicongpu.output.openpmd_sources import DerivedAttributes as PyPIConGPUDerivedAttributes
import unittest


# Test cases for valid DerivedAttributes inputs
TESTCASES_VALID = [
    (
        {"filter": None},
        {"filter": None},
    ),
    (
        {"filter": "all"},
        {"filter": "all"},
    ),
    (
        {"filter": "electrons"},
        {"filter": "electrons"},
    ),
    (
        {"filter": "ions"},
        {"filter": "ions"},
    ),
    (
        {},  # Default filter
        {"filter": None},
    ),
]

# Invalid test cases for instantiation
TESTCASES_INVALID = [
    (
        {"filter": 123},
        "Filter must be a string or None",
    ),
]


class PICMI_TestDerivedAttributes(unittest.TestCase):
    def test_derived_attributes_instantiation(self):
        """Test DerivedAttributes instantiation and validation."""
        for params, _ in TESTCASES_VALID:
            with self.subTest(params=params):
                source = DerivedAttributes(**params)
                self.assertEqual(source.filter, params.get("filter", None))
                source.check()  # Should not raise

        for params, expected_error in TESTCASES_INVALID:
            with self.subTest(params=params, expected_error=expected_error):
                with self.assertRaisesRegex(ValueError, expected_error):
                    source = DerivedAttributes(**params)
                    source.check()

    def test_derived_attributes_serialization(self):
        """Test DerivedAttributes serialization to PyPIConGPUDerivedAttributes."""
        for params, expected_serialized in TESTCASES_VALID:
            with self.subTest(params=params, expected_serialized=expected_serialized):
                source = DerivedAttributes(**params)
                pypicongpu_source = source.get_as_pypicongpu()
                self.assertIsInstance(pypicongpu_source, PyPIConGPUDerivedAttributes)
                self.assertEqual(pypicongpu_source.filter, expected_serialized["filter"])


if __name__ == "__main__":
    unittest.main()
