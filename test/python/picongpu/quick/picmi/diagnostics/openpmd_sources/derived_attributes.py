"""
This file is part of PIConGPU.
Copyright 2025 PIConGPU contributors
Authors: Masoud Afshari
License: GPLv3+
"""

from picongpu.picmi.diagnostics.openpmd_sources import DerivedAttributes
from picongpu.pypicongpu.output.openpmd_sources import DerivedAttributes as PyPIConGPUDerivedAttributes
import unittest
import typeguard


TESTCASES_VALID = [
    ({"filter": "species_all"}, {"filter": "species_all"}),
    ({"filter": "fields_all"}, {"filter": "fields_all"}),
    ({"filter": "custom_filter"}, {"filter": "custom_filter"}),
    ({}, {"filter": "species_all"}),
]

TESTCASES_INVALID = [
    (
        {"filter": "invalid"},
        r"Filter must be one of \['species_all', 'fields_all', 'custom_filter'\], got invalid",
    ),
    (
        {"filter": 123},
        r'argument "filter" \(int\) is not an instance of str',
    ),
]


class PICMI_TestDerivedAttributes(unittest.TestCase):
    def test_derived_attributes_instantiation(self):
        """Test DerivedAttributes instantiation and validation."""
        for params, _ in TESTCASES_VALID:
            with self.subTest(params=params):
                source = DerivedAttributes(**params)
                self.assertEqual(source.filter, params.get("filter", "species_all"))
                source.check()

        for params, expected_error in TESTCASES_INVALID:
            with self.subTest(params=params, expected_error=expected_error):
                with self.assertRaisesRegex((ValueError, typeguard.TypeCheckError), expected_error):
                    DerivedAttributes(**params)

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
