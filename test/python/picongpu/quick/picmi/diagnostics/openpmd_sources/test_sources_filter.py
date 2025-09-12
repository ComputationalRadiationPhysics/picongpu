"""
This file is part of PIConGPU.
Copyright 2025 PIConGPU contributors
Authors: Masoud Afshari
License: GPLv3+
"""

import unittest
import typeguard

from picongpu.picmi.diagnostics.openpmd_sources import (
    Auto,
    DerivedAttributes,
)
from picongpu.pypicongpu.output.openpmd_sources import (
    Auto as PyPIConGPUAuto,
    DerivedAttributes as PyPIConGPUDerivedAttributes,
)

# List all the pairs to test
SOURCE_CLASSES = [
    (Auto, PyPIConGPUAuto),
    (DerivedAttributes, PyPIConGPUDerivedAttributes),
]

TESTCASES_VALID = [
    ({"filter": "species_all"}, {"filter": "species_all"}),
    ({"filter": "fields_all"}, {"filter": "fields_all"}),
    ({"filter": "custom_filter"}, {"filter": "custom_filter"}),
    ({}, {"filter": "species_all"}),  # default
]

TESTCASES_INVALID = [
    ({"filter": "invalid"}, r"Filter must be one of \['species_all', 'fields_all', 'custom_filter'\], got invalid"),
    ({"filter": 123}, r"argument \"filter\" \(int\) is not an instance of str"),
]


class PICMI_TestFilterOnlySources(unittest.TestCase):
    def test_instantiation_and_validation(self):
        """Test instantiation and validation of filter-only sources."""
        for SourceClass, _ in SOURCE_CLASSES:
            # Valid cases
            for params, expected_serialized in TESTCASES_VALID:
                with self.subTest(Source=SourceClass.__name__, params=params):
                    source = SourceClass(**params)
                    self.assertEqual(source.filter, params.get("filter", "species_all"))
                    source.check()

            # Invalid cases
            for params, expected_error in TESTCASES_INVALID:
                with self.subTest(Source=SourceClass.__name__, params=params):
                    with self.assertRaisesRegex((ValueError, typeguard.TypeCheckError), expected_error):
                        SourceClass(**params)

    def test_serialization(self):
        """Test get_as_pypicongpu returns correct PyPIConGPU object and filter."""
        for SourceClass, PySourceClass in SOURCE_CLASSES:
            for params, expected_serialized in TESTCASES_VALID:
                with self.subTest(Source=SourceClass.__name__, params=params):
                    source = SourceClass(**params)
                    pypicongpu_source = source.get_as_pypicongpu()
                    self.assertIsInstance(pypicongpu_source, PySourceClass)
                    self.assertEqual(pypicongpu_source.filter, expected_serialized["filter"])


if __name__ == "__main__":
    unittest.main()
