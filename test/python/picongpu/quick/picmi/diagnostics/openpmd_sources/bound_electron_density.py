"""
This file is part of PIConGPU.
Copyright 2025 PIConGPU contributors
Authors: Masoud Afshari
License: GPLv3+
"""

from picongpu.picmi.diagnostics.openpmd_sources import BoundElectronDensity
from picongpu.pypicongpu.output.openpmd_sources import BoundElectronDensity as PyPIConGPUBoundElectronDensity
from picongpu.picmi.species import Species as PICMISpecies
from picongpu.pypicongpu.species import Species as PyPIConGPUSpecies
import unittest


# Test cases for valid BoundElectronDensity inputs
TESTCASES_VALID = [
    (
        {"species": PICMISpecies(name="electrons"), "filter": "all"},
        {"filter": "all", "species": PyPIConGPUSpecies(name="electrons")},
    ),
    (
        {"species": PICMISpecies(name="ions"), "filter": "electrons"},
        {"filter": "electrons", "species": PyPIConGPUSpecies(name="ions")},
    ),
    (
        {"species": PICMISpecies(name="protons"), "filter": "ions"},
        {"filter": "ions", "species": PyPIConGPUSpecies(name="protons")},
    ),
]

# Invalid test cases for instantiation
TESTCASES_INVALID = [
    (
        {"species": PICMISpecies(name="electrons"), "filter": 123},
        "Filter must be a string",
    ),
    (
        {"species": "not_a_species", "filter": "all"},
        "Species must be a PICMISpecies",
    ),
    (
        {"species": PICMISpecies(name="electrons"), "filter": ""},
        "Filter must be a non-empty string",
    ),
]

# Invalid test cases for unmapped species
TESTCASES_INVALID_MAPPING = [
    (
        {"species": PICMISpecies(name="electrons"), "filter": "all", "mapping": {}},
        "Species .* is not known to Simulation",
    ),
    (
        {
            "species": PICMISpecies(name="electrons"),
            "filter": "all",
            "mapping": {PICMISpecies(name="ions"): PyPIConGPUSpecies(name="ions")},
        },
        "Species .* is not known to Simulation",
    ),
]


class PICMI_TestBoundElectronDensity(unittest.TestCase):
    def test_bound_electron_density_instantiation(self):
        """Test BoundElectronDensity instantiation and validation."""
        for params, _ in TESTCASES_VALID:
            with self.subTest(params=params):
                source = BoundElectronDensity(**params)
                for key, value in params.items():
                    self.assertEqual(getattr(source, key), value)
                source.check()  # Should not raise

        for params, expected_error in TESTCASES_INVALID:
            with self.subTest(params=params, expected_error=expected_error):
                with self.assertRaisesRegex(ValueError, expected_error):
                    source = BoundElectronDensity(**params)
                    source.check()

    def test_bound_electron_density_serialization(self):
        """Test BoundElectronDensity serialization to PyPIConGPUBoundElectronDensity."""
        for params, expected_serialized in TESTCASES_VALID:
            with self.subTest(params=params, expected_serialized=expected_serialized):
                source = BoundElectronDensity(**params)
                mapping = {params["species"]: expected_serialized["species"]}
                pypicongpu_source = source.get_as_pypicongpu(mapping)
                self.assertIsInstance(pypicongpu_source, PyPIConGPUBoundElectronDensity)
                serialized = pypicongpu_source._get_serialized()
                self.assertEqual(serialized["typeID"], {"boundElectronDensity": True})
                serialized_data = serialized["data"]
                self.assertEqual(serialized_data["filter"], expected_serialized["filter"])
                self.assertEqual(serialized_data["species"], expected_serialized["species"])

    def test_bound_electron_density_invalid_mapping(self):
        """Test BoundElectronDensity with invalid species mapping."""
        for params, expected_error in TESTCASES_INVALID_MAPPING:
            with self.subTest(params=params, expected_error=expected_error):
                source = BoundElectronDensity(species=params["species"], filter=params["filter"])
                with self.assertRaisesRegex(ValueError, expected_error):
                    source.get_as_pypicongpu(params["mapping"])


if __name__ == "__main__":
    unittest.main()
