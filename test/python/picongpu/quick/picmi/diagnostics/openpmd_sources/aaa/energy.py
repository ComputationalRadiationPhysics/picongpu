"""
This file is part of PIConGPU.
Copyright 2025 PIConGPU contributors
Authors: Masoud Afshari
License: GPLv3+
"""

from picongpu.picmi.diagnostics.openpmd_sources import Energy
from picongpu.pypicongpu.output.openpmd_sources import Energy as PyPIConGPUEnergy
from picongpu.picmi.species import Species as PICMISpecies
from picongpu.pypicongpu.species import Species as PyPIConGPUSpecies
import unittest


# Test cases for valid Energy inputs
TESTCASES_VALID = [
    (
        {"species": PICMISpecies(name="electrons"), "filter": "all"},
        {"filter": "all"},
    ),
    (
        {"species": PICMISpecies(name="ions"), "filter": "electrons"},
        {"filter": "electrons"},
    ),
    (
        {"species": PICMISpecies(name="protons"), "filter": "ions"},
        {"filter": "ions"},
    ),
    (
        {"species": PICMISpecies(name="electrons")},
        {"filter": "all"},
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
            "mapping": {PICMISpecies(name="ions"): PyPIConGPUSpecies()},
        },
        "Species .* is not known to Simulation",
    ),
]


class PICMI_TestEnergy(unittest.TestCase):
    def test_energy_instantiation(self):
        """Test Energy instantiation and validation."""
        for params, _ in TESTCASES_VALID:
            with self.subTest(params=params):
                source = Energy(**params)
                self.assertEqual(source.species, params["species"])
                self.assertEqual(source.filter, params.get("filter", "all"))
                source.check()  # Should not raise

        for params, expected_error in TESTCASES_INVALID:
            with self.subTest(params=params, expected_error=expected_error):
                with self.assertRaisesRegex(ValueError, expected_error):
                    source = Energy(**params)
                    source.check()

    def test_energy_serialization(self):
        """Test Energy serialization to PyPIConGPUEnergy."""
        for params, expected_serialized in TESTCASES_VALID:
            with self.subTest(params=params, expected_serialized=expected_serialized):
                source = Energy(**params)
                pypicongpu_species = PyPIConGPUSpecies()
                mapping = {params["species"]: pypicongpu_species}
                pypicongpu_source = source.get_as_pypicongpu(mapping)
                self.assertIsInstance(pypicongpu_source, PyPIConGPUEnergy)
                self.assertEqual(pypicongpu_source.filter, expected_serialized["filter"])
                self.assertIsInstance(pypicongpu_source.species, PyPIConGPUSpecies)

    def test_energy_invalid_mapping(self):
        """Test Energy with invalid species mapping."""
        for params, expected_error in TESTCASES_INVALID_MAPPING:
            with self.subTest(params=params, expected_error=expected_error):
                source = Energy(species=params["species"], filter=params["filter"])
                with self.assertRaisesRegex(ValueError, expected_error):
                    source.get_as_pypicongpu(params["mapping"])


if __name__ == "__main__":
    unittest.main()
