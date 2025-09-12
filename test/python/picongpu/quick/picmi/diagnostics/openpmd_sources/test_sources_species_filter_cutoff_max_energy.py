"""
This file is part of PIConGPU.
Copyright 2025 PIConGPU contributors
Authors: Masoud Afshari
License: GPLv3+
"""

import unittest
import typeguard

from picongpu.picmi.diagnostics.openpmd_sources import (
    EnergyDensityCutoff,
)
from picongpu.pypicongpu.output.openpmd_sources import (
    EnergyDensityCutoff as PyPIConGPUEnergyDensityCutoff,
)
from picongpu.picmi.species import Species as PICMISpecies
from picongpu.pypicongpu.species import Species as PyPIConGPUSpecies

# List of source classes to test
SOURCE_CLASSES = [
    (EnergyDensityCutoff, PyPIConGPUEnergyDensityCutoff),
]

# Valid test cases
TESTCASES_VALID = [
    (
        {"species": PICMISpecies(name="electrons"), "filter": "species_all", "cutoff_max_energy": 10.0},
        {"filter": "species_all", "cutoff_max_energy": 10.0},
    ),
    (
        {"species": PICMISpecies(name="ions"), "cutoff_max_energy": 1.5},
        {"filter": "species_all", "cutoff_max_energy": 1.5},
    ),
]

# Invalid test cases for instantiation / validation
TESTCASES_INVALID = [
    (
        {"species": PICMISpecies(name="electrons"), "filter": "invalid", "cutoff_max_energy": 10.0},
        r"Filter must be one of \['species_all', 'fields_all', 'custom_filter'\], got invalid",
    ),
    (
        {"species": PICMISpecies(name="electrons"), "cutoff_max_energy": -5},
        r"cutoff_max_energy must be positive, got -5",
    ),
    (
        {"species": PICMISpecies(name="electrons"), "cutoff_max_energy": "not_a_number"},
        r"argument \"cutoff_max_energy\" .* did not match any element in the union",
    ),
    (
        {"species": "not_a_species", "cutoff_max_energy": 1.0},
        r"argument \"species\" \(str\) is not an instance of picongpu.picmi.species.Species",
    ),
    (
        {"species": PICMISpecies(name="electrons"), "cutoff_max_energy": None},  # None triggers check in __init__
        r"cutoff_max_energy is required",
    ),
]

# Invalid species mapping for get_as_pypicongpu
TESTCASES_INVALID_MAPPING = [
    (
        {"species": PICMISpecies(name="electrons"), "filter": "species_all", "cutoff_max_energy": 10.0, "mapping": {}},
        "Species .* is not known to Simulation",
    ),
    (
        {
            "species": PICMISpecies(name="electrons"),
            "filter": "species_all",
            "cutoff_max_energy": 10.0,
            "mapping": {PICMISpecies(name="ions"): PyPIConGPUSpecies()},
        },
        "Species .* is not known to Simulation",
    ),
]


class PICMI_TestSpeciesFilterCutoffMaxEnergy(unittest.TestCase):
    def test_instantiation_and_validation(self):
        """Test instantiation, validation, and check() for cutoff_max_energy sources."""
        for SourceClass, _ in SOURCE_CLASSES:
            # Valid cases
            for params, _ in TESTCASES_VALID:
                with self.subTest(Source=SourceClass.__name__, params=params):
                    source = SourceClass(**params)
                    self.assertEqual(source.species, params["species"])
                    self.assertEqual(source.filter, params.get("filter", "species_all"))
                    self.assertEqual(source.cutoff_max_energy, params["cutoff_max_energy"])
                    # explicitly test check()
                    source.check()

            # Invalid cases
            for params, expected_error in TESTCASES_INVALID:
                with self.subTest(Source=SourceClass.__name__, params=params):
                    with self.assertRaisesRegex((ValueError, TypeError, typeguard.TypeCheckError), expected_error):
                        SourceClass(**params)

    def test_serialization(self):
        """Test serialization and PyPIConGPU conversion."""
        for SourceClass, PySourceClass in SOURCE_CLASSES:
            for params, expected_serialized in TESTCASES_VALID:
                with self.subTest(Source=SourceClass.__name__, params=params):
                    source = SourceClass(**params)
                    pypicongpu_species = PyPIConGPUSpecies()
                    mapping = {params["species"]: pypicongpu_species}
                    pypicongpu_source = source.get_as_pypicongpu(mapping)
                    self.assertIsInstance(pypicongpu_source, PySourceClass)
                    self.assertEqual(pypicongpu_source.filter, expected_serialized["filter"])
                    self.assertEqual(pypicongpu_source.cutoff_max_energy, expected_serialized["cutoff_max_energy"])
                    self.assertIsInstance(pypicongpu_source.species, PyPIConGPUSpecies)

    def test_invalid_mapping(self):
        """Test invalid species mapping for cutoff_max_energy sources."""
        for SourceClass, _ in SOURCE_CLASSES:
            for params, expected_error in TESTCASES_INVALID_MAPPING:
                with self.subTest(Source=SourceClass.__name__, params=params):
                    source = SourceClass(
                        species=params["species"],
                        filter=params["filter"],
                        cutoff_max_energy=params["cutoff_max_energy"],
                    )
                    with self.assertRaisesRegex(ValueError, expected_error):
                        source.get_as_pypicongpu(params["mapping"])


if __name__ == "__main__":
    unittest.main()
