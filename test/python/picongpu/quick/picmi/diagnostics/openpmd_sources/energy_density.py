"""
This file is part of PIConGPU.
Copyright 2025 PIConGPU contributors
Authors: Masoud Afshari
License: GPLv3+
"""

from picongpu.picmi.diagnostics.openpmd_sources import EnergyDensity
from picongpu.pypicongpu.output.openpmd_sources import EnergyDensity as PyPIConGPUEnergyDensity
from picongpu.picmi.species import Species as PICMISpecies
from picongpu.pypicongpu.species import Species as PyPIConGPUSpecies
import unittest
import typeguard


TESTCASES_VALID = [
    (
        {"species": PICMISpecies(name="electrons"), "filter": "species_all"},
        {"filter": "species_all"},
    ),
    (
        {"species": PICMISpecies(name="ions"), "filter": "fields_all"},
        {"filter": "fields_all"},
    ),
    (
        {"species": PICMISpecies(name="protons"), "filter": "custom_filter"},
        {"filter": "custom_filter"},
    ),
    (
        {"species": PICMISpecies(name="electrons")},
        {"filter": "species_all"},
    ),
]

TESTCASES_INVALID = [
    (
        {"species": PICMISpecies(name="electrons"), "filter": "invalid"},
        r"Filter must be one of \['species_all', 'fields_all', 'custom_filter'\], got invalid",
    ),
    (
        {"species": PICMISpecies(name="electrons"), "filter": 123},
        r"argument \"filter\" \(int\) is not an instance of str",
    ),
    (
        {"species": "not_a_species", "filter": "species_all"},
        r"argument \"species\" \(str\) is not an instance of picongpu.picmi.species.Species",
    ),
]

TESTCASES_INVALID_MAPPING = [
    (
        {"species": PICMISpecies(name="electrons"), "filter": "species_all", "mapping": {}},
        "Species .* is not known to Simulation",
    ),
    (
        {
            "species": PICMISpecies(name="electrons"),
            "filter": "species_all",
            "mapping": {PICMISpecies(name="ions"): PyPIConGPUSpecies()},
        },
        "Species .* is not known to Simulation",
    ),
]


class PICMI_TestEnergyDensity(unittest.TestCase):
    def test_energy_density_instantiation(self):
        """Test EnergyDensity instantiation and validation."""
        for params, _ in TESTCASES_VALID:
            with self.subTest(params=params):
                source = EnergyDensity(**params)
                self.assertEqual(source.species, params["species"])
                self.assertEqual(source.filter, params.get("filter", "species_all"))
                source.check()

        for params, expected_error in TESTCASES_INVALID:
            with self.subTest(params=params, expected_error=expected_error):
                with self.assertRaisesRegex((ValueError, typeguard.TypeCheckError), expected_error):
                    EnergyDensity(**params)

    def test_energy_density_serialization(self):
        """Test EnergyDensity serialization to PyPIConGPUEnergyDensity."""
        for params, expected_serialized in TESTCASES_VALID:
            with self.subTest(params=params, expected_serialized=expected_serialized):
                source = EnergyDensity(**params)
                pypicongpu_species = PyPIConGPUSpecies()
                mapping = {params["species"]: pypicongpu_species}
                pypicongpu_source = source.get_as_pypicongpu(mapping)
                self.assertIsInstance(pypicongpu_source, PyPIConGPUEnergyDensity)
                self.assertEqual(pypicongpu_source.filter, expected_serialized["filter"])
                self.assertIsInstance(pypicongpu_source.species, PyPIConGPUSpecies)

    def test_energy_density_invalid_mapping(self):
        """Test EnergyDensity with invalid species mapping."""
        for params, expected_error in TESTCASES_INVALID_MAPPING:
            with self.subTest(params=params, expected_error=expected_error):
                source = EnergyDensity(species=params["species"], filter=params["filter"])
                with self.assertRaisesRegex(ValueError, expected_error):
                    source.get_as_pypicongpu(params["mapping"])


if __name__ == "__main__":
    unittest.main()
