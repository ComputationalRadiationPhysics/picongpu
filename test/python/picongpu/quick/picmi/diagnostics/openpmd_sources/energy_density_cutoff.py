"""
This file is part of PIConGPU.
Copyright 2025 PIConGPU contributors
Authors: Masoud Afshari
License: GPLv3+
"""

from picongpu.picmi.diagnostics.openpmd_sources import EnergyDensityCutoff
from picongpu.pypicongpu.output.openpmd_sources import EnergyDensityCutoff as PyPIConGPUEnergyDensityCutoff
from picongpu.picmi.species import Species as PICMISpecies
from picongpu.pypicongpu.species import Species as PyPIConGPUSpecies
import unittest
import typeguard


TESTCASES_VALID = [
    (
        {"species": PICMISpecies(name="electrons"), "filter": "species_all", "cutoff_max_energy": 1e-18},
        {"filter": "species_all", "cutoff_max_energy": 1e-18},
    ),
    (
        {"species": PICMISpecies(name="ions"), "filter": "fields_all", "cutoff_max_energy": 2e-17},
        {"filter": "fields_all", "cutoff_max_energy": 2e-17},
    ),
    (
        {"species": PICMISpecies(name="protons"), "filter": "custom_filter"},
        {"filter": "custom_filter", "cutoff_max_energy": None},
    ),
    (
        {"species": PICMISpecies(name="electrons")},
        {"filter": "species_all", "cutoff_max_energy": None},
    ),
]

TESTCASES_INVALID = [
    (
        {"species": PICMISpecies(name="electrons"), "filter": "invalid", "cutoff_max_energy": 1e-18},
        r"Filter must be one of \['species_all', 'fields_all', 'custom_filter'\], got invalid",
    ),
    (
        {"species": PICMISpecies(name="electrons"), "filter": 123, "cutoff_max_energy": 1e-18},
        r"argument \"filter\" \(int\) is not an instance of str",
    ),
    (
        {"species": "not_a_species", "filter": "species_all", "cutoff_max_energy": 1e-18},
        r"argument \"species\" \(str\) is not an instance of picongpu.picmi.species.Species",
    ),
    (
        {"species": PICMISpecies(name="electrons"), "filter": "species_all", "cutoff_max_energy": -1e-18},
        r"cutoff_max_energy must be positive, got -1e-18",
    ),
    (
        {"species": PICMISpecies(name="electrons"), "filter": "species_all", "cutoff_max_energy": "invalid"},
        r"argument \"cutoff_max_energy\" \(str\) did not match any element in the union:\n\s+float: is neither float or int\n\s+NoneType: is not an instance of NoneType",
    ),
]

TESTCASES_INVALID_MAPPING = [
    (
        {"species": PICMISpecies(name="electrons"), "filter": "species_all", "cutoff_max_energy": 1e-18, "mapping": {}},
        "Species .* is not known to Simulation",
    ),
    (
        {
            "species": PICMISpecies(name="electrons"),
            "filter": "species_all",
            "cutoff_max_energy": 1e-18,
            "mapping": {PICMISpecies(name="ions"): PyPIConGPUSpecies()},
        },
        "Species .* is not known to Simulation",
    ),
]


class PICMI_TestEnergyDensityCutoff(unittest.TestCase):
    def test_energy_density_cutoff_instantiation(self):
        """Test EnergyDensityCutoff instantiation and validation."""
        for params, _ in TESTCASES_VALID:
            with self.subTest(params=params):
                source = EnergyDensityCutoff(**params)
                self.assertEqual(source.species, params["species"])
                self.assertEqual(source.filter, params.get("filter", "species_all"))
                self.assertEqual(source.cutoff_max_energy, params.get("cutoff_max_energy", None))
                source.check()

        for params, expected_error in TESTCASES_INVALID:
            with self.subTest(params=params, expected_error=expected_error):
                with self.assertRaisesRegex((ValueError, typeguard.TypeCheckError), expected_error):
                    EnergyDensityCutoff(**params)

    def test_energy_density_cutoff_serialization(self):
        """Test EnergyDensityCutoff serialization to PyPIConGPUEnergyDensityCutoff."""
        for params, expected_serialized in TESTCASES_VALID:
            with self.subTest(params=params, expected_serialized=expected_serialized):
                source = EnergyDensityCutoff(**params)
                pypicongpu_species = PyPIConGPUSpecies()
                mapping = {params["species"]: pypicongpu_species}
                pypicongpu_source = source.get_as_pypicongpu(mapping)
                self.assertIsInstance(pypicongpu_source, PyPIConGPUEnergyDensityCutoff)
                self.assertEqual(pypicongpu_source.filter, expected_serialized["filter"])
                self.assertEqual(pypicongpu_source.cutoff_max_energy, expected_serialized["cutoff_max_energy"])
                self.assertIsInstance(pypicongpu_source.species, PyPIConGPUSpecies)

    def test_energy_density_cutoff_invalid_mapping(self):
        """Test EnergyDensityCutoff with invalid species mapping."""
        for params, expected_error in TESTCASES_INVALID_MAPPING:
            with self.subTest(params=params, expected_error=expected_error):
                source = EnergyDensityCutoff(
                    species=params["species"], filter=params["filter"], cutoff_max_energy=params["cutoff_max_energy"]
                )
                with self.assertRaisesRegex(ValueError, expected_error):
                    source.get_as_pypicongpu(params["mapping"])


if __name__ == "__main__":
    unittest.main()
