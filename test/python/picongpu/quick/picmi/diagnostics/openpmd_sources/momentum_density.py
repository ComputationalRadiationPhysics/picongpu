"""
This file is part of PIConGPU.
Copyright 2025 PIConGPU contributors
Authors: Masoud Afshari
License: GPLv3+
"""

from picongpu.picmi.diagnostics.openpmd_sources import MomentumDensity
from picongpu.pypicongpu.output.openpmd_sources import MomentumDensity as PyPIConGPUMomentumDensity
from picongpu.picmi.species import Species as PICMISpecies
from picongpu.pypicongpu.species import Species as PyPIConGPUSpecies
import unittest
import typeguard


TESTCASES_VALID = [
    (
        {"species": PICMISpecies(name="electrons"), "filter": "species_all", "direction": "x"},
        {"filter": "species_all", "direction": "x"},
    ),
    (
        {"species": PICMISpecies(name="ions"), "filter": "fields_all", "direction": "y"},
        {"filter": "fields_all", "direction": "y"},
    ),
    (
        {"species": PICMISpecies(name="protons"), "filter": "custom_filter", "direction": "z"},
        {"filter": "custom_filter", "direction": "z"},
    ),
    (
        {"species": PICMISpecies(name="electrons")},
        {"filter": "species_all", "direction": "x"},
    ),
]

TESTCASES_INVALID = [
    (
        {"species": PICMISpecies(name="electrons"), "filter": "invalid", "direction": "x"},
        r"Filter must be one of \['species_all', 'fields_all', 'custom_filter'\], got invalid",
    ),
    (
        {"species": PICMISpecies(name="electrons"), "filter": 123, "direction": "x"},
        r"argument \"filter\" \(int\) is not an instance of str",
    ),
    (
        {"species": "not_a_species", "filter": "species_all", "direction": "x"},
        r"argument \"species\" \(str\) is not an instance of picongpu.picmi.species.Species",
    ),
    (
        {"species": PICMISpecies(name="electrons"), "filter": "species_all", "direction": "invalid"},
        r"Direction must be 'x', 'y', or 'z', got invalid",
    ),
    (
        {"species": PICMISpecies(name="electrons"), "filter": "species_all", "direction": 0},
        r"argument \"direction\" \(int\) is not an instance of str",
    ),
]

TESTCASES_INVALID_MAPPING = [
    (
        {"species": PICMISpecies(name="electrons"), "filter": "species_all", "direction": "x", "mapping": {}},
        "Species .* is not known to Simulation",
    ),
    (
        {
            "species": PICMISpecies(name="electrons"),
            "filter": "species_all",
            "direction": "x",
            "mapping": {PICMISpecies(name="ions"): PyPIConGPUSpecies()},
        },
        "Species .* is not known to Simulation",
    ),
]


class PICMI_TestMomentumDensity(unittest.TestCase):
    def test_momentum_density_instantiation(self):
        """Test MomentumDensity instantiation and validation."""
        for params, _ in TESTCASES_VALID:
            with self.subTest(params=params):
                source = MomentumDensity(**params)
                self.assertEqual(source.species, params["species"])
                self.assertEqual(source.filter, params.get("filter", "species_all"))
                self.assertEqual(source.direction, params.get("direction", "x"))
                source.check()

        for params, expected_error in TESTCASES_INVALID:
            with self.subTest(params=params, expected_error=expected_error):
                with self.assertRaisesRegex((ValueError, typeguard.TypeCheckError), expected_error):
                    MomentumDensity(**params)

    def test_momentum_density_serialization(self):
        """Test MomentumDensity serialization to PyPIConGPUMomentumDensity."""
        for params, expected_serialized in TESTCASES_VALID:
            with self.subTest(params=params, expected_serialized=expected_serialized):
                source = MomentumDensity(**params)
                pypicongpu_species = PyPIConGPUSpecies()
                mapping = {params["species"]: pypicongpu_species}
                pypicongpu_source = source.get_as_pypicongpu(mapping)
                self.assertIsInstance(pypicongpu_source, PyPIConGPUMomentumDensity)
                self.assertEqual(pypicongpu_source.filter, expected_serialized["filter"])
                self.assertEqual(pypicongpu_source.direction, expected_serialized["direction"])
                self.assertIsInstance(pypicongpu_source.species, PyPIConGPUSpecies)

    def test_momentum_density_invalid_mapping(self):
        """Test MomentumDensity with invalid species mapping."""
        for params, expected_error in TESTCASES_INVALID_MAPPING:
            with self.subTest(params=params, expected_error=expected_error):
                source = MomentumDensity(
                    species=params["species"], filter=params["filter"], direction=params["direction"]
                )
                with self.assertRaisesRegex(ValueError, expected_error):
                    source.get_as_pypicongpu(params["mapping"])


if __name__ == "__main__":
    unittest.main()
