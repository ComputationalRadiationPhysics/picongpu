"""
This file is part of PIConGPU.
Copyright 2025 PIConGPU contributors
Authors: Masoud Afshari
License: GPLv3+
"""

import unittest
import typeguard

from picongpu.picmi.diagnostics.openpmd_sources import (
    Momentum,
    MidCurrentDensityComponent,
    MomentumDensity,
    WeightedVelocity,
)
from picongpu.pypicongpu.output.openpmd_sources import (
    Momentum as PyPIConGPUMomentum,
    MidCurrentDensityComponent as PyPIConGPUMidCurrentDensityComponent,
    MomentumDensity as PyPIConGPUMomentumDensity,
    WeightedVelocity as PyPIConGPUWeightedVelocity,
)
from picongpu.picmi.species import Species as PICMISpecies
from picongpu.pypicongpu.species import Species as PyPIConGPUSpecies


# List all the pairs to test
SOURCE_CLASSES = [
    (Momentum, PyPIConGPUMomentum),
    (MidCurrentDensityComponent, PyPIConGPUMidCurrentDensityComponent),
    (MomentumDensity, PyPIConGPUMomentumDensity),
    (WeightedVelocity, PyPIConGPUWeightedVelocity),
]

# Valid cases: species + filter + direction
TESTCASES_VALID = [
    (
        {"species": PICMISpecies(name="electrons"), "filter": "species_all", "direction": "x"},
        {"filter": "species_all", "direction": "x"},
    ),
    ({"species": PICMISpecies(name="electrons"), "direction": "y"}, {"filter": "species_all", "direction": "y"}),
    ({"species": PICMISpecies(name="electrons"), "direction": "z"}, {"filter": "species_all", "direction": "z"}),
]

# Invalid cases: wrong filter, wrong species, wrong direction
TESTCASES_INVALID = [
    # Invalid filter
    (
        {"species": PICMISpecies(name="electrons"), "filter": "invalid", "direction": "x"},
        r"Filter must be one of \['species_all', 'fields_all', 'custom_filter'\], got invalid",
    ),
    (
        {"species": PICMISpecies(name="electrons"), "filter": 123, "direction": "x"},
        r"argument \"filter\" \(int\) is not an instance of str",
    ),
    # Invalid species
    (
        {"species": "not_a_species", "filter": "species_all", "direction": "x"},
        r"argument \"species\" \(str\) is not an instance of picongpu.picmi.species.Species",
    ),
    # Invalid direction
    (
        {"species": PICMISpecies(name="electrons"), "filter": "species_all", "direction": "invalid"},
        r"Direction must be 'x', 'y', or 'z', got invalid",
    ),
    (
        {"species": PICMISpecies(name="electrons"), "filter": "species_all", "direction": 123},
        r"argument \"direction\" \(int\) is not an instance of str",
    ),
]

# Invalid mapping cases
TESTCASES_INVALID_MAPPING = [
    (
        {"species": PICMISpecies(name="electrons"), "filter": "species_all", "direction": "x", "mapping": {}},
        "Species .* is not known to Simulation",
    ),
    (
        {
            "species": PICMISpecies(name="electrons"),
            "filter": "species_all",
            "direction": "y",
            "mapping": {PICMISpecies(name="ions"): PyPIConGPUSpecies()},
        },
        "Species .* is not known to Simulation",
    ),
]


class PICMI_TestSpeciesFilterDirectionSources(unittest.TestCase):
    def test_instantiation_and_validation(self):
        """Test instantiation and validation, including direction."""
        for SourceClass, _ in SOURCE_CLASSES:
            # Valid cases
            for params, _ in TESTCASES_VALID:
                with self.subTest(Source=SourceClass.__name__, params=params):
                    source = SourceClass(**params)
                    self.assertEqual(source.species, params["species"])
                    self.assertEqual(source.filter, params.get("filter", "species_all"))
                    self.assertEqual(source.direction, params["direction"])
                    source.check()

            # Invalid cases
            for params, expected_error in TESTCASES_INVALID:
                with self.subTest(Source=SourceClass.__name__, params=params):
                    with self.assertRaisesRegex((ValueError, typeguard.TypeCheckError), expected_error):
                        SourceClass(**params)

    def test_serialization(self):
        """Test serialization to PyPIConGPU sources, including direction."""
        for SourceClass, PySourceClass in SOURCE_CLASSES:
            for params, expected_serialized in TESTCASES_VALID:
                with self.subTest(Source=SourceClass.__name__, params=params):
                    source = SourceClass(**params)
                    pypicongpu_species = PyPIConGPUSpecies()
                    mapping = {params["species"]: pypicongpu_species}
                    pypicongpu_source = source.get_as_pypicongpu(mapping)
                    self.assertIsInstance(pypicongpu_source, PySourceClass)
                    self.assertEqual(pypicongpu_source.filter, expected_serialized["filter"])
                    self.assertEqual(pypicongpu_source.direction, expected_serialized["direction"])
                    self.assertIsInstance(pypicongpu_source.species, PyPIConGPUSpecies)

    def test_invalid_mapping(self):
        """Test invalid species mapping for direction sources."""
        for SourceClass, _ in SOURCE_CLASSES:
            for params, expected_error in TESTCASES_INVALID_MAPPING:
                with self.subTest(Source=SourceClass.__name__, params=params):
                    source = SourceClass(
                        species=params["species"],
                        filter=params["filter"],
                        direction=params["direction"],
                    )
                    with self.assertRaisesRegex(ValueError, expected_error):
                        source.get_as_pypicongpu(params["mapping"])


if __name__ == "__main__":
    unittest.main()
