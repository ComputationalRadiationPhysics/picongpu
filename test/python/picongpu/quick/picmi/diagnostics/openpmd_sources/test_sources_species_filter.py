"""
This file is part of PIConGPU.
Copyright 2025 PIConGPU contributors
Authors: Masoud Afshari
License: GPLv3+
"""

import unittest
import typeguard

from picongpu.picmi.diagnostics.openpmd_sources import (
    BoundElectronDensity,
    ChargeDensity,
    Counter,
    Density,
    Energy,
    EnergyDensity,
    LarmorPower,
    MacroCounter,
)
from picongpu.pypicongpu.output.openpmd_sources import (
    BoundElectronDensity as PyPIConGPUBoundElectronDensity,
    ChargeDensity as PyPIConGPUChargeDensity,
    Counter as PyPIConGPUCounter,
    Density as PyPIConGPUDensity,
    Energy as PyPIConGPUEnergy,
    EnergyDensity as PyPIConGPUEnergyDensity,
    LarmorPower as PyPIConGPULarmorPower,
    MacroCounter as PyPIConGPUMacroCounter,
)
from picongpu.picmi.species import Species as PICMISpecies
from picongpu.pypicongpu.species import Species as PyPIConGPUSpecies


# List all the pairs to test
SOURCE_CLASSES = [
    (BoundElectronDensity, PyPIConGPUBoundElectronDensity),
    (ChargeDensity, PyPIConGPUChargeDensity),
    (Counter, PyPIConGPUCounter),
    (Density, PyPIConGPUDensity),
    (Energy, PyPIConGPUEnergy),
    (EnergyDensity, PyPIConGPUEnergyDensity),
    (LarmorPower, PyPIConGPULarmorPower),
    (MacroCounter, PyPIConGPUMacroCounter),
]

TESTCASES_VALID = [
    ({"species": PICMISpecies(name="electrons"), "filter": "species_all"}, {"filter": "species_all"}),
    ({"species": PICMISpecies(name="electrons")}, {"filter": "species_all"}),
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


class PICMI_TestSpeciesFilterSources(unittest.TestCase):
    def test_instantiation_and_validation(self):
        """Test instantiation and validation."""
        for SourceClass, _ in SOURCE_CLASSES:
            for params, _ in TESTCASES_VALID:
                with self.subTest(Source=SourceClass.__name__, params=params):
                    source = SourceClass(**params)
                    self.assertEqual(source.species, params["species"])
                    self.assertEqual(source.filter, params.get("filter", "species_all"))
                    source.check()

            for params, expected_error in TESTCASES_INVALID:
                with self.subTest(Source=SourceClass.__name__, params=params):
                    with self.assertRaisesRegex((ValueError, typeguard.TypeCheckError), expected_error):
                        SourceClass(**params)

    def test_serialization(self):
        """Test serialization."""
        for SourceClass, PySourceClass in SOURCE_CLASSES:
            for params, expected_serialized in TESTCASES_VALID:
                with self.subTest(Source=SourceClass.__name__, params=params):
                    source = SourceClass(**params)
                    pypicongpu_species = PyPIConGPUSpecies()
                    mapping = {params["species"]: pypicongpu_species}
                    pypicongpu_source = source.get_as_pypicongpu(mapping)
                    self.assertIsInstance(pypicongpu_source, PySourceClass)
                    self.assertEqual(pypicongpu_source.filter, expected_serialized["filter"])
                    self.assertIsInstance(pypicongpu_source.species, PyPIConGPUSpecies)

    def test_invalid_mapping(self):
        """Test invalid species mapping.
        mapping to convert PICMI species to PyPIConGPU species"""
        for SourceClass, _ in SOURCE_CLASSES:
            for params, expected_error in TESTCASES_INVALID_MAPPING:
                with self.subTest(Source=SourceClass.__name__, params=params):
                    source = SourceClass(species=params["species"], filter=params["filter"])
                    with self.assertRaisesRegex(ValueError, expected_error):
                        source.get_as_pypicongpu(params["mapping"])


if __name__ == "__main__":
    unittest.main()
