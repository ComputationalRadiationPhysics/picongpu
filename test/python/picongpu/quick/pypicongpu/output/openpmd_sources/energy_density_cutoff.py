"""
This file is part of PIConGPU.
Copyright 2025 PIConGPU contributors
Authors: Masoud Afshari
License: GPLv3+
"""

from picongpu.pypicongpu.output import OpenPMD
from picongpu.pypicongpu.output.timestepspec import TimeStepSpec
from picongpu.pypicongpu.output.openpmd_sources.energy_density_cutoff import EnergyDensityCutoff
from picongpu.pypicongpu.species import Species
from picongpu.pypicongpu.species.attribute import Position, Momentum
import unittest
import typeguard
import typing


class MockSpecies(Species):
    def __init__(self):
        self.name = "electron"
        self.attributes = [Position(), Momentum()]
        self.constants = []

    def get_rendering_context(self) -> typing.Dict:
        return {
            "name": self.name,
            "typename": "Electron",
            "attributes": [{"picongpu_name": attr.__class__.__name__.lower()} for attr in self.attributes],
            "constants": {
                "mass": None,
                "charge": None,
                "density_ratio": None,
                "ground_state_ionization": None,
                "element_properties": None,
            },
        }

    def check(self) -> None:
        pass


class TestEnergyDensityCutoff(unittest.TestCase):
    def test_source_energy_density_cutoff(self):
        """Test EnergyDensityCutoff instantiation and serialization."""
        source = EnergyDensityCutoff(species=MockSpecies())
        self.assertIsInstance(source.species, MockSpecies)
        self.assertEqual(source.filter, "species_all")
        self.assertIsNone(source.cutoff_max_energy)
        source.check()

        source = EnergyDensityCutoff(species=MockSpecies(), filter="custom_filter", cutoff_max_energy=100.0)
        self.assertEqual(source.filter, "custom_filter")
        self.assertEqual(source.cutoff_max_energy, 100.0)
        source.check()

        source = EnergyDensityCutoff(species=MockSpecies(), filter="fields_all", cutoff_max_energy=50.0)
        self.assertEqual(source.filter, "fields_all")
        self.assertEqual(source.cutoff_max_energy, 50.0)
        source.check()

        with self.assertRaisesRegex(
            ValueError, r"Filter must be one of \['species_all', 'fields_all', 'custom_filter'\], got invalid"
        ):
            EnergyDensityCutoff(species=MockSpecies(), filter="invalid").check()

        with self.assertRaisesRegex(typeguard.TypeCheckError, r"argument \"filter\" \(int\) is not an instance of str"):
            EnergyDensityCutoff(species=MockSpecies(), filter=123)

        with self.assertRaisesRegex(
            typeguard.TypeCheckError,
            r"argument \"species\" \(str\) is not an instance of picongpu.pypicongpu.species.species.Species",
        ):
            EnergyDensityCutoff(species="invalid")

        with self.assertRaisesRegex(
            typeguard.TypeCheckError, r"argument \"cutoff_max_energy\" \(str\) did not match any element in the union"
        ):
            EnergyDensityCutoff(species=MockSpecies(), cutoff_max_energy="100").check()

        with self.assertRaisesRegex(ValueError, r"cutoff_max_energy must be positive, got -10.0"):
            EnergyDensityCutoff(species=MockSpecies(), cutoff_max_energy=-10.0).check()

        openpmd = OpenPMD(
            period=TimeStepSpec([slice(0, None, 100)]),
            source=[EnergyDensityCutoff(species=MockSpecies(), filter="custom_filter", cutoff_max_energy=100.0)],
        )
        context = openpmd.get_rendering_context()
        self.assertTrue(context["typeID"]["openpmd"])
        context = context["data"]
        self.assertEqual(len(context["source"]), 1)
        self.assertEqual(context["source"][0]["type"], "energydensitycutoff")
        self.assertEqual(context["source"][0]["filter"], "custom_filter")
        self.assertEqual(context["source"][0]["cutoff_max_energy"], 100.0)
        self.assertEqual(context["source"][0]["species"]["name"], "electron")
        self.assertEqual(context["source"][0]["species"]["typename"], "Electron")
        self.assertEqual(len(context["source"][0]["species"]["attributes"]), 2)
        self.assertEqual(
            context["source"][0]["species"]["constants"],
            {
                "mass": None,
                "charge": None,
                "density_ratio": None,
                "ground_state_ionization": None,
                "element_properties": None,
            },
        )

        openpmd = OpenPMD(
            period=TimeStepSpec([slice(0, None, 100)]), source=[EnergyDensityCutoff(species=MockSpecies())]
        )
        context = openpmd.get_rendering_context()
        self.assertEqual(context["data"]["source"][0]["type"], "energydensitycutoff")
        self.assertEqual(context["data"]["source"][0]["filter"], "species_all")
        self.assertIsNone(context["data"]["source"][0]["cutoff_max_energy"])
        self.assertEqual(context["data"]["source"][0]["species"]["name"], "electron")
        self.assertEqual(context["data"]["source"][0]["species"]["typename"], "Electron")
        self.assertEqual(len(context["data"]["source"][0]["species"]["attributes"]), 2)
        self.assertEqual(
            context["data"]["source"][0]["species"]["constants"],
            {
                "mass": None,
                "charge": None,
                "density_ratio": None,
                "ground_state_ionization": None,
                "element_properties": None,
            },
        )


if __name__ == "__main__":
    unittest.main()
