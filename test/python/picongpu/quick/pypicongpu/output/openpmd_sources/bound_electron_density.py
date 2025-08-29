"""
This file is part of PIConGPU.
Copyright 2025 PIConGPU contributors
Authors: Masoud Afshari
License: GPLv3+
"""

from picongpu.pypicongpu.output import OpenPMD
from picongpu.pypicongpu.output.timestepspec import TimeStepSpec
from picongpu.pypicongpu.output.openpmd_sources.bound_electron_density import BoundElectronDensity
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


class TestBoundElectronDensity(unittest.TestCase):
    def test_source_bound_electron_density(self):
        """Test BoundElectronDensity instantiation and serialization."""
        source = BoundElectronDensity(species=MockSpecies(), filter="species_all")
        self.assertIsInstance(source.species, MockSpecies)
        self.assertEqual(source.filter, "species_all")
        source.check()

        source = BoundElectronDensity(species=MockSpecies(), filter="fields_all")
        self.assertEqual(source.filter, "fields_all")
        source.check()

        source = BoundElectronDensity(species=MockSpecies(), filter="custom_filter")
        self.assertEqual(source.filter, "custom_filter")
        source.check()

        with self.assertRaisesRegex(
            ValueError, r"Filter must be one of \['species_all', 'fields_all', 'custom_filter'\], got invalid"
        ):
            BoundElectronDensity(species=MockSpecies(), filter="invalid").check()

        with self.assertRaisesRegex(typeguard.TypeCheckError, r"argument \"filter\" \(int\) is not an instance of str"):
            BoundElectronDensity(species=MockSpecies(), filter=123)

        with self.assertRaisesRegex(
            typeguard.TypeCheckError,
            r"argument \"species\" \(str\) is not an instance of picongpu.pypicongpu.species.species.Species",
        ):
            BoundElectronDensity(species="invalid")

        openpmd = OpenPMD(
            period=TimeStepSpec([slice(0, None, 100)]),
            source=[BoundElectronDensity(species=MockSpecies(), filter="custom_filter")],
        )
        context = openpmd.get_rendering_context()
        self.assertTrue(context["typeID"]["openpmd"])
        context = context["data"]
        self.assertEqual(len(context["source"]), 1)
        self.assertEqual(context["source"][0]["type"], "boundelectrondensity")
        self.assertEqual(context["source"][0]["filter"], "custom_filter")
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
            period=TimeStepSpec([slice(0, None, 100)]),
            source=[BoundElectronDensity(species=MockSpecies())],
        )
        context = openpmd.get_rendering_context()
        self.assertEqual(context["data"]["source"][0]["type"], "boundelectrondensity")
        self.assertEqual(context["data"]["source"][0]["filter"], "species_all")
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
