"""
This file is part of PIConGPU.
Copyright 2025 PIConGPU contributors
Authors: Masoud Afshari
License: GPLv3+
"""

from picongpu.pypicongpu.output import OpenPMD
from picongpu.pypicongpu.output.timestepspec import TimeStepSpec
from picongpu.pypicongpu.output.openpmd_sources.momentum_density import MomentumDensity
from picongpu.pypicongpu.species import Species
from picongpu.pypicongpu.species.attribute import Position, Momentum as MomentumAttr
import unittest
import typeguard
import typing


class MockSpecies(Species):
    def __init__(self):
        self.name = "electron"
        self.attributes = [Position(), MomentumAttr()]
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


class TestMomentumDensity(unittest.TestCase):
    def test_source_momentum_density(self):
        """Test MomentumDensity instantiation and serialization."""
        source = MomentumDensity(species=MockSpecies())
        self.assertIsInstance(source.species, MockSpecies)
        self.assertEqual(source.filter, "species_all")
        self.assertEqual(source.direction, "x")
        source.check()

        source = MomentumDensity(species=MockSpecies(), filter="custom_filter", direction="y")
        self.assertEqual(source.filter, "custom_filter")
        self.assertEqual(source.direction, "y")
        source.check()

        source = MomentumDensity(species=MockSpecies(), filter="fields_all", direction="z")
        self.assertEqual(source.filter, "fields_all")
        self.assertEqual(source.direction, "z")
        source.check()

        with self.assertRaisesRegex(
            ValueError, r"Filter must be one of \['species_all', 'fields_all', 'custom_filter'\], got invalid"
        ):
            MomentumDensity(species=MockSpecies(), filter="invalid").check()

        with self.assertRaisesRegex(typeguard.TypeCheckError, r"argument \"filter\" \(int\) is not an instance of str"):
            MomentumDensity(species=MockSpecies(), filter=123)

        with self.assertRaisesRegex(
            typeguard.TypeCheckError,
            r"argument \"species\" \(str\) is not an instance of picongpu.pypicongpu.species.species.Species",
        ):
            MomentumDensity(species="invalid")

        with self.assertRaisesRegex(
            typeguard.TypeCheckError, r"argument \"direction\" \(str\) is not any of \('x', 'y', 'z'\)"
        ):
            MomentumDensity(species=MockSpecies(), direction="invalid").check()

        openpmd = OpenPMD(
            period=TimeStepSpec([slice(0, None, 100)]),
            source=[MomentumDensity(species=MockSpecies(), filter="custom_filter", direction="z")],
        )
        context = openpmd.get_rendering_context()
        self.assertTrue(context["typeID"]["openpmd"])
        context = context["data"]
        self.assertEqual(len(context["source"]), 1)
        self.assertEqual(context["source"][0]["type"], "momentumdensity")
        self.assertEqual(context["source"][0]["filter"], "custom_filter")
        self.assertEqual(context["source"][0]["direction"], "z")
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

        openpmd = OpenPMD(period=TimeStepSpec([slice(0, None, 100)]), source=[MomentumDensity(species=MockSpecies())])
        context = openpmd.get_rendering_context()
        self.assertEqual(context["data"]["source"][0]["type"], "momentumdensity")
        self.assertEqual(context["data"]["source"][0]["filter"], "species_all")
        self.assertEqual(context["data"]["source"][0]["direction"], "x")
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
