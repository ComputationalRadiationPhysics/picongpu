"""
This file is part of PIConGPU.
Copyright 2025 PIConGPU contributors
Authors: Masoud Afshari
License: GPLv3+
"""

from picongpu.pypicongpu.output import OpenPMD
from picongpu.pypicongpu.output.timestepspec import TimeStepSpec
from picongpu.pypicongpu.output.openpmd_sources.density import Density
from picongpu.pypicongpu.species import Species
from picongpu.pypicongpu.species.attribute import Position, Momentum

import unittest
import typing


# Mock Species class for testing
class MockSpecies(Species):
    def __init__(self):
        self.name = "electron"
        self.attributes = [Position(), Momentum()]
        self.constants = []

    def get_rendering_context(self) -> typing.Dict:
        return {}

    def check(self) -> None:
        pass


class TestDensity(unittest.TestCase):
    def test_source_density(self):
        """Test Density instantiation and serialization."""
        # Test instantiation with default filter
        source = Density(species=MockSpecies())
        self.assertIsInstance(source.species, MockSpecies)
        self.assertEqual(source.filter, "all")
        source.check()

        # Test instantiation with custom filter
        source = Density(species=MockSpecies(), filter="custom")
        self.assertEqual(source.filter, "custom")
        source.check()

        # Test invalid filter type
        with self.assertRaises(ValueError):
            Density(species=MockSpecies(), filter=123).check()

        # Test invalid species type
        with self.assertRaises(ValueError):
            Density(species="invalid").check()

        # Test serialization
        openpmd = OpenPMD(
            period=TimeStepSpec([slice(0, None, 100)]), source=[Density(species=MockSpecies(), filter="custom")]
        )
        context = openpmd.get_rendering_context()
        self.assertTrue(context["typeID"]["openpmd"])
        context = context["data"]
        self.assertEqual(len(context["source"]), 1)
        self.assertTrue(isinstance(context["source"][0], dict))
        self.assertEqual(context["source"][0]["filter"], "custom")
        self.assertEqual(context["source"][0]["species"], {})

        # Test serialization with default filter
        openpmd = OpenPMD(period=TimeStepSpec([slice(0, None, 100)]), source=[Density(species=MockSpecies())])
        context = openpmd.get_rendering_context()
        self.assertEqual(context["data"]["source"][0]["filter"], "all")
        self.assertEqual(context["data"]["source"][0]["species"], {})
