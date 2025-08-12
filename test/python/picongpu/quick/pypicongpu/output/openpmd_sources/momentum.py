"""
This file is part of PIConGPU.
Copyright 2025 PIConGPU contributors
Authors: Masoud Afshari
License: GPLv3+
"""

from picongpu.pypicongpu.output import OpenPMD
from picongpu.pypicongpu.output.timestepspec import TimeStepSpec
from picongpu.pypicongpu.output.openpmd_sources.momentum import Momentum
from picongpu.pypicongpu.species import Species
from picongpu.pypicongpu.species.attribute import Position, Momentum as MomentumAttr

import unittest
import typing


# Mock Species class for testing
class MockSpecies(Species):
    def __init__(self):
        self.name = "electron"
        self.attributes = [Position(), MomentumAttr()]
        self.constants = []

    def get_rendering_context(self) -> typing.Dict:
        return {}  # Minimal context to avoid schema conflicts

    def check(self) -> None:
        pass


class TestMomentum(unittest.TestCase):
    def test_source_momentum(self):
        """Test Momentum instantiation and serialization."""
        # Test instantiation with default filter and direction
        source = Momentum(species=MockSpecies())
        self.assertIsInstance(source.species, MockSpecies)
        self.assertEqual(source.filter, "all")
        self.assertEqual(source.direction, "x")
        source.check()

        # Test instantiation with custom filter and direction
        source = Momentum(species=MockSpecies(), filter="custom", direction="y")
        self.assertEqual(source.filter, "custom")
        self.assertEqual(source.direction, "y")
        source.check()

        # Test invalid filter type
        with self.assertRaises(ValueError):
            Momentum(species=MockSpecies(), filter=123).check()

        # Test invalid species type
        with self.assertRaises(ValueError):
            Momentum(species="invalid").check()

        # Test invalid direction
        with self.assertRaises(ValueError):
            Momentum(species=MockSpecies(), direction="invalid").check()

        # Test serialization
        openpmd = OpenPMD(
            period=TimeStepSpec([slice(0, None, 100)]),
            source=[Momentum(species=MockSpecies(), filter="custom", direction="z")],
        )
        context = openpmd.get_rendering_context()
        self.assertTrue(context["typeID"]["openpmd"])
        context = context["data"]
        self.assertEqual(len(context["source"], 1))
        self.assertTrue(isinstance(context["source"][0], dict))
        self.assertEqual(context["source"][0]["filter"], "custom")
        self.assertEqual(context["source"][0]["species"], {})
        self.assertEqual(context["source"][0]["direction"], "z")

        # Test serialization with default filter and direction
        openpmd = OpenPMD(period=TimeStepSpec([slice(0, None, 100)]), source=[Momentum(species=MockSpecies())])
        context = openpmd.get_rendering_context()
        self.assertEqual(context["data"]["source"][0]["filter"], "all")
        self.assertEqual(context["data"]["source"][0]["species"], {})
        self.assertEqual(context["data"]["source"][0]["direction"], "x")
