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
import typing


# Mock Species class for testing
class MockSpecies(Species):
    def __init__(self):
        self.name = "electron"
        self.attributes = [Position(), Momentum()]
        self.constants = []

    def get_rendering_context(self) -> typing.Dict:
        return {}  # Minimal context to avoid schema conflicts

    def check(self) -> None:
        pass


class TestEnergyDensityCutoff(unittest.TestCase):
    def test_source_energy_density_cutoff(self):
        """Test EnergyDensityCutoff instantiation and serialization."""
        # Test instantiation with default filter and cutoff_max_energy
        source = EnergyDensityCutoff(species=MockSpecies())
        self.assertIsInstance(source.species, MockSpecies)
        self.assertEqual(source.filter, "all")
        self.assertIsNone(source.cutoff_max_energy)
        source.check()

        # Test instantiation with custom filter and cutoff_max_energy
        source = EnergyDensityCutoff(species=MockSpecies(), filter="custom", cutoff_max_energy=1.0)
        self.assertEqual(source.filter, "custom")
        self.assertEqual(source.cutoff_max_energy, 1.0)
        source.check()

        # Test invalid filter type
        with self.assertRaises(ValueError):
            EnergyDensityCutoff(species=MockSpecies(), filter=123).check()

        # Test invalid species type
        with self.assertRaises(ValueError):
            EnergyDensityCutoff(species="invalid").check()

        # Test invalid cutoff_max_energy type
        with self.assertRaises(ValueError):
            EnergyDensityCutoff(species=MockSpecies(), cutoff_max_energy="invalid").check()

        # Test non-positive cutoff_max_energy
        with self.assertRaises(ValueError):
            EnergyDensityCutoff(species=MockSpecies(), cutoff_max_energy=0).check()

        # Test serialization
        openpmd = OpenPMD(
            period=TimeStepSpec([slice(0, None, 100)]),
            source=[EnergyDensityCutoff(species=MockSpecies(), filter="custom", cutoff_max_energy=1.0)],
        )
        context = openpmd.get_rendering_context()
        self.assertTrue(context["typeID"]["openpmd"])
        context = context["data"]
        self.assertEqual(len(context["source"], 1))
        self.assertTrue(isinstance(context["source"][0], dict))
        self.assertEqual(context["source"][0]["filter"], "custom")
        self.assertEqual(context["source"][0]["species"], {})
        self.assertEqual(context["source"][0]["cutoff_max_energy"], 1.0)

        # Test serialization with default filter and cutoff_max_energy
        openpmd = OpenPMD(
            period=TimeStepSpec([slice(0, None, 100)]), source=[EnergyDensityCutoff(species=MockSpecies())]
        )
        context = openpmd.get_rendering_context()
        self.assertEqual(context["data"]["source"][0]["filter"], "all")
        self.assertEqual(context["data"]["source"][0]["species"], {})
        self.assertIsNone(context["data"]["source"][0]["cutoff_max_energy"])
