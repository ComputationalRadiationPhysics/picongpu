"""
This file is part of PIConGPU.
Copyright 2025 PIConGPU contributors
Authors: Masoud Afshari
License: GPLv3+
"""

from picongpu.pypicongpu.output import OpenPMD
from picongpu.pypicongpu.output.timestepspec import TimeStepSpec
from picongpu.pypicongpu.output.openpmd_sources import (
    BoundElectronDensity,
    ChargeDensity,
    Density,
    Energy,
    EnergyDensity,
    LarmorPower,
    MacroCounter,
    Counter,
)
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


# ---------------------------------------------------------------------------
# Helper function to reduce duplication
# ---------------------------------------------------------------------------


def _check_species_filter_source(testcase: unittest.TestCase, source_cls):
    """Generic test routine for (species, filter) sources."""
    # Valid filters
    for f in ["species_all", "fields_all", "custom_filter"]:
        src = source_cls(species=MockSpecies(), filter=f)
        testcase.assertIsInstance(src.species, MockSpecies)
        testcase.assertEqual(src.filter, f)
        src.check()

    # Invalid filter
    with testcase.assertRaisesRegex(
        ValueError, r"Filter must be one of \['species_all', 'fields_all', 'custom_filter'\], got invalid"
    ):
        source_cls(species=MockSpecies(), filter="invalid").check()

    # Wrong filter type
    with testcase.assertRaisesRegex(typeguard.TypeCheckError, r"argument \"filter\" \(int\) is not an instance of str"):
        source_cls(species=MockSpecies(), filter=123)

    # Wrong species type
    with testcase.assertRaisesRegex(
        typeguard.TypeCheckError,
        r"argument \"species\" \(str\) is not an instance of picongpu.pypicongpu.species.species.Species",
    ):
        source_cls(species="invalid")

    # OpenPMD serialization
    openpmd = OpenPMD(
        period=TimeStepSpec([slice(0, None, 100)]),
        source=[source_cls(species=MockSpecies(), filter="custom_filter")],
    )
    context = openpmd.get_rendering_context()
    testcase.assertTrue(context["typeID"]["openpmd"])
    context = context["data"]
    testcase.assertEqual(len(context["source"]), 1)
    testcase.assertEqual(context["source"][0]["type"], source_cls.__name__.lower())
    testcase.assertEqual(context["source"][0]["filter"], "custom_filter")
    testcase.assertEqual(context["source"][0]["species"]["name"], "electron")
    testcase.assertEqual(context["source"][0]["species"]["typename"], "Electron")
    testcase.assertEqual(len(context["source"][0]["species"]["attributes"]), 2)
    testcase.assertEqual(
        context["source"][0]["species"]["constants"],
        {
            "mass": None,
            "charge": None,
            "density_ratio": None,
            "ground_state_ionization": None,
            "element_properties": None,
        },
    )

    # Default filter = "species_all"
    openpmd = OpenPMD(period=TimeStepSpec([slice(0, None, 100)]), source=[source_cls(species=MockSpecies())])
    context = openpmd.get_rendering_context()
    testcase.assertEqual(context["data"]["source"][0]["type"], source_cls.__name__.lower())
    testcase.assertEqual(context["data"]["source"][0]["filter"], "species_all")
    testcase.assertEqual(context["data"]["source"][0]["species"]["name"], "electron")


# ---------------------------------------------------------------------------
# Test classes for each species+filter source
# ---------------------------------------------------------------------------


class TestBoundElectronDensity(unittest.TestCase):
    def test_source(self):
        _check_species_filter_source(self, BoundElectronDensity)


class TestChargeDensity(unittest.TestCase):
    def test_source(self):
        _check_species_filter_source(self, ChargeDensity)


class TestDensity(unittest.TestCase):
    def test_source(self):
        _check_species_filter_source(self, Density)


class TestEnergy(unittest.TestCase):
    def test_source(self):
        _check_species_filter_source(self, Energy)


class TestEnergyDensity(unittest.TestCase):
    def test_source(self):
        _check_species_filter_source(self, EnergyDensity)


class TestLarmorPower(unittest.TestCase):
    def test_source(self):
        _check_species_filter_source(self, LarmorPower)


class TestMacroCounter(unittest.TestCase):
    def test_source(self):
        _check_species_filter_source(self, MacroCounter)


class TestCounter(unittest.TestCase):
    def test_source(self):
        _check_species_filter_source(self, Counter)


if __name__ == "__main__":
    unittest.main()
