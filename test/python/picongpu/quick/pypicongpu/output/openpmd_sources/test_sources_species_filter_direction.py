"""
This file is part of PIConGPU.
Copyright 2025 PIConGPU contributors
Authors: Masoud Afshari
License: GPLv3+
"""

from picongpu.pypicongpu.output import OpenPMD
from picongpu.pypicongpu.output.timestepspec import TimeStepSpec
from picongpu.pypicongpu.output.openpmd_sources import (
    MidCurrentDensityComponent,
    Momentum,
    MomentumDensity,
    WeightedVelocity,
)
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


# ---------------------------------------------------------------------------
# Helper function
# ---------------------------------------------------------------------------


def _check_species_filter_direction_source(testcase: unittest.TestCase, source_cls):
    """Generic test routine for (species, filter, direction) sources."""
    directions = ["x", "y", "z"]
    filters = ["species_all", "fields_all", "custom_filter"]

    # Test all combinations of valid filters and directions
    for f in filters:
        for d in directions:
            src = source_cls(species=MockSpecies(), filter=f, direction=d)
            testcase.assertIsInstance(src.species, MockSpecies)
            testcase.assertEqual(src.filter, f)
            testcase.assertEqual(src.direction, d)
            src.check()

    # Invalid direction
    with testcase.assertRaisesRegex(ValueError, r"Direction must be 'x', 'y', or 'z', got invalid"):
        source_cls(species=MockSpecies(), direction="invalid")

    # Invalid type for direction
    with testcase.assertRaisesRegex(
        typeguard.TypeCheckError, r"argument \"direction\" \(int\) is not an instance of str"
    ):
        source_cls(species=MockSpecies(), direction=123)

    # Invalid species type
    with testcase.assertRaisesRegex(
        typeguard.TypeCheckError, r"argument \"species\" \(str\) is not an instance of .*Species"
    ):
        source_cls(species="invalid", direction="x")

    # OpenPMD serialization
    src = source_cls(species=MockSpecies(), filter="custom_filter", direction="y")
    openpmd = OpenPMD(period=TimeStepSpec([slice(0, None, 100)]), source=[src])
    context = openpmd.get_rendering_context()
    testcase.assertTrue(context["typeID"]["openpmd"])
    context = context["data"]
    testcase.assertEqual(len(context["source"]), 1)
    testcase.assertEqual(context["source"][0]["type"], source_cls.__name__.lower())
    testcase.assertEqual(context["source"][0]["filter"], "custom_filter")
    testcase.assertEqual(context["source"][0]["direction"], "y")
    testcase.assertEqual(context["source"][0]["species"]["name"], "electron")


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------


class PICMI_TestSpeciesFilterDirection(unittest.TestCase):
    def test_mid_current_density_component(self):
        _check_species_filter_direction_source(self, MidCurrentDensityComponent)

    def test_momentum(self):
        _check_species_filter_direction_source(self, Momentum)

    def test_momentum_density(self):
        _check_species_filter_direction_source(self, MomentumDensity)

    def test_weighted_velocity(self):
        _check_species_filter_direction_source(self, WeightedVelocity)


if __name__ == "__main__":
    unittest.main()
