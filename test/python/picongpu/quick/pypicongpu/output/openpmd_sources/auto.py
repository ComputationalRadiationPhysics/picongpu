"""
This file is part of PIConGPU.
Copyright 2025 PIConGPU contributors
Authors: Masoud Afshari
License: GPLv3+
"""

from picongpu.pypicongpu.output import OpenPMD
from picongpu.pypicongpu.output.timestepspec import TimeStepSpec
from picongpu.pypicongpu.output.openpmd_sources.auto import Auto
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


class TestAuto(unittest.TestCase):
    def test_source_auto(self):
        """Test Auto instantiation and serialization."""
        # Test instantiation with default filter (None)
        source = Auto()
        self.assertIsNone(source.filter)
        source.check()

        # Test instantiation with custom filter
        source = Auto(filter="custom")
        self.assertEqual(source.filter, "custom")
        source.check()

        # Test invalid filter type for non-string/non-None
        with self.assertRaises(ValueError):
            Auto(filter=123).check()

        # Test serialization
        openpmd = OpenPMD(period=TimeStepSpec([slice(0, None, 100)]), source=[Auto(filter="custom")])
        context = openpmd.get_rendering_context()
        self.assertTrue(context["typeID"]["openpmd"])
        context = context["data"]
        self.assertEqual(len(context["source"]), 1)
        self.assertTrue(isinstance(context["source"][0], dict))
        self.assertEqual(context["source"][0]["filter"], "custom")

        # Test serialization with default filter
        openpmd = OpenPMD(period=TimeStepSpec([slice(0, None, 100)]), source=[Auto()])
        context = openpmd.get_rendering_context()
        self.assertEqual(context["data"]["source"][0]["filter"], None)


if __name__ == "__main__":
    unittest.main()
