"""
This file is part of PIConGPU.
Copyright 2025 PIConGPU contributors
Authors: Masoud Afshari
License: GPLv3+
"""

from picongpu.pypicongpu.output import OpenPMD
from picongpu.pypicongpu.output.timestepspec import TimeStepSpec
from picongpu.pypicongpu.output.openpmd_sources.derived_attributes import DerivedAttributes

import unittest


class TestDerivedAttributes(unittest.TestCase):
    def test_source_derived_attributes(self):
        """Test DerivedAttributes instantiation and serialization."""
        # Test instantiation with default filter (None)
        source = DerivedAttributes()
        self.assertIsNone(source.filter)
        source.check()

        # Test instantiation with custom filter
        source = DerivedAttributes(filter="custom")
        self.assertEqual(source.filter, "custom")
        source.check()

        # Test invalid filter type
        with self.assertRaises(ValueError):
            DerivedAttributes(filter=123).check()

        # Test serialization
        openpmd = OpenPMD(period=TimeStepSpec([slice(0, None, 100)]), source=[DerivedAttributes(filter="custom")])
        context = openpmd.get_rendering_context()
        self.assertTrue(context["typeID"]["openpmd"])
        context = context["data"]
        self.assertEqual(len(context["source"], 1))
        self.assertTrue(isinstance(context["source"][0], dict))
        self.assertEqual(context["source"][0]["filter"], "custom")

        # Test serialization with default filter
        openpmd = OpenPMD(period=TimeStepSpec([slice(0, None, 100)]), source=[DerivedAttributes()])
        context = openpmd.get_rendering_context()
        self.assertEqual(context["data"]["source"][0]["filter"], None)
