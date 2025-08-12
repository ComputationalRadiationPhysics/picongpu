"""
This file is part of PIConGPU.
Copyright 2025 PIConGPU contributors
Authors: Masoud Afshari
License: GPLv3+
"""

from picongpu.pypicongpu.output import OpenPMD
from picongpu.pypicongpu.output.timestepspec import TimeStepSpec
from picongpu.pypicongpu.output.openpmd_sources.source_base import SourceBase
from picongpu.pypicongpu.output.openpmd_sources.auto import Auto

import unittest


class TestSourceBase(unittest.TestCase):
    def test_source_base_abstract(self):
        """Test SourceBase cannot be instantiated and Auto implements required methods."""
        # Test that SourceBase cannot be instantiated
        with self.assertRaises(TypeError):
            SourceBase()

        # Test Auto as a concrete subclass
        source = Auto(filter="custom")
        self.assertEqual(source.filter, "custom")
        source.check()  # Should not raise

        # Test invalid filter type
        with self.assertRaises(ValueError):
            Auto(filter=123).check()

        # Test serialization
        openpmd = OpenPMD(period=TimeStepSpec([slice(0, None, 100)]), source=[Auto(filter="custom")])
        context = openpmd.get_rendering_context()
        self.assertTrue(context["typeID"]["openpmd"])
        context = context["data"]
        self.assertEqual(len(context["source"], 1))
        self.assertTrue(isinstance(context["source"][0], dict))
        self.assertEqual(context["source"][0]["filter"], "custom")

        # Test serialization with default filter
        openpmd = OpenPMD(period=TimeStepSpec([slice(0, None, 100)]), source=[Auto()])
        context = openpmd.get_rendering_context()
        self.assertEqual(context["data"]["source"][0]["filter"], None)
