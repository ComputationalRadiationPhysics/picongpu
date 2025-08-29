"""
This file is part of PIConGPU.
Copyright 2025 PIConGPU contributors
Authors: Masoud Afshari
License: GPLv3+
"""

from picongpu.pypicongpu.output import OpenPMD
from picongpu.pypicongpu.output.timestepspec import TimeStepSpec
from picongpu.pypicongpu.output.openpmd_sources.auto import Auto
import unittest
import typeguard


class TestAuto(unittest.TestCase):
    def test_source_auto(self):
        """Test Auto instantiation and serialization."""
        source = Auto(filter="species_all")
        self.assertEqual(source.filter, "species_all")
        source.check()

        source = Auto(filter="fields_all")
        self.assertEqual(source.filter, "fields_all")
        source.check()

        source = Auto(filter="custom_filter")
        self.assertEqual(source.filter, "custom_filter")
        source.check()

        with self.assertRaisesRegex(
            ValueError, r"Filter must be one of \['species_all', 'fields_all', 'custom_filter'\], got invalid"
        ):
            Auto(filter="invalid").check()

        with self.assertRaisesRegex(typeguard.TypeCheckError, r"argument \"filter\" \(int\) is not an instance of str"):
            Auto(filter=123)

        openpmd = OpenPMD(period=TimeStepSpec([slice(0, None, 100)]), source=[Auto(filter="custom_filter")])
        context = openpmd.get_rendering_context()
        self.assertTrue(context["typeID"]["openpmd"])
        context = context["data"]
        self.assertEqual(len(context["source"]), 1)
        self.assertEqual(context["source"][0]["type"], "auto")
        self.assertEqual(context["source"][0]["filter"], "custom_filter")

        openpmd = OpenPMD(period=TimeStepSpec([slice(0, None, 100)]), source=[Auto()])
        context = openpmd.get_rendering_context()
        self.assertEqual(context["data"]["source"][0]["type"], "auto")
        self.assertEqual(context["data"]["source"][0]["filter"], "species_all")


if __name__ == "__main__":
    unittest.main()
