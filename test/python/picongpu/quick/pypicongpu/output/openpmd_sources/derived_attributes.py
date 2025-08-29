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
import typeguard


class TestDerivedAttributes(unittest.TestCase):
    def test_source_derived_attributes(self):
        """Test DerivedAttributes instantiation and serialization."""
        source = DerivedAttributes()
        self.assertEqual(source.filter, "species_all")
        source.check()

        source = DerivedAttributes(filter="custom_filter")
        self.assertEqual(source.filter, "custom_filter")
        source.check()

        source = DerivedAttributes(filter="fields_all")
        self.assertEqual(source.filter, "fields_all")
        source.check()

        with self.assertRaisesRegex(
            ValueError, r"Filter must be one of \['species_all', 'fields_all', 'custom_filter'\], got invalid"
        ):
            DerivedAttributes(filter="invalid").check()

        with self.assertRaisesRegex(typeguard.TypeCheckError, r"argument \"filter\" \(int\) is not an instance of str"):
            DerivedAttributes(filter=123)

        openpmd = OpenPMD(
            period=TimeStepSpec([slice(0, None, 100)]), source=[DerivedAttributes(filter="custom_filter")]
        )
        context = openpmd.get_rendering_context()
        self.assertTrue(context["typeID"]["openpmd"])
        context = context["data"]
        self.assertEqual(len(context["source"]), 1)
        self.assertEqual(context["source"][0]["type"], "derivedattributes")
        self.assertEqual(context["source"][0]["filter"], "custom_filter")

        openpmd = OpenPMD(period=TimeStepSpec([slice(0, None, 100)]), source=[DerivedAttributes()])
        context = openpmd.get_rendering_context()
        self.assertEqual(context["data"]["source"][0]["type"], "derivedattributes")
        self.assertEqual(context["data"]["source"][0]["filter"], "species_all")


if __name__ == "__main__":
    unittest.main()
