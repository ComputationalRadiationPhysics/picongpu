"""
This file is part of PIConGPU.
Copyright 2025 PIConGPU contributors
Authors: Masoud Afshari
License: GPLv3+
"""

import unittest
import typeguard
from picongpu.pypicongpu.output import OpenPMD
from picongpu.pypicongpu.output.timestepspec import TimeStepSpec
from picongpu.pypicongpu.output.openpmd_sources import Auto, DerivedAttributes

# ---------------------------------------------------------------------------
# Helper function to reduce duplication
# ---------------------------------------------------------------------------


def _check_filter_only_source(testcase: unittest.TestCase, source_cls):
    """Generic test routine for filter-only sources."""
    # Valid filters
    for f in ["species_all", "fields_all", "custom_filter"]:
        src = source_cls(filter=f)
        testcase.assertEqual(src.filter, f)
        src.check()

    # Invalid filter type
    with testcase.assertRaisesRegex(typeguard.TypeCheckError, r"argument \"filter\" \(int\) is not an instance of str"):
        source_cls(filter=123)

    # Invalid filter value
    with testcase.assertRaisesRegex(
        ValueError, r"Filter must be one of \['species_all', 'fields_all', 'custom_filter'\], got invalid"
    ):
        source_cls(filter="invalid")

    # Test OpenPMD serialization
    # Custom filter
    src = source_cls(filter="custom_filter")
    openpmd = OpenPMD(period=TimeStepSpec([slice(0, None, 100)]), source=[src])
    context = openpmd.get_rendering_context()
    testcase.assertTrue(context["typeID"]["openpmd"])
    context = context["data"]
    testcase.assertEqual(len(context["source"]), 1)
    testcase.assertEqual(context["source"][0]["type"], source_cls.__name__.lower())
    testcase.assertEqual(context["source"][0]["filter"], "custom_filter")

    # Default filter
    src = source_cls()
    openpmd = OpenPMD(period=TimeStepSpec([slice(0, None, 100)]), source=[src])
    context = openpmd.get_rendering_context()
    testcase.assertEqual(context["data"]["source"][0]["type"], source_cls.__name__.lower())
    testcase.assertEqual(context["data"]["source"][0]["filter"], "species_all")


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------


class PICMI_TestFilterOnlySources(unittest.TestCase):
    def test_auto(self):
        _check_filter_only_source(self, Auto)

    def test_derived_attributes(self):
        _check_filter_only_source(self, DerivedAttributes)


if __name__ == "__main__":
    unittest.main()
