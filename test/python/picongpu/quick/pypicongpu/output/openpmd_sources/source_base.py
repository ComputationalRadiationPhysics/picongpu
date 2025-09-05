"""
This file is part of PIConGPU.
Copyright 2025 PIConGPU contributors
Authors: Masoud Afshari
License: GPLv3+
"""

from picongpu.pypicongpu.output import OpenPMD
from picongpu.pypicongpu.output.timestepspec import TimeStepSpec
from picongpu.pypicongpu.output.openpmd_sources import SourceBase, BoundElectronDensity
from picongpu.pypicongpu.species import Species
from picongpu.pypicongpu.species.attribute import Position, Momentum
import unittest
import typeguard


def create_species() -> Species:
    """Helper function to create a test species."""
    s = Species()
    s.name = "electron"
    s.attributes = [Position(), Momentum()]
    s.constants = []
    return s


class TestSourceBase(unittest.TestCase):
    def setUp(self):
        self.species = create_species()
        self.period = TimeStepSpec([slice(0, None, 100)])

    def test_source_base_abstract(self):
        """Test that SourceBase is abstract and BoundElectronDensity implements required methods."""
        with self.assertRaisesRegex(TypeError, "Can't instantiate abstract class"):
            SourceBase()

    def test_bound_electron_density_filters(self):
        """Test different filter values for BoundElectronDensity."""
        for filter_value in ["custom_filter", "species_all", "fields_all"]:
            source = BoundElectronDensity(species=self.species, filter=filter_value)
            self.assertEqual(source.filter, filter_value)
            source.check()

        # Invalid filter
        with self.assertRaisesRegex(
            ValueError, r"Filter must be one of \['species_all', 'fields_all', 'custom_filter'\], got invalid"
        ):
            BoundElectronDensity(species=self.species, filter="invalid").check()

        # Type check for filter
        with self.assertRaisesRegex(typeguard.TypeCheckError, r"argument \"filter\" \(int\) is not an instance of str"):
            BoundElectronDensity(species=self.species, filter=123)

        # Type check for species
        with self.assertRaisesRegex(
            typeguard.TypeCheckError,
            r"argument \"species\" \(str\) is not an instance of picongpu.pypicongpu.species.species.Species",
        ):
            BoundElectronDensity(species="invalid")

    def test_openpmd_rendering(self):
        """Test OpenPMD rendering with custom and default filters."""
        # Custom filter
        openpmd = OpenPMD(
            period=self.period, source=[BoundElectronDensity(species=self.species, filter="custom_filter")]
        )
        context = openpmd.get_rendering_context()
        self.assertTrue(context["typeID"]["openpmd"])
        context_data = context["data"]
        self.assertEqual(len(context_data["source"]), 1)
        self.assertEqual(context_data["source"][0]["type"], "boundelectrondensity")
        self.assertEqual(context_data["source"][0]["filter"], "custom_filter")
        self.assertEqual(context_data["source"][0]["species"]["name"], "electron")

        # Default filter
        openpmd = OpenPMD(period=self.period, source=[BoundElectronDensity(species=self.species)])
        context_data = openpmd.get_rendering_context()["data"]
        self.assertEqual(context_data["source"][0]["type"], "boundelectrondensity")
        self.assertEqual(context_data["source"][0]["filter"], "species_all")
        self.assertEqual(context_data["source"][0]["species"]["name"], "electron")


if __name__ == "__main__":
    unittest.main()
