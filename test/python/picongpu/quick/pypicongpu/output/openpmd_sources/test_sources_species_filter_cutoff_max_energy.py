"""
This file is part of PIConGPU.
Copyright 2025 PIConGPU contributors
Authors: Masoud Afshari
License: GPLv3+
"""

from picongpu.pypicongpu.output import OpenPMD
from picongpu.pypicongpu.output.timestepspec import TimeStepSpec
from picongpu.pypicongpu.output.openpmd_sources import EnergyDensityCutoff
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
# Helper function
# ---------------------------------------------------------------------------


def _check_species_filter_cutoff_source(testcase: unittest.TestCase, source_cls):
    """Generic test routine for EnergyDensityCutoff source."""
    filters = ["species_all", "fields_all", "custom_filter"]
    cutoff_values = [50.0, 100.0]  # example valid cutoffs

    # Test all combinations of valid filters and cutoff_max_energy
    for f in filters:
        for cutoff in cutoff_values:
            src = source_cls(species=MockSpecies(), filter=f, cutoff_max_energy=cutoff)
            testcase.assertIsInstance(src.species, MockSpecies)
            testcase.assertEqual(src.filter, f)
            testcase.assertEqual(src.cutoff_max_energy, cutoff)
            src.check()

    # Missing cutoff_max_energy
    with testcase.assertRaisesRegex(ValueError, "cutoff_max_energy is required"):
        source_cls(species=MockSpecies())

    # Invalid types
    with testcase.assertRaisesRegex(typeguard.TypeCheckError, r"argument \"filter\" \(int\) is not an instance of str"):
        source_cls(species=MockSpecies(), filter=123, cutoff_max_energy=1.0)

    with testcase.assertRaisesRegex(
        typeguard.TypeCheckError, r"argument \"species\" \(str\) is not an instance of .*Species"
    ):
        source_cls(species="invalid", cutoff_max_energy=1.0)

    with testcase.assertRaisesRegex(
        typeguard.TypeCheckError, r"argument \"cutoff_max_energy\" \(str\) did not match any element in the union"
    ):
        source_cls(species=MockSpecies(), cutoff_max_energy="100")

    # Negative cutoff
    with testcase.assertRaisesRegex(ValueError, r"cutoff_max_energy must be positive, got -10.0"):
        source_cls(species=MockSpecies(), cutoff_max_energy=-10.0).check()

    # OpenPMD serialization for one combination
    src = source_cls(species=MockSpecies(), filter="custom_filter", cutoff_max_energy=100.0)
    openpmd = OpenPMD(period=TimeStepSpec([slice(0, None, 100)]), source=[src])
    context = openpmd.get_rendering_context()
    testcase.assertTrue(context["typeID"]["openpmd"])
    context = context["data"]
    testcase.assertEqual(len(context["source"]), 1)
    testcase.assertEqual(context["source"][0]["type"], source_cls.__name__.lower())
    testcase.assertEqual(context["source"][0]["filter"], "custom_filter")
    testcase.assertEqual(context["source"][0]["cutoff_max_energy"], 100.0)
    testcase.assertEqual(context["source"][0]["species"]["name"], "electron")


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------


class PICMI_TestSpeciesFilterCutoff(unittest.TestCase):
    def test_energy_density_cutoff(self):
        _check_species_filter_cutoff_source(self, EnergyDensityCutoff)


if __name__ == "__main__":
    unittest.main()
