"""
This file is part of PIConGPU.
Copyright 2025 PIConGPU contributors
Authors: Masoud Afshari
License: GPLv3+
"""

from picongpu.pypicongpu.output import EnergyHistogram
from picongpu.pypicongpu.species import Species
from picongpu.pypicongpu.output.timestepspec import TimeStepSpec
from picongpu.pypicongpu.species.attribute import Position, Momentum
import unittest
import typeguard


def create_species():
    """Helper function to create a valid Species object."""
    species = Species()
    species.name = "electron"
    species.attributes = [Position(), Momentum()]
    species.constants = []
    return species


class TestEnergyHistogram(unittest.TestCase):
    def setUp(self):
        self.species = create_species()
        self.period = TimeStepSpec([slice(0, None, 100)])

    def test_instantiation_and_types(self):
        """Test instantiation, type safety, and valid serialization."""
        # Valid configuration
        eh = EnergyHistogram(
            species=self.species,
            period=self.period,
            bin_count=1024,
            min_energy=0.0,
            max_energy=1000.0,
        )
        eh.check()
        context = eh.get_rendering_context()
        self.assertTrue(context["typeID"]["energyhistogram"])
        self.assertEqual(context["data"]["bin_count"], 1024)
        self.assertEqual(context["data"]["min_energy"], 0.0)
        self.assertEqual(context["data"]["max_energy"], 1000.0)
        self.assertEqual(context["data"]["species"]["name"], "electron")
        self.assertEqual(context["data"]["period"]["specs"][0]["step"], 100)

        # Type safety
        invalid_types = {
            "species": ["string", 1],
            "period": ["string", 1],
            "bin_count": ["string", 1.0],
            "min_energy": ["string", []],
            "max_energy": ["string", []],
        }
        for attr, invalid_values in invalid_types.items():
            for value in invalid_values:
                with self.subTest(attr=attr, value=value):
                    kwargs = {
                        "species": self.species,
                        "period": self.period,
                        "bin_count": 1024,
                        "min_energy": 0.0,
                        "max_energy": 1000.0,
                    }
                    kwargs[attr] = value
                    with self.assertRaises(typeguard.TypeCheckError):
                        EnergyHistogram(**kwargs)

    def test_rendering_and_validation(self):
        """Test serialization output, validation errors, and disabled state."""
        # Valid serialization
        eh = EnergyHistogram(
            species=self.species,
            period=self.period,
            bin_count=1024,
            min_energy=0.0,
            max_energy=1000.0,
        )
        context = eh.get_rendering_context()
        self.assertTrue(context["typeID"]["energyhistogram"])
        self.assertEqual(context["data"]["bin_count"], 1024)
        self.assertEqual(context["data"]["min_energy"], 0.0)
        self.assertEqual(context["data"]["max_energy"], 1000.0)

        # Validation errors
        eh = EnergyHistogram(
            species=self.species,
            period=self.period,
            bin_count=0,
            min_energy=0.0,
            max_energy=1000.0,
        )
        with self.assertRaisesRegex(ValueError, "bin_count must be positive"):
            eh.get_rendering_context()

        eh = EnergyHistogram(
            species=self.species,
            period=self.period,
            bin_count=1024,
            min_energy=1000.0,
            max_energy=0.0,
        )
        with self.assertRaisesRegex(ValueError, "min_energy must be less than max_energy"):
            eh.get_rendering_context()


if __name__ == "__main__":
    unittest.main()
