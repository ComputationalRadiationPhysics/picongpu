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
    def test_empty(self):
        """Empty or incomplete configurations are handled correctly."""
        eh = EnergyHistogram()
        # Unset attributes should raise an exception
        with self.assertRaises(Exception):
            eh._get_serialized()

        # Set valid attributes
        eh.species = create_species()
        eh.period = TimeStepSpec([slice(0, None, 100)])
        eh.bin_count = 1024
        eh.min_energy = 0.0
        eh.max_energy = 1000.0
        # Should succeed
        serialized = eh._get_serialized()
        # confirm the serialized dictionary has the correct values.
        self.assertEqual(serialized["bin_count"], 1024)
        self.assertEqual(serialized["min_energy"], 0.0)
        self.assertEqual(serialized["max_energy"], 1000.0)

    def test_types(self):
        """Type safety is ensured for all attributes."""
        eh = EnergyHistogram()

        # Invalid species
        invalid_species = ["string", 1, 1.0, None, {}]
        for invalid in invalid_species:
            with self.assertRaises(typeguard.TypeCheckError):
                eh.species = invalid

        # Invalid period
        invalid_periods = [13.2, [], "2", None, {}]
        for invalid in invalid_periods:
            with self.assertRaises(typeguard.TypeCheckError):
                eh.period = invalid

        # Invalid bin_count
        invalid_bin_counts = ["string", 1.0, None, {}]
        for invalid in invalid_bin_counts:
            with self.assertRaises(typeguard.TypeCheckError):
                eh.bin_count = invalid

        # Invalid min_energy
        invalid_min_energy = ["string", (1,), None, {}]
        for invalid in invalid_min_energy:
            with self.assertRaises(typeguard.TypeCheckError):
                eh.min_energy = invalid

        # Invalid max_energy
        invalid_max_energy = ["string", (1,), None, {}]
        for invalid in invalid_max_energy:
            with self.assertRaises(typeguard.TypeCheckError):
                eh.max_energy = invalid

        # Valid configuration
        eh.species = create_species()
        eh.period = TimeStepSpec([slice(0, None, 100)])
        eh.bin_count = 1024
        eh.min_energy = 0.0
        eh.max_energy = 1000.0
        eh._get_serialized()  # Should succeed

    def test_rendering(self):
        """Serialized data is correctly formatted for template consumption."""
        eh = EnergyHistogram()
        eh.species = create_species()
        eh.period = TimeStepSpec([slice(0, None, 100)])
        eh.bin_count = 1024
        eh.min_energy = 0.0
        eh.max_energy = 1000.0

        context = eh.get_rendering_context()
        self.assertTrue(context["typeID"]["energyhistogram"])
        context = context["data"]
        self.assertEqual(100, context["period"]["specs"][0]["step"])
        self.assertEqual(1024, context["bin_count"])
        self.assertEqual(0.0, context["min_energy"])
        self.assertEqual(1000.0, context["max_energy"])

        # Unset attributes should fail
        eh = EnergyHistogram()
        with self.assertRaises(Exception):
            eh.get_rendering_context()

    def test_validation(self):
        """Constraints on bin_count and energy range are enforced."""
        eh = EnergyHistogram()
        eh.species = create_species()
        eh.period = TimeStepSpec([slice(0, None, 100)])

        # Test invalid bin_count (bin_count <= 0)
        eh.bin_count = 0
        eh.min_energy = 0.0
        eh.max_energy = 1000.0
        with self.assertRaises(ValueError, match="bin_count must be positive"):
            eh.check()
        with self.assertRaises(ValueError, match="bin_count must be positive"):
            eh._get_serialized()  # Should fail due to calls check() internally

        # Test negative bin_count
        eh.bin_count = -1
        with self.assertRaises(ValueError, match="bin_count must be positive"):
            eh.check()

        # Test invalid energy range (min_energy >= max_energy)
        eh.bin_count = 1024
        eh.min_energy = 1000.0
        eh.max_energy = 0.0
        with self.assertRaises(ValueError, match="min_energy must be less than max_energy"):
            eh.check()
        with self.assertRaises(ValueError, match="min_energy must be less than max_energy"):
            eh._get_serialized()  # Should fail due to calls check() internally

        # Test equal energy range
        eh.min_energy = 500.0
        eh.max_energy = 500.0
        with self.assertRaises(ValueError, match="min_energy must be less than max_energy"):
            eh.check()

        # Test unset attributes
        eh = EnergyHistogram()
        eh.species = create_species()
        eh.period = TimeStepSpec([slice(0, None, 100)])
        eh.bin_count = 1024
        # min_energy and max_energy unset
        with self.assertRaises(ValueError, match="min_energy must be less than max_energy"):
            eh.check()

        # Valid configuration
        eh.min_energy = 0.0
        eh.max_energy = 1000.0
        eh.check()  # Should succeed
        serialized = eh._get_serialized()
        self.assertEqual(serialized["bin_count"], 1024)
        self.assertEqual(serialized["min_energy"], 0.0)
        self.assertEqual(serialized["max_energy"], 1000.0)
