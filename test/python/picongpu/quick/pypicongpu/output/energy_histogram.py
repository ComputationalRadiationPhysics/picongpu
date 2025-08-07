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
        """Invalid configurations are handled correctly."""
        # Invalid bin_count
        with self.assertRaisesRegex(ValueError, "bin_count must be positive"):
            eh = EnergyHistogram(
                species=create_species(),
                period=TimeStepSpec([slice(0, None, 100)]),
                bin_count=0,
                min_energy=0.0,
                max_energy=1000.0,
            )
            eh._get_serialized()

        # Invalid energy range
        with self.assertRaisesRegex(ValueError, "min_energy must be less than max_energy"):
            eh = EnergyHistogram(
                species=create_species(),
                period=TimeStepSpec([slice(0, None, 100)]),
                bin_count=1024,
                min_energy=1000.0,
                max_energy=0.0,
            )
            eh._get_serialized()

        # Valid configuration
        eh = EnergyHistogram(
            species=create_species(),
            period=TimeStepSpec([slice(0, None, 100)]),
            bin_count=1024,
            min_energy=0.0,
            max_energy=1000.0,
        )
        serialized = eh._get_serialized()
        self.assertEqual(serialized["bin_count"], 1024)
        self.assertEqual(serialized["min_energy"], 0.0)
        self.assertEqual(serialized["max_energy"], 1000.0)

    def test_types(self):
        """Type safety is ensured for all attributes."""
        # Invalid species
        invalid_species = ["string", 1, 1.0, {}]
        for invalid in invalid_species:
            with self.assertRaises(typeguard.TypeCheckError):
                eh = EnergyHistogram(
                    species=invalid,
                    period=TimeStepSpec([slice(0, None, 100)]),
                    bin_count=1024,
                    min_energy=0.0,
                    max_energy=1000.0,
                )

        # Invalid period
        invalid_periods = [13.2, [], "2", {}]
        for invalid in invalid_periods:
            with self.assertRaises(typeguard.TypeCheckError):
                eh = EnergyHistogram(
                    species=create_species(), period=invalid, bin_count=1024, min_energy=0.0, max_energy=1000.0
                )

        # Invalid bin_count
        invalid_bin_counts = ["string", 1.0, {}]
        for invalid in invalid_bin_counts:
            with self.assertRaises(typeguard.TypeCheckError):
                eh = EnergyHistogram(
                    species=create_species(),
                    period=TimeStepSpec([slice(0, None, 100)]),
                    bin_count=invalid,
                    min_energy=0.0,
                    max_energy=1000.0,
                )

        # Invalid min_energy
        invalid_min_energy = ["string", (1,), {}]
        for invalid in invalid_min_energy:
            with self.assertRaises(typeguard.TypeCheckError):
                eh = EnergyHistogram(
                    species=create_species(),
                    period=TimeStepSpec([slice(0, None, 100)]),
                    bin_count=1024,
                    min_energy=invalid,
                    max_energy=1000.0,
                )

        # Invalid max_energy
        invalid_max_energy = ["string", (1,), {}]
        for invalid in invalid_max_energy:
            with self.assertRaises(typeguard.TypeCheckError):
                eh = EnergyHistogram(
                    species=create_species(),
                    period=TimeStepSpec([slice(0, None, 100)]),
                    bin_count=1024,
                    min_energy=0.0,
                    max_energy=invalid,
                )

        # Valid configuration
        eh = EnergyHistogram(
            species=create_species(),
            period=TimeStepSpec([slice(0, None, 100)]),
            bin_count=1024,
            min_energy=0.0,
            max_energy=1000.0,
        )
        eh._get_serialized()  # Should succeed

    def test_rendering(self):
        """Serialized data is correctly formatted for template consumption."""
        eh = EnergyHistogram(
            species=create_species(),
            period=TimeStepSpec([slice(0, None, 100)]),
            bin_count=1024,
            min_energy=0.0,
            max_energy=1000.0,
        )

        context = eh.get_rendering_context()
        self.assertTrue(context["typeID"]["energyhistogram"])
        context = context["data"]
        self.assertEqual(context["period"]["specs"][0]["step"], 100)
        self.assertEqual(context["bin_count"], 1024)
        self.assertEqual(context["min_energy"], 0.0)
        self.assertEqual(context["max_energy"], 1000.0)

    def test_validation(self):
        """Constraints on bin_count and energy range are enforced."""
        # Invalid bin_count
        eh = EnergyHistogram(
            species=create_species(),
            period=TimeStepSpec([slice(0, None, 100)]),
            bin_count=0,
            min_energy=0.0,
            max_energy=1000.0,
        )
        with self.assertRaisesRegex(ValueError, "bin_count must be positive"):
            eh.check()

        # Negative bin_count
        eh = EnergyHistogram(
            species=create_species(),
            period=TimeStepSpec([slice(0, None, 100)]),
            bin_count=-1,
            min_energy=0.0,
            max_energy=1000.0,
        )
        with self.assertRaisesRegex(ValueError, "bin_count must be positive"):
            eh.check()

        # Invalid energy range
        eh = EnergyHistogram(
            species=create_species(),
            period=TimeStepSpec([slice(0, None, 100)]),
            bin_count=1024,
            min_energy=1000.0,
            max_energy=0.0,
        )
        with self.assertRaisesRegex(ValueError, "min_energy must be less than max_energy"):
            eh.check()

        # Equal energy range
        eh = EnergyHistogram(
            species=create_species(),
            period=TimeStepSpec([slice(0, None, 100)]),
            bin_count=1024,
            min_energy=500.0,
            max_energy=500.0,
        )
        with self.assertRaisesRegex(ValueError, "min_energy must be less than max_energy"):
            eh.check()

        # Valid configuration
        eh = EnergyHistogram(
            species=create_species(),
            period=TimeStepSpec([slice(0, None, 100)]),
            bin_count=1024,
            min_energy=0.0,
            max_energy=1000.0,
        )
        eh.check()  # Should succeed
        serialized = eh._get_serialized()
        self.assertEqual(serialized["bin_count"], 1024)
        self.assertEqual(serialized["min_energy"], 0.0)
        self.assertEqual(serialized["max_energy"], 1000.0)


if __name__ == "__main__":
    unittest.main()
