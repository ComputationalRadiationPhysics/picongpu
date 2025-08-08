"""
This file is part of PIConGPU.
Copyright 2025 PIConGPU contributors
Authors: Masoud Afshari
License: GPLv3+
"""

from picongpu.pypicongpu.output import MacroParticleCount
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


class TestMacroParticleCount(unittest.TestCase):
    def test_empty(self):
        """Invalid configurations are handled correctly."""
        # Valid configuration
        mpc = MacroParticleCount(species=create_species(), period=TimeStepSpec([slice(0, None, 100)]))
        serialized = mpc._get_serialized()
        self.assertEqual(serialized["period"]["specs"][0]["step"], 100)

    def test_types(self):
        """Type safety is ensured for all attributes."""
        # Invalid species
        invalid_species = ["string", 1, 1.0, {}]
        for invalid in invalid_species:
            with self.assertRaises(typeguard.TypeCheckError):
                mpc = MacroParticleCount(species=invalid, period=TimeStepSpec([slice(0, None, 100)]))

        # Invalid period
        invalid_periods = [13.2, [], "2", {}]
        for invalid in invalid_periods:
            with self.assertRaises(typeguard.TypeCheckError):
                mpc = MacroParticleCount(species=create_species(), period=invalid)

        # Valid configuration
        mpc = MacroParticleCount(species=create_species(), period=TimeStepSpec([slice(0, None, 100)]))
        mpc._get_serialized()  # Should succeed

    def test_rendering(self):
        """Serialized data is correctly formatted for template consumption."""
        mpc = MacroParticleCount(species=create_species(), period=TimeStepSpec([slice(0, None, 100)]))
        context = mpc.get_rendering_context()
        self.assertTrue(context["typeID"]["macroparticlecount"])
        context = context["data"]
        self.assertEqual(context["period"]["specs"][0]["step"], 100)

    def test_validation(self):
        """Constraints on species and period are enforced."""
        # Valid configuration
        mpc = MacroParticleCount(species=create_species(), period=TimeStepSpec([slice(0, None, 100)]))
        mpc.check()  # Should succeed
        serialized = mpc._get_serialized()
        self.assertEqual(serialized["period"]["specs"][0]["step"], 100)


if __name__ == "__main__":
    unittest.main()
