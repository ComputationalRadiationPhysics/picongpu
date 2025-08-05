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
        """Empty or incomplete configurations are handled correctly."""
        mpc = MacroParticleCount()
        # Unset attributes should raise an exception
        with self.assertRaises(ValueError, match="species must be set"):
            mpc._get_serialized()

        # Set species but not period
        mpc.species = create_species()
        with self.assertRaises(ValueError, match="period must be set"):
            mpc._get_serialized()

        # Set valid attributes
        mpc.species = create_species()
        mpc.period = TimeStepSpec([slice(0, None, 100)])
        # Should succeed
        serialized = mpc._get_serialized()
        self.assertEqual(serialized["species"]["name"], "electron")
        self.assertEqual(serialized["period"]["specs"][0]["step"], 100)

    def test_types(self):
        """Type safety is ensured for all attributes."""
        mpc = MacroParticleCount()

        # Invalid species
        invalid_species = ["string", 1, 1.0, None, {}]
        for invalid in invalid_species:
            with self.assertRaises(typeguard.TypeCheckError):
                mpc.species = invalid

        # Invalid period
        invalid_periods = [13.2, [], "2", None, {}]
        for invalid in invalid_periods:
            with self.assertRaises(typeguard.TypeCheckError):
                mpc.period = invalid

        # Valid configuration
        mpc.species = create_species()
        mpc.period = TimeStepSpec([slice(0, None, 100)])
        mpc._get_serialized()  # Should succeed

    def test_rendering(self):
        """Serialized data is correctly formatted for template consumption."""
        mpc = MacroParticleCount()
        mpc.species = create_species()
        mpc.period = TimeStepSpec([slice(0, None, 100)])

        context = mpc.get_rendering_context()
        self.assertTrue(context["typeID"]["macroparticlecount"])
        context = context["data"]
        self.assertEqual(context["species"]["name"], "electron")
        self.assertEqual(context["period"]["specs"][0]["step"], 100)

        # Unset attributes should fail
        mpc = MacroParticleCount()
        with self.assertRaises(ValueError, match="species must be set"):
            mpc.get_rendering_context()

    def test_validation(self):
        """Constraints on species and period are enforced."""
        mpc = MacroParticleCount()

        # Test unset species
        mpc.period = TimeStepSpec([slice(0, None, 100)])
        with self.assertRaises(ValueError, match="species must be set"):
            mpc.check()
        with self.assertRaises(ValueError, match="species must be set"):
            mpc._get_serialized()

        # Test unset period
        mpc = MacroParticleCount()
        mpc.species = create_species()
        with self.assertRaises(ValueError, match="period must be set"):
            mpc.check()
        with self.assertRaises(ValueError, match="period must be set"):
            mpc._get_serialized()

        # Valid configuration
        mpc.species = create_species()
        mpc.period = TimeStepSpec([slice(0, None, 100)])
        mpc.check()  # Should succeed
        serialized = mpc._get_serialized()
        self.assertEqual(serialized["species"]["name"], "electron")
        self.assertEqual(serialized["period"]["specs"][0]["step"], 100)


if __name__ == "__main__":
    unittest.main()
