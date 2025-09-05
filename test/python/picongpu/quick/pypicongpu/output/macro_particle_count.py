"""
This file is part of PIConGPU.
Copyright 2021-2025 PIConGPU contributors
Authors: Masoud Afshari
License: GPLv3+
"""

from picongpu.pypicongpu.output.timestepspec import TimeStepSpec
from picongpu.pypicongpu.output import MacroParticleCount
from picongpu.pypicongpu.species import Species
from picongpu.pypicongpu.species.attribute import Position, Momentum
import unittest
import typeguard


def create_species():
    species = Species()
    species.name = "electron"
    species.attributes = [Position(), Momentum()]
    species.constants = []
    return species


class TestMacroParticleCount(unittest.TestCase):
    def setUp(self):
        self.species = create_species()

    def test_instantiation_and_types(self):
        """Test instantiation, type safety, and valid serialization."""
        # Valid case
        mpc = MacroParticleCount()
        mpc.species = self.species
        mpc.period = TimeStepSpec([slice(0, None, 17)])
        mpc.check()
        context = mpc.get_rendering_context()  # Use public API
        self.assertTrue(context["typeID"]["macroparticlecount"])
        self.assertEqual(context["data"]["species"]["name"], "electron")
        self.assertEqual(context["data"]["period"]["specs"][0]["step"], 17)

        # Type safety for species
        invalid_species = ["string", 1, 1.0, None, {}]
        for invalid in invalid_species:
            with self.subTest(invalid_species=invalid):
                mpc = MacroParticleCount()
                with self.assertRaises(typeguard.TypeCheckError):
                    mpc.species = invalid  # Expect error during assignment

        # Type safety for period
        invalid_periods = [13.2, [], "2", None, {}]
        for invalid in invalid_periods:
            with self.subTest(invalid_period=invalid):
                mpc = MacroParticleCount()
                mpc.species = self.species  # Set valid species first
                with self.assertRaises(typeguard.TypeCheckError):
                    mpc.period = invalid  # Expect error during assignment

    def test_rendering_and_validation(self):
        """Test serialization output, disabled state, and validation errors."""
        # Valid serialization
        mpc = MacroParticleCount()
        mpc.species = self.species
        mpc.period = TimeStepSpec([slice(0, None, 42)])
        context = mpc.get_rendering_context()  # Use public API
        self.assertTrue(context["typeID"]["macroparticlecount"])
        context = context["data"]
        self.assertEqual(42, context["period"]["specs"][0]["step"])
        self.assertEqual(0, context["period"]["specs"][0]["start"])
        self.assertEqual("electron", context["species"]["name"])

        # Empty period warning
        mpc.period = TimeStepSpec([])
        with self.assertWarnsRegex(UserWarning, "MacroParticleCount is disabled"):
            mpc.get_rendering_context()  # Use public API

        # Validation errors
        mpc = MacroParticleCount()
        with self.assertRaisesRegex(ValueError, "species must be set"):
            mpc.get_rendering_context()  # Calls check() internally

        mpc.species = self.species
        with self.assertRaisesRegex(ValueError, "period must be set"):
            mpc.get_rendering_context()  # Calls check() internally


if __name__ == "__main__":
    unittest.main()
