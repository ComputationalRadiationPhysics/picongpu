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

    def test_instantiation_valid(self):
        """Test instantiation and validation for valid inputs."""
        TESTCASES_VALID = [
            (
                {
                    "species": self.species,
                    "period": TimeStepSpec([slice(0, None, 17)]),
                },
                None,
            ),
            (
                {
                    "species": self.species,
                    "period": TimeStepSpec([]),
                },
                "MacroParticleCount is disabled",
            ),
        ]

        for params, warning_msg in TESTCASES_VALID:
            with self.subTest(params=params):
                mpc = MacroParticleCount()
                for key, value in params.items():
                    setattr(mpc, key, value)
                for key, value in params.items():
                    self.assertEqual(getattr(mpc, key), value)
                mpc.check()  # Ensure attributes are set correctly
                if warning_msg:
                    with self.assertWarnsRegex(UserWarning, warning_msg):
                        mpc._get_serialized()
                else:
                    mpc._get_serialized()

    def test_types(self):
        """Type safety is ensured."""
        mpc = MacroParticleCount()

        invalid_species = ["string", 1, 1.0, None, {}]
        for invalid in invalid_species:
            with self.assertRaises(typeguard.TypeCheckError):
                mpc.species = invalid

        invalid_periods = [13.2, [], "2", None, {}]
        for invalid in invalid_periods:
            with self.assertRaises(typeguard.TypeCheckError):
                mpc.period = invalid

        # Valid case
        mpc.species = self.species
        mpc.period = TimeStepSpec([slice(0, None, 17)])
        mpc.check()
        mpc._get_serialized()

    def test_rendering(self):
        """Data transformed to template-consumable version."""
        mpc = MacroParticleCount()
        mpc.species = self.species
        mpc.period = TimeStepSpec([slice(0, None, 42)])

        context = mpc.get_rendering_context()
        self.assertTrue(context["typeID"]["macroparticlecount"])
        context = context["data"]
        self.assertEqual(42, context["period"]["specs"][0]["step"])
        self.assertEqual(0, context["period"]["specs"][0]["start"])
        self.assertEqual("electron", context["species"]["name"])

        # Empty period
        mpc.period = TimeStepSpec([])
        with self.assertWarnsRegex(UserWarning, "MacroParticleCount is disabled"):
            mpc.get_rendering_context()

        # Invalid attributes
        mpc = MacroParticleCount()
        with self.assertRaises(ValueError, msg="species must be set"):
            mpc.get_rendering_context()

    def test_validation(self):
        """Test validation for unset attributes."""
        mpc = MacroParticleCount()
        with self.assertRaises(ValueError, msg="species must be set"):
            mpc.check()

        mpc.species = self.species
        with self.assertRaises(ValueError, msg="period must be set"):
            mpc.check()

        mpc.period = TimeStepSpec([slice(0, None, 1)])
        mpc.check()  # Should pass

    def test_period_warning(self):
        """Test warning for disabled MacroParticleCount output."""
        mpc = MacroParticleCount()
        mpc.species = self.species
        mpc.period = TimeStepSpec([])
        with self.assertWarnsRegex(UserWarning, "MacroParticleCount is disabled"):
            mpc._get_serialized()


if __name__ == "__main__":
    unittest.main()
