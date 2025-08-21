"""
This file is part of PIConGPU.
Copyright 2021-2025 PIConGPU contributors
Authors: Masoud Afshari
License: GPLv3+
"""

from picongpu.picmi.diagnostics import MacroParticleCount, TimeStepSpec
from picongpu.picmi.species import Species as PICMISpecies
from picongpu.pypicongpu.species import Species as PyPIConGPUSpecies
from picongpu.pypicongpu.species.attribute import Position, Momentum
import unittest
import typeguard


def create_picmi_species():
    species = PICMISpecies()
    species.name = "electron"
    return species


def create_pypicongpu_species():
    species = PyPIConGPUSpecies()
    species.name = "electron"
    species.attributes = [Position(), Momentum()]
    species.constants = []
    return species


class PICMI_TestMacroParticleCount(unittest.TestCase):
    def setUp(self):
        self.species = create_picmi_species()
        self.pypicongpu_species = create_pypicongpu_species()
        self.species_map = {self.species: self.pypicongpu_species}

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
                    "period": 10,
                },
                None,
            ),
            (
                {
                    "species": self.species,
                    "period": 0,
                },
                "MacroParticleCount is disabled",
            ),
            (
                {
                    "species": self.species,
                },
                None,
            ),
        ]

        for params, warning_msg in TESTCASES_VALID:
            with self.subTest(params=params):
                mpc = MacroParticleCount(**params)
                for key, value in params.items():
                    if key == "period" and isinstance(value, int):
                        expected = (
                            TimeStepSpec([slice(None, None, value)])("steps")
                            if value > 0
                            else TimeStepSpec([])("steps")
                        )
                        expected_context = expected.get_as_pypicongpu(0.5, 200).get_rendering_context()
                        self.assertEqual(
                            mpc.period.get_as_pypicongpu(0.5, 200).get_rendering_context(),
                            expected_context,
                        )
                    elif key == "period" and value is None:
                        expected = TimeStepSpec([slice(0, None, 1)])("steps")
                        expected_context = expected.get_as_pypicongpu(0.5, 200).get_rendering_context()
                        self.assertEqual(
                            mpc.period.get_as_pypicongpu(0.5, 200).get_rendering_context(),
                            expected_context,
                        )
                    else:
                        self.assertEqual(getattr(mpc, key), value)
                if warning_msg:
                    with self.assertWarnsRegex(UserWarning, "MacroParticleCount is disabled"):
                        mpc.get_as_pypicongpu(self.species_map, 0.5, 200)
                else:
                    mpc.get_as_pypicongpu(self.species_map, 0.5, 200)

    def test_types(self):
        """Test type safety for species and period."""
        invalid_species = ["string", 1, 1.0, None, {}]
        for invalid in invalid_species:
            with self.assertRaises(typeguard.TypeCheckError):
                MacroParticleCount(
                    species=invalid,
                    period=TimeStepSpec([slice(0, None, 1)]),
                )

        invalid_periods = [[], "2", {}]
        for invalid in invalid_periods:
            with self.assertRaises(typeguard.TypeCheckError):
                MacroParticleCount(
                    species=self.species,
                    period=invalid,
                )

    def test_rendering(self):
        """Test serialization to PyPIConGPU MacroParticleCount."""
        mpc = MacroParticleCount(
            species=self.species,
            period=TimeStepSpec([slice(0, None, 42)]),
        )
        pypicongpu_mpc = mpc.get_as_pypicongpu(self.species_map, 0.5, 200)
        pypicongpu_mpc.check()  # Ensure attributes are set
        context = pypicongpu_mpc.get_rendering_context()
        self.assertTrue(context["typeID"]["macroparticlecount"])
        context = context["data"]
        self.assertEqual(42, context["period"]["specs"][0]["step"])
        self.assertEqual(0, context["period"]["specs"][0]["start"])
        self.assertEqual(199, context["period"]["specs"][0]["stop"])
        self.assertEqual("electron", context["species"]["name"])

        # Integer period
        mpc = MacroParticleCount(
            species=self.species,
            period=10,
        )
        pypicongpu_mpc = mpc.get_as_pypicongpu(self.species_map, 0.5, 200)
        pypicongpu_mpc.check()
        context = pypicongpu_mpc.get_rendering_context()
        self.assertTrue(context["typeID"]["macroparticlecount"])
        context = context["data"]
        self.assertEqual(10, context["period"]["specs"][0]["step"])
        self.assertEqual(0, context["period"]["specs"][0]["start"])
        self.assertEqual(199, context["period"]["specs"][0]["stop"])

        # Default period
        mpc = MacroParticleCount(
            species=self.species,
        )
        pypicongpu_mpc = mpc.get_as_pypicongpu(self.species_map, 0.5, 200)
        pypicongpu_mpc.check()
        context = pypicongpu_mpc.get_rendering_context()
        self.assertTrue(context["typeID"]["macroparticlecount"])
        context = context["data"]
        self.assertEqual(1, context["period"]["specs"][0]["step"])
        self.assertEqual(0, context["period"]["specs"][0]["start"])
        self.assertEqual(199, context["period"]["specs"][0]["stop"])

        # Test invalid species mapping
        with self.assertRaises(ValueError):
            mpc.get_as_pypicongpu({}, 0.5, 200)

    def test_invalid_period(self):
        """Test invalid period values."""
        for period, expected_error in [
            (-10, "period must be non-negative"),
            (TimeStepSpec([slice(None, None, -10)]), "Step size must be >= 1"),
            (TimeStepSpec([slice(5, 10, -2)]), "Step size must be >= 1"),
        ]:
            with self.subTest(period=period, expected_error=expected_error):
                with self.assertRaisesRegex(ValueError, expected_error):
                    MacroParticleCount(
                        species=self.species,
                        period=period,
                    ).check()

    def test_period_warning(self):
        """Test warning for disabled MacroParticleCount output."""
        mpc = MacroParticleCount(
            species=self.species,
            period=0,
        )
        with self.assertWarnsRegex(UserWarning, "MacroParticleCount is disabled"):
            pypicongpu_mpc = mpc.get_as_pypicongpu(self.species_map, 0.5, 200)
        pypicongpu_mpc.check()
        with self.assertWarnsRegex(UserWarning, "MacroParticleCount is disabled"):
            pypicongpu_mpc._get_serialized()


if __name__ == "__main__":
    unittest.main()
