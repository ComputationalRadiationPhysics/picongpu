"""
This file is part of PIConGPU.
Copyright 2025 PIConGPU contributors
Authors: Masoud Afshari
License: GPLv3+
"""

from picongpu.picmi.diagnostics import MacroParticleCount, TimeStepSpec
from picongpu.pypicongpu.output.macro_particle_count import MacroParticleCount as PyPIConGPUMacroParticleCount
from picongpu.picmi.species import Species as PICMISpecies
from picongpu.pypicongpu.species import Species as PyPIConGPUSpecies
from picongpu.pypicongpu.species.attribute import Position, Momentum
import unittest
import typeguard


class PICMI_TestMacroParticleCount(unittest.TestCase):
    def setUp(self):
        self.species = PICMISpecies(name="electron")
        self.pypicongpu_species = PyPIConGPUSpecies()
        self.pypicongpu_species.name = "electron"
        self.pypicongpu_species.attributes = [Position(), Momentum()]
        self.pypicongpu_species.constants = []
        self.species_map = {self.species: self.pypicongpu_species}
        self.time_step_size = 0.5
        self.num_steps = 200

    def test_macro_particle_count(self):
        """Test MacroParticleCount instantiation, validation, and serialization."""
        TESTCASES_VALID = [
            (
                {"species": self.species, "period": 10},
                {"period_specs": [{"start": 0, "stop": 199, "step": 10}], "species_name": "electron"},
            ),
            (
                {"species": self.species, "period": TimeStepSpec([slice(0, None, 17)])},
                {"period_specs": [{"start": 0, "stop": 199, "step": 17}], "species_name": "electron"},
            ),
        ]
        for params, expected in TESTCASES_VALID:
            with self.subTest(params=params):
                mpc = MacroParticleCount(**params)
                self.assertEqual(mpc.species, params["species"])
                if isinstance(params["period"], int):
                    expected_period = TimeStepSpec(
                        [slice(None, None, params["period"])] if params["period"] > 0 else []
                    )("steps")
                    self.assertEqual(mpc.period.specs, expected_period.specs)
                else:
                    self.assertEqual(mpc.period.specs, params["period"].specs)
                mpc.check()
                pypicongpu_mpc = mpc.get_as_pypicongpu(self.species_map, self.time_step_size, self.num_steps)
                self.assertIsInstance(pypicongpu_mpc, PyPIConGPUMacroParticleCount)
                self.assertEqual(pypicongpu_mpc.species, self.pypicongpu_species)
                context = pypicongpu_mpc.get_rendering_context()
                self.assertTrue(context["typeID"]["macroparticlecount"])
                self.assertEqual(context["data"]["period"]["specs"], expected["period_specs"])
                self.assertEqual(context["data"]["species"]["name"], expected["species_name"])

        # Test default period
        mpc = MacroParticleCount(species=self.species)
        expected_period = TimeStepSpec([slice(0, None, 1)])("steps")
        self.assertEqual(mpc.period.specs, expected_period.specs)
        context = mpc.get_as_pypicongpu(self.species_map, self.time_step_size, self.num_steps).get_rendering_context()
        self.assertEqual(context["data"]["period"]["specs"], [{"start": 0, "stop": 199, "step": 1}])

        # Test invalid species mapping
        mpc = MacroParticleCount(species=self.species, period=10)
        with self.assertRaisesRegex(ValueError, f"Species {self.species.name} is not known to Simulation"):
            mpc.get_as_pypicongpu({}, self.time_step_size, self.num_steps)

    def test_macro_particle_count_invalid(self):
        """Test invalid MacroParticleCount inputs and warnings."""
        TESTCASES_INVALID = [
            (
                {"species": "invalid", "period": 10},
                'argument "species" .* is not an instance of picongpu\.picmi\.species\.Species',
            ),
            (
                {"species": self.species, "period": "invalid"},
                'argument "period" .* did not match any element in the union',
            ),
            ({"species": self.species, "period": -10}, "period must be non-negative"),
            ({"species": self.species, "period": TimeStepSpec([slice(None, None, -10)])}, "Step size must be >= 1"),
            ({"species": PICMISpecies(), "period": 10}, "species must have a non-empty name", True),
            (
                {"species": self.species, "period": 0},
                "MacroParticleCount is disabled because period is set to 0 or an empty TimeStepSpec",
                True,
            ),
        ]
        for params, expected_error, *skip in TESTCASES_INVALID:
            with self.subTest(params=params, expected_error=expected_error):
                if skip and skip[0]:
                    mpc = MacroParticleCount(**params)
                    if "MacroParticleCount is disabled" in expected_error:
                        with self.assertWarnsRegex(UserWarning, expected_error):
                            mpc.get_as_pypicongpu(self.species_map, self.time_step_size, self.num_steps)
                    else:
                        mpc.check()  # No error for empty name
                else:
                    with self.assertRaisesRegex((ValueError, TypeError, typeguard.TypeCheckError), expected_error):
                        mpc = MacroParticleCount(**params)
                        mpc.check()


if __name__ == "__main__":
    unittest.main()
