"""
This file is part of PIConGPU.
Copyright 2025 PIConGPU contributors
Authors: Masoud Afshari
License: GPLv3+
"""

from picongpu.picmi.diagnostics import MacroParticleCount, TimeStepSpec
from picongpu.pypicongpu.output.macro_particle_count import MacroParticleCount as PyPIConGPUMacroParticleCount
from picongpu.pypicongpu.output.timestepspec import TimeStepSpec as PyPIConGPUTimeStepSpec
from picongpu.pypicongpu.species.species import Species as PyPIConGPUSpecies
from picongpu.picmi.species import Species as PICMISpecies

import unittest

# Test cases for valid MacroParticleCount inputs
TESTCASES_VALID = [
    (
        {"species": PICMISpecies(name="electron"), "period": 10},
        {"species": {"name": "electron"}, "period": {"specs": [{"start": 0, "stop": 199, "step": 10}]}},
    ),
    (
        {"species": PICMISpecies(name="proton"), "period": 0},
        {"species": {"name": "proton"}, "period": {"specs": []}},
    ),
    (
        {"species": PICMISpecies(name="ion"), "period": TimeStepSpec([slice(None, None, 10)])},
        {"species": {"name": "ion"}, "period": {"specs": [{"start": 0, "stop": 199, "step": 10}]}},
    ),
    (
        {"species": PICMISpecies(name="electron"), "period": TimeStepSpec([5, 10])},
        {
            "species": {"name": "electron"},
            "period": {"specs": [{"start": 5, "stop": 6, "step": 1}, {"start": 10, "stop": 11, "step": 1}]},
        },
    ),
    (
        {"species": PICMISpecies(name="proton"), "period": TimeStepSpec([slice(-10, None, 1)])},
        {"species": {"name": "proton"}, "period": {"specs": [{"start": 190, "stop": 199, "step": 1}]}},
    ),
    (
        {"species": PICMISpecies(name="ion"), "period": TimeStepSpec()},
        {"species": {"name": "ion"}, "period": {"specs": []}},
    ),
]

# Invalid test cases for instantiation
TESTCASES_INVALID = [
    ({"species": "electron", "period": 10}, "species must be a Species"),
    ({"species": 123, "period": 10}, "species must be a Species"),
    ({"species": PICMISpecies(name="electron"), "period": "10"}, "period must be an integer or TimeStepSpec"),
    ({"species": PICMISpecies(name="electron"), "period": -10}, "period must be non-negative"),
    (
        {"species": PICMISpecies(name="electron"), "period": 10, "name": "counter"},
        "got unexpected keyword argument 'name'",
    ),
]

# Invalid test cases for TimeStepSpec with negative steps
TESTCASES_INVALID_TIMESTEPS = [
    (
        {"species": PICMISpecies(name="electron"), "period": TimeStepSpec([slice(None, None, -10)])},
        "Step size must be >= 1",
    ),
]

# Test cases for warning when period is disabled
TESTCASES_WARNING = [
    (
        {"species": PICMISpecies(name="electron"), "period": 0},
        "MacroParticleCount is disabled because period is set to 0 or an empty TimeStepSpec",
    ),
    (
        {"species": PICMISpecies(name="ion"), "period": TimeStepSpec()},
        "MacroParticleCount is disabled because period is set to 0 or an empty TimeStepSpec",
    ),
]


class TestMacroParticleCount(unittest.TestCase):
    def test_macroparticlecount_instantiation(self):
        """Test MacroParticleCount instantiation and validation."""
        for params, _ in TESTCASES_VALID:
            with self.subTest(params=params):
                macro_count = MacroParticleCount(**params)
                for key, value in params.items():
                    if key == "period" and isinstance(value, int):
                        expected = TimeStepSpec([slice(None, None, value)]) if value > 0 else TimeStepSpec()
                        self.assertEqual(
                            macro_count.period.get_as_pypicongpu(0.5, 200).get_rendering_context(),
                            expected.get_as_pypicongpu(0.5, 200).get_rendering_context(),
                        )
                    else:
                        self.assertEqual(getattr(macro_count, key), value)
                if not params["period"] or (
                    isinstance(params["period"], TimeStepSpec)
                    and not params["period"].get_as_pypicongpu(0.5, 200).get_rendering_context().get("specs", [])
                ):
                    with self.assertWarnsRegex(UserWarning, "MacroParticleCount is disabled"):
                        macro_count.check()
                else:
                    macro_count.check()  # Should not raise or warn

        for params, expected_error in TESTCASES_INVALID:
            with self.subTest(params=params, expected_error=expected_error):
                with self.assertRaisesRegex((ValueError, TypeError), expected_error):
                    MacroParticleCount(**params).check()

    def test_macroparticlecount_serialization(self):
        """Test MacroParticleCount serialization to PyPIConGPUMacroParticleCount."""
        species_map = {
            PICMISpecies(name="electron"): PyPIConGPUSpecies(name="electron"),
            PICMISpecies(name="proton"): PyPIConGPUSpecies(name="proton"),
            PICMISpecies(name="ion"): PyPIConGPUSpecies(name="ion"),
        }
        for params, expected_serialized in TESTCASES_VALID:
            with self.subTest(params=params, expected_serialized=expected_serialized):
                macro_count = MacroParticleCount(**params)
                pypicongpu_macro_count = macro_count.get_as_pypicongpu(species_map, 0.5, 200)
                self.assertIsInstance(pypicongpu_macro_count, PyPIConGPUMacroParticleCount)
                self.assertIsInstance(pypicongpu_macro_count.species, PyPIConGPUSpecies)
                self.assertEqual(pypicongpu_macro_count.species.name, params["species"].name)
                self.assertIsInstance(pypicongpu_macro_count.period, PyPIConGPUTimeStepSpec)
                serialized = pypicongpu_macro_count.get_rendering_context()
                self.assertEqual(serialized, expected_serialized)

    def test_macroparticlecount_warning(self):
        """Test warning for disabled MacroParticleCount."""
        for params, expected_warning in TESTCASES_WARNING:
            with self.subTest(params=params, expected_warning=expected_warning):
                macro_count = MacroParticleCount(**params)
                with self.assertWarnsRegex(UserWarning, "MacroParticleCount is disabled"):
                    macro_count.check()

    def test_macroparticlecount_invalid_species(self):
        """Test invalid species in get_as_pypicongpu."""
        macro_count = MacroParticleCount(species=PICMISpecies(name="unknown"), period=10)
        with self.assertRaisesRegex(ValueError, "Species unknown is not known to Simulation"):
            macro_count.get_as_pypicongpu({}, 0.5, 200)

    def test_macroparticlecount_invalid_timestepspec(self):
        """Test invalid TimeStepSpec with negative steps."""
        for params, expected_error in TESTCASES_INVALID_TIMESTEPS:
            with self.subTest(params=params, expected_error=expected_error):
                macro_count = MacroParticleCount(**params)
                species_map = {params["species"]: PyPIConGPUSpecies(name=params["species"].name)}
                with self.assertRaisesRegex(ValueError, "Step size must be >= 1"):
                    macro_count.get_as_pypicongpu(species_map, 0.5, 200)

    def test_macroparticlecount_invalid_simulation_parameters(self):
        """Test invalid simulation parameters in get_as_pypicongpu."""
        macro_count = MacroParticleCount(species=PICMISpecies(name="electron"), period=10)
        species_map = {PICMISpecies(name="electron"): PyPIConGPUSpecies(name="electron")}
        with self.assertRaisesRegex(ValueError, "Time step size must be strictly positive"):
            macro_count.get_as_pypicongpu(species_map, -0.5, 200)
        with self.assertRaisesRegex(ValueError, "Time step size must be strictly positive"):
            macro_count.get_as_pypicongpu(species_map, 0, 200)

    def test_macroparticlecount_plugin_name(self):
        """Test that the plugin name is correctly set."""
        macro_count = MacroParticleCount(species=PICMISpecies(name="electron"), period=10)
        pypicongpu_macro_count = macro_count.get_as_pypicongpu(
            {PICMISpecies(name="electron"): PyPIConGPUSpecies(name="electron")}, 0.5, 200
        )
        self.assertEqual(pypicongpu_macro_count._name, "macroparticlecount")


if __name__ == "__main__":
    unittest.main()
