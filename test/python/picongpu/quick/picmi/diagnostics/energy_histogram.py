"""
This file is part of PIConGPU.
Copyright 2025 PIConGPU contributors
Authors: Masoud Afshari
License: GPLv3+
"""

from picongpu.picmi.diagnostics import EnergyHistogram, TimeStepSpec
from picongpu.pypicongpu.output.energy_histogram import EnergyHistogram as PyPIConGPUEnergyHistogram
from picongpu.pypicongpu.output.timestepspec import TimeStepSpec as PyPIConGPUTimeStepSpec
from picongpu.pypicongpu.species.species import Species as PyPIConGPUSpecies
from picongpu.picmi.species import Species as PICMISpecies

import unittest

# Test cases for valid EnergyHistogram inputs
TESTCASES_VALID = [
    (
        {
            "species": PICMISpecies(name="electron"),
            "period": 10,
            "bin_count": 100,
            "min_energy": 0.0,
            "max_energy": 1000.0,
        },
        {
            "species": {"name": "electron"},
            "period": {"specs": [{"start": 0, "stop": -1, "step": 10}]},
            "bin_count": 100,
            "min_energy": 0.0,
            "max_energy": 1000.0,
        },
    ),
    (
        {
            "species": PICMISpecies(name="proton"),
            "period": 0,
            "bin_count": 50,
            "min_energy": 10.0,
            "max_energy": 5000.0,
        },
        {
            "species": {"name": "proton"},
            "period": {"specs": []},
            "bin_count": 50,
            "min_energy": 10.0,
            "max_energy": 5000.0,
        },
    ),
    (
        {
            "species": PICMISpecies(name="ion"),
            "period": TimeStepSpec[::10],
            "bin_count": 200,
            "min_energy": 0.0,
            "max_energy": 1e6,
        },
        {
            "species": {"name": "ion"},
            "period": {"specs": [{"start": 0, "stop": -1, "step": 10}]},
            "bin_count": 200,
            "min_energy": 0.0,
            "max_energy": 1e6,
        },
    ),
    (
        {
            "species": PICMISpecies(name="electron"),
            "period": TimeStepSpec[5, 10],
            "bin_count": 1000,
            "min_energy": 0.0,
            "max_energy": 10000.0,
        },
        {
            "species": {"name": "electron"},
            "period": {"specs": [{"start": 5, "stop": 5, "step": 1}, {"start": 10, "stop": 10, "step": 1}]},
            "bin_count": 1000,
            "min_energy": 0.0,
            "max_energy": 10000.0,
        },
    ),
    (
        {
            "species": PICMISpecies(name="proton"),
            "period": TimeStepSpec[-10:],
            "bin_count": 150,
            "min_energy": 0.0,
            "max_energy": 2000.0,
        },
        {
            "species": {"name": "proton"},
            "period": {"specs": [{"start": 90, "stop": 99, "step": 1}]},
            "bin_count": 150,
            "min_energy": 0.0,
            "max_energy": 2000.0,
        },
    ),
    (
        {
            "species": PICMISpecies(name="ion"),
            "period": TimeStepSpec(),
            "bin_count": 300,
            "min_energy": 0.0,
            "max_energy": 5000.0,
        },
        {
            "species": {"name": "ion"},
            "period": {"specs": []},
            "bin_count": 300,
            "min_energy": 0.0,
            "max_energy": 5000.0,
        },
    ),
]

# Invalid test cases for instantiation
TESTCASES_INVALID = [
    (
        {
            "species": PICMISpecies(name="electron"),
            "period": 10,
            "bin_count": 100,
            "min_energy": 1000.0,
            "max_energy": 0.0,
        },
        "min_energy must be less than max_energy",
    ),
    (
        {
            "species": PICMISpecies(name="electron"),
            "period": 10,
            "bin_count": 0,
            "min_energy": 0.0,
            "max_energy": 1000.0,
        },
        "bin_count must be > 0",
    ),
    (
        {"species": "electron", "period": 10, "bin_count": 100, "min_energy": 0.0, "max_energy": 1000.0},
        "species must be a Species",
    ),
    (
        {"species": 123, "period": 10, "bin_count": 100, "min_energy": 0.0, "max_energy": 1000.0},
        "species must be a Species",
    ),
    (
        {
            "species": PICMISpecies(name="electron"),
            "period": "10",
            "bin_count": 100,
            "min_energy": 0.0,
            "max_energy": 1000.0,
        },
        "period must be an integer or TimeStepSpec",
    ),
    (
        {
            "species": PICMISpecies(name="electron"),
            "period": -10,
            "bin_count": 100,
            "min_energy": 0.0,
            "max_energy": 1000.0,
        },
        "period must be non-negative",
    ),
    (
        {
            "species": PICMISpecies(name="electron"),
            "period": 10,
            "bin_count": 100,
            "min_energy": 0.0,
            "max_energy": 1000.0,
            "name": "histogram",
        },
        "got unexpected keyword argument 'name'",
    ),
]

# Invalid test cases for TimeStepSpec with negative steps
TESTCASES_INVALID_TIMESTEPS = [
    (
        {
            "species": PICMISpecies(name="electron"),
            "period": TimeStepSpec[::-10],
            "bin_count": 100,
            "min_energy": 0.0,
            "max_energy": 1000.0,
        },
        "Step size must be >= 1",
    ),
]

# Test cases for warning when period is disabled
TESTCASES_WARNING = [
    (
        {
            "species": PICMISpecies(name="electron"),
            "period": 0,
            "bin_count": 100,
            "min_energy": 0.0,
            "max_energy": 1000.0,
        },
        "EnergyHistogram is disabled because period is set to 0 or an empty TimeStepSpec",
    ),
    (
        {
            "species": PICMISpecies(name="ion"),
            "period": TimeStepSpec(),
            "bin_count": 100,
            "min_energy": 0.0,
            "max_energy": 1000.0,
        },
        "EnergyHistogram is disabled because period is set to 0 or an empty TimeStepSpec",
    ),
]


class TestEnergyHistogram(unittest.TestCase):
    def test_energyhistogram_instantiation(self):
        """Test EnergyHistogram instantiation and validation."""
        for params, _ in TESTCASES_VALID:
            with self.subTest(params=params):
                histogram = EnergyHistogram(**params)
                for key, value in params.items():
                    if key == "period" and isinstance(value, int):
                        expected = TimeStepSpec[::value] if value > 0 else TimeStepSpec()
                        self.assertEqual(
                            histogram.period.get_as_pypicongpu(0.5, 100).get_rendering_context(),
                            expected.get_as_pypicongpu(0.5, 100).get_rendering_context(),
                        )
                    else:
                        self.assertEqual(getattr(histogram, key), value)
                if not params["period"] or (
                    isinstance(params["period"], TimeStepSpec)
                    and not params["period"].get_as_pypicongpu(0.5, 100).get_rendering_context().get("specs", [])
                ):
                    with self.assertWarnsRegex(UserWarning, "EnergyHistogram is disabled"):
                        histogram.check()
                else:
                    histogram.check()  # Should not raise or warn

        for params, expected_error in TESTCASES_INVALID:
            with self.subTest(params=params, expected_error=expected_error):
                with self.assertRaisesRegex((ValueError, TypeError), expected_error):
                    EnergyHistogram(**params).check()

    def test_energyhistogram_serialization(self):
        """Test EnergyHistogram serialization to PyPIConGPUEnergyHistogram."""
        species_map = {
            PICMISpecies(name="electron"): PyPIConGPUSpecies(name="electron"),
            PICMISpecies(name="proton"): PyPIConGPUSpecies(name="proton"),
            PICMISpecies(name="ion"): PyPIConGPUSpecies(name="ion"),
        }
        for params, expected_serialized in TESTCASES_VALID:
            with self.subTest(params=params, expected_serialized=expected_serialized):
                histogram = EnergyHistogram(**params)
                pypicongpu_histogram = histogram.get_as_pypicongpu(species_map, 0.5, 100)
                self.assertIsInstance(pypicongpu_histogram, PyPIConGPUEnergyHistogram)
                self.assertIsInstance(pypicongpu_histogram.species, PyPIConGPUSpecies)
                self.assertEqual(pypicongpu_histogram.species.name, params["species"].name)
                self.assertIsInstance(pypicongpu_histogram.period, PyPIConGPUTimeStepSpec)
                serialized = pypicongpu_histogram.get_rendering_context()
                self.assertEqual(serialized, expected_serialized)

    def test_energyhistogram_warning(self):
        """Test warning for disabled EnergyHistogram."""
        for params, expected_warning in TESTCASES_WARNING:
            with self.subTest(params=params, expected_warning=expected_warning):
                histogram = EnergyHistogram(**params)
                with self.assertWarnsRegex(UserWarning, expected_warning):
                    histogram.check()

    def test_energyhistogram_invalid_species(self):
        """Test invalid species in get_as_pypicongpu."""
        histogram = EnergyHistogram(
            species=PICMISpecies(name="unknown"), period=10, bin_count=100, min_energy=0.0, max_energy=1000.0
        )
        with self.assertRaisesRegex(ValueError, "Species unknown is not known to Simulation"):
            histogram.get_as_pypicongpu({}, 0.5, 100)

    def test_energyhistogram_invalid_timestepspec(self):
        """Test invalid TimeStepSpec with negative steps."""
        for params, expected_error in TESTCASES_INVALID_TIMESTEPS:
            with self.subTest(params=params, expected_error=expected_error):
                histogram = EnergyHistogram(**params)
                species_map = {params["species"]: PyPIConGPUSpecies(name=params["species"].name)}
                with self.assertRaisesRegex(ValueError, expected_error):
                    histogram.get_as_pypicongpu(species_map, 0.5, 100)

    def test_energyhistogram_invalid_simulation_parameters(self):
        """Test invalid simulation parameters in get_as_pypicongpu."""
        histogram = EnergyHistogram(
            species=PICMISpecies(name="electron"), period=10, bin_count=100, min_energy=0.0, max_energy=1000.0
        )
        species_map = {PICMISpecies(name="electron"): PyPIConGPUSpecies(name="electron")}
        with self.assertRaisesRegex(ValueError, "Time step size must be strictly positive"):
            histogram.get_as_pypicongpu(species_map, -0.5, 100)
        with self.assertRaisesRegex(ValueError, "Time step size must be strictly positive"):
            histogram.get_as_pypicongpu(species_map, 0, 100)

    def test_energyhistogram_plugin_name(self):
        """Test that the plugin name is correctly set."""
        histogram = EnergyHistogram(
            species=PICMISpecies(name="electron"), period=10, bin_count=100, min_energy=0.0, max_energy=1000.0
        )
        pypicongpu_histogram = histogram.get_as_pypicongpu(
            {PICMISpecies(name="electron"): PyPIConGPUSpecies(name="electron")}, 0.5, 100
        )
        self.assertEqual(pypicongpu_histogram._name, "energyhistogram")


if __name__ == "__main__":
    unittest.main()
