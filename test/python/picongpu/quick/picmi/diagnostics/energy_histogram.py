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
from picongpu.pypicongpu.species.attribute import Position, Momentum, Weighting
from picongpu.picmi.species import Species as PICMISpecies
import unittest
import re

# Pre-create PICMISpecies instances for consistent use
electron_species = PICMISpecies(name="electron")
proton_species = PICMISpecies(name="proton")
ion_species = PICMISpecies(name="ion")

# Test cases for valid EnergyHistogram inputs
TESTCASES_VALID = [
    (
        {
            "species": electron_species,
            "period": 10,
            "bin_count": 100,
            "min_energy": 0.0,
            "max_energy": 1000.0,
        },
        {
            "typeID": {
                "auto": False,
                "phasespace": False,
                "energyhistogram": True,
                "macroparticlecount": False,
                "png": False,
                "checkpoint": False,
                "openpmd": False,
            },
            "data": {
                "species": {
                    "name": "electron",
                    "typename": "species_electron",
                    "attributes": [
                        {"picongpu_name": "position<position_pic>"},
                        {"picongpu_name": "weighting"},
                        {"picongpu_name": "momentum"},
                    ],
                    "constants": {
                        "mass": None,
                        "charge": None,
                        "density_ratio": None,
                        "element_properties": None,
                        "ground_state_ionization": None,
                    },
                },
                "period": {"specs": [{"start": 0, "stop": 199, "step": 10}]},
                "bin_count": 100,
                "min_energy": 0.0,
                "max_energy": 1000.0,
            },
        },
    ),
    (
        {
            "species": proton_species,
            "period": 0,
            "bin_count": 50,
            "min_energy": 10.0,
            "max_energy": 5000.0,
        },
        {
            "typeID": {
                "auto": False,
                "phasespace": False,
                "energyhistogram": True,
                "macroparticlecount": False,
                "png": False,
                "checkpoint": False,
                "openpmd": False,
            },
            "data": {
                "species": {
                    "name": "proton",
                    "typename": "species_proton",
                    "attributes": [
                        {"picongpu_name": "position<position_pic>"},
                        {"picongpu_name": "weighting"},
                        {"picongpu_name": "momentum"},
                    ],
                    "constants": {
                        "mass": None,
                        "charge": None,
                        "density_ratio": None,
                        "element_properties": None,
                        "ground_state_ionization": None,
                    },
                },
                "period": {"specs": []},
                "bin_count": 50,
                "min_energy": 10.0,
                "max_energy": 5000.0,
            },
        },
    ),
    (
        {
            "species": ion_species,
            "period": TimeStepSpec([slice(None, None, 10)]),
            "bin_count": 200,
            "min_energy": 0.0,
            "max_energy": 1e6,
        },
        {
            "typeID": {
                "auto": False,
                "phasespace": False,
                "energyhistogram": True,
                "macroparticlecount": False,
                "png": False,
                "checkpoint": False,
                "openpmd": False,
            },
            "data": {
                "species": {
                    "name": "ion",
                    "typename": "species_ion",
                    "attributes": [
                        {"picongpu_name": "position<position_pic>"},
                        {"picongpu_name": "weighting"},
                        {"picongpu_name": "momentum"},
                    ],
                    "constants": {
                        "mass": None,
                        "charge": None,
                        "density_ratio": None,
                        "element_properties": None,
                        "ground_state_ionization": None,
                    },
                },
                "period": {"specs": [{"start": 0, "stop": 199, "step": 10}]},
                "bin_count": 200,
                "min_energy": 0.0,
                "max_energy": 1e6,
            },
        },
    ),
    (
        {
            "species": electron_species,
            "period": TimeStepSpec([5, 10]),
            "bin_count": 1000,
            "min_energy": 0.0,
            "max_energy": 10000.0,
        },
        {
            "typeID": {
                "auto": False,
                "phasespace": False,
                "energyhistogram": True,
                "macroparticlecount": False,
                "png": False,
                "checkpoint": False,
                "openpmd": False,
            },
            "data": {
                "species": {
                    "name": "electron",
                    "typename": "species_electron",
                    "attributes": [
                        {"picongpu_name": "position<position_pic>"},
                        {"picongpu_name": "weighting"},
                        {"picongpu_name": "momentum"},
                    ],
                    "constants": {
                        "mass": None,
                        "charge": None,
                        "density_ratio": None,
                        "element_properties": None,
                        "ground_state_ionization": None,
                    },
                },
                "period": {"specs": [{"start": 5, "stop": 6, "step": 1}, {"start": 10, "stop": 11, "step": 1}]},
                "bin_count": 1000,
                "min_energy": 0.0,
                "max_energy": 10000.0,
            },
        },
    ),
    (
        {
            "species": proton_species,
            "period": TimeStepSpec([slice(-10, None, 1)]),
            "bin_count": 150,
            "min_energy": 0.0,
            "max_energy": 2000.0,
        },
        {
            "typeID": {
                "auto": False,
                "phasespace": False,
                "energyhistogram": True,
                "macroparticlecount": False,
                "png": False,
                "checkpoint": False,
                "openpmd": False,
            },
            "data": {
                "species": {
                    "name": "proton",
                    "typename": "species_proton",
                    "attributes": [
                        {"picongpu_name": "position<position_pic>"},
                        {"picongpu_name": "weighting"},
                        {"picongpu_name": "momentum"},
                    ],
                    "constants": {
                        "mass": None,
                        "charge": None,
                        "density_ratio": None,
                        "element_properties": None,
                        "ground_state_ionization": None,
                    },
                },
                "period": {"specs": [{"start": 190, "stop": 199, "step": 1}]},
                "bin_count": 150,
                "min_energy": 0.0,
                "max_energy": 2000.0,
            },
        },
    ),
    (
        {
            "species": ion_species,
            "period": TimeStepSpec(),
            "bin_count": 300,
            "min_energy": 0.0,
            "max_energy": 5000.0,
        },
        {
            "typeID": {
                "auto": False,
                "phasespace": False,
                "energyhistogram": True,
                "macroparticlecount": False,
                "png": False,
                "checkpoint": False,
                "openpmd": False,
            },
            "data": {
                "species": {
                    "name": "ion",
                    "typename": "species_ion",
                    "attributes": [
                        {"picongpu_name": "position<position_pic>"},
                        {"picongpu_name": "weighting"},
                        {"picongpu_name": "momentum"},
                    ],
                    "constants": {
                        "mass": None,
                        "charge": None,
                        "density_ratio": None,
                        "element_properties": None,
                        "ground_state_ionization": None,
                    },
                },
                "period": {"specs": []},
                "bin_count": 300,
                "min_energy": 0.0,
                "max_energy": 5000.0,
            },
        },
    ),
]

# Invalid test cases for instantiation
TESTCASES_INVALID = [
    (
        {
            "species": electron_species,
            "period": 10,
            "bin_count": 100,
            "min_energy": 1000.0,
            "max_energy": 0.0,
        },
        "min_energy must be less than max_energy",
    ),
    (
        {
            "species": electron_species,
            "period": 10,
            "bin_count": 0,
            "min_energy": 0.0,
            "max_energy": 1000.0,
        },
        "bin_count must be > 0",
    ),
    (
        {"species": "electron", "period": 10, "bin_count": 100, "min_energy": 0.0, "max_energy": 1000.0},
        r'argument "species" \(str\) is not an instance of picongpu\.picmi\.species\.Species',
    ),
    (
        {"species": 123, "period": 10, "bin_count": 100, "min_energy": 0.0, "max_energy": 1000.0},
        r'argument "species" \(int\) is not an instance of picongpu\.picmi\.species\.Species',
    ),
    (
        {
            "species": electron_species,
            "period": "10",
            "bin_count": 100,
            "min_energy": 0.0,
            "max_energy": 1000.0,
        },
        r'argument "period" \(str\) did not match any element in the union',
    ),
    (
        {
            "species": electron_species,
            "period": -10,
            "bin_count": 100,
            "min_energy": 0.0,
            "max_energy": 1000.0,
        },
        "period must be non-negative",
    ),
    (
        {
            "species": electron_species,
            "period": 10,
            "bin_count": 100,
            "min_energy": 0.0,
            "max_energy": 1000.0,
            "name": "histogram",
        },
        r"Auto\.__init__\(\) got an unexpected keyword argument 'name'",
    ),
]

# Invalid test cases for TimeStepSpec with negative steps
TESTCASES_INVALID_TIMESTEPS = [
    (
        {
            "species": electron_species,
            "period": TimeStepSpec([slice(0, 100, -10)]),
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
            "species": electron_species,
            "period": 0,
            "bin_count": 100,
            "min_energy": 0.0,
            "max_energy": 1000.0,
        },
        "EnergyHistogram is disabled because period is set to 0 or an empty TimeStepSpec",
    ),
    (
        {
            "species": ion_species,
            "period": TimeStepSpec(),
            "bin_count": 100,
            "min_energy": 0.0,
            "max_energy": 1000.0,
        },
        "EnergyHistogram is disabled because period is set to 0 or an empty TimeStepSpec",
    ),
]


class TestEnergyHistogram(unittest.TestCase):
    def test_energyhistogram_instantiation_valid(self):
        """Test EnergyHistogram instantiation and validation for valid inputs."""
        for params, _ in TESTCASES_VALID:
            with self.subTest(params=params):
                histogram = EnergyHistogram(**params)
                for key, value in params.items():
                    if key == "period" and isinstance(value, int):
                        expected = (
                            TimeStepSpec([slice(None, None, value)])("steps") if value > 0 else TimeStepSpec()("steps")
                        )
                        expected_context = expected.get_as_pypicongpu(0.5, 200).get_rendering_context()
                        self.assertEqual(
                            histogram.period.get_as_pypicongpu(0.5, 200).get_rendering_context(),
                            expected_context,
                        )
                    else:
                        self.assertEqual(getattr(histogram, key), value)
                if not params["period"] or (
                    isinstance(params["period"], TimeStepSpec)
                    and not params["period"].get_as_pypicongpu(0.5, 200).get_rendering_context().get("specs", [])
                ):
                    with self.assertWarnsRegex(UserWarning, "EnergyHistogram is disabled"):
                        histogram.check()
                else:
                    histogram.check()  # Should not raise or warn

    def test_energyhistogram_instantiation_invalid(self):
        """Test EnergyHistogram instantiation and validation for invalid inputs."""
        for params, expected_error in TESTCASES_INVALID:
            try:
                histogram = EnergyHistogram(**params)
                histogram.check()
                self.fail(
                    f"Expected error matching '{expected_error}' but no exception was raised for params: {params}"
                )
            except Exception as e:
                self.assertTrue(
                    re.search(expected_error, str(e), re.IGNORECASE),
                    f"Expected error matching '{expected_error}' but got '{str(e)}' for params: {params}",
                )

    def test_energyhistogram_serialization(self):
        """Test EnergyHistogram serialization to PyPIConGPUEnergyHistogram."""
        species_map = {
            electron_species: PyPIConGPUSpecies(),
            proton_species: PyPIConGPUSpecies(),
            ion_species: PyPIConGPUSpecies(),
        }
        for species in species_map:
            species_map[species].name = species.name
            species_map[species].attributes = [Position(), Weighting(), Momentum()]
            species_map[species].constants = []
        for params, expected_serialized in TESTCASES_VALID:
            with self.subTest(params=params, expected_serialized=expected_serialized):
                histogram = EnergyHistogram(**params)
                pypicongpu_histogram = histogram.get_as_pypicongpu(species_map, 0.5, 200)
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
                with self.assertWarnsRegex(UserWarning, "EnergyHistogram is disabled"):
                    histogram.check()

    def test_energyhistogram_invalid_species(self):
        """Test invalid species in get_as_pypicongpu."""
        histogram = EnergyHistogram(
            species=PICMISpecies(name="unknown"), period=10, bin_count=100, min_energy=0.0, max_energy=1000.0
        )
        with self.assertRaisesRegex(ValueError, "Species unknown is not known to Simulation"):
            histogram.get_as_pypicongpu({}, 0.5, 200)

    def test_energyhistogram_invalid_timestepspec(self):
        """Test invalid TimeStepSpec with negative steps."""
        for params, expected_error in TESTCASES_INVALID_TIMESTEPS:
            with self.subTest(params=params, expected_error=expected_error):
                with self.assertRaisesRegex(ValueError, expected_error):
                    histogram = EnergyHistogram(**params)
                    histogram.check()

    def test_energyhistogram_invalid_simulation_parameters(self):
        """Test invalid simulation parameters in get_as_pypicongpu."""
        histogram = EnergyHistogram(
            species=electron_species, period=10, bin_count=100, min_energy=0.0, max_energy=1000.0
        )
        species_map = {electron_species: PyPIConGPUSpecies()}
        species_map[electron_species].name = "electron"
        species_map[electron_species].attributes = [Position(), Weighting(), Momentum()]
        species_map[electron_species].constants = []
        with self.assertRaisesRegex(ValueError, "time_step_size must be positive"):
            histogram.get_as_pypicongpu(species_map, -0.5, 200)
        with self.assertRaisesRegex(ValueError, "time_step_size must be positive"):
            histogram.get_as_pypicongpu(species_map, 0, 200)

    def test_energyhistogram_plugin_name(self):
        """Test that the plugin name is correctly set."""
        histogram = EnergyHistogram(
            species=electron_species, period=10, bin_count=100, min_energy=0.0, max_energy=1000.0
        )
        species_map = {electron_species: PyPIConGPUSpecies()}
        species_map[electron_species].name = "electron"
        species_map[electron_species].attributes = [Position(), Weighting(), Momentum()]
        species_map[electron_species].constants = []
        pypicongpu_histogram = histogram.get_as_pypicongpu(species_map, 0.5, 200)
        self.assertEqual(pypicongpu_histogram._name, "energyhistogram")


if __name__ == "__main__":
    unittest.main()
