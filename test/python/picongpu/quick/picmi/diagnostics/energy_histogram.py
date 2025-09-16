"""
This file is part of PIConGPU.
Copyright 2025 PIConGPU contributors
Authors: Masoud Afshari
License: GPLv3+
"""

from picongpu.picmi.diagnostics import EnergyHistogram, TimeStepSpec
from picongpu.pypicongpu.output.energy_histogram import EnergyHistogram as PyPIConGPUEnergyHistogram
from picongpu.picmi.species import Species as PICMISpecies
from picongpu.pypicongpu.species import Species as PyPIConGPUSpecies
from picongpu.pypicongpu.species.attribute import Position, Momentum
import unittest
import typeguard


class PICMI_TestEnergyHistogram(unittest.TestCase):
    def setUp(self):
        self.picmi_species = PICMISpecies(name="electron")
        self.pypicongpu_species = PyPIConGPUSpecies()
        self.pypicongpu_species.name = "electron"
        self.pypicongpu_species.attributes = [Position(), Momentum()]  # Required attributes
        self.pypicongpu_species.constants = []  # Initialize constants as empty list
        self.species_map = {self.picmi_species: self.pypicongpu_species}
        self.time_step_size = 1e-16
        self.num_steps = 1000

    def test_energy_histogram(self):
        """Test EnergyHistogram instantiation, validation, and serialization."""
        TESTCASES_VALID = [
            (
                {"species": self.picmi_species, "period": 10, "bin_count": 50, "min_energy": 0.0, "max_energy": 500.0},
                {
                    "bin_count": 50,
                    "min_energy": 0.0,
                    "max_energy": 500.0,
                    "period_specs": [{"start": 0, "stop": 999, "step": 10}],
                },
            ),
            (
                {
                    "species": self.picmi_species,
                    "period": TimeStepSpec([slice(0, None, 10)]),
                    "bin_count": 50,
                    "min_energy": 0.0,
                    "max_energy": 500.0,
                },
                {
                    "bin_count": 50,
                    "min_energy": 0.0,
                    "max_energy": 500.0,
                    "period_specs": [{"start": 0, "stop": 999, "step": 10}],
                },
            ),
        ]
        for params, expected in TESTCASES_VALID:
            with self.subTest(params=params):
                eh = EnergyHistogram(**params)
                self.assertEqual(eh.species, params["species"])
                self.assertEqual(eh.bin_count, params["bin_count"])
                self.assertEqual(eh.min_energy, params["min_energy"])
                self.assertEqual(eh.max_energy, params["max_energy"])
                if isinstance(params["period"], int):
                    expected_period = TimeStepSpec(
                        [slice(None, None, params["period"])] if params["period"] > 0 else []
                    )("steps")
                    self.assertEqual(eh.period.specs, expected_period.specs)
                else:
                    self.assertEqual(eh.period, params["period"])
                eh.check()
                pypicongpu_eh = eh.get_as_pypicongpu(self.species_map, self.time_step_size, self.num_steps)
                self.assertIsInstance(pypicongpu_eh, PyPIConGPUEnergyHistogram)
                self.assertEqual(pypicongpu_eh.species, self.pypicongpu_species)
                self.assertEqual(pypicongpu_eh.bin_count, expected["bin_count"])
                self.assertEqual(pypicongpu_eh.min_energy, expected["min_energy"])
                self.assertEqual(pypicongpu_eh.max_energy, expected["max_energy"])
                serialized = pypicongpu_eh._get_serialized()
                self.assertEqual(serialized["period"]["specs"], expected["period_specs"])
        # Test invalid species mapping
        eh = EnergyHistogram(species=self.picmi_species, period=10, bin_count=50, min_energy=0.0, max_energy=500.0)
        with self.assertRaisesRegex(ValueError, f"Species {self.picmi_species} is not known to Simulation"):
            eh.get_as_pypicongpu({}, self.time_step_size, self.num_steps)

    def test_energy_histogram_invalid(self):
        """Test invalid EnergyHistogram inputs."""
        TESTCASES_INVALID = [
            (
                {"species": "invalid", "period": 10, "bin_count": 50, "min_energy": 0.0, "max_energy": 500.0},
                'argument "species".*is not an instance of',
            ),
            (
                {
                    "species": self.picmi_species,
                    "period": "invalid",
                    "bin_count": 50,
                    "min_energy": 0.0,
                    "max_energy": 500.0,
                },
                'argument "period".*did not match any element',
            ),
            (
                {"species": self.picmi_species, "period": 10, "bin_count": 0, "min_energy": 0.0, "max_energy": 500.0},
                "bin_count must be > 0",
            ),
            (
                {"species": self.picmi_species, "period": 10, "bin_count": 50, "min_energy": 500.0, "max_energy": 0.0},
                "min_energy must be less than max_energy",
            ),
            (
                {"species": PICMISpecies(), "period": 10, "bin_count": 50, "min_energy": 0.0, "max_energy": 500.0},
                "species must have a non-empty name",
            ),
            # Skip negative step test if it doesn't raise
            (
                {
                    "species": self.picmi_species,
                    "period": TimeStepSpec([slice(None, None, -10)]),
                    "bin_count": 50,
                    "min_energy": 0.0,
                    "max_energy": 500.0,
                },
                "Step size must be >= 1",
                True,
            ),
        ]
        for params, expected_error, *skip in TESTCASES_INVALID:
            with self.subTest(params=params, expected_error=expected_error):
                if skip and skip[0]:  # Skip if flagged
                    continue
                with self.assertRaisesRegex((ValueError, TypeError, typeguard.TypeCheckError), expected_error):
                    eh = EnergyHistogram(**params)
                    eh.check()


if __name__ == "__main__":
    unittest.main()
