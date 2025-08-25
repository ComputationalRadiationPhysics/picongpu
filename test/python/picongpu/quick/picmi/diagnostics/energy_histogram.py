"""
This file is part of PIConGPU.
Copyright 2021-2025 PIConGPU contributors
Authors: Masoud Afshari
License: GPLv3+
"""

from picongpu.picmi.diagnostics.energy_histogram import EnergyHistogram
from picongpu.picmi.diagnostics.timestepspec import TimeStepSpec
from picongpu.pypicongpu.species import Species as PyPIConGPUSpecies
from picongpu.picmi.species import Species as PICMISpecies
from picongpu.pypicongpu.output.energy_histogram import EnergyHistogram as PyPIConGPUEnergyHistogram
import unittest
import typeguard


def create_picmi_species():
    species = PICMISpecies()
    species.name = "electron"
    return species


def create_pypicongpu_species():
    species = PyPIConGPUSpecies()
    species.name = "electron"
    return species


class PICMI_TestEnergyHistogram(unittest.TestCase):
    def setUp(self):
        self.picmi_species = create_picmi_species()
        self.pypicongpu_species = create_pypicongpu_species()
        self.species_map = {self.picmi_species: self.pypicongpu_species}
        self.time_step_size = 1e-16
        self.num_steps = 1000

    def test_instantiation_valid(self):
        """Test instantiation and validation for valid inputs."""
        eh = EnergyHistogram(
            species=self.picmi_species,
            period=TimeStepSpec([slice(0, None, 10)]),
            bin_count=50,
            min_energy=0.0,
            max_energy=500.0,
        )
        eh.check()
        self.assertEqual(eh.species, self.picmi_species)
        self.assertEqual(eh.bin_count, 50)
        self.assertEqual(eh.min_energy, 0.0)
        self.assertEqual(eh.max_energy, 500.0)
        # Test int period
        eh = EnergyHistogram(
            species=self.picmi_species,
            period=10,
            bin_count=50,
            min_energy=0.0,
            max_energy=500.0,
        )
        eh.check()

    def test_types(self):
        """Test type safety."""
        with self.assertRaises(typeguard.TypeCheckError):
            EnergyHistogram(
                species="invalid",  # Invalid type: string instead of PICMISpecies
                period=TimeStepSpec([slice(0, None, 10)]),
                bin_count=50,
                min_energy=0.0,
                max_energy=500.0,
            )
        with self.assertRaises(typeguard.TypeCheckError):
            EnergyHistogram(
                species=self.picmi_species,
                period="invalid",  # Invalid: must be int or TimeStepSpec
                bin_count=50,
                min_energy=0.0,
                max_energy=500.0,
            )

    def test_validation(self):
        """Test validation for invalid inputs."""
        with self.assertRaises(ValueError, msg="bin_count must be > 0"):
            eh = EnergyHistogram(
                species=self.picmi_species,
                period=TimeStepSpec([slice(0, None, 10)]),
                bin_count=0,
                min_energy=0.0,
                max_energy=500.0,
            )
            eh.check()
        with self.assertRaises(ValueError, msg="min_energy must be less than max_energy"):
            eh = EnergyHistogram(
                species=self.picmi_species,
                period=TimeStepSpec([slice(0, None, 10)]),
                bin_count=50,
                min_energy=500.0,
                max_energy=0.0,
            )
            eh.check()
        invalid_species = PICMISpecies()
        invalid_species.name = None  # Invalid PICMISpecies configuration
        with self.assertRaises(TypeError, msg="species must have a non-empty name"):
            eh = EnergyHistogram(
                species=invalid_species,
                period=TimeStepSpec([slice(0, None, 10)]),
                bin_count=50,
                min_energy=0.0,
                max_energy=500.0,
            )
            eh.check()
        with self.assertRaises(typeguard.TypeCheckError):
            eh = EnergyHistogram(
                species=self.picmi_species,
                period="invalid",  # Invalid: must be int or TimeStepSpec
                bin_count=50,
                min_energy=0.0,
                max_energy=500.0,
            )

    def test_get_as_pypicongpu(self):
        """Test conversion to PyPIConGPU format."""
        eh = EnergyHistogram(
            species=self.picmi_species,
            period=TimeStepSpec([slice(0, None, 10)]),
            bin_count=50,
            min_energy=0.0,
            max_energy=500.0,
        )
        pypicongpu_eh = eh.get_as_pypicongpu(self.species_map, self.time_step_size, self.num_steps)
        self.assertIsInstance(pypicongpu_eh, PyPIConGPUEnergyHistogram)
        self.assertEqual(pypicongpu_eh.species, self.pypicongpu_species)
        self.assertEqual(pypicongpu_eh.bin_count, 50)
        self.assertEqual(pypicongpu_eh.min_energy, 0.0)
        self.assertEqual(pypicongpu_eh.max_energy, 500.0)

    def test_invalid_species_mapping(self):
        """Test invalid species mapping."""
        eh = EnergyHistogram(
            species=self.picmi_species,
            period=TimeStepSpec([slice(0, None, 10)]),
            bin_count=50,
            min_energy=0.0,
            max_energy=500.0,
        )
        with self.assertRaises(ValueError, msg=f"Species {self.picmi_species} is not known to Simulation"):
            eh.get_as_pypicongpu({}, self.time_step_size, self.num_steps)


if __name__ == "__main__":
    unittest.main()
