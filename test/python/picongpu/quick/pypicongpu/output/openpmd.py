"""
This file is part of PIConGPU.
Copyright 2025 PIConGPU contributors
Authors: Masoud Afshari
License: GPLv3+
"""

from picongpu.pypicongpu.output import OpenPMD
from picongpu.pypicongpu.output.timestepspec import TimeStepSpec
from picongpu.pypicongpu.output.rangespec import RangeSpec as PyPIConGPURangeSpec
from picongpu.pypicongpu.output.openpmd_sources.charge_density import ChargeDensity
from picongpu.pypicongpu.output.openpmd_sources.bound_electron_density import BoundElectronDensity
from picongpu.pypicongpu.output.openpmd_sources.counter import Counter
from picongpu.pypicongpu.output.openpmd_sources.density import Density
from picongpu.pypicongpu.output.openpmd_sources.derived_attributes import DerivedAttributes
from picongpu.pypicongpu.output.openpmd_sources.energy import Energy
from picongpu.pypicongpu.output.openpmd_sources.energy_density import EnergyDensity
from picongpu.pypicongpu.output.openpmd_sources.energy_density_cutoff import EnergyDensityCutoff
from picongpu.pypicongpu.output.openpmd_sources.larmor_power import LarmorPower
from picongpu.pypicongpu.output.openpmd_sources.macro_counter import MacroCounter
from picongpu.pypicongpu.output.openpmd_sources.mid_current_density_component import MidCurrentDensityComponent
from picongpu.pypicongpu.output.openpmd_sources.momentum import Momentum
from picongpu.pypicongpu.output.openpmd_sources.momentum_density import MomentumDensity
from picongpu.pypicongpu.output.openpmd_sources.weighted_velocity import WeightedVelocity
from picongpu.pypicongpu.species import Species

import unittest
import typeguard
import typing


# Mock Species class for testing
class MockSpecies(Species):
    def __init__(self):
        pass

    def get_rendering_context(self) -> typing.Dict:
        return {}  # Minimal context to avoid schema conflicts

    def check(self) -> None:
        pass


class TestOpenPMD(unittest.TestCase):
    def test_empty(self):
        """Minimal configuration with only period is handled correctly."""
        # Missing period should fail
        with self.assertRaises(typeguard.TypeCheckError):
            OpenPMD(period=None)

        # Valid minimal configuration with period
        openpmd = OpenPMD(period=TimeStepSpec([slice(None, None, None)]))
        serialized = openpmd._get_serialized()
        self.assertEqual(serialized["period"]["specs"][0]["step"], 1)  # Default step is 1
        self.assertIsNone(serialized["source"])
        self.assertIsNone(serialized["range"])
        self.assertIsNone(serialized["file"])
        self.assertEqual(serialized["ext"], "bp")
        self.assertEqual(serialized["infix"], "NULL")
        self.assertIsNone(serialized["json"])
        self.assertIsNone(serialized["json_restart"])
        self.assertIsNone(serialized["data_preparation_strategy"])
        self.assertIsNone(serialized["toml"])
        self.assertIsNone(serialized["particle_io_chunk_size"])
        self.assertEqual(serialized["file_writing"], "create")

    def test_types(self):
        """Type safety is ensured for all attributes."""
        openpmd = OpenPMD(period=TimeStepSpec([slice(None, None, None)]))

        # Invalid period
        invalid_periods = ["string", 1, [], {}, None]
        for invalid in invalid_periods:
            with self.assertRaises(typeguard.TypeCheckError):
                openpmd.period = invalid

        # Invalid source
        invalid_sources = ["string", 1, [1, 2], [None]]
        for invalid in invalid_sources:
            with self.assertRaises(typeguard.TypeCheckError):
                openpmd.source = invalid

        # Invalid range
        invalid_ranges = ["string", 1, [], slice(0.0, 10), slice("a", 10)]
        for invalid in invalid_ranges:
            with self.assertRaises(TypeError):
                openpmd.range = PyPIConGPURangeSpec[invalid]

        # Invalid file
        invalid_files = [1, []]
        for invalid in invalid_files:
            with self.assertRaises(typeguard.TypeCheckError):
                openpmd.file = invalid

        # Invalid ext
        invalid_exts = ["invalid", 1, []]
        for invalid in invalid_exts:
            with self.assertRaises(typeguard.TypeCheckError):
                openpmd.ext = invalid

        # Invalid infix
        invalid_infixes = [1, []]
        for invalid in invalid_infixes:
            with self.assertRaises(typeguard.TypeCheckError):
                openpmd.infix = invalid

        # Invalid json
        invalid_jsons = [1, []]
        for invalid in invalid_jsons:
            with self.assertRaises(typeguard.TypeCheckError):
                openpmd.json = invalid

        # Invalid json_restart
        invalid_json_restarts = [1, []]
        for invalid in invalid_json_restarts:
            with self.assertRaises(typeguard.TypeCheckError):
                openpmd.json_restart = invalid

        # Invalid data_preparation_strategy
        invalid_strategies = ["invalid", 1, []]
        for invalid in invalid_strategies:
            with self.assertRaises(typeguard.TypeCheckError):
                openpmd.data_preparation_strategy = invalid

        # Invalid toml
        invalid_tomls = [1, []]
        for invalid in invalid_tomls:
            with self.assertRaises(typeguard.TypeCheckError):
                openpmd.toml = invalid

        # Invalid particle_io_chunk_size
        invalid_chunk_sizes = ["string", 1.5, []]
        for invalid in invalid_chunk_sizes:
            with self.assertRaises(typeguard.TypeCheckError):
                openpmd.particle_io_chunk_size = invalid

        # Invalid file_writing
        invalid_file_writings = ["invalid", 1, []]
        for invalid in invalid_file_writings:
            with self.assertRaises(typeguard.TypeCheckError):
                openpmd.file_writing = invalid

        # Valid configuration
        openpmd.period = TimeStepSpec([slice(0, None, 100)])
        openpmd.source = [ChargeDensity(species=MockSpecies())]
        openpmd.range = PyPIConGPURangeSpec[0:10]
        openpmd.file = "output"
        openpmd.ext = "h5"
        openpmd.infix = "prefix"
        openpmd.json = {"key": "value"}
        openpmd.json_restart = {"key": "restart"}
        openpmd.data_preparation_strategy = "doubleBuffer"
        openpmd.toml = "config.toml"
        openpmd.particle_io_chunk_size = 1024
        openpmd.file_writing = "append"
        openpmd.check()  # Validate instead of serializing

    def test_validation(self):
        """Constraints on parameters are enforced."""
        openpmd = OpenPMD(period=TimeStepSpec([slice(None, None, None)]))

        # Invalid particle_io_chunk_size
        openpmd.particle_io_chunk_size = 0
        with self.assertRaisesRegex(ValueError, "particle_io_chunk_size \(in MiB\) must be positive"):
            openpmd.check()

        # Invalid infix with ext="sst"
        openpmd.particle_io_chunk_size = None
        openpmd.ext = "sst"
        openpmd.infix = "prefix"
        with self.assertRaisesRegex(ValueError, "infix must be 'NULL' when ext is 'sst'"):
            openpmd.check()

        # Invalid source
        openpmd.ext = "bp"
        openpmd.infix = "NULL"
        with self.assertRaises(typeguard.TypeCheckError):
            openpmd.source = ["invalid"]

        # Valid configuration
        openpmd.period = TimeStepSpec([slice(0, None, 100)])
        openpmd.source = [ChargeDensity(species=MockSpecies())]
        openpmd.range = PyPIConGPURangeSpec[0:10]
        openpmd.file = "output"
        openpmd.ext = "h5"
        openpmd.infix = "prefix"
        openpmd.json = {"key": "value"}
        openpmd.json_restart = {"key": "restart"}
        openpmd.data_preparation_strategy = "doubleBuffer"
        openpmd.toml = "config.toml"
        openpmd.particle_io_chunk_size = 1024
        openpmd.file_writing = "append"
        openpmd.check()  # Should succeed

    def test_rendering(self):
        """Serialized data is correctly formatted for template consumption."""
        openpmd = OpenPMD(period=TimeStepSpec([slice(0, None, 100)]))
        openpmd.source = None  # Avoid schema conflict
        openpmd.range = PyPIConGPURangeSpec[0:10]
        openpmd.file = "output"
        openpmd.ext = "h5"
        openpmd.infix = "prefix"
        openpmd.json = {"key": "value"}
        openpmd.json_restart = {"key": "restart"}
        openpmd.data_preparation_strategy = "doubleBuffer"
        openpmd.toml = "config.toml"
        openpmd.particle_io_chunk_size = 1024
        openpmd.file_writing = "append"

        context = openpmd.get_rendering_context()
        self.assertTrue(context["typeID"]["openpmd"], "typeID should be openpmd")
        context = context["data"]
        self.assertEqual(context["period"]["specs"][0]["step"], 100)
        self.assertIsNone(context["source"])
        self.assertEqual(context["range"], "0:10")
        self.assertEqual(context["file"], "output")
        self.assertEqual(context["ext"], "h5")
        self.assertEqual(context["infix"], "prefix")
        self.assertEqual(context["json"], {"key": "value"})
        self.assertEqual(context["json_restart"], {"key": "restart"})
        self.assertEqual(context["data_preparation_strategy"], "doubleBuffer")
        self.assertEqual(context["toml"], "config.toml")
        self.assertEqual(context["particle_io_chunk_size"], 1024)
        self.assertEqual(context["file_writing"], "append")

        # Minimal configuration
        openpmd = OpenPMD(period=TimeStepSpec([slice(None, None, None)]))
        context = openpmd.get_rendering_context()
        context = context["data"]
        self.assertEqual(context["period"]["specs"][0]["step"], 1)  # Default step is 1
        self.assertIsNone(context["source"])
        self.assertIsNone(context["range"])
        self.assertIsNone(context["file"])
        self.assertEqual(context["ext"], "bp")
        self.assertEqual(context["infix"], "NULL")
        self.assertIsNone(context["json"])
        self.assertIsNone(context["json_restart"])
        self.assertIsNone(context["data_preparation_strategy"])
        self.assertIsNone(context["toml"])
        self.assertIsNone(context["particle_io_chunk_size"])
        self.assertEqual(context["file_writing"], "create")

    def test_source_bound_electron_density(self):
        """Test BoundElectronDensity instantiation."""
        source = BoundElectronDensity(species=MockSpecies())
        openpmd = OpenPMD(period=TimeStepSpec([slice(0, None, 100)]), source=[source])
        openpmd.check()  # Ensure instantiation and check pass

    def test_source_charge_density(self):
        """Test ChargeDensity instantiation."""
        source = ChargeDensity(species=MockSpecies())
        openpmd = OpenPMD(period=TimeStepSpec([slice(0, None, 100)]), source=[source])
        openpmd.check()  # Ensure instantiation and check pass

    def test_source_counter(self):
        """Test Counter instantiation."""
        source = Counter(species=MockSpecies())
        openpmd = OpenPMD(period=TimeStepSpec([slice(0, None, 100)]), source=[source])
        openpmd.check()  # Ensure instantiation and check pass

    def test_source_density(self):
        """Test Density instantiation."""
        source = Density(species=MockSpecies())
        openpmd = OpenPMD(period=TimeStepSpec([slice(0, None, 100)]), source=[source])
        openpmd.check()  # Ensure instantiation and check pass

    def test_source_derived_attributes(self):
        """Test DerivedAttributes instantiation."""
        source = DerivedAttributes()
        openpmd = OpenPMD(period=TimeStepSpec([slice(0, None, 100)]), source=[source])
        openpmd.check()  # Ensure instantiation and check pass

    def test_source_energy(self):
        """Test Energy instantiation."""
        source = Energy(species=MockSpecies())
        openpmd = OpenPMD(period=TimeStepSpec([slice(0, None, 100)]), source=[source])
        openpmd.check()  # Ensure instantiation and check pass

    def test_source_energy_density(self):
        """Test EnergyDensity instantiation."""
        source = EnergyDensity(species=MockSpecies())
        openpmd = OpenPMD(period=TimeStepSpec([slice(0, None, 100)]), source=[source])
        openpmd.check()  # Ensure instantiation and check pass

    def test_source_energy_density_cutoff(self):
        """Test EnergyDensityCutoff instantiation."""
        source = EnergyDensityCutoff(species=MockSpecies(), cutoff_max_energy=1.0)
        openpmd = OpenPMD(period=TimeStepSpec([slice(0, None, 100)]), source=[source])
        openpmd.check()  # Ensure instantiation and check pass

    def test_source_larmor_power(self):
        """Test LarmorPower instantiation."""
        source = LarmorPower(species=MockSpecies())
        openpmd = OpenPMD(period=TimeStepSpec([slice(0, None, 100)]), source=[source])
        openpmd.check()  # Ensure instantiation and check pass

    def test_source_macro_counter(self):
        """Test MacroCounter instantiation."""
        source = MacroCounter(species=MockSpecies())
        openpmd = OpenPMD(period=TimeStepSpec([slice(0, None, 100)]), source=[source])
        openpmd.check()  # Ensure instantiation and check pass

    def test_source_mid_current_density_component(self):
        """Test MidCurrentDensityComponent instantiation."""
        source = MidCurrentDensityComponent(species=MockSpecies(), direction="x")
        openpmd = OpenPMD(period=TimeStepSpec([slice(0, None, 100)]), source=[source])
        openpmd.check()  # Ensure instantiation and check pass

    def test_source_momentum(self):
        """Test Momentum instantiation."""
        source = Momentum(species=MockSpecies())
        openpmd = OpenPMD(period=TimeStepSpec([slice(0, None, 100)]), source=[source])
        openpmd.check()  # Ensure instantiation and check pass

    def test_source_momentum_density(self):
        """Test MomentumDensity instantiation."""
        source = MomentumDensity(species=MockSpecies())
        openpmd = OpenPMD(period=TimeStepSpec([slice(0, None, 100)]), source=[source])
        openpmd.check()  # Ensure instantiation and check pass

    def test_source_weighted_velocity(self):
        """Test WeightedVelocity instantiation."""
        source = WeightedVelocity(species=MockSpecies())
        openpmd = OpenPMD(period=TimeStepSpec([slice(0, None, 100)]), source=[source])
        openpmd.check()  # Ensure instantiation and check pass


if __name__ == "__main__":
    unittest.main()
