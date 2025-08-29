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


class MockSpecies(Species):
    def __init__(self):
        self.name = "electron"
        self.constants = []
        self.attributes = []

    def get_rendering_context(self) -> typing.Dict:
        return {
            "name": self.name,
            "typename": self.__class__.__name__,
            "attributes": [{"picongpu_name": attr.__class__.__name__.lower()} for attr in self.attributes],
            "constants": {
                "mass": None,
                "charge": None,
                "density_ratio": None,
                "ground_state_ionization": None,
                "element_properties": None,
            },
        }

    def check(self) -> None:
        if not self.name:
            raise ValueError("Species name must not be empty or None")


class TestOpenPMD(unittest.TestCase):
    def test_empty(self):
        """Minimal configuration with only period is handled correctly."""
        with self.assertRaises(typeguard.TypeCheckError):
            OpenPMD(period=None)

        openpmd = OpenPMD(period=TimeStepSpec([slice(None, None, None)]))
        serialized = openpmd._get_serialized()
        self.assertEqual(serialized["period"]["specs"][0]["step"], 1)
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
        invalid_periods = ["string", 1, [], {}, None]
        for invalid in invalid_periods:
            with self.assertRaises(typeguard.TypeCheckError):
                openpmd.period = invalid

        invalid_sources = ["string", 1, [1, 2], [None]]
        for invalid in invalid_sources:
            with self.assertRaises(typeguard.TypeCheckError):
                openpmd.source = invalid

        invalid_ranges = ["string", 1, [], slice(0.0, 10), slice("a", 10)]
        for invalid in invalid_ranges:
            with self.assertRaises(TypeError):
                openpmd.range = PyPIConGPURangeSpec[invalid]

        invalid_files = [1, []]
        for invalid in invalid_files:
            with self.assertRaises(typeguard.TypeCheckError):
                openpmd.file = invalid

        invalid_exts = ["invalid", 1, []]
        for invalid in invalid_exts:
            with self.assertRaises(typeguard.TypeCheckError):
                openpmd.ext = invalid

        invalid_infixes = [1, []]
        for invalid in invalid_infixes:
            with self.assertRaises(typeguard.TypeCheckError):
                openpmd.infix = invalid

        invalid_jsons = [1, []]
        for invalid in invalid_jsons:
            with self.assertRaises(typeguard.TypeCheckError):
                openpmd.json = invalid

        invalid_json_restarts = [1, []]
        for invalid in invalid_json_restarts:
            with self.assertRaises(typeguard.TypeCheckError):
                openpmd.json_restart = invalid

        invalid_strategies = ["invalid", 1, []]
        for invalid in invalid_strategies:
            with self.assertRaises(typeguard.TypeCheckError):
                openpmd.data_preparation_strategy = invalid

        invalid_tomls = [1, []]
        for invalid in invalid_tomls:
            with self.assertRaises(typeguard.TypeCheckError):
                openpmd.toml = invalid

        invalid_chunk_sizes = ["string", 1.5, []]
        for invalid in invalid_chunk_sizes:
            with self.assertRaises(typeguard.TypeCheckError):
                openpmd.particle_io_chunk_size = invalid

        invalid_file_writings = ["invalid", 1, []]
        for invalid in invalid_file_writings:
            with self.assertRaises(typeguard.TypeCheckError):
                openpmd.file_writing = invalid

        openpmd.period = TimeStepSpec([slice(0, None, 100)])
        openpmd.source = [ChargeDensity(species=MockSpecies(), filter="species_all")]
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
        openpmd.check()

    def test_validation(self):
        """Constraints on parameters are enforced."""
        openpmd = OpenPMD(period=TimeStepSpec([slice(None, None, None)]))
        openpmd.particle_io_chunk_size = 0
        with self.assertRaisesRegex(ValueError, "particle_io_chunk_size \(in MiB\) must be positive"):
            openpmd.check()

        openpmd.particle_io_chunk_size = None
        openpmd.ext = "sst"
        openpmd.infix = "prefix"
        with self.assertRaisesRegex(ValueError, "infix must be 'NULL' when ext is 'sst'"):
            openpmd.check()

        openpmd.ext = "bp"
        openpmd.infix = "NULL"
        with self.assertRaises(typeguard.TypeCheckError):
            openpmd.source = ["invalid"]

        openpmd.period = TimeStepSpec([slice(0, None, 100)])
        openpmd.source = [ChargeDensity(species=MockSpecies(), filter="species_all")]
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
        openpmd.check()

    def test_rendering(self):
        """Serialized data is correctly formatted for template consumption."""
        openpmd = OpenPMD(period=TimeStepSpec([slice(0, None, 100)]))
        openpmd.source = None
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
        self.assertTrue(context["typeID"]["openpmd"])
        context = context["data"]
        self.assertEqual(context["period"]["specs"][0]["step"], 100)
        self.assertIsNone(context["source"])
        self.assertEqual(context["range"], {"ranges": [{"begin": 0, "end": 10}]})
        self.assertEqual(context["file"], "output")
        self.assertEqual(context["ext"], "h5")
        self.assertEqual(context["infix"], "prefix")
        self.assertEqual(context["json"], {"key": "value"})
        self.assertEqual(context["json_restart"], {"key": "restart"})
        self.assertEqual(context["data_preparation_strategy"], "doubleBuffer")
        self.assertEqual(context["toml"], "config.toml")
        self.assertEqual(context["particle_io_chunk_size"], 1024)
        self.assertEqual(context["file_writing"], "append")

        openpmd = OpenPMD(period=TimeStepSpec([slice(None, None, None)]))
        context = openpmd.get_rendering_context()
        context = context["data"]
        self.assertEqual(context["period"]["specs"][0]["step"], 1)
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
        """Test BoundElectronDensity instantiation and filter."""
        source = BoundElectronDensity(species=MockSpecies(), filter="species_all")
        openpmd = OpenPMD(period=TimeStepSpec([slice(0, None, 100)]), source=[source])
        openpmd.check()
        context = openpmd.get_rendering_context()
        self.assertEqual(context["data"]["source"][0]["filter"], "species_all")

    def test_source_charge_density(self):
        """Test ChargeDensity instantiation and filter."""
        source = ChargeDensity(species=MockSpecies(), filter="species_all")
        openpmd = OpenPMD(period=TimeStepSpec([slice(0, None, 100)]), source=[source])
        openpmd.check()
        context = openpmd.get_rendering_context()
        self.assertEqual(context["data"]["source"][0]["filter"], "species_all")

    def test_source_counter(self):
        """Test Counter instantiation and filter."""
        source = Counter(species=MockSpecies(), filter="species_all")
        openpmd = OpenPMD(period=TimeStepSpec([slice(0, None, 100)]), source=[source])
        openpmd.check()
        context = openpmd.get_rendering_context()
        self.assertEqual(context["data"]["source"][0]["filter"], "species_all")

    def test_source_density(self):
        """Test Density instantiation and filter."""
        source = Density(species=MockSpecies(), filter="species_all")
        openpmd = OpenPMD(period=TimeStepSpec([slice(0, None, 100)]), source=[source])
        openpmd.check()
        context = openpmd.get_rendering_context()
        self.assertEqual(context["data"]["source"][0]["filter"], "species_all")

    def test_source_derived_attributes(self):
        """Test DerivedAttributes instantiation and filter."""
        source = DerivedAttributes(filter="species_all")
        openpmd = OpenPMD(period=TimeStepSpec([slice(0, None, 100)]), source=[source])
        openpmd.check()
        context = openpmd.get_rendering_context()
        self.assertEqual(context["data"]["source"][0]["filter"], "species_all")

    def test_source_energy(self):
        """Test Energy instantiation and filter."""
        source = Energy(species=MockSpecies(), filter="species_all")
        openpmd = OpenPMD(period=TimeStepSpec([slice(0, None, 100)]), source=[source])
        openpmd.check()
        context = openpmd.get_rendering_context()
        self.assertEqual(context["data"]["source"][0]["filter"], "species_all")

    def test_source_energy_density(self):
        """Test EnergyDensity instantiation and filter."""
        source = EnergyDensity(species=MockSpecies(), filter="species_all")
        openpmd = OpenPMD(period=TimeStepSpec([slice(0, None, 100)]), source=[source])
        openpmd.check()
        context = openpmd.get_rendering_context()
        self.assertEqual(context["data"]["source"][0]["filter"], "species_all")

    def test_source_energy_density_cutoff(self):
        """Test EnergyDensityCutoff instantiation and filter."""
        source = EnergyDensityCutoff(species=MockSpecies(), cutoff_max_energy=1.0, filter="species_all")
        openpmd = OpenPMD(period=TimeStepSpec([slice(0, None, 100)]), source=[source])
        openpmd.check()
        context = openpmd._get_serialized()
        self.assertEqual(context["source"][0]["filter"], "species_all")
        self.assertEqual(context["source"][0]["type"], "energydensitycutoff")
        self.assertEqual(context["source"][0]["species"]["name"], "electron")
        self.assertEqual(context["source"][0]["cutoff_max_energy"], 1.0)

    def test_source_larmor_power(self):
        """Test LarmorPower instantiation and filter."""
        source = LarmorPower(species=MockSpecies(), filter="species_all")
        openpmd = OpenPMD(period=TimeStepSpec([slice(0, None, 100)]), source=[source])
        openpmd.check()
        context = openpmd.get_rendering_context()
        self.assertEqual(context["data"]["source"][0]["filter"], "species_all")

    def test_source_macro_counter(self):
        """Test MacroCounter instantiation and filter."""
        source = MacroCounter(species=MockSpecies(), filter="species_all")
        openpmd = OpenPMD(period=TimeStepSpec([slice(0, None, 100)]), source=[source])
        openpmd.check()
        context = openpmd.get_rendering_context()
        self.assertEqual(context["data"]["source"][0]["filter"], "species_all")

    def test_source_mid_current_density_component(self):
        """Test MidCurrentDensityComponent instantiation and filter."""
        source = MidCurrentDensityComponent(species=MockSpecies(), direction="x", filter="species_all")
        openpmd = OpenPMD(period=TimeStepSpec([slice(0, None, 100)]), source=[source])
        openpmd.check()
        context = openpmd.get_rendering_context()
        self.assertEqual(context["data"]["source"][0]["filter"], "species_all")

    def test_source_momentum(self):
        """Test Momentum instantiation and filter."""
        source = Momentum(species=MockSpecies(), filter="species_all")
        openpmd = OpenPMD(period=TimeStepSpec([slice(0, None, 100)]), source=[source])
        openpmd.check()
        context = openpmd.get_rendering_context()
        self.assertEqual(context["data"]["source"][0]["filter"], "species_all")

    def test_source_momentum_density(self):
        """Test MomentumDensity instantiation and filter."""
        source = MomentumDensity(species=MockSpecies(), filter="species_all")
        openpmd = OpenPMD(period=TimeStepSpec([slice(0, None, 100)]), source=[source])
        openpmd.check()
        context = openpmd.get_rendering_context()
        self.assertEqual(context["data"]["source"][0]["filter"], "species_all")

    def test_source_weighted_velocity(self):
        """Test WeightedVelocity instantiation and filter."""
        source = WeightedVelocity(species=MockSpecies(), filter="species_all")
        openpmd = OpenPMD(period=TimeStepSpec([slice(0, None, 100)]), source=[source])
        openpmd.check()
        context = openpmd.get_rendering_context()
        self.assertEqual(context["data"]["source"][0]["filter"], "species_all")


if __name__ == "__main__":
    unittest.main()
