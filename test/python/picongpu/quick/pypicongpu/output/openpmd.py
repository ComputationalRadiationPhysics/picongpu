"""
This file is part of PIConGPU.
Copyright 2025 PIConGPU contributors
Authors: Masoud Afshari
License: GPLv3+
"""

from picongpu.pypicongpu.output import OpenPMD
from picongpu.pypicongpu.output.timestepspec import TimeStepSpec
from picongpu.pypicongpu.output.rangespec import RangeSpec
from picongpu.pypicongpu.output.openpmd_sources.source_base import SourceBase
import unittest
import typeguard
import typing


# Mock SourceBase for testing
class MockSourceBase(SourceBase):
    def _get_serialized(self) -> typing.Dict:
        return {"mock_source": True}

    def check(self) -> None:
        pass


class TestOpenPMD(unittest.TestCase):
    def test_empty(self):
        """Minimal configuration with only period is handled correctly."""
        # Missing period should fail
        openpmd = OpenPMD(period=None)
        with self.assertRaises(TypeError, msg="period must be a TimeStepSpec object"):
            openpmd._get_serialized()

        # Valid minimal configuration with period
        openpmd = OpenPMD(period=TimeStepSpec([slice(0, None, 100)]))
        serialized = openpmd._get_serialized()
        self.assertEqual(serialized["period"]["specs"][0]["step"], 100)
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
        openpmd = OpenPMD(period=TimeStepSpec([slice(0, None, 100)]))

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
        invalid_ranges = ["string", 1, []]
        for invalid in invalid_ranges:
            with self.assertRaises(typeguard.TypeCheckError):
                openpmd.range = invalid

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
        openpmd.period = TimeStepSpec([slice(0, None, 100)])  # Required parameter
        openpmd.source = [MockSourceBase()]
        openpmd.range = RangeSpec([slice(0, 10, None)])
        openpmd.file = "output"
        openpmd.ext = "h5"
        openpmd.infix = "prefix"
        openpmd.json = {"key": "value"}
        openpmd.json_restart = {"key": "restart"}
        openpmd.data_preparation_strategy = "doubleBuffer"
        openpmd.toml = "config.toml"
        openpmd.particle_io_chunk_size = 1024
        openpmd.file_writing = "append"
        openpmd._get_serialized()  # Should succeed

    def test_validation(self):
        """Constraints on parameters are enforced."""
        openpmd = OpenPMD(period=TimeStepSpec([slice(0, None, 100)]))

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
        openpmd.source = ["invalid"]
        with self.assertRaisesRegex(ValueError, "source must be a list of SourceBase objects"):
            openpmd.check()

        # Valid configuration
        openpmd.period = TimeStepSpec([slice(0, None, 100)])  # Required parameter
        openpmd.source = [MockSourceBase()]
        openpmd.range = RangeSpec([slice(0, 10, None)])
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
        openpmd.source = [MockSourceBase()]
        openpmd.range = RangeSpec([slice(0, 10, None)])
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
        self.assertEqual(context["source"], [{"mock_source": True}])
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
        openpmd = OpenPMD(period=TimeStepSpec([slice(0, None, 100)]))
        context = openpmd.get_rendering_context()
        context = context["data"]
        self.assertEqual(context["period"]["specs"][0]["step"], 100)
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


if __name__ == "__main__":
    unittest.main()
