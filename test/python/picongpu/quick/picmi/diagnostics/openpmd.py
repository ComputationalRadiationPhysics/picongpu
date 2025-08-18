"""
This file is part of PIConGPU.
Copyright 2021-2025 PIConGPU contributors
Authors: Masoud Afshari
License: GPLv3+
"""

from picongpu.picmi.diagnostics.openpmd import OpenPMD
from picongpu.picmi.diagnostics.timestepspec import TimeStepSpec
from picongpu.picmi.diagnostics.rangespec import RangeSpec
from picongpu.pypicongpu.output.openpmd import OpenPMD as PyPIConGPUOpenPMD
from picongpu.pypicongpu.output.openpmd_sources.source_base import SourceBase
import unittest
import unittest.mock


# Mock SourceBase for testing
class MockSource(SourceBase):
    def __init__(self, name="mock"):
        self.name = name

    @property
    def filter(self) -> str:
        return "all"

    def check(self) -> None:
        pass

    def _get_serialized(self):
        return {"name": self.name}


# Valid test cases for OpenPMD instantiation and serialization
TESTCASES_VALID = [
    (
        # Basic case: period, default range, no source
        {
            "period": TimeStepSpec("1:100:2"),
            "source": None,
            "range": ":,:,:",
            "file": None,
            "ext": "bp",
            "infix": "NULL",
            "json": {},
            "json_restart": {},
            "data_preparation_strategy": None,
            "toml": None,
            "particle_io_chunk_size": None,
            "file_writing": "create",
        },
        (20, 30, 40),  # simulation_box
        0.001,  # time_step_size
        1000,  # num_steps
        {
            "period": {"specs": [{"start": 1, "stop": 100, "step": 2}]},
            "source": None,
            "range": [{"begin": 0, "end": 19}, {"begin": 0, "end": 29}, {"begin": 0, "end": 39}],
            "file": None,
            "ext": "bp",
            "infix": "NULL",
            "json": {},
            "json_restart": {},
            "data_preparation_strategy": None,
            "toml": None,
            "particle_io_chunk_size": None,
            "file_writing": "create",
        },
    ),
    (
        # With source, string range, custom file and ext
        {
            "period": TimeStepSpec("0::1"),
            "source": [MockSource("chargeDensity"), MockSource("energyHistogram")],
            "range": "0:10,5:15,2:8",
            "file": "output/data",
            "ext": "h5",
            "infix": "_%06T",
            "json": {"key": "value"},
            "json_restart": "@restart.json",
            "data_preparation_strategy": "doubleBuffer",
            "toml": "config.toml",
            "particle_io_chunk_size": 1024,
            "file_writing": "append",
        },
        (20, 30, 40),
        0.001,
        1000,
        {
            "period": {"specs": [{"start": 0, "stop": 999, "step": 1}]},
            "source": [{"name": "chargeDensity"}, {"name": "energyHistogram"}],
            "range": [{"begin": 0, "end": 10}, {"begin": 5, "end": 15}, {"begin": 2, "end": 8}],
            "file": "output/data",
            "ext": "h5",
            "infix": "_%06T",
            "json": {"key": "value"},
            "json_restart": "@restart.json",
            "data_preparation_strategy": "doubleBuffer",
            "toml": "config.toml",
            "particle_io_chunk_size": 1024,
            "file_writing": "append",
        },
    ),
    (
        # RangeSpec object, minimal parameters
        {
            "period": TimeStepSpec("10"),
            "source": [MockSource("field")],
            "range": RangeSpec([slice(0, 5), slice(10, 20)]),
            "file": None,
            "ext": "sst",
            "infix": "NULL",
            "json": None,
            "json_restart": None,
            "data_preparation_strategy": None,
            "toml": None,
            "particle_io_chunk_size": 512,
            "file_writing": "create",
        },
        (20, 30),
        0.001,
        100,
        {
            "period": {"specs": [{"start": 10, "stop": 11, "step": 1}]},
            "source": [{"name": "field"}],
            "range": [{"begin": 0, "end": 5}, {"begin": 10, "end": 20}],
            "file": None,
            "ext": "sst",
            "infix": "NULL",
            "json": {},
            "json_restart": {},
            "data_preparation_strategy": None,
            "toml": None,
            "particle_io_chunk_size": 512,
            "file_writing": "create",
        },
    ),
]

# Invalid test cases for instantiation
TESTCASES_INVALID = [
    (
        {"period": TimeStepSpec("1"), "particle_io_chunk_size": 0},
        "particle_io_chunk_size.*must be positive",
    ),
    (
        {"period": TimeStepSpec("1"), "ext": "sst", "infix": "_%06T"},
        "infix must be 'NULL' when ext is 'sst'",
    ),
    (
        {"period": TimeStepSpec("1"), "source": [MockSource(), "invalid"]},
        "source must be a list of SourceBase objects",
    ),
    (
        {"period": TimeStepSpec("1"), "ext": "invalid"},
        "ext.*must be one of.*bp.*h5.*sst",
    ),
    (
        {"period": TimeStepSpec("1"), "file_writing": "invalid"},
        "file_writing.*must be one of.*create.*append",
    ),
]


class TestOpenPMD(unittest.TestCase):
    def test_openpmd_instantiation(self):
        """Test OpenPMD instantiation and validation."""
        for params, sim_box, time_step_size, num_steps, _ in TESTCASES_VALID:
            with self.subTest(params=params, sim_box=sim_box):
                openpmd = OpenPMD(**params)
                openpmd.check()
                self.assertIsInstance(openpmd.range, RangeSpec)
                if params["source"] is not None:
                    self.assertTrue(all(isinstance(s, SourceBase) for s in params["source"]))

        for params, expected_error in TESTCASES_INVALID:
            with self.subTest(params=params, expected_error=expected_error):
                with self.assertRaisesRegex(ValueError, expected_error):
                    OpenPMD(**params)

    def test_openpmd_serialization(self):
        """Test OpenPMD serialization to PyPIConGPUOpenPMD."""
        for params, sim_box, time_step_size, num_steps, expected_serialized in TESTCASES_VALID:
            with self.subTest(params=params, sim_box=sim_box):
                openpmd = OpenPMD(**params)
                pypicongpu_openpmd = openpmd.get_as_pypicongpu({}, time_step_size, num_steps, sim_box)
                self.assertIsInstance(pypicongpu_openpmd, PyPIConGPUOpenPMD)
                serialized = pypicongpu_openpmd.get_rendering_context()
                self.assertEqual(serialized, expected_serialized)

    def test_openpmd_invalid_simulation_box(self):
        """Test invalid simulation box dimensions."""
        openpmd = OpenPMD(period=TimeStepSpec("1"), range=RangeSpec([slice(0, 10), slice(5, 15)]))
        with self.assertRaisesRegex(ValueError, "Number of range specifications"):
            openpmd.get_as_pypicongpu({}, 0.001, 100, (20,))  # Too few dimensions
        with self.assertRaisesRegex(ValueError, "Number of range specifications"):
            openpmd.get_as_pypicongpu({}, 0.001, 100, (20, 30, 40))  # Too many dimensions


if __name__ == "__main__":
    unittest.main()
