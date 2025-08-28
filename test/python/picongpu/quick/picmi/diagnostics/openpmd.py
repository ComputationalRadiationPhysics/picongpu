"""
This file is part of PIConGPU.
Copyright 2025 PIConGPU contributors
Authors: Masoud Afshari
License: GPLv3+
"""

from picongpu.picmi.diagnostics.openpmd import OpenPMD
from picongpu.picmi.diagnostics.timestepspec import TimeStepSpec
from picongpu.picmi.diagnostics.rangespec import RangeSpec
from picongpu.pypicongpu.output.openpmd import OpenPMD as PyPIConGPUOpenPMD
from picongpu.picmi.diagnostics.openpmd_sources.source_base import SourceBase
from picongpu.picmi.diagnostics.openpmd_sources.bound_electron_density import BoundElectronDensity
from picongpu.picmi.species import Species as PICMISpecies
from picongpu.pypicongpu.species import Species as PyPIConGPUSpecies
from picongpu.pypicongpu.output.openpmd_sources.source_base import SourceBase as PyPIConGPUSourceBase
import unittest
import typeguard
import typing


class MockSource(SourceBase):
    def __init__(self, name="mock", filter_value="species_all"):
        self.name = name
        self._filter = filter_value

    @property
    def filter(self) -> str:
        return self._filter

    def check(self) -> None:
        valid_filters = ["species_all", "fields_all", "custom_filter"]
        if not isinstance(self._filter, str):
            raise ValueError(f"Filter must be a string, got {type(self._filter)}")
        if self._filter not in valid_filters:
            raise ValueError(f"Filter must be one of {valid_filters}, got {self._filter}")

    def get_as_pypicongpu(self, mapping: typing.Dict[PICMISpecies, typing.Any] = None) -> typing.Any:
        class MockPyPIConGPUSource(PyPIConGPUSourceBase):
            def __init__(self, name, filter_value):
                self.name = name
                self.filter = filter_value

            def _get_serialized(self):
                return {"name": self.name, "filter": self.filter}

            def check(self):
                valid_filters = ["species_all", "fields_all", "custom_filter"]
                if not isinstance(self.filter, str):
                    raise ValueError(f"Filter must be a string, got {type(self.filter)}")
                if self.filter not in valid_filters:
                    raise ValueError(f"Filter must be one of {valid_filters}, got {self.filter}")

        return MockPyPIConGPUSource(self.name, self._filter)


TESTCASES_VALID = [
    (
        {
            "period": TimeStepSpec([slice(1, 100, 2)]),
            "source": None,
            "range": RangeSpec[:, :, :],
            "file": None,
            "ext": "bp",
            "infix": "NULL",
            "json": None,
            "json_restart": None,
            "data_preparation_strategy": None,
            "toml": None,
            "particle_io_chunk_size": None,
            "file_writing": "create",
        },
        (20, 30, 40),
        0.001,
        1000,
        {
            "period": {"specs": [{"start": 1, "stop": 100, "step": 2}]},
            "source": None,
            "range": {"ranges": [{"begin": 0, "end": 19}, {"begin": 0, "end": 29}, {"begin": 0, "end": 39}]},
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
        {
            "period": TimeStepSpec([slice(0, None, 1)]),
            "source": [MockSource("chargeDensity", "species_all"), MockSource("energyHistogram", "fields_all")],
            "range": RangeSpec[0:10, 5:15, 2:8],
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
            "source": [
                {"name": "chargeDensity", "filter": "species_all"},
                {"name": "energyHistogram", "filter": "fields_all"},
            ],
            "range": {"ranges": [{"begin": 0, "end": 10}, {"begin": 5, "end": 15}, {"begin": 2, "end": 8}]},
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
        {
            "period": TimeStepSpec([slice(0.0, 0.1, 0.02)])("seconds"),
            "source": [MockSource("density", "custom_filter")],
            "range": RangeSpec[0:10, 5:15, 2:8],
            "file": "output/ions",
            "ext": "bp",
            "infix": "_%06T",
            "json": None,
            "json_restart": None,
            "data_preparation_strategy": "adios",
            "toml": None,
            "particle_io_chunk_size": 256,
            "file_writing": "append",
        },
        (20, 30, 40),
        0.001,
        200,
        {
            "period": {"specs": [{"start": 0, "stop": 100, "step": 20}]},
            "source": [{"name": "density", "filter": "custom_filter"}],
            "range": {"ranges": [{"begin": 0, "end": 10}, {"begin": 5, "end": 15}, {"begin": 2, "end": 8}]},
            "file": "output/ions",
            "ext": "bp",
            "infix": "_%06T",
            "json": {},
            "json_restart": {},
            "data_preparation_strategy": "adios",
            "toml": None,
            "particle_io_chunk_size": 256,
            "file_writing": "append",
        },
    ),
]

TESTCASES_INVALID = [
    (
        {"period": TimeStepSpec([slice(0, 10, 1)]), "particle_io_chunk_size": 0},
        r"particle_io_chunk_size \(in MiB\) must be positive",
    ),
    (
        {"period": TimeStepSpec([slice(0, 10, 1)]), "ext": "sst", "infix": "_%06T"},
        r"infix must be 'NULL' when ext is 'sst'",
    ),
    (
        {"period": TimeStepSpec([slice(0, 10, 1)]), "ext": "invalid"},
        r"argument \"ext\" \(str\) did not match any element in the union:",
        typeguard.TypeCheckError,
    ),
    (
        {"period": TimeStepSpec([slice(0, 10, 1)]), "file_writing": "invalid"},
        r"argument \"file_writing\" \(str\) did not match any element in the union:",
        typeguard.TypeCheckError,
    ),
    (
        {"period": TimeStepSpec([slice(0, 10, 1)]), "data_preparation_strategy": "invalid"},
        r"argument \"data_preparation_strategy\" \(str\) did not match any element in the union:",
        typeguard.TypeCheckError,
    ),
    (
        {"period": "invalid"},
        r"argument \"period\" \(str\) is not an instance of picongpu\.picmi\.diagnostics\.timestepspec\.TimeStepSpec",
        typeguard.TypeCheckError,
    ),
    (
        {"period": TimeStepSpec([slice(0, 10, 1)]), "range": "invalid"},
        r"argument \"range\" \(str\) did not match any element in the union:",
        typeguard.TypeCheckError,
    ),
]

TESTCASES_INVALID_TIMESTEPS = [
    (
        {"period": TimeStepSpec([slice(0, 10, -1)])},
        r"Step size must be >= 1",
    ),
]

TESTCASES_INVALID_MAPPING = [
    (
        {
            "period": TimeStepSpec([slice(0, 10, 1)]),
            "source": [BoundElectronDensity(species=PICMISpecies(name="electrons"), filter="species_all")],
            "range": RangeSpec[:, :],
        },
        {},
        r"Species .* is not known to Simulation",
    ),
    (
        {
            "period": TimeStepSpec([slice(0, 10, 1)]),
            "source": [BoundElectronDensity(species=PICMISpecies(name="electrons"), filter="species_all")],
            "range": RangeSpec[:, :],
        },
        {PICMISpecies(name="ions"): (lambda s=PyPIConGPUSpecies(): (setattr(s, "name", "ions"), s)[1])()},
        r"Species .* is not known to Simulation",
    ),
]

TESTCASES_INVALID_SIMBOX = [
    (
        {"period": TimeStepSpec([slice(0, 10, 1)]), "range": RangeSpec[slice(0, 10), slice(5, 15)]},
        (20,),
        r"Number of range specifications must match simulation box dimensions",
    ),
    (
        {"period": TimeStepSpec([slice(0, 10, 1)]), "range": RangeSpec[slice(0, 10), slice(5, 15)]},
        (20, 30, 40),
        r"Number of range specifications must match simulation box dimensions",
    ),
]


class PICMI_TestOpenPMD(unittest.TestCase):
    def test_openpmd_instantiation(self):
        """Test OpenPMD instantiation and validation."""
        for params, sim_box, time_step_size, num_steps, _ in TESTCASES_VALID:
            with self.subTest(params=params, sim_box=sim_box):
                openpmd = OpenPMD(**params)
                openpmd.check()
                self.assertIsInstance(openpmd.period, TimeStepSpec)
                self.assertIsInstance(openpmd.range, RangeSpec)
                if params["source"] is not None:
                    self.assertTrue(all(isinstance(s, SourceBase) for s in params["source"]))
                for key, value in params.items():
                    if key not in ["period", "source", "range"]:
                        expected_value = {} if value is None and key in ("json", "json_restart") else value
                        self.assertEqual(getattr(openpmd, key), expected_value)
                    elif key == "source" and value is not None:
                        self.assertEqual(len(openpmd.source), len(value))
                    elif key == "range":
                        self.assertEqual(len(openpmd.range), len(value))

        for params, expected_error, *extra_exceptions in TESTCASES_INVALID:
            with self.subTest(params=params, expected_error=expected_error):
                exceptions = (ValueError, TypeError) + tuple(extra_exceptions or [])
                with self.assertRaisesRegex(exceptions, expected_error):
                    openpmd = OpenPMD(**params)
                    openpmd.check()

    def test_openpmd_serialization(self):
        """Test OpenPMD serialization to PyPIConGPUOpenPMD."""
        self.maxDiff = None
        for params, sim_box, time_step_size, num_steps, expected_serialized in TESTCASES_VALID:
            with self.subTest(params=params, sim_box=sim_box):
                openpmd = OpenPMD(**params)
                pypicongpu_openpmd = openpmd.get_as_pypicongpu({}, time_step_size, num_steps, sim_box)
                self.assertIsInstance(pypicongpu_openpmd, PyPIConGPUOpenPMD)
                serialized = pypicongpu_openpmd._get_serialized()
                self.assertEqual(serialized, expected_serialized)

    def test_openpmd_invalid_timestepspec(self):
        """Test invalid TimeStepSpec with negative steps."""
        for params, expected_error in TESTCASES_INVALID_TIMESTEPS:
            with self.subTest(params=params, expected_error=expected_error):
                with self.assertRaisesRegex(ValueError, expected_error):
                    openpmd = OpenPMD(**params)
                    openpmd.check()

    def test_openpmd_invalid_species_mapping(self):
        """Test invalid species mapping for BoundElectronDensity."""
        for params, mapping, expected_error in TESTCASES_INVALID_MAPPING:
            with self.subTest(params=params, mapping=mapping, expected_error=expected_error):
                openpmd = OpenPMD(**params)
                sim_box = (20, 30)  # Match RangeSpec[:, :] dimensions
                with self.assertRaisesRegex(ValueError, expected_error):
                    openpmd.get_as_pypicongpu(mapping, 0.001, 100, sim_box)

    def test_openpmd_invalid_simulation_box(self):
        """Test invalid simulation box dimensions."""
        for params, sim_box, expected_error in TESTCASES_INVALID_SIMBOX:
            with self.subTest(params=params, sim_box=sim_box, expected_error=expected_error):
                openpmd = OpenPMD(**params)
                with self.assertRaisesRegex(ValueError, expected_error):
                    openpmd.get_as_pypicongpu({}, 0.001, 100, sim_box)

    def test_openpmd_invalid_species_instantiation(self):
        """Test invalid species instantiation for BoundElectronDensity."""
        params = {"period": TimeStepSpec([slice(0, 10, 1)])}
        with self.subTest(params=params):
            with self.assertRaisesRegex(
                typeguard.TypeCheckError,
                r"argument \"species\" \(str\) is not an instance of picongpu\.picmi\.species\.Species",
            ):
                openpmd = OpenPMD(**params, source=[BoundElectronDensity(species="invalid", filter="species_all")])
                openpmd.check()

    def test_range_spec_len(self):
        """Test RangeSpec length property."""
        r1 = RangeSpec[0:10]
        self.assertEqual(len(r1), 1)
        r2 = RangeSpec[0:10, 5:15]
        self.assertEqual(len(r2), 2)
        r3 = RangeSpec[0:10, 5:15, 2:8]
        self.assertEqual(len(r3), 3)


if __name__ == "__main__":
    unittest.main()
