"""
This file is part of PIConGPU.
Copyright 2025 PIConGPU contributors
Authors: Masoud Afshari
License: GPLv3+
"""

from picongpu.picmi.diagnostics.timestepspec import TimeStepSpec
from picongpu.picmi.diagnostics.rangespec import RangeSpec
from picongpu.picmi.diagnostics.openpmd import OpenPMD
from picongpu.picmi.diagnostics.openpmd_sources.source_base import SourceBase
from picongpu.pypicongpu.output.openpmd_sources.source_base import SourceBase as PySourceBase
from picongpu.picmi.diagnostics.openpmd_sources.sources import BoundElectronDensity, EnergyDensityCutoff, Momentum
from picongpu.pypicongpu.output.openpmd import OpenPMD as PyPIConGPUOpenPMD
from picongpu.pypicongpu.species import Species as PyPIConGPUSpecies
from picongpu.pypicongpu.species.attribute import Position, Momentum as PyMomentum
from picongpu.picmi.species import Species as PICMISpecies
import unittest
import typeguard


def create_picmi_species():
    species = PICMISpecies()
    species.name = "electrons"
    return species


def create_pypicongpu_species():
    species = PyPIConGPUSpecies()
    species.name = "electrons"
    species.attributes = [Position(), PyMomentum()]
    species.constants = []
    return species


class MockPySource(PySourceBase):
    def __init__(self, name="mock", filter_value="species_all"):
        self.name = name
        self._filter = filter_value

    def check(self):
        pass

    def _get_serialized(self):
        return {"name": self.name, "filter": self._filter, "type": "mock"}

    @property
    def filter(self):
        return self._filter


class MockSource(SourceBase):
    def __init__(self, name="mock", filter_value="species_all"):
        self.name = name
        self._filter = filter_value

    def check(self):
        pass

    def get_as_pypicongpu(self, _):
        return MockPySource(self.name, self._filter)

    def _get_serialized(self):
        return {"name": self.name, "filter": self._filter, "type": "mock"}

    @property
    def filter(self):
        return self._filter


class PICMI_TestOpenPMD(unittest.TestCase):
    def setUp(self):
        self.species = create_picmi_species()
        self.pypicongpu_species = create_pypicongpu_species()
        self.species_map = {self.species: self.pypicongpu_species}

    def test_openpmd(self):
        """Test OpenPMD instantiation, serialization, and RangeSpec length."""
        species_context = {
            "name": "electrons",
            "typename": "species_electrons",
            "attributes": [{"picongpu_name": "position<position_pic>"}, {"picongpu_name": "momentum"}],
            "constants": {
                "mass": None,
                "charge": None,
                "density_ratio": None,
                "element_properties": None,
                "ground_state_ionization": None,
            },
        }
        TESTCASES_VALID = [
            (
                {"period": TimeStepSpec([slice(0, 100, 2)]), "range": RangeSpec[:, :, :], "ext": "bp"},
                (20, 30, 40),
                0.001,
                1000,
                {
                    "period": {"specs": [{"start": 0, "stop": 100, "step": 2}]},
                    "source": None,
                    "range": {"ranges": [{"begin": 0, "end": 19}, {"begin": 0, "end": 29}, {"begin": 0, "end": 39}]},
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
            ),
            (
                {
                    "period": TimeStepSpec([slice(0, None, 1)]),
                    "source": [
                        MockSource("density", "custom_filter"),
                        EnergyDensityCutoff(self.species, "species_all", 1e6),
                        Momentum(self.species, "species_all", "x"),
                    ],
                    "range": RangeSpec[0:10, 5:15, 2:8],
                    "file": "output/data",
                    "ext": "h5",
                },
                (20, 30, 40),
                0.001,
                1000,
                {
                    "period": {"specs": [{"start": 0, "stop": 999, "step": 1}]},
                    "source": [
                        {"name": "density", "filter": "custom_filter", "type": "mock"},
                        {
                            "species": species_context,
                            "filter": "species_all",
                            "cutoff_max_energy": 1e6,
                            "type": "energydensitycutoff",
                        },
                        {"species": species_context, "filter": "species_all", "direction": "x", "type": "momentum"},
                    ],
                    "range": {"ranges": [{"begin": 0, "end": 10}, {"begin": 5, "end": 15}, {"begin": 2, "end": 8}]},
                    "file": "output/data",
                    "ext": "h5",
                    "infix": "NULL",
                    "json": None,
                    "json_restart": None,
                    "data_preparation_strategy": None,
                    "toml": None,
                    "particle_io_chunk_size": None,
                    "file_writing": "create",
                },
            ),
        ]

        for params, sim_box, time_step_size, num_steps, expected in TESTCASES_VALID:
            with self.subTest(params=params):
                openpmd = OpenPMD(**params)
                openpmd.check()
                self.assertIsInstance(openpmd.period, TimeStepSpec)
                self.assertIsInstance(openpmd.range, RangeSpec)
                self.assertEqual(len(openpmd.range), len(sim_box))
                if params.get("source"):
                    self.assertTrue(all(isinstance(s, SourceBase) for s in params["source"]))
                pypicongpu_openpmd = openpmd.get_as_pypicongpu(self.species_map, time_step_size, num_steps, sim_box)
                self.assertIsInstance(pypicongpu_openpmd, PyPIConGPUOpenPMD)
                self.assertEqual(pypicongpu_openpmd._get_serialized(), expected)

    def test_openpmd_invalid(self):
        """Test invalid OpenPMD inputs, timesteps, mapping, and simulation box."""
        TESTCASES_INVALID = [
            (
                {"period": TimeStepSpec([slice(0, 10, 1)]), "particle_io_chunk_size": 0},
                r"particle_io_chunk_size \(in MiB\) must be positive",
                ValueError,
            ),
            (
                {"period": TimeStepSpec([slice(0, 10, 1)]), "ext": "invalid"},
                r'argument "ext" \(str\) did not match any element in the union:',
                typeguard.TypeCheckError,
            ),
            (
                {"period": "invalid"},
                r'argument "period" \(str\) is not an instance of.*TimeStepSpec',
                typeguard.TypeCheckError,
            ),
            ({"period": TimeStepSpec([slice(0, 10, -1)])}, r"Step size must be >= 1", ValueError),
            (
                {
                    "period": TimeStepSpec([slice(0, 10, 1)]),
                    "source": [BoundElectronDensity(species=self.species, filter="species_all")],
                },
                r"Species .* is not known to Simulation",
                ValueError,
            ),
            (
                {"period": TimeStepSpec([slice(0, 10, 1)]), "range": RangeSpec[0:10, 5:15]},
                r"Number of range specifications must match simulation box dimensions",
                ValueError,
            ),
            (
                {
                    "period": TimeStepSpec([slice(0, 10, 1)]),
                    "source": lambda: [EnergyDensityCutoff(self.species, "species_all", -1)],
                },
                r"cutoff_max_energy must be positive",
                ValueError,
            ),
        ]

        for params, error, exception in TESTCASES_INVALID:
            with self.subTest(params=params, error=error):
                with self.assertRaisesRegex(exception, error):
                    if callable(params.get("source")):
                        params = dict(params, source=params["source"]())
                    openpmd = OpenPMD(**params)
                    sim_box = (
                        (20,)
                        if "range" in params and isinstance(params["range"], RangeSpec) and len(params["range"]) == 2
                        else (20, 30, 40)
                    )
                    mapping = {} if error.startswith("Species") else self.species_map
                    openpmd.get_as_pypicongpu(mapping, 0.001, 100, sim_box)


if __name__ == "__main__":
    unittest.main()
