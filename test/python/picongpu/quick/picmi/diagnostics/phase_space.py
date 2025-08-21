"""
This file is part of PIConGPU.
Copyright 2025 PIConGPU contributors
Authors: Masoud Afshari
License: GPLv3+
"""

from picongpu.picmi.diagnostics import PhaseSpace, TimeStepSpec
from picongpu.picmi.species import Species as PICMISpecies
from picongpu.pypicongpu.species import Species as PyPIConGPUSpecies
from picongpu.pypicongpu.species.attribute import Position, Momentum

import unittest
import typeguard


def create_picmi_species():
    species = PICMISpecies()
    species.name = "electron"
    return species


def create_pypicongpu_species():
    species = PyPIConGPUSpecies()
    species.name = "electron"
    species.attributes = [Position(), Momentum()]
    species.constants = []
    return species


class PICMI_TestPhaseSpace(unittest.TestCase):
    def setUp(self):
        self.species = create_picmi_species()
        self.pypicongpu_species = create_pypicongpu_species()
        self.species_map = {self.species: self.pypicongpu_species}

    def test_instantiation_valid(self):
        TESTCASES_VALID = [
            (
                {
                    "species": self.species,
                    "period": TimeStepSpec([slice(0, None, 17)]),
                    "spatial_coordinate": "x",
                    "momentum_coordinate": "px",
                    "min_momentum": 0.0,
                    "max_momentum": 1.0,
                },
                None,
            ),
            (
                {
                    "species": self.species,
                    "period": 10,
                    "spatial_coordinate": "y",
                    "momentum_coordinate": "py",
                    "min_momentum": -1.0,
                    "max_momentum": 1.0,
                },
                None,
            ),
            (
                {
                    "species": self.species,
                    "period": 0,
                    "spatial_coordinate": "z",
                    "momentum_coordinate": "pz",
                    "min_momentum": 0.0,
                    "max_momentum": 2.0,
                },
                "PhaseSpace is disabled",
            ),
        ]

        for params, warning_msg in TESTCASES_VALID:
            with self.subTest(params=params):
                ps = PhaseSpace(**params)
                for key, value in params.items():
                    if key == "period" and isinstance(value, int):
                        expected = (
                            TimeStepSpec([slice(None, None, value)])("steps")
                            if value > 0
                            else TimeStepSpec([])("steps")
                        )
                        expected_context = expected.get_as_pypicongpu(0.5, 200).get_rendering_context()
                        self.assertEqual(
                            ps.period.get_as_pypicongpu(0.5, 200).get_rendering_context(),
                            expected_context,
                        )
                    else:
                        self.assertEqual(getattr(ps, key), value)
                if warning_msg:
                    with self.assertWarnsRegex(UserWarning, warning_msg):
                        ps.get_as_pypicongpu(self.species_map, 0.5, 200)
                else:
                    ps.get_as_pypicongpu(self.species_map, 0.5, 200)

    def test_types(self):
        invalid_species = ["string", 1, 1.0, None, {}]
        for invalid in invalid_species:
            with self.assertRaises(typeguard.TypeCheckError):
                PhaseSpace(
                    species=invalid,
                    period=TimeStepSpec([slice(0, None, 1)]),
                    spatial_coordinate="x",
                    momentum_coordinate="px",
                    min_momentum=0.0,
                    max_momentum=1.0,
                )

        invalid_periods = [[], "2", {}]
        for invalid in invalid_periods:
            with self.assertRaises(typeguard.TypeCheckError):
                PhaseSpace(
                    species=self.species,
                    period=invalid,
                    spatial_coordinate="x",
                    momentum_coordinate="px",
                    min_momentum=0.0,
                    max_momentum=1.0,
                )

        invalid_spatial = ["a", "b", "c", (1,), None, {}]
        for invalid in invalid_spatial:
            with self.assertRaises(typeguard.TypeCheckError):
                PhaseSpace(
                    species=self.species,
                    period=TimeStepSpec([slice(0, None, 1)]),
                    spatial_coordinate=invalid,
                    momentum_coordinate="px",
                    min_momentum=0.0,
                    max_momentum=1.0,
                )

        invalid_momentum = ["a", "b", "c", (1,), None, {}]
        for invalid in invalid_momentum:
            with self.assertRaises(typeguard.TypeCheckError):
                PhaseSpace(
                    species=self.species,
                    period=TimeStepSpec([slice(0, None, 1)]),
                    spatial_coordinate="x",
                    momentum_coordinate=invalid,
                    min_momentum=0.0,
                    max_momentum=1.0,
                )

        invalid_min_momentum = ["string", (1,), None, {}]
        for invalid in invalid_min_momentum:
            with self.assertRaises(typeguard.TypeCheckError):
                PhaseSpace(
                    species=self.species,
                    period=TimeStepSpec([slice(0, None, 1)]),
                    spatial_coordinate="x",
                    momentum_coordinate="px",
                    min_momentum=invalid,
                    max_momentum=1.0,
                )

        invalid_max_momentum = ["string", (1,), None, {}]
        for invalid in invalid_max_momentum:
            with self.assertRaises(typeguard.TypeCheckError):
                PhaseSpace(
                    species=self.species,
                    period=TimeStepSpec([slice(0, None, 1)]),
                    spatial_coordinate="x",
                    momentum_coordinate="px",
                    min_momentum=0.0,
                    max_momentum=invalid,
                )

    def test_rendering(self):
        ps = PhaseSpace(
            species=self.species,
            period=TimeStepSpec([slice(0, None, 42)]),
            spatial_coordinate="x",
            momentum_coordinate="px",
            min_momentum=0.0,
            max_momentum=1.0,
        )

        pypicongpu_ps = ps.get_as_pypicongpu(self.species_map, 0.5, 200)
        context = pypicongpu_ps.get_rendering_context()
        self.assertTrue(context["typeID"]["phasespace"])
        context = context["data"]
        self.assertEqual(42, context["period"]["specs"][0]["step"])
        self.assertEqual(0, context["period"]["specs"][0]["start"])
        self.assertEqual(199, context["period"]["specs"][0]["stop"])
        self.assertEqual("x", context["spatial_coordinate"])
        self.assertEqual("px", context["momentum_coordinate"])
        self.assertEqual(0.0, context["min_momentum"])
        self.assertEqual(1.0, context["max_momentum"])
        self.assertEqual("electron", context["species"]["name"])

        # Integer period
        ps = PhaseSpace(
            species=self.species,
            period=10,
            spatial_coordinate="x",
            momentum_coordinate="px",
            min_momentum=0.0,
            max_momentum=1.0,
        )
        pypicongpu_ps = ps.get_as_pypicongpu(self.species_map, 0.5, 200)
        context = pypicongpu_ps.get_rendering_context()
        self.assertTrue(context["typeID"]["phasespace"])
        context = context["data"]
        self.assertEqual(10, context["period"]["specs"][0]["step"])
        self.assertEqual(0, context["period"]["specs"][0]["start"])
        self.assertEqual(199, context["period"]["specs"][0]["stop"])

        # Test invalid species mapping
        with self.assertRaises(ValueError):
            ps.get_as_pypicongpu({}, 0.5, 200)

    def test_momentum_values(self):
        ps = PhaseSpace(
            species=self.species,
            period=TimeStepSpec([slice(0, None, 1)]),
            spatial_coordinate="x",
            momentum_coordinate="px",
            min_momentum=2.0,
            max_momentum=1.0,
        )

        with self.assertRaises(ValueError):
            ps.check()

        with self.assertRaises(ValueError):
            ps.get_as_pypicongpu(self.species_map, 0.5, 200)

    def test_period_warning(self):
        ps = PhaseSpace(
            species=self.species,
            period=0,
            spatial_coordinate="x",
            momentum_coordinate="px",
            min_momentum=0.0,
            max_momentum=1.0,
        )

        with self.assertWarnsRegex(UserWarning, "PhaseSpace is disabled"):
            ps.get_as_pypicongpu(self.species_map, 0.5, 200)


if __name__ == "__main__":
    unittest.main()
