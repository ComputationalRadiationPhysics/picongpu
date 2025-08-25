"""
This file is part of PIConGPU.
Copyright 2025 PIConGPU contributors
Authors: Julian Lenz, Masoud Afshari
License: GPLv3+
"""

from picongpu.pypicongpu.output.timestepspec import TimeStepSpec
from picongpu.pypicongpu.output import PhaseSpace
from picongpu.pypicongpu.species import Species
from picongpu.pypicongpu.species.attribute import Position, Momentum
import unittest
import typeguard


def create_species():
    species = Species()
    species.name = "electron"
    species.attributes = [Position(), Momentum()]
    species.constants = []
    return species


class TestPhaseSpace(unittest.TestCase):
    def setUp(self):
        self.species = create_species()

    def test_instantiation_valid(self):
        """Test instantiation and validation for valid inputs."""
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
                    "period": TimeStepSpec([]),
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
                ps = PhaseSpace()
                for key, value in params.items():
                    setattr(ps, key, value)
                for key, value in params.items():
                    self.assertEqual(getattr(ps, key), value)
                if warning_msg:
                    with self.assertWarnsRegex(UserWarning, warning_msg):
                        ps._get_serialized()
                else:
                    ps._get_serialized()

    def test_types(self):
        """Type safety is ensured."""
        ps = PhaseSpace()

        invalid_species = ["string", 1, 1.0, None, {}]
        for invalid in invalid_species:
            with self.assertRaises(typeguard.TypeCheckError):
                ps.species = invalid

        invalid_periods = [13.2, [], "2", None, {}]
        for invalid in invalid_periods:
            with self.assertRaises(typeguard.TypeCheckError):
                ps.period = invalid

        invalid_spatial_coordinates = ["a", "b", "c", (1,), None, {}]
        for invalid in invalid_spatial_coordinates:
            with self.assertRaises(typeguard.TypeCheckError):
                ps.spatial_coordinate = invalid

        invalid_momentum_coordinates = ["a", "b", "c", (1,), None, {}]
        for invalid in invalid_momentum_coordinates:
            with self.assertRaises(typeguard.TypeCheckError):
                ps.momentum_coordinate = invalid

        invalid_min_momentum = ["string", (1,), None, {}]
        for invalid in invalid_min_momentum:
            with self.assertRaises(typeguard.TypeCheckError):
                ps.min_momentum = invalid

        invalid_max_momentum = ["string", (1,), None, {}]
        for invalid in invalid_max_momentum:
            with self.assertRaises(typeguard.TypeCheckError):
                ps.max_momentum = invalid

        # Valid case
        ps.species = self.species
        ps.period = TimeStepSpec([slice(0, None, 17)])
        ps.spatial_coordinate = "x"
        ps.momentum_coordinate = "px"
        ps.min_momentum = 0.0
        ps.max_momentum = 1.0
        ps._get_serialized()

    def test_rendering(self):
        """Data transformed to template-consumable version."""
        ps = PhaseSpace()
        ps.species = self.species
        ps.period = TimeStepSpec([slice(0, None, 42)])
        ps.spatial_coordinate = "x"
        ps.momentum_coordinate = "px"
        ps.min_momentum = 0.0
        ps.max_momentum = 1.0

        context = ps.get_rendering_context()
        self.assertTrue(context["typeID"]["phasespace"])
        context = context["data"]
        self.assertEqual(42, context["period"]["specs"][0]["step"])
        self.assertEqual(0, context["period"]["specs"][0]["start"])
        self.assertEqual("x", context["spatial_coordinate"])
        self.assertEqual("px", context["momentum_coordinate"])
        self.assertEqual(0.0, context["min_momentum"])
        self.assertEqual(1.0, context["max_momentum"])

        # Empty period
        ps.period = TimeStepSpec([])
        with self.assertWarnsRegex(UserWarning, "PhaseSpace is disabled"):
            ps.get_rendering_context()

        # Invalid attributes
        ps = PhaseSpace()
        with self.assertRaises(Exception):
            ps.get_rendering_context()

    def test_momentum_values(self):
        """Min_momentum and max_momentum values are valid."""
        ps = PhaseSpace()
        ps.species = self.species
        ps.period = TimeStepSpec([slice(0, None, 1)])
        ps.spatial_coordinate = "x"
        ps.momentum_coordinate = "px"
        ps.min_momentum = 2.0
        ps.max_momentum = 1.0

        with self.assertRaises(ValueError):
            ps.check()

        with self.assertRaises(ValueError):
            ps._get_serialized()


if __name__ == "__main__":
    unittest.main()
