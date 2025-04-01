"""
This file is part of PIConGPU.
Copyright 2025 PIConGPU contributors
Authors: Julian Lenz
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
    def test_empty(self):
        """empty args handled correctly"""
        ps = PhaseSpace()
        # unset args
        with self.assertRaises(Exception):
            ps._get_serialized()

        ps.species = create_species()
        ps.period = TimeStepSpec([slice(0, None, 17)])
        ps.spatial_coordinate = "x"
        ps.momentum_coordinate = "px"
        ps.min_momentum = 0.0
        ps.max_momentum = 1.0

        # ok:
        ps._get_serialized()

    def test_types(self):
        """type safety is ensured"""
        ps = PhaseSpace()

        invalid_species = ["string", 1, 1.0, None, {}]
        for invalid_species_ in invalid_species:
            with self.assertRaises(typeguard.TypeCheckError):
                ps.species = invalid_species_

        invalid_periods = [13.2, [], "2", None, {}]
        for invalid_period in invalid_periods:
            with self.assertRaises(typeguard.TypeCheckError):
                ps.period = invalid_period

        invalid_spatial_coordinates = ["a", "b", "c", (1,), None, {}]
        for invalid_spatial_coordinate in invalid_spatial_coordinates:
            with self.assertRaises(typeguard.TypeCheckError):
                ps.spatial_coordinate = invalid_spatial_coordinate

        invalid_momentum_coordinates = ["a", "b", "c", (1,), None, {}]
        for invalid_momentum_coordinate in invalid_momentum_coordinates:
            with self.assertRaises(typeguard.TypeCheckError):
                ps.momentum_coordinate = invalid_momentum_coordinate

        invalid_min_momentum = ["string", (1,), None, {}]
        for invalid_min_momentum_ in invalid_min_momentum:
            with self.assertRaises(typeguard.TypeCheckError):
                ps.min_momentum = invalid_min_momentum_

        invalid_max_momentum = ["string", (1,), None, {}]
        for invalid_max_momentum_ in invalid_max_momentum:
            with self.assertRaises(typeguard.TypeCheckError):
                ps.max_momentum = invalid_max_momentum_

        # ok
        ps.species = create_species()
        ps.period = TimeStepSpec([slice(0, None, 17)])
        ps.spatial_coordinate = "x"
        ps.momentum_coordinate = "px"
        ps.min_momentum = 0.0
        ps.max_momentum = 1.0

    def test_rendering(self):
        """data transformed to template-consumable version"""
        ps = PhaseSpace()
        ps.species = create_species()
        ps.period = TimeStepSpec([slice(0, None, 42)])
        ps.spatial_coordinate = "x"
        ps.momentum_coordinate = "px"
        ps.min_momentum = 0.0
        ps.max_momentum = 1.0

        # normal rendering
        context = ps.get_rendering_context()
        self.assertTrue(context["typeID"]["phasespace"])
        context = context["data"]
        self.assertEqual(42, context["period"]["specs"][0]["step"])
        self.assertEqual("x", context["spatial_coordinate"])
        self.assertEqual("px", context["momentum_coordinate"])
        self.assertEqual(0.0, context["min_momentum"])
        self.assertEqual(1.0, context["max_momentum"])

        # refuses to render if attributes are not set
        ps = PhaseSpace()
        with self.assertRaises(Exception):
            ps.get_rendering_context()

    def test_momentum_values(self):
        """min_momentum and max_momentum values are valid"""
        ps = PhaseSpace()
        ps.species = create_species()
        ps.period = TimeStepSpec([slice(0, None, 1)])
        ps.spatial_coordinate = "x"
        ps.momentum_coordinate = "px"

        # Min is larger than max, that's not allowed
        ps.min_momentum = 2.0
        ps.max_momentum = 1.0

        with self.assertRaises(ValueError):
            ps.check()

        # get_rendering_context calls check internally, so this should also fail:
        with self.assertRaises(ValueError):
            ps.get_rendering_context()
