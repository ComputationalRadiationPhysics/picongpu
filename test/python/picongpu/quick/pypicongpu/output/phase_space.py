"""
This file is part of PIConGPU.
Copyright 2025 PIConGPU contributors
Authors: Julian Lenz, Masoud Afshari
License: GPLv3+
"""

from picongpu.pypicongpu.output import PhaseSpace
from picongpu.pypicongpu.output.timestepspec import TimeStepSpec
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
        self.period = TimeStepSpec([slice(0, None, 17)])

    def test_instantiation_and_types(self):
        """Test instantiation, type safety, and valid serialization."""
        # Valid configuration
        ps = PhaseSpace()
        ps.species = self.species
        ps.period = self.period
        ps.spatial_coordinate = "x"
        ps.momentum_coordinate = "px"
        ps.min_momentum = 0.0
        ps.max_momentum = 1.0
        ps.check()
        context = ps.get_rendering_context()
        self.assertTrue(context["typeID"]["phasespace"])
        self.assertEqual(context["data"]["species"]["name"], "electron")
        self.assertEqual(context["data"]["period"]["specs"][0]["step"], 17)
        self.assertEqual(context["data"]["spatial_coordinate"], "x")
        self.assertEqual(context["data"]["momentum_coordinate"], "px")
        self.assertEqual(context["data"]["min_momentum"], 0.0)
        self.assertEqual(context["data"]["max_momentum"], 1.0)

        # Type safety
        invalid_types = {
            "species": ["string", 1],
            "period": ["string", 1],
            "spatial_coordinate": ["a", 1],
            "momentum_coordinate": ["b", 1],
            "min_momentum": ["string", []],
            "max_momentum": ["string", []],
        }
        for attr, invalid_values in invalid_types.items():
            for value in invalid_values:
                with self.subTest(attr=attr, value=value):
                    ps = PhaseSpace()
                    with self.assertRaises(typeguard.TypeCheckError):
                        setattr(ps, attr, value)

    def test_rendering_and_validation(self):
        """Test serialization output, validation errors, and disabled state."""
        # Valid serialization
        ps = PhaseSpace()
        ps.species = self.species
        ps.period = TimeStepSpec([slice(0, None, 42)])
        ps.spatial_coordinate = "z"
        ps.momentum_coordinate = "pz"
        ps.min_momentum = 0.0
        ps.max_momentum = 2.0
        context = ps.get_rendering_context()
        self.assertTrue(context["typeID"]["phasespace"])
        self.assertEqual(context["data"]["period"]["specs"][0]["step"], 42)
        self.assertEqual(context["data"]["spatial_coordinate"], "z")
        self.assertEqual(context["data"]["momentum_coordinate"], "pz")
        self.assertEqual(context["data"]["min_momentum"], 0.0)
        self.assertEqual(context["data"]["max_momentum"], 2.0)

        # Empty period warning
        ps.period = TimeStepSpec([])
        with self.assertWarnsRegex(UserWarning, "PhaseSpace is disabled"):
            ps.get_rendering_context()

        # Validation error
        ps = PhaseSpace()
        ps.species = self.species
        ps.period = self.period
        ps.spatial_coordinate = "x"
        ps.momentum_coordinate = "px"
        ps.min_momentum = 2.0
        ps.max_momentum = 1.0
        with self.assertRaisesRegex(ValueError, "min_momentum should be smaller than max_momentum"):
            ps.get_rendering_context()

        # Invalid attributes (low-level check)
        ps = PhaseSpace()
        with self.assertRaises(Exception):
            ps._get_serialized()


if __name__ == "__main__":
    unittest.main()
