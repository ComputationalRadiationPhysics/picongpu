"""
This file is part of PIConGPU.
Copyright 2021-2023 PIConGPU contributors
Authors: Hannes Troepgen, Brian Edward Marre
License: GPLv3+
"""

from picongpu.pypicongpu.species.operation import NotPlaced

import unittest
import typeguard

from picongpu.pypicongpu.species import Species
from picongpu.pypicongpu.species.attribute import Position, Weighting


class TestNotPlaced(unittest.TestCase):
    def setUp(self):
        self.species1 = Species()
        self.species1.name = "species1"
        self.species1.constants = []

    def test_not_rendering_context(self):
        """get_rendering_context raises"""
        np = NotPlaced()
        np.species = self.species1

        # check okay:
        np.check_preconditions()

        # but can't be represented as context either way:
        with self.assertRaises(RuntimeError):
            np.get_rendering_context()

    def test_types(self):
        """typesafety is ensured"""
        np = NotPlaced()

        for invalid_species in ["s", [], {"species1"}, 1, None, {}]:
            with self.assertRaises(typeguard.TypeCheckError):
                np.species = invalid_species

    def test_empty(self):
        """at least one species is required"""
        np = NotPlaced()

        # nothing set at all -> raises
        with self.assertRaises(Exception):
            np.check_postconditions()

    def test_position_added(self):
        """provides position attribute"""
        np = NotPlaced()
        np.species = self.species1

        np.check_preconditions()
        np.prebook_species_attributes()

        self.assertEqual(1, len(np.attributes_by_species))

        attributes = np.attributes_by_species[np.species]
        self.assertEqual(2, len(attributes))
        attr_types = set(map(lambda attr: type(attr), attributes))
        self.assertEqual({Position, Weighting}, attr_types)
