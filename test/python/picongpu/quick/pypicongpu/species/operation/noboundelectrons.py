"""
This file is part of the PIConGPU.
Copyright 2021-2023 PIConGPU contributors
Authors: Hannes Troepgen, Brian Edward Marre
License: GPLv3+
"""

from picongpu.pypicongpu.species.operation import NoBoundElectrons

import unittest

from picongpu.pypicongpu.species import Species
from picongpu.pypicongpu.species.constant import Ionizers
from picongpu.pypicongpu.species.attribute import BoundElectrons


class TestNoBoundElectrons(unittest.TestCase):
    def setUp(self):
        self.species1 = Species()
        self.species1.name = "ion"
        self.species1.constants = [Ionizers()]

    def test_no_rendering_context(self):
        """results in no rendered code, hence no rendering context available"""
        # works:
        nbe = NoBoundElectrons()
        nbe.species = self.species1
        nbe.check_preconditions()

        with self.assertRaises(RuntimeError):
            nbe.get_rendering_context()

    def test_types(self):
        """typesafety is ensured"""
        nbe = NoBoundElectrons()
        for invalid_species in ["x", 0, None, []]:
            with self.assertRaises(TypeError):
                nbe.species = invalid_species

        # works:
        nbe.species = self.species1

    def test_ionizers_required(self):
        """species must have ionizers constant"""
        nbe = NoBoundElectrons()
        nbe.species = self.species1

        self.assertTrue(self.species1.has_constant_of_type(Ionizers))

        # passes
        nbe.check_preconditions()

        # remove constant:
        self.species1.constants = []

        # now raises b/c ionizers constant is missing
        with self.assertRaisesRegex(AssertionError, ".*[Ii]onizers.*"):
            nbe.check_preconditions()

    def test_empty(self):
        """species is mandatory"""
        nbe = NoBoundElectrons()
        with self.assertRaises(Exception):
            nbe.check_preconditions()

        nbe.species = self.species1
        # now works:
        nbe.check_preconditions()

    def test_bound_electrons_attr_added(self):
        """adds attribute BoundElectrons"""
        nbe = NoBoundElectrons()
        nbe.species = self.species1

        # emulate initmanager behavior
        self.species1.attributes = []
        nbe.check_preconditions()
        nbe.prebook_species_attributes()

        self.assertTrue(self.species1 in nbe.attributes_by_species)
        self.assertEqual(1, len(nbe.attributes_by_species))

        self.assertEqual(1, len(nbe.attributes_by_species[self.species1]))
        self.assertTrue(isinstance(nbe.attributes_by_species[self.species1][0],
                                   BoundElectrons))
