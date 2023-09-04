"""
This file is part of the PIConGPU.
Copyright 2021-2023 PIConGPU contributors
Authors: Hannes Troepgen, Brian Edward Marre
License: GPLv3+
"""

from picongpu.pypicongpu.species.operation import SetBoundElectrons

import unittest

from picongpu.pypicongpu.species import Species
from picongpu.pypicongpu.species.constant import Ionizers
from picongpu.pypicongpu.species.attribute import \
    BoundElectrons, Position, Momentum


class TestSetBoundElectrons(unittest.TestCase):
    def setUp(self):
        self.species1 = Species()
        self.species1.name = "ion"
        self.species1.constants = [Ionizers()]

    def test_basic(self):
        """basic operation"""
        sbe = SetBoundElectrons()
        sbe.species = self.species1
        sbe.bound_electrons = 2

        # checks pass
        sbe.check_preconditions()

    def test_typesafety(self):
        """typesafety is ensured"""
        sbe = SetBoundElectrons()
        for invalid_species in [None, 1, "a", []]:
            with self.assertRaises(TypeError):
                sbe.species = invalid_species

        for invalid_number in [None, "a", [], self.species1, 2.3]:
            with self.assertRaises(TypeError):
                sbe.bound_electrons = invalid_number

        # works:
        sbe.species = self.species1
        sbe.bound_electrons = 1

    def test_empty(self):
        """all parameters are mandatory"""
        for set_species in [True, False]:
            for set_bound_electrons in [True, False]:
                sbe = SetBoundElectrons()

                if set_species:
                    sbe.species = self.species1
                if set_bound_electrons:
                    sbe.bound_electrons = 1

                if set_species and set_bound_electrons:
                    # must pass
                    sbe.check_preconditions()
                else:
                    # mandatory missing -> must raise
                    with self.assertRaises(Exception):
                        sbe.check_preconditions()

    def test_attribute_generated(self):
        """creates bound electrons attribute"""
        sbe = SetBoundElectrons()
        sbe.species = self.species1
        sbe.bound_electrons = 1

        # emulate initmanager
        sbe.check_preconditions()
        self.species1.attributes = []
        sbe.prebook_species_attributes()

        self.assertEqual(1, len(sbe.attributes_by_species))
        self.assertTrue(self.species1 in sbe.attributes_by_species)
        self.assertEqual(1, len(sbe.attributes_by_species[self.species1]))
        self.assertTrue(isinstance(sbe.attributes_by_species[self.species1][0],
                                   BoundElectrons))

    def test_ionizers_required(self):
        """ionizers constant must be present"""
        sbe = SetBoundElectrons()
        sbe.species = self.species1
        sbe.bound_electrons = 1

        # passes:
        self.assertTrue(sbe.species.has_constant_of_type(Ionizers))
        sbe.check_preconditions()

        # without constants does not pass:
        sbe.species.constants = []
        with self.assertRaisesRegex(AssertionError, ".*[Ii]onizers.*"):
            sbe.check_preconditions()

    def test_values(self):
        """bound electrons must be >0"""
        sbe = SetBoundElectrons()
        sbe.species = self.species1

        with self.assertRaisesRegex(ValueError, ".*>0.*"):
            sbe.bound_electrons = -1
            sbe.check_preconditions()

        with self.assertRaisesRegex(ValueError, ".*NoBoundElectrons.*"):
            sbe.bound_electrons = 0
            sbe.check_preconditions()

        # silently passes
        sbe.bound_electrons = 1
        sbe.check_preconditions()

    def test_rendering(self):
        """rendering works"""
        # create full electron species
        electron = Species()
        electron.name = "e"
        electron.constants = []
        electron.attributes = [Position(), Momentum()]

        # can be rendered:
        self.assertNotEqual({}, electron.get_rendering_context())

        ion = Species()
        ion.name = "ion"
        ionizers_const = Ionizers()
        ionizers_const.electron_species = electron
        ion.constants = [ionizers_const]
        ion.attributes = [Position(), Momentum(), BoundElectrons()]

        # can be rendered
        self.assertNotEqual({}, ion.get_rendering_context())

        sbe = SetBoundElectrons()
        sbe.species = ion
        sbe.bound_electrons = 1

        context = sbe.get_rendering_context()
        self.assertEqual(1, context["bound_electrons"])
        self.assertEqual(ion.get_rendering_context(),
                         context["species"])
