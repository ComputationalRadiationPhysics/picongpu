"""
This file is part of PIConGPU.
Copyright 2024 PIConGPU contributors
Authors: Brian Edward Marre
License: GPLv3+
"""

from picongpu.pypicongpu.species.constant.ionizationmodel import IonizationModel

from picongpu.pypicongpu.species import Species
from picongpu.pypicongpu.species.constant import Mass, Charge, ElementProperties, GroundStateIonization
from picongpu.pypicongpu.species.attribute import Position, Momentum, BoundElectrons
from picongpu.picmi import constants

import unittest


# raw implementation for testing
class Implementation(IonizationModel):
    PICONGPU_NAME: str = "test"


class Test_IonizationModel(unittest.TestCase):
    def setUp(self):
        electron = Species()
        electron.name = "e"
        mass_constant = Mass()
        mass_constant.mass_si = constants.m_e
        charge_constant = Charge()
        charge_constant.charge_si = constants.m_e
        electron.constants = [
            charge_constant,
            mass_constant,
        ]
        # note: attributes not set yet (as would be in init manager)

        self.electron = electron

    def test_not_constructible(self):
        with self.assertRaises(Exception):
            IonizationModel()

    def test_basic(self):
        """simple operation"""
        # note: electrons are not checked, because they are not fully
        # initialized yet

        instance = Implementation()
        instance.ionization_electron_species = self.electron
        instance.check()

        self.assertEqual("test", instance.PICONGPU_NAME)

        self.assertEqual([self.electron], instance.get_species_dependencies())
        self.assertEqual([BoundElectrons], instance.get_attribute_dependencies())
        self.assertEqual([ElementProperties], instance.get_constant_dependencies())

    def test_empty(self):
        """electron species is mandatory"""
        instance = Implementation()

        # must fail:
        with self.assertRaises(Exception):
            instance.check()
        with self.assertRaises(Exception):
            instance.get_species_dependencies()

        # now passes
        instance.ionization_electron_species = self.electron
        instance.check()

    def test_typesafety(self):
        """types are checked"""
        instance = Implementation()
        for invalid in ["electron", {}, [], 0, None]:
            with self.assertRaises(TypeError):
                # note: circular imports would be required to use the
                # pypicongpu-standard build_typesafe_property, hence the type
                # is checked by check() instead of on assignment (as usual)
                instance.ionization_electron_species = invalid
                instance.check()

        for invalid in ["ionization_current", {}, [], 0]:
            with self.assertRaises(TypeError):
                # note: circular imports would be required to use the
                # pypicongpu-standard build_typesafe_property, hence the type
                # is checked by check() instead of on assignment (as usual)
                instance.ionization_electron_species = self.electron
                instance.ionization_current = invalid
                instance.check()

    def test_circular_ionization(self):
        """electron species must not be ionizable itself"""
        other_electron = Species()
        other_electron.name = "e"
        mass_constant = Mass()
        mass_constant.mass_si = constants.m_e
        charge_constant = Charge()
        charge_constant.charge_si = constants.m_e
        other_electron.constants = [
            charge_constant,
            mass_constant,
        ]
        # note: attributes not set yet, as would be case in init manager

        instance_transitive_const = Implementation()
        instance_transitive_const.ionization_electron_species = other_electron

        self.electron.constants.append(GroundStateIonization(ionization_model_list=[instance_transitive_const]))

        # original instance is valid
        instance_transitive_const.check()

        # ...but a constant using an ionizable species as electrons must reject
        instance = Implementation()
        instance.ionization_electron_species = self.electron
        with self.assertRaisesRegex(ValueError, ".*ionizable.*"):
            instance.check()

    def test_check_passthru(self):
        """calls check of electron species & checks during rendering"""
        instance = Implementation()
        instance.ionization_electron_species = self.electron

        # both pass:
        instance.check()
        self.assertNotEqual([], instance.get_species_dependencies())

        # with a broken species...
        instance.ionization_electron_species = None
        # ...check()...
        with self.assertRaises(Exception):
            instance.check()

        # ...and get dependencies fail
        with self.assertRaises(Exception):
            instance.get_species_dependencies()

    def test_rendering(self):
        """renders to rendering context"""
        # prepare electron species s.t. it can be rendered
        self.electron.attributes = [Position(), Momentum()]
        # must pass
        self.electron.check()
        self.assertNotEqual({}, self.electron.get_rendering_context())

        instance = Implementation()
        instance.ionization_electron_species = self.electron

        context = instance.get_rendering_context()
        self.assertNotEqual({}, context)
        self.assertEqual(self.electron.get_rendering_context(), context["ionization_electron_species"])

        # do *NOT* render if check() does not pass
        instance.ionization_electron_species = None
        with self.assertRaises(TypeError):
            instance.check()
        with self.assertRaises(TypeError):
            instance.get_rendering_context()

        # pass again
        instance.ionization_electron_species = self.electron
        instance.check()

        # do *NOT* render if electron species is broken
        instance.ionization_electron_species.attributes = []
        with self.assertRaises(ValueError):
            instance.ionization_electron_species.check()
        with self.assertRaises(ValueError):
            instance.get_rendering_context()
