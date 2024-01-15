"""
This file is part of the PIConGPU.
Copyright 2021-2023 PIConGPU contributors
Authors: Hannes Troepgen, Brian Edward Marre
License: GPLv3+
"""

from picongpu.pypicongpu.species.constant import Ionizers, ElementProperties

import unittest

from picongpu.pypicongpu.species import Species
from picongpu.pypicongpu.species.constant import Mass, Charge
from picongpu.pypicongpu.species.attribute import Position, Momentum, BoundElectrons
from picongpu.picmi import constants


class TestIonizers(unittest.TestCase):
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

    def test_basic(self):
        """simple operation"""
        # note: electrons are not checked, because they are not fully
        # initialized yet

        ionizers = Ionizers()
        ionizers.electron_species = self.electron
        ionizers.check()

        self.assertEqual([self.electron], ionizers.get_species_dependencies())
        self.assertEqual([BoundElectrons], ionizers.get_attribute_dependencies())
        self.assertEqual([ElementProperties], ionizers.get_constant_dependencies())

    def test_typesafety(self):
        """types are checked"""
        ionizers = Ionizers()
        for invalid in ["electron", {}, [], 0, None]:
            with self.assertRaises(TypeError):
                # note: circular imports would be required to use the
                # pypicongpu-standard build_typesafe_property, hence the type
                # is checked by check() instead of on assignment (as usual)
                ionizers.electron_species = invalid
                ionizers.check()

    def test_empty(self):
        """electron species is mandatory"""
        ionizers = Ionizers()

        # must fail:
        with self.assertRaises(Exception):
            ionizers.check()
        with self.assertRaises(Exception):
            ionizers.get_species_dependencies()

        # now passes
        ionizers.electron_species = self.electron
        ionizers.check()

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

        ionizers_transitive_const = Ionizers()
        ionizers_transitive_const.electron_species = other_electron

        self.electron.constants.append(ionizers_transitive_const)

        # original ionizers is valid
        ionizers_transitive_const.check()

        # ...but a constant using an ionizable species as electrons must reject
        ionizers = Ionizers()
        ionizers.electron_species = self.electron
        with self.assertRaisesRegex(ValueError, ".*ionizable.*"):
            ionizers.check()

    def test_check_passthru(self):
        """calls check of electron species & checks during rendering"""
        ionizers = Ionizers()

        # must raise (b/c no electron species)
        with self.assertRaises(Exception):
            ionizers.check()

        # subsequently, dependency retrieval mus also raise
        with self.assertRaises(Exception):
            ionizers.get_species_dependencies()

        ionizers.electron_species = self.electron

        # both pass:
        ionizers.check()
        self.assertNotEqual([], ionizers.get_species_dependencies())

        # with a broken species...
        ionizers.electron_species = None
        # ...check()...
        with self.assertRaises(Exception):
            ionizers.check()

        # ...and get dependencies fail
        with self.assertRaises(Exception):
            ionizers.get_species_dependencies()

    def test_rendering(self):
        """renders to rendering context"""
        # prepare electron species s.t. it can be rendered
        self.electron.attributes = [Position(), Momentum()]
        # must pass
        self.electron.check()
        self.assertNotEqual({}, self.electron.get_rendering_context())

        ionizers = Ionizers()
        ionizers.electron_species = self.electron

        context = ionizers.get_rendering_context()
        self.assertNotEqual({}, context)
        self.assertEqual(self.electron.get_rendering_context(), context["electron_species"])

        # do *NOT* render if check() does not pass
        ionizers.electron_species = None
        with self.assertRaises(TypeError):
            ionizers.check()
        with self.assertRaises(TypeError):
            ionizers.get_rendering_context()

        # pass again
        ionizers.electron_species = self.electron
        ionizers.check()

        # do *NOT* render if electron species is broken
        ionizers.electron_species.attributes = []
        with self.assertRaises(ValueError):
            ionizers.electron_species.check()
        with self.assertRaises(ValueError):
            ionizers.get_rendering_context()
