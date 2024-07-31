"""
This file is part of PIConGPU.
Copyright 2021-2023 PIConGPU contributors
Authors: Hannes Troepgen, Brian Edward Marre
License: GPLv3+
"""

from picongpu.pypicongpu.species.operation import SetChargeState

import unittest
import typeguard

from picongpu.pypicongpu.species import Species
from picongpu.pypicongpu.species.constant import GroundStateIonization
from picongpu.pypicongpu.species.constant.ionizationmodel import BSI
from picongpu.pypicongpu.species.constant.ionizationcurrent import None_
from picongpu.pypicongpu.species.attribute import BoundElectrons, Position, Momentum


class TestSetChargeState(unittest.TestCase):
    def setUp(self):
        electron = Species()
        electron.name = "e"
        # note: attributes not set yet (as would be in init manager)

        self.electron = electron

        self.species1 = Species()
        self.species1.name = "ion"
        self.species1.constants = [
            GroundStateIonization(
                ionization_model_list=[BSI(ionization_electron_species=self.electron, ionization_current=None_())]
            )
        ]

    def test_basic(self):
        """basic operation"""
        scs = SetChargeState()
        scs.species = self.species1
        scs.charge_state = 2

        # checks pass
        scs.check_preconditions()

    def test_typesafety(self):
        """typesafety is ensured"""
        scs = SetChargeState()
        for invalid_species in [None, 1, "a", []]:
            with self.assertRaises(typeguard.TypeCheckError):
                scs.species = invalid_species

        for invalid_number in [None, "a", [], self.species1, 2.3]:
            with self.assertRaises(typeguard.TypeCheckError):
                scs.charge_state = invalid_number

        # works:
        scs.species = self.species1
        scs.charge_state = 1

    def test_empty(self):
        """all parameters are mandatory"""
        for set_species in [True, False]:
            for set_charge_state in [True, False]:
                scs = SetChargeState()

                if set_species:
                    scs.species = self.species1
                if set_charge_state:
                    scs.charge_state = 1

                if set_species and set_charge_state:
                    # must pass
                    scs.check_preconditions()
                else:
                    # mandatory missing -> must raise
                    with self.assertRaises(Exception):
                        scs.check_preconditions()

    def test_attribute_generated(self):
        """creates bound electrons attribute"""
        scs = SetChargeState()
        scs.species = self.species1
        scs.charge_state = 1

        # emulate initmanager
        scs.check_preconditions()
        self.species1.attributes = []
        scs.prebook_species_attributes()

        self.assertEqual(1, len(scs.attributes_by_species))
        self.assertTrue(self.species1 in scs.attributes_by_species)
        self.assertEqual(1, len(scs.attributes_by_species[self.species1]))
        self.assertTrue(isinstance(scs.attributes_by_species[self.species1][0], BoundElectrons))

    def test_ionizers_required(self):
        """ionizers constant must be present"""
        scs = SetChargeState()
        scs.species = self.species1
        scs.charge_state = 1

        # passes:
        self.assertTrue(scs.species.has_constant_of_type(GroundStateIonization))
        scs.check_preconditions()

        # without constants does not pass:
        scs.species.constants = []
        with self.assertRaisesRegex(AssertionError, ".*BoundElectrons requires GroundStateIonization.*"):
            scs.check_preconditions()

    def test_values(self):
        """bound electrons must be >0"""
        scs = SetChargeState()
        scs.species = self.species1

        with self.assertRaisesRegex(ValueError, ".*> 0.*"):
            scs.charge_state = -1
            scs.check_preconditions()

        # silently passes
        scs.charge_state = 1
        scs.check_preconditions()

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
        ion.constants = [
            GroundStateIonization(
                ionization_model_list=[BSI(ionization_electron_species=electron, ionization_current=None_())]
            ),
        ]
        ion.attributes = [Position(), Momentum(), BoundElectrons()]

        # can be rendered
        self.assertNotEqual({}, ion.get_rendering_context())

        scs = SetChargeState()
        scs.species = ion
        scs.charge_state = 1

        context = scs.get_rendering_context()
        self.assertEqual(1, context["charge_state"])
        self.assertEqual(ion.get_rendering_context(), context["species"])
