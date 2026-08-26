"""
This file is part of PIConGPU.
Copyright 2026 PIConGPU contributors
Authors: Julian Lenz
License: GPLv3+
"""

from pydantic import ValidationError
from unittest import TestCase

from picongpu.pypicongpu.species.attribute.momentum import Momentum
from picongpu.pypicongpu.species.attribute.position import Position
from picongpu.pypicongpu.species.attribute.weighting import Weighting
from picongpu.pypicongpu.species.constant.charge import Charge
from picongpu.pypicongpu.species.constant.elementproperties import ElementProperties
from picongpu.pypicongpu.species.constant.mass import Mass
from picongpu.pypicongpu.species.operation.setchargestate import SetChargeState
from picongpu.pypicongpu.species.species import Species
from picongpu.pypicongpu.species.util.element import Element


def _element_species(symbol):
    return Species(
        name="ion",
        constants=[ElementProperties(element=Element(symbol)), Mass(mass_si=1.0), Charge(charge_si=1.0)],
        attributes=[Position(), Momentum(), Weighting()],
    )


def _custom_species():
    return Species(
        name="custom",
        constants=[Mass(mass_si=1.0), Charge(charge_si=1.0)],
        attributes=[Position(), Momentum(), Weighting()],
    )


class TestSetChargeState(TestCase):
    def test_charge_state_below_atomic_number_accepted(self):
        # Carbon has 6 protons
        op = SetChargeState(species=_element_species("C"), charge_state=2)
        self.assertEqual(op.charge_state, 2)

    def test_charge_state_equal_to_atomic_number_accepted(self):
        # fully stripped
        op = SetChargeState(species=_element_species("C"), charge_state=6)
        self.assertEqual(op.charge_state, 6)

    def test_charge_state_above_atomic_number_rejected(self):
        with self.assertRaises(ValidationError) as ctx:
            SetChargeState(species=_element_species("C"), charge_state=7)
        self.assertIn("atomic number", str(ctx.exception))
        self.assertIn("C", str(ctx.exception))

    def test_charge_state_of_neutral_atom_accepted(self):
        op = SetChargeState(species=_element_species("H"), charge_state=0)
        self.assertEqual(op.charge_state, 0)

    def test_charge_state_without_element_knowledge_accepted(self):
        # no element properties -> no atomic number to check against
        op = SetChargeState(species=_custom_species(), charge_state=1)
        self.assertEqual(op.charge_state, 1)

    def test_negative_charge_state_rejected(self):
        with self.assertRaises(ValidationError):
            SetChargeState(species=_element_species("C"), charge_state=-1)
