"""
This file is part of PIConGPU.
Copyright 2026 PIConGPU contributors
Authors: Julian Lenz
License: GPLv3+
"""

from unittest import TestCase

from pydantic import ValidationError

from picongpu.pypicongpu.species.attribute.momentum import Momentum
from picongpu.pypicongpu.species.attribute.position import Position
from picongpu.pypicongpu.species.attribute.weighting import Weighting
from picongpu.pypicongpu.species.constant.charge import Charge
from picongpu.pypicongpu.species.constant.mass import Mass
from picongpu.pypicongpu.species.species import Species


def _attributes():
    return [Position(), Momentum(), Weighting()]


class TestSpeciesConstantsNormalization(TestCase):
    def test_dict_constants_are_preserved(self):
        # Regression: passing constants as a dict (direct pypicongpu construction)
        # used to be misread as a list of keys, so every constant was dropped to
        # None by the constants field validator.
        mass = Mass(mass_si=9.109e-31)
        charge = Charge(charge_si=1.602e-19)
        species = Species(name="electron", constants={"mass": mass, "charge": charge}, attributes=_attributes())
        self.assertIs(species.constants.mass, mass)
        self.assertIs(species.constants.charge, charge)

    def test_list_constants_are_preserved(self):
        # The PICMI path passes constants as a list of Constant objects and must
        # keep working.
        mass = Mass(mass_si=9.109e-31)
        charge = Charge(charge_si=1.602e-19)
        species = Species(name="electron", constants=[mass, charge], attributes=_attributes())
        self.assertIs(species.constants.mass, mass)
        self.assertIs(species.constants.charge, charge)

    def test_dict_constants_reject_unknown_keys(self):
        # A typo'd constant name in the dict form must not be silently dropped
        # (e.g. "mss" instead of "mass" would leave mass as None without error).
        mass = Mass(mass_si=9.109e-31)
        with self.assertRaises(ValidationError) as ctx:
            Species(name="electron", constants={"mss": mass}, attributes=_attributes())
        self.assertIn("mss", str(ctx.exception))
        self.assertIn("mass", str(ctx.exception))

    def test_duplicate_list_constants_last_wins(self):
        # Duplicates are invalid per the "each constant type may only be defined
        # once" contract; the effective (documented) semantics are last-wins.
        first = Mass(mass_si=1.0)
        second = Mass(mass_si=2.0)
        species = Species(name="electron", constants=[first, second], attributes=_attributes())
        self.assertIs(species.constants.mass, second)

    def test_absent_constants_default_to_none(self):
        species = Species(name="electron", constants={"mass": Mass(mass_si=9.109e-31)}, attributes=_attributes())
        self.assertIsNotNone(species.constants.mass)
        self.assertIsNone(species.constants.charge)
        self.assertIsNone(species.constants.density_ratio)

    def test_check_passes_with_unique_constants(self):
        species = Species(
            name="electron",
            constants={"mass": Mass(mass_si=9.109e-31), "charge": Charge(charge_si=1.602e-19)},
            attributes=_attributes(),
        )
        # Sanity-check should not trip the (previously always-firing) unique-constant check.
        species.check()
