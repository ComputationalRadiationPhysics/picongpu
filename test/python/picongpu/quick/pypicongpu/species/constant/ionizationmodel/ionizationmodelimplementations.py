"""
This file is part of PIConGPU.
Copyright 2024 PIConGPU contributors
Authors: Brian Edward Marre
License: GPLv3+
"""

from picongpu.pypicongpu.species.constant.ionizationmodel import BSI, BSIEffectiveZ, BSIStarkShifted
from picongpu.pypicongpu.species.constant.ionizationmodel import ADKLinearPolarization, ADKCircularPolarization
from picongpu.pypicongpu.species.constant.ionizationmodel import Keldysh, ThomasFermi
from picongpu.pypicongpu.species.constant.ionizationcurrent import None_
from picongpu.pypicongpu.species.constant import Charge, Mass
from picongpu.pypicongpu.species import Species
from picongpu.picmi import constants

import unittest


class Test_IonizationModelImplementations(unittest.TestCase):
    implementations_withIonizationCurrent = {
        BSI: "BSI",
        BSIEffectiveZ: "BSIEffectiveZ",
        BSIStarkShifted: "BSIStarkShifted",
        ADKCircularPolarization: "ADKLinPol",
        ADKLinearPolarization: "ADKCircPol",
        Keldysh: "Keldysh",
    }

    implementations_withoutIonizationCurrent = {ThomasFermi: "ThomasFermi"}

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

    def test_ionizationCurrentRequired(self):
        """ionization current must be explicitly configured"""
        for Implementation in self.implementations_withIonizationCurrent.keys():
            with self.assertRaisesRegex(Exception, ".*ionization_current.*"):
                implementation = Implementation(ionization_electron_species=self.electron)
                # do not call get_rendering_context, since species not completely initialized yet
                implementation.check()

    def test_basic(self):
        """may create and serialize"""
        for Implementation in self.implementations_withIonizationCurrent.keys():
            implementation = Implementation(ionization_electron_species=self.electron, ionization_current=None_())
            implementation.check()

        for Implementation in self.implementations_withoutIonizationCurrent.keys():
            implementation = Implementation(ionization_electron_species=self.electron)
            implementation.check()

    def test_picongpu_name(self):
        for Implementation, name in self.implementations_withoutIonizationCurrent.items():
            self.assertEqual(
                name,
                Implementation(ionization_electron_species=self.electron, ionization_current=None_()).PICONGPU_NAME,
            )
        for Implementation, name in self.implementations_withoutIonizationCurrent.items():
            self.assertEqual(name, Implementation(ionization_electron_species=self.electron))
