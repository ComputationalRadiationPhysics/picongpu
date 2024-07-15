"""
This file is part of PIConGPU.
Copyright 2021-2023 PIConGPU contributors
Authors: Hannes Troepgen, Brian Edward Marre
License: GPLv3+
"""


from picongpu.pypicongpu.species import Species
from picongpu.pypicongpu.species.constant import Mass, Charge
from picongpu.picmi import constants


class TestGroundStateIonization:
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
