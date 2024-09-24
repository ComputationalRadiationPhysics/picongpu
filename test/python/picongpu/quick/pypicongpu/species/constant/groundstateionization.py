"""
This file is part of PIConGPU.
Copyright 2021-2024 PIConGPU contributors
Authors: Hannes Troepgen, Brian Edward Marre
License: GPLv3+
"""


from picongpu.pypicongpu.species import Species
from picongpu.pypicongpu.species.attribute import Position, Momentum, BoundElectrons
from picongpu.pypicongpu.species.constant import Mass, Charge, GroundStateIonization, ElementProperties
from picongpu.pypicongpu.species.constant.ionizationmodel import BSI, BSIStarkShifted, ThomasFermi
from picongpu.pypicongpu.species.constant.ionizationcurrent import None_
from picongpu.picmi import constants

import unittest
import pydantic_core


class TestGroundStateIonization(unittest.TestCase):
    # set maximum length to infinite to get sensible error message on fail
    maxDiff = None

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

        self.electron = electron

        self.BSI_instance = BSI(ionization_electron_species=self.electron, ionization_current=None_())
        self.BSIstark_instance = BSIStarkShifted(ionization_electron_species=self.electron, ionization_current=None_())
        self.thomas_fermi_instance = ThomasFermi(ionization_electron_species=self.electron)

    def test_basic(self):
        """we may create basic Instance"""
        # test we may create GroundStateIonization
        GroundStateIonization(ionization_model_list=[self.BSI_instance])

    def test_type_safety(self):
        """may only add list of IonizationModel instances"""

        for invalid in ["BSI", ["BSI"], [1], 1.0, self.BSI_instance]:
            with self.assertRaises(pydantic_core._pydantic_core.ValidationError):
                GroundStateIonization(ionization_model_list=invalid)

    def test_check_empty_ionization_model_list(self):
        """empty ionization model list is not allowed"""

        # assignment is possible
        instance = GroundStateIonization(ionization_model_list=[])

        with self.assertRaisesRegex(
            ValueError, ".*at least one ionization model must be specified if ground_state_ionization is not none.*"
        ):
            # but check throws error
            instance.check()

    def test_check_doubled_up_model_group(self):
        """may not assign more than one ionization model from the same group"""

        # assignment is possible
        instance = GroundStateIonization(
            ionization_model_list=[self.BSI_instance, self.BSIstark_instance, self.thomas_fermi_instance]
        )

        with self.assertRaisesRegex(ValueError, ".*ionization model group already represented: BSI.*"):
            # but check throws
            instance.check()

    def test_check_call_on_ionization_model(self):
        """check method of ionization models is called"""

        # creation is possible will only raise in check method
        invalid_ionization_model = BSI(ionization_electron_species=None, ionization_current=None_())

        # assignment is allowed
        instance = GroundStateIonization(ionization_model_list=[invalid_ionization_model])
        with self.assertRaisesRegex(TypeError, ".*ionization_electron_species must be of type pypicongpu Species.*"):
            # but check throws error
            instance.check()

    def test_species_dependencies(self):
        """correct return"""
        self.assertEqual(
            GroundStateIonization(ionization_model_list=[self.BSI_instance]).get_species_dependencies(), [self.electron]
        )

    def test_attribute_dependencies(self):
        """correct return"""
        self.assertEqual(
            GroundStateIonization(ionization_model_list=[self.BSI_instance]).get_attribute_dependencies(),
            [BoundElectrons],
        )

    def test_constant_dependencies(self):
        """correct return"""
        self.assertEqual(
            GroundStateIonization(ionization_model_list=[self.BSI_instance]).get_constant_dependencies(),
            [ElementProperties],
        )

    def test_rendering(self):
        """rendering may be called and returns correct context"""
        # complete configuration of electron species
        electron = self.BSI_instance.ionization_electron_species
        electron.attributes = [Position(), Momentum()]

        context = GroundStateIonization(ionization_model_list=[self.BSI_instance]).get_rendering_context()

        expected_context = {
            "ionization_model_list": [
                {
                    "ionizer_picongpu_name": "BSI",
                    "ionization_electron_species": {
                        "name": "e",
                        "typename": "species_e",
                        "attributes": [
                            {"picongpu_name": "position<position_pic>"},
                            {"picongpu_name": "momentum"},
                        ],
                        "constants": {
                            "mass": {"mass_si": constants.m_e},
                            "charge": {"charge_si": constants.m_e},
                            "density_ratio": None,
                            "element_properties": None,
                            "ground_state_ionization": None,
                        },
                    },
                    "ionization_current": {"picongpu_name": "None"},
                }
            ]
        }

        self.assertEqual(context, expected_context)
