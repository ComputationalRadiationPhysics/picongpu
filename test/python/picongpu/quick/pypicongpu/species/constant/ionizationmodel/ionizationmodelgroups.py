"""
This file is part of PIConGPU.
Copyright 2024 PIConGPU contributors
Authors: Brian Edward Marre
License: GPLv3+
"""

from picongpu.pypicongpu.species.constant.ionizationmodel import IonizationModelGroups

from picongpu.pypicongpu.species.constant.ionizationmodel import BSI, BSIEffectiveZ, BSIStarkShifted
from picongpu.pypicongpu.species.constant.ionizationmodel import ADKLinearPolarization, ADKCircularPolarization
from picongpu.pypicongpu.species.constant.ionizationmodel import Keldysh, ThomasFermi

import unittest
import copy


class Test_IonizationModelGroups(unittest.TestCase):
    def setUp(self):
        self.expected_groups_custom = {
            "1": [BSI],
            "2": [ADKLinearPolarization, ADKCircularPolarization],
        }

        self.expected_groups_standard = {
            "BSI_like": [BSI, BSIEffectiveZ, BSIStarkShifted],
            "ADK_like": [ADKLinearPolarization, ADKCircularPolarization],
            "Keldysh_like": [Keldysh],
            "electronic_collisional_equilibrium": [ThomasFermi],
        }

        self.expected_by_model_custom = {
            BSI: "1",
            ADKCircularPolarization: "2",
            ADKLinearPolarization: "2",
        }

    def test_creation(self):
        """may be constructed"""
        # default value construction
        IonizationModelGroups()

        # custom value construction
        IonizationModelGroups(by_group=self.expected_groups_custom)

    def test_get_by_group(self):
        """by_group is correctly returned"""
        self.assertEqual(IonizationModelGroups().get_by_group(), self.expected_groups_standard)
        self.assertEqual(
            IonizationModelGroups(by_group=self.expected_groups_custom).get_by_group(), self.expected_groups_custom
        )

    def test_get_by_model(self):
        """by_group is correctly converted to by_model"""
        self.assertEqual(
            IonizationModelGroups(by_group=self.expected_groups_custom).get_by_model(), self.expected_by_model_custom
        )

    def _switch_groups(self, result, one, two):
        keys = list(result.keys())
        values = list(result.values())

        first_group = keys[one]
        second_group = keys[two]

        first_models = values[one]
        second_models = values[two]

        result[first_group] = second_models
        result[second_group] = first_models

        return result

    def test_get_by_group_returns_copy(self):
        """get_by_group() return copies only"""
        ionization_model_group = IonizationModelGroups(by_group=self.expected_groups_custom)

        # get result
        result = ionization_model_group.get_by_group()

        # make copy for reference
        result_copy = copy.copy(result)

        # manipulate result
        result = self._switch_groups(result, 0, 1)

        # check output is unchanged
        self.assertEqual(result_copy, ionization_model_group.get_by_group())

    def test_get_by_model_returns_copy(self):
        """get_by_model returns copies only"""
        ionization_model_group = IonizationModelGroups(by_group=self.expected_groups_custom)

        # get result
        result = ionization_model_group.get_by_model()

        # make copy for reference
        result_copy = copy.copy(result)

        # manipulate result
        result = self._switch_groups(result, 0, 1)

        # check output is unchanged
        self.assertEqual(result_copy, ionization_model_group.get_by_model())
