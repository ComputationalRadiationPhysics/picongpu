"""
This file is part of the PIConGPU.
Copyright 2021-2023 PIConGPU contributors
Authors: Hannes Troepgen, Brian Edward Marre
License: GPLv3+
"""

from picongpu.pypicongpu.species.operation import SimpleMomentum

import unittest
import typeguard

from picongpu.pypicongpu.species.operation.momentum import Temperature, Drift
from picongpu.pypicongpu.species import Species
from picongpu.pypicongpu.species.attribute import Momentum, Position

from copy import deepcopy


class TestSimpleMomentum(unittest.TestCase):
    def setUp(self):
        self.temperature = Temperature()
        self.temperature.temperature_kev = 42

        self.drift = Drift()
        self.drift.direction_normalized = (1, 0, 0)
        self.drift.gamma = 1

        self.species = Species()
        self.species.name = "mockname"
        self.species.constants = []
        self.species.attributes = [Position()]

        self.sm = SimpleMomentum()
        self.sm.species = self.species
        self.sm.temperature = self.temperature
        self.sm.drift = self.drift

    def test_rendering_context(self):
        """renders to context object"""
        self.sm.prebook_species_attributes()
        self.sm.bake_species_attributes()
        context = self.sm.get_rendering_context()
        self.assertEqual(context["species"], self.species.get_rendering_context())
        self.assertEqual(context["temperature"], self.temperature.get_rendering_context())
        self.assertEqual(context["drift"], self.drift.get_rendering_context())

    def test_check_passthru(self):
        """calls check of children"""
        # drift check called:
        self.drift.gamma = -1
        with self.assertRaises(ValueError):
            self.drift.check()
        with self.assertRaises(ValueError):
            self.sm.check_preconditions()
        self.drift.gamma = 1

        # temperature check is called:
        self.temperature.temperature_kev = -1
        with self.assertRaises(ValueError):
            self.temperature.check()
        with self.assertRaises(ValueError):
            self.sm.check_preconditions()
        self.temperature.temperature_kev = 42

        # works again:
        self.sm.check_preconditions()

    def test_attribute(self):
        """actually provides an attribute"""
        for temp in [None, self.temperature]:
            for drift in [None, self.drift]:
                sm = SimpleMomentum()
                sm.species = self.species
                sm.temperature = temp
                sm.drift = drift

                sm.check_preconditions()
                sm.prebook_species_attributes()

                self.assertEqual(1, len(sm.attributes_by_species))
                attrs = sm.attributes_by_species[self.species]
                self.assertEqual(1, len(attrs))
                self.assertTrue(isinstance(attrs[0], Momentum))

    def test_types(self):
        """typesafety is ensured"""
        for invalid in [1, "", [], {}]:
            with self.assertRaises(typeguard.TypeCheckError):
                self.sm.temperature = invalid
            with self.assertRaises(typeguard.TypeCheckError):
                self.sm.drift = invalid
            with self.assertRaises(typeguard.TypeCheckError):
                self.sm.species = invalid

        with self.assertRaises(typeguard.TypeCheckError):
            self.sm.temperature = self.drift
        with self.assertRaises(typeguard.TypeCheckError):
            self.sm.temperature = self.species

        with self.assertRaises(typeguard.TypeCheckError):
            self.sm.drift = self.temperature
        with self.assertRaises(typeguard.TypeCheckError):
            self.sm.drift = self.species

        with self.assertRaises(typeguard.TypeCheckError):
            self.sm.species = self.temperature
        with self.assertRaises(typeguard.TypeCheckError):
            self.sm.species = self.drift

    def test_optional(self):
        """temperature and drift may be left at None"""
        for temp in [None, self.temperature]:
            for drift in [None, self.drift]:
                # would accumulate multiple momentum attributes over multiple
                # iterations (which throws)
                sm = deepcopy(self.sm)
                sm.temperature = temp
                sm.drift = drift

                # checks pass
                sm.check_preconditions()

                # renders without problems
                sm.prebook_species_attributes()
                sm.bake_species_attributes()
                sm.get_rendering_context()

    def test_species_mandatory(self):
        """species must be set"""
        sm = SimpleMomentum()
        sm.temperature = self.temperature
        sm.drift = self.drift

        with self.assertRaises(Exception):
            sm.check_preconditions()

        # ok with species:
        sm.species = self.species
        sm.check_preconditions()

    def test_explicit_attributes(self):
        """even if optional, parameters have to be given explicitly"""
        for set_temp in [True, False]:
            for set_drift in [True, False]:
                sm = SimpleMomentum()
                sm.species = self.species
                if set_temp:
                    sm.temperature = self.temperature
                if set_drift:
                    sm.drift = self.drift

                if not (set_temp and set_drift):
                    # refuse to work if not both are explicitly set
                    with self.assertRaises(Exception):
                        sm.check_preconditions()

    def test_rendering_checks(self):
        """rendering calls check"""
        sm = SimpleMomentum()
        sm.species = self.species
        sm.temperature = None
        sm.drift = self.drift
        sm.drift.gamma = -1

        # raises...
        with self.assertRaises(ValueError):
            sm.check_preconditions()

        # ...hence context also raises
        with self.assertRaises(ValueError):
            sm.get_rendering_context()
