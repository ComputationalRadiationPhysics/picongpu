"""
This file is part of the PIConGPU.
Copyright 2021-2023 PIConGPU contributors
Authors: Hannes Troepgen, Brian Edward Marre
License: GPLv3+
"""

from picongpu.pypicongpu.species.operation import Operation

from picongpu.pypicongpu.species import Species

from ..attribute import DummyAttribute

import unittest


class DummyOperation(Operation):
    def __init__(self):
        pass


class TestOperation(unittest.TestCase):
    def setUp(self):
        self.op = DummyOperation()
        self.species1 = Species()
        self.species1.name = "species1"
        self.species1.attributes = []
        self.species1.constants = []
        self.species2 = Species()
        self.species2.name = "species2"
        self.species2.attributes = []
        self.species2.constants = []
        self.attribute1 = DummyAttribute()
        self.attribute1.PICONGPU_NAME = "attribute1"
        self.attribute1_copy = DummyAttribute()
        self.attribute1_copy.PICONGPU_NAME = "attribute1"
        self.attribute2 = DummyAttribute()
        self.attribute2.PICONGPU_NAME = "attribute2"
        self.attribute3 = DummyAttribute()
        self.attribute3.PICONGPU_NAME = "attribute3"

    def test_abstract(self):
        """constructor is not implemented"""
        with self.assertRaises(NotImplementedError):
            Operation()

        # other methods are also not implemented
        op = DummyOperation()
        with self.assertRaises(NotImplementedError):
            op.check_preconditions()
        with self.assertRaises(NotImplementedError):
            op.prebook_species_attributes()

    def test_bake_species_attributes_basic(self):
        """valid example works"""
        op = self.op
        # Note: all these examples access Operation.attributes_by_species
        # directly. This is forbidden. *Normally* this would be performed
        # inside Operation.prebook_species_attributes(), but it is more concise
        # to access Operation.attributes_by_species directly.
        op.attributes_by_species = {
            self.species1: [
                self.attribute1,
            ],
            self.species2: [
                self.attribute2,
                self.attribute3,
            ]
        }
        op.bake_species_attributes()

        self.assertEqual(self.species1.attributes, [self.attribute1])

        # note: check if in lists b/c there is no defined order
        self.assertEqual(2, len(self.species2.attributes))
        self.assertTrue(self.attribute2 in self.species2.attributes)
        self.assertTrue(self.attribute3 in self.species2.attributes)

    def test_bake_species_attributes_empty(self):
        """rejects if nothing is prebooked"""
        # not defined rejects (no default is set)
        op = DummyOperation()
        with self.assertRaises(Exception):
            op.bake_species_attributes()

        # explicitly empty {} is not allowed
        op.attributes_by_species = {}
        with self.assertRaisesRegex(ValueError, ".*at least one.*"):
            op.bake_species_attributes()

        # species without attributes is not allowed
        op.attributes_by_species = {
            self.species1: [],
        }
        with self.assertRaisesRegex(ValueError, ".*at least one.*"):
            op.bake_species_attributes()

    def test_bake_species_attributes_attribute_repetition(self):
        """every attribute type may only be prebooked once per species"""
        # check preconditions from setUp()
        self.assertEqual(self.attribute1.PICONGPU_NAME,
                         self.attribute1_copy.PICONGPU_NAME)
        self.assertTrue(self.attribute1 is not self.attribute1_copy)

        # provide two objects defining the same attribute "attribute1"
        op = self.op
        op.attributes_by_species = {
            self.species1: [
                self.attribute1,
                self.attribute1_copy]}

        with self.assertRaisesRegex(ValueError, ".*attribute1.*"):
            op.bake_species_attributes()

    def test_bake_species_attributes_attributes_exclusive(self):
        """same attribute object can't be prebooked to multiple species"""
        # try to assign same attribute **object** to multiple species
        op = self.op
        op.attributes_by_species = {
            self.species1: [self.attribute1],
            self.species2: [self.attribute1],
        }
        self.assertTrue(op.attributes_by_species[self.species1][0] is
                        op.attributes_by_species[self.species2][0])
        with self.assertRaisesRegex(ValueError, ".*exclusive.*"):
            op.bake_species_attributes()

    def test_bake_species_attributes_species_checked(self):
        """species can not have the same attribute already"""
        # note: do use same attribute "attribute1", but not identical object
        self.species1.attributes = [self.attribute1_copy]

        op = self.op
        op.attributes_by_species = {self.species1: [self.attribute1]}

        with self.assertRaisesRegex(ValueError, ".*attribute1.*"):
            op.bake_species_attributes()
