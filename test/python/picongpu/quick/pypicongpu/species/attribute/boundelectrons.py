"""
This file is part of the PIConGPU.
Copyright 2021-2023 PIConGPU contributors
Authors: Hannes Troepgen, Brian Edward Marre
License: GPLv3+
"""

from picongpu.pypicongpu.species.attribute import BoundElectrons, Attribute

import unittest


class TestPosition(unittest.TestCase):
    def test_is_attr(self):
        """is an attribute"""
        self.assertTrue(isinstance(BoundElectrons(), Attribute))

    def test_basic(self):
        be = BoundElectrons()
        self.assertNotEqual("", be.PICONGPU_NAME)
