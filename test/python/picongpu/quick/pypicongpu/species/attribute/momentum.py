"""
This file is part of PIConGPU.
Copyright 2021-2023 PIConGPU contributors
Authors: Hannes Troepgen, Brian Edward Marre
License: GPLv3+
"""

from picongpu.pypicongpu.species.attribute import Momentum, Attribute

import unittest


class TestMomentum(unittest.TestCase):
    def test_is_attr(self):
        """is an attribute"""
        self.assertTrue(isinstance(Momentum(), Attribute))

    def test_basic(self):
        m = Momentum()
        self.assertNotEqual("", m.PICONGPU_NAME)
