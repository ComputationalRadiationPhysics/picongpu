"""
This file is part of the PIConGPU.
Copyright 2021-2022 PIConGPU contributors
Authors: Hannes Tröpgen, Brian Edward Marré
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
