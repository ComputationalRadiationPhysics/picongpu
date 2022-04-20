"""
This file is part of the PIConGPU.
Copyright 2021-2022 PIConGPU contributors
Authors: Hannes Tröpgen, Brian Edward Marré
License: GPLv3+
"""

from picongpu.pypicongpu.species.attribute import Position, Attribute

import unittest


class TestPosition(unittest.TestCase):
    def test_is_attr(self):
        """is an attribute"""
        self.assertTrue(isinstance(Position(), Attribute))

    def test_basic(self):
        pos = Position()
        self.assertNotEqual("", pos.PICONGPU_NAME)
