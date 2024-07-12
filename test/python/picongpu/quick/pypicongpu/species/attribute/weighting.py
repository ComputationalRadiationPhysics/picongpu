"""
This file is part of PIConGPU.
Copyright 2021-2023 PIConGPU contributors
Authors: Hannes Troepgen, Brian Edward Marre
License: GPLv3+
"""

from picongpu.pypicongpu.species.attribute import Weighting, Attribute

import unittest


class TestWeighting(unittest.TestCase):
    def test_is_attr(self):
        """is an attribute"""
        self.assertTrue(isinstance(Weighting(), Attribute))

    def test_basic(self):
        w = Weighting()
        self.assertNotEqual("", w.PICONGPU_NAME)
