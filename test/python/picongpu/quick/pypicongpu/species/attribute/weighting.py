"""
This file is part of the PIConGPU.
Copyright 2021-2022 PIConGPU contributors
Authors: Hannes Tröpgen, Brian Edward Marré
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
