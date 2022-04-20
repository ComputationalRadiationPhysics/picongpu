"""
This file is part of the PIConGPU.
Copyright 2021-2022 PIConGPU contributors
Authors: Hannes Tröpgen, Brian Edward Marré
License: GPLv3+
"""

from picongpu.pypicongpu.species.attribute import Attribute

import unittest


class DummyAttribute(Attribute):
    def __init__(self):
        pass


class TestSpeciesAttribute(unittest.TestCase):
    def test_abstract(self):
        """methods are not implemented"""
        with self.assertRaises(NotImplementedError):
            Attribute()

        # must not raise
        DummyAttribute()
