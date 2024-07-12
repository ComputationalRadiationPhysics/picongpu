"""
This file is part of PIConGPU.
Copyright 2021-2023 PIConGPU contributors
Authors: Hannes Troepgen, Brian Edward Marre
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
