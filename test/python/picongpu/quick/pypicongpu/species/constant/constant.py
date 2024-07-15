"""
This file is part of PIConGPU.
Copyright 2021-2023 PIConGPU contributors
Authors: Hannes Troepgen, Brian Edward Marre
License: GPLv3+
"""

from picongpu.pypicongpu.species.constant import Constant

import unittest


class TestConstant(unittest.TestCase):
    def test_rendering_abstract(self):
        """rendering context not implemented, but available"""
        dc = Constant()
        with self.assertRaises(NotImplementedError):
            dc.get_rendering_context()

    def test_dependencies_abstract(self):
        dc = Constant()
        with self.assertRaises(NotImplementedError):
            Constant().check()
        with self.assertRaises(NotImplementedError):
            dc.get_species_dependencies()
        with self.assertRaises(NotImplementedError):
            dc.get_attribute_dependencies()
        with self.assertRaises(NotImplementedError):
            dc.get_constant_dependencies()
