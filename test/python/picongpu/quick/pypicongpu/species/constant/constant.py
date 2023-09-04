"""
This file is part of the PIConGPU.
Copyright 2021-2023 PIConGPU contributors
Authors: Hannes Troepgen, Brian Edward Marre
License: GPLv3+
"""

from picongpu.pypicongpu.species.constant import Constant

import unittest


class DummyConstant(Constant):
    def __init__(self):
        pass

    def check(self):
        pass


class TestConstant(unittest.TestCase):
    def test_abstract(self):
        """methods are not implemented"""
        with self.assertRaises(NotImplementedError):
            Constant()

        # must pass silently
        dc = DummyConstant()
        dc.check()

    def test_check_abstract(self):
        class ConstantCheckAbstract(Constant):
            def __init__(self):
                pass

            # check() not overwritten

        cca = ConstantCheckAbstract()
        with self.assertRaises(NotImplementedError):
            cca.check()

    def test_rendering_abstract(self):
        """rendering context not implemented, but available"""
        dc = DummyConstant()
        with self.assertRaises(NotImplementedError):
            dc.get_rendering_context()

    def test_dependencies_abstract(self):
        dc = DummyConstant()
        with self.assertRaises(NotImplementedError):
            dc.get_species_dependencies()
        with self.assertRaises(NotImplementedError):
            dc.get_attribute_dependencies()
        with self.assertRaises(NotImplementedError):
            dc.get_constant_dependencies()
