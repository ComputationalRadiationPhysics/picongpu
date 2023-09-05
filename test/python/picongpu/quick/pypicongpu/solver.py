"""
This file is part of the PIConGPU.
Copyright 2021-2023 PIConGPU contributors
Authors: Hannes Troepgen, Brian Edward Marre
License: GPLv3+
"""

from picongpu.pypicongpu.solver import Solver, YeeSolver

import unittest


class TestSolver(unittest.TestCase):
    def test_basic(self):
        # the parent class must raise an error when using
        # note: the error is that this class does not exist
        with self.assertRaises(Exception):
            Solver().get_rendering_context()


class TestYeeSolver(unittest.TestCase):
    def test_basic(self):
        # basically only check the type -- which actually happens automatically
        yee = YeeSolver()
        self.assertTrue(isinstance(yee, Solver))

        self.assertEqual("Yee", yee.get_rendering_context()["name"])
