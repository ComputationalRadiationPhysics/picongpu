"""
This file is part of PIConGPU.
Copyright 2021-2024 PIConGPU contributors
Authors: Hannes Troepgen, Brian Edward Marre
License: GPLv3+
"""

from picongpu.pypicongpu.field_solver import Solver, YeeSolver, LeheSolver

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


class TestLeheSolver(unittest.TestCase):
    def test_basic(self):
        # basically only check the type -- which actually happens automatically
        lehe = LeheSolver()
        self.assertTrue(isinstance(lehe, Solver))

        self.assertEqual("Lehe<>", lehe.get_rendering_context()["name"])
