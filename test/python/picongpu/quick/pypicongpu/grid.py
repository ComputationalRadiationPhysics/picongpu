"""
This file is part of the PIConGPU.
Copyright 2021-2023 PIConGPU contributors
Authors: Hannes Troepgen, Brian Edward Marre, Richard Pausch
License: GPLv3+
"""

from picongpu.pypicongpu.grid import Grid3D, BoundaryCondition

import unittest


class TestGrid3D(unittest.TestCase):
    def setUp(self):
        """setup default grid"""
        self.g = Grid3D()
        self.g.cell_size_x_si = 1.2
        self.g.cell_size_y_si = 2.3
        self.g.cell_size_z_si = 4.5
        self.g.cell_cnt_x = 6
        self.g.cell_cnt_y = 7
        self.g.cell_cnt_z = 8
        self.g.boundary_condition_x = BoundaryCondition.PERIODIC
        self.g.boundary_condition_y = BoundaryCondition.ABSORBING
        self.g.boundary_condition_z = BoundaryCondition.PERIODIC
        self.g.n_gpus = tuple([2, 4, 1])

    def test_basic(self):
        """test default setup"""
        g = self.g
        self.assertEqual(1.2, g.cell_size_x_si)
        self.assertEqual(2.3, g.cell_size_y_si)
        self.assertEqual(4.5, g.cell_size_z_si)
        self.assertEqual(6, g.cell_cnt_x)
        self.assertEqual(7, g.cell_cnt_y)
        self.assertEqual(8, g.cell_cnt_z)
        self.assertEqual(BoundaryCondition.PERIODIC, g.boundary_condition_x)
        self.assertEqual(BoundaryCondition.ABSORBING, g.boundary_condition_y)
        self.assertEqual(BoundaryCondition.PERIODIC, g.boundary_condition_z)

    def test_types(self):
        """test raising errors if types are wrong"""
        g = self.g
        with self.assertRaises(TypeError):
            g.cell_size_x_si = "54.3"
        with self.assertRaises(TypeError):
            g.cell_size_y_si = "2"
        with self.assertRaises(TypeError):
            g.cell_size_z_si = "126"
        with self.assertRaises(TypeError):
            g.cell_cnt_x = 11.1
        with self.assertRaises(TypeError):
            g.cell_cnt_y = 11.412
        with self.assertRaises(TypeError):
            g.cell_cnt_z = 16781123173.12637183
        with self.assertRaises(TypeError):
            g.boundary_condition_x = "open"
        with self.assertRaises(TypeError):
            g.boundary_condition_y = 1
        with self.assertRaises(TypeError):
            g.boundary_condition_z = {}
        with self.assertRaises(TypeError):
            # list not accepted - tuple needed
            g.n_gpus = [1, 1, 1]

    def test_gpu_and_cell_cnt_positive(self):
        """test if n_gpus and cell number s are >0"""
        g = self.g
        with self.assertRaisesRegex(Exception,
                                    ".*cell_cnt_x.*greater than 0.*"):
            g.cell_cnt_x = -1
            g._get_serialized()
        # revert changes
        g.cell_cnt_x = 6

        with self.assertRaisesRegex(Exception,
                                    ".*cell_cnt_y.*greater than 0.*"):
            g.cell_cnt_y = -2
            g._get_serialized()
        # revert changes
        g.cell_cnt_y = 7

        with self.assertRaisesRegex(Exception,
                                    ".*cell_cnt_z.*greater than 0.*"):
            g.cell_cnt_z = 0
            g._get_serialized()
        # revert changes
        g.cell_cnt_z = 8

        for wrong_n_gpus in [tuple([-1, 1, 1]), tuple([1, 1, 0])]:
            with self.assertRaisesRegex(Exception, ".*greater than 0.*"):
                g.n_gpus = wrong_n_gpus
                g._get_serialized()

    def test_mandatory(self):
        """test if None as content fails"""
        # check that mandatory arguments can't be none
        g = self.g
        with self.assertRaises(TypeError):
            g.cell_size_x_si = None
        with self.assertRaises(TypeError):
            g.cell_size_y_si = None
        with self.assertRaises(TypeError):
            g.cell_size_z_si = None
        with self.assertRaises(TypeError):
            g.cell_cnt_x = None
        with self.assertRaises(TypeError):
            g.cell_cnt_y = None
        with self.assertRaises(TypeError):
            g.cell_cnt_x = None
        with self.assertRaises(TypeError):
            g.boundary_condition_x = None
        with self.assertRaises(TypeError):
            g.boundary_condition_y = None
        with self.assertRaises(TypeError):
            g.boundary_condition_z = None
        with self.assertRaises(TypeError):
            g.n_gpus = None

    def test_get_rendering_context(self):
        """object is correctly serialized"""
        # automatically checks against schema
        context = self.g.get_rendering_context()
        self.assertEqual(1.2, context["cell_size"]["x"])
        self.assertEqual(2.3, context["cell_size"]["y"])
        self.assertEqual(4.5, context["cell_size"]["z"])
        self.assertEqual(6, context["cell_cnt"]["x"])
        self.assertEqual(7, context["cell_cnt"]["y"])
        self.assertEqual(8, context["cell_cnt"]["z"])

        # boundary condition translated to numbers for cfgfiles
        self.assertEqual("1", context["boundary_condition"]["x"])
        self.assertEqual("0", context["boundary_condition"]["y"])
        self.assertEqual("1", context["boundary_condition"]["z"])

        # n_gpus ouput
        self.assertEqual(2, context["gpu_cnt"]["x"])
        self.assertEqual(4, context["gpu_cnt"]["y"])
        self.assertEqual(1, context["gpu_cnt"]["z"])


class TestBoundaryCondition(unittest.TestCase):
    def test_cfg_translation(self):
        """test boundary condition strings"""
        p = BoundaryCondition.PERIODIC
        a = BoundaryCondition.ABSORBING
        self.assertEqual("0", a.get_cfg_str())
        self.assertEqual("1", p.get_cfg_str())
