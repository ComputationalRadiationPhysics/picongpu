"""
This file is part of the PIConGPU.
Copyright 2021-2023 PIConGPU contributors
Authors: Brian Edward Marre
License: GPLv3+
"""

from picongpu.pypicongpu.species.operation.densityprofile import Gaussian, DensityProfile

import unittest
import typeguard


class TestGaussian(unittest.TestCase):
    def test_inheritance(self):
        """foil is a density profile"""
        self.assertTrue(isinstance(Gaussian(), DensityProfile))

    def test_basic(self):
        """simple scenario works, other plasma ramps are tested separately"""
        g = Gaussian()

        g.gas_center_front = 1.0
        g.gas_center_rear = 2.0
        g.gas_sigma_front = 3.0
        g.gas_sigma_rear = 4.0
        g.gas_power = 1.0
        g.gas_factor = -1.0
        g.vacuum_cells_front = 50
        g.density = 1.0e25

        # passes
        g.check()

    def test_value_pass_through(self):
        """values are passed through"""
        g = Gaussian()

        g.gas_center_front = 1.0
        g.gas_center_rear = 2.0
        g.gas_sigma_front = 3.0
        g.gas_sigma_rear = 4.0
        g.gas_power = 5.0
        g.gas_factor = -6.0
        g.vacuum_cells_front = 50
        g.density = 1.0e25

        self.assertAlmostEqual(1.0, g.gas_center_front)
        self.assertAlmostEqual(2.0, g.gas_center_rear)
        self.assertAlmostEqual(3.0, g.gas_sigma_front)
        self.assertAlmostEqual(4.0, g.gas_sigma_rear)
        self.assertAlmostEqual(5.0, g.gas_power)
        self.assertAlmostEqual(-6.0, g.gas_factor)
        self.assertEqual(50, g.vacuum_cells_front)
        self.assertAlmostEqual(1.0e25, g.density)

    def test_typesafety(self):
        """typesafety is ensured"""
        g = Gaussian()

        for invalid in [None, "1", [], {}]:
            with self.assertRaises(typeguard.TypeCheckError):
                g.density = invalid

        for invalid in [None, "1", [], {}, 1.0]:
            with self.assertRaises(typeguard.TypeCheckError):
                g.vacuum_cells_front = invalid

        for invalid in [None, "1", [], {}]:
            with self.assertRaises(typeguard.TypeCheckError):
                g.gas_factor = invalid

        for invalid in [None, "1", [], {}]:
            with self.assertRaises(typeguard.TypeCheckError):
                g.gas_power = invalid

        for invalid in [None, "1", [], {}]:
            with self.assertRaises(typeguard.TypeCheckError):
                g.gas_sigma_front = invalid

        for invalid in [None, "1", [], {}]:
            with self.assertRaises(typeguard.TypeCheckError):
                g.gas_sigma_rear = invalid

        for invalid in [None, "1", [], {}]:
            with self.assertRaises(typeguard.TypeCheckError):
                g.gas_center_front = invalid

        for invalid in [None, "1", [], {}]:
            with self.assertRaises(typeguard.TypeCheckError):
                g.gas_center_rear = invalid

    def test_check_unsetParameters(self):
        """validity check on self for no parameters set"""

        g = Gaussian()

        # parameters unset:
        with self.assertRaises(Exception):
            g.check()

    def test_check_density(self):
        """validity check on self for invalid density"""
        g = Gaussian()
        g.gas_center_front = 1.0
        g.gas_center_rear = 2.0
        g.gas_sigma_front = 3.0
        g.gas_sigma_rear = 4.0
        g.gas_power = 5.0
        g.gas_factor = -6.0
        g.vacuum_cells_front = 50

        # invalid density
        for invalid in [-1, 0, -0.00000003]:
            # assignment passes, but check catches the error
            g.density = invalid
            with self.assertRaisesRegex(ValueError, ".*density.* > 0.*"):
                g.check()

    def test_check_vacuum_cells_front(self):
        """validity check on self for invalid vacuum_cells_front"""
        g = Gaussian()
        g.gas_center_front = 1.0
        g.gas_center_rear = 2.0
        g.gas_sigma_front = 3.0
        g.gas_sigma_rear = 4.0
        g.gas_power = 5.0
        g.gas_factor = -6.0
        g.density = 1.0e25

        # invalid vacuum_cells_front
        for invalid in [-1, -15]:
            # assignment passes, but check catches the error
            g.vacuum_cells_front = invalid
            with self.assertRaisesRegex(ValueError, ".*vacuum_cells_front.* >= 0.*"):
                g.check()

    def test_check_gas_factor(self):
        """validity check on self for invalid gas_factor"""
        g = Gaussian()
        g.gas_center_front = 1.0
        g.gas_center_rear = 2.0
        g.gas_sigma_front = 3.0
        g.gas_sigma_rear = 4.0
        g.gas_power = 5.0
        g.vacuum_cell_front = 50
        g.density = 1.0e25

        # invalid gas_factor
        for invalid in [0.0, 1.0]:
            # assignment passes, but check catches the error
            g.gas_factor = invalid
            with self.assertRaisesRegex(ValueError, ".*gas_factor.* < 0.*"):
                g.check()

    def test_check_gas_power(self):
        """validity check on self for invalid gas_power"""
        g = Gaussian()
        g.gas_center_front = 1.0
        g.gas_center_rear = 2.0
        g.gas_sigma_front = 3.0
        g.gas_sigma_rear = 4.0
        g.gas_factor = -6.0
        g.vacuum_cells_front = 50
        g.density = 1.0e25

        # invalid gas_power
        for invalid in [0.0]:
            # assignment passes, but check catches the error
            g.gas_power = invalid
            with self.assertRaisesRegex(ValueError, ".*gas_power.* != 0.*"):
                g.check()

    def test_check_gas_sigma_rear(self):
        """validity check on self for invalid gas_sigma_rear"""
        g = Gaussian()
        g.gas_center_front = 1.0
        g.gas_center_rear = 2.0
        g.gas_sigma_front = 3.0
        g.gas_power = 5.0
        g.gas_factor = -6.0
        g.vacuum_cells_front = 50
        g.density = 1.0e25

        # invalid gas_sigma_rear
        for invalid in [0.0]:
            # assignment passes, but check catches the error
            g.gas_sigma_rear = invalid
            with self.assertRaisesRegex(ValueError, ".*gas_sigma_rear.* != 0.*"):
                g.check()

    def test_check_gas_sigma_front(self):
        """validity check on self for invalid gas_sigma_front"""
        g = Gaussian()
        g.gas_center_front = 1.0
        g.gas_center_rear = 2.0
        g.gas_sigma_rear = 4.0
        g.gas_power = 5.0
        g.gas_factor = -6.0
        g.vacuum_cells_front = 50
        g.density = 1.0e25

        # invalid gas_sigma_front
        for invalid in [0.0]:
            # assignment passes, but check catches the error
            g.gas_sigma_front = invalid
            with self.assertRaisesRegex(ValueError, ".*gas_sigma_front.* != 0.*"):
                g.check()

    def test_check_gas_center_rear(self):
        """validity check on self for invalid gas_center_rear"""
        g = Gaussian()
        g.gas_center_front = 1.0
        g.gas_sigma_front = 3.0
        g.gas_sigma_rear = 4.0
        g.gas_power = 5.0
        g.gas_factor = -6.0
        g.vacuum_cells_front = 50
        g.density = 1.0e25

        # invalid gas_center_rear
        for invalid in [-1.0]:
            # assignment passes, but check catches the error
            g.gas_center_rear = invalid
            with self.assertRaisesRegex(ValueError, ".*gas_center_rear.* >= 0.*"):
                g.check()

        # rear < front
        # assignment passes, but check catches the error
        g.gas_center_rear = 0.5
        with self.assertRaisesRegex(ValueError, ".*gas_center_rear.* >= gas_center_front.*"):
            g.check()

    def test_check_gas_center_front(self):
        """validity check on self for invalid gas_center_front"""
        g = Gaussian()
        g.gas_center_rear = 2.0
        g.gas_sigma_front = 3.0
        g.gas_sigma_rear = 4.0
        g.gas_power = 5.0
        g.gas_factor = -6.0
        g.vacuum_cells_front = 50
        g.density = 1.0e25

        # invalid gas_center_front
        for invalid in [-1.0]:
            # assignment passes, but check catches the error
            g.gas_center_front = invalid
            with self.assertRaisesRegex(ValueError, ".*gas_center_front.* >= 0.*"):
                g.check()

        # front > rear
        # assignment passes, but check catches the error
        g.gas_center_front = 3.0
        with self.assertRaisesRegex(ValueError, ".*gas_center_rear.* >= gas_center_front.*"):
            g.check()

    def test_rendering(self):
        """value passed through from rendering"""
        g = Gaussian()
        g.gas_center_front = 1.0
        g.gas_center_rear = 2.0
        g.gas_sigma_front = 3.0
        g.gas_sigma_rear = 4.0
        g.gas_power = 5.0
        g.gas_factor = -6.0
        g.vacuum_cells_front = 50
        g.density = 1.0e25

        context = g.get_rendering_context()
        self.assertAlmostEqual(g.gas_center_front, context["gas_center_front"])
        self.assertAlmostEqual(g.gas_center_rear, context["gas_center_rear"])
        self.assertAlmostEqual(g.gas_sigma_front, context["gas_sigma_front"])
        self.assertAlmostEqual(g.gas_sigma_rear, context["gas_sigma_rear"])
        self.assertAlmostEqual(g.gas_power, context["gas_power"])
        self.assertAlmostEqual(g.gas_factor, context["gas_factor"])
        self.assertEqual(g.vacuum_cells_front, context["vacuum_cells_front"])
        self.assertAlmostEqual(g.density, context["density"])

        # ensure check() is performed
        g.density = -1
        with self.assertRaisesRegex(ValueError, ".*> 0.*"):
            g.get_rendering_context()
