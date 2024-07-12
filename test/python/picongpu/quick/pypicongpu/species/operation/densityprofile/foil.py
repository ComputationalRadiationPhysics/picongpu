"""
This file is part of PIConGPU.
Copyright 2024 PIConGPU contributors
Authors: Hannes Troepgen, Brian Edward Marre
License: GPLv3+
"""

from picongpu.pypicongpu.species.operation.densityprofile import Foil, DensityProfile
from picongpu.pypicongpu.species.operation.densityprofile.plasmaramp import None_

import unittest
import typeguard


class TestFoil(unittest.TestCase):
    def test_inheritance(self):
        """foil is a density profile"""
        self.assertTrue(isinstance(Foil(), DensityProfile))

    def test_basic(self):
        """simple scenario works, other plasma ramps are tested separately"""
        f = Foil()
        f.density_si = 10e-27
        f.y_value_front_foil_si = 0.0
        f.thickness_foil_si = 1.0e-5
        f.pre_foil_plasmaRamp = None_()
        f.post_foil_plasmaRamp = None_()

        # passes
        f.check()

    def test_value_pass_through(self):
        """values are passed through"""
        f = Foil()
        f.density_si = 10e-27
        f.y_value_front_foil_si = 0.0
        f.thickness_foil_si = 1.0e-5

        front = None_()
        back = None_()
        f.pre_foil_plasmaRamp = front
        f.post_foil_plasmaRamp = back

        self.assertAlmostEqual(10e-27, f.density_si)
        self.assertAlmostEqual(0.0, f.y_value_front_foil_si)
        self.assertAlmostEqual(1.0e-5, f.thickness_foil_si)
        self.assertEqual(front, f.pre_foil_plasmaRamp)
        self.assertEqual(back, f.post_foil_plasmaRamp)

    def test_typesafety(self):
        """typesafety is ensured"""
        f = Foil()

        for invalid in [None, "1", [], {}]:
            with self.assertRaises(typeguard.TypeCheckError):
                f.density_si = invalid

        for invalid in [None, "1", [], {}]:
            with self.assertRaises(typeguard.TypeCheckError):
                f.y_value_front_foil_si = invalid

        for invalid in [None, "1", [], {}]:
            with self.assertRaises(typeguard.TypeCheckError):
                f.thickness_foil_si = invalid

        for invalid in [None, "1", [], {}]:
            with self.assertRaises(typeguard.TypeCheckError):
                f.pre_foil_plasmaRamp = invalid

        for invalid in [None, "1", [], {}]:
            with self.assertRaises(typeguard.TypeCheckError):
                f.post_foil_plasmaRamp = invalid

    def test_check_unsetParameters(self):
        """validity check on self for no parameters set"""

        f = Foil()

        # parameters unset:
        with self.assertRaises(Exception):
            f.check()

    def test_check_density(self):
        """validity check on self for invalid density"""
        f = Foil()
        f.post_foil_plasmaRamp = None_()
        f.pre_foil_plasmaRamp = None_()
        f.y_value_front_foil_si = 0
        f.thickness_foil_si = 1.0e-5

        # invalid density
        for invalid in [-1, 0, -0.00000003]:
            # assignment passes, but check catches the error
            f.density_si = invalid
            with self.assertRaisesRegex(ValueError, ".*density.* > 0.*"):
                f.check()

    def test_check_y_value_front_foil(self):
        """validity check on self for invalid y_value_front_foil_si"""
        f = Foil()
        f.post_foil_plasmaRamp = None_()
        f.pre_foil_plasmaRamp = None_()
        f.density_si = 1.0e-5
        f.thickness_foil_si = 1.0e-5

        # invalid density
        for invalid in [-1, -0.00000003]:
            # assignment passes, but check catches the error
            f.y_value_front_foil_si = invalid
            with self.assertRaisesRegex(ValueError, ".*y_value_front.* >= 0.*"):
                f.check()

    def test_check_thickness(self):
        """validity check on self for invalid y_value_front_foil_si"""
        f = Foil()
        f.post_foil_plasmaRamp = None_()
        f.pre_foil_plasmaRamp = None_()
        f.y_value_front_foil_si = 0
        f.density_si = 1.0e-5

        # invalid density
        for invalid in [-1, -0.00000003]:
            # assignment passes, but check catches the error
            f.thickness_foil_si = invalid
            with self.assertRaisesRegex(ValueError, ".*thickness.* >= 0.*"):
                f.check()

    def test_rendering(self):
        """value passed through from rendering"""
        f = Foil()
        f.density_si = 42.17
        f.post_foil_plasmaRamp = None_()
        f.pre_foil_plasmaRamp = None_()
        f.y_value_front_foil_si = 0
        f.thickness_foil_si = 1.0e-5

        expectedContextNoRamp = {"type": {"exponential": False, "none": True}, "data": None}

        context = f.get_rendering_context()
        self.assertAlmostEqual(f.density_si, context["density_si"])
        self.assertAlmostEqual(f.y_value_front_foil_si, context["y_value_front_foil_si"])
        self.assertAlmostEqual(f.thickness_foil_si, context["thickness_foil_si"])
        self.assertEqual(expectedContextNoRamp, context["pre_foil_plasmaRamp"])
        self.assertEqual(expectedContextNoRamp, context["post_foil_plasmaRamp"])

        # ensure check() is performed
        f.density_si = -1
        with self.assertRaisesRegex(ValueError, ".*> 0.*"):
            f.get_rendering_context()
