"""
This file is part of the PIConGPU.
Copyright 2021-2023 PIConGPU contributors
Authors: Hannes Troepgen, Brian Edward Marre
License: GPLv3+
"""

from picongpu.pypicongpu.species.operation.densityprofile import Uniform, DensityProfile

import unittest
import typeguard


class TestUniform(unittest.TestCase):
    def test_inheritance(self):
        """uniform is a density profile"""
        self.assertTrue(isinstance(Uniform(), DensityProfile))

    def test_basic(self):
        """simple scenario works"""
        u = Uniform()
        u.density_si = 10e-27
        # passes
        u.check()

    def test_typesafety(self):
        """typesafety is ensured"""
        u = Uniform()

        for invalid in [None, "1", [], {}]:
            with self.assertRaises(typeguard.TypeCheckError):
                u.density_si = invalid

    def test_check(self):
        """validity check on self"""
        u = Uniform()

        # density unset:
        with self.assertRaises(Exception):
            u.check()

        for invalid in [-1, 0, -0.00000003]:
            # assignment passes, but check catches the error
            u.density_si = invalid
            with self.assertRaises(ValueError):
                u.check()

    def test_rendering(self):
        """value passed through from rendering"""
        u = Uniform()
        u.density_si = 42.17

        context = u.get_rendering_context()
        self.assertAlmostEqual(u.density_si, context["density_si"])

        # ensure check() is performed
        u.density_si = -1
        with self.assertRaisesRegex(ValueError, ".*>0.*"):
            u.get_rendering_context()
