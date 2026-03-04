"""
This file is part of PIConGPU.
Copyright 2021-2024 PIConGPU contributors
Authors: Hannes Troepgen, Brian Edward Marre
License: GPLv3+
"""

from picongpu import picmi

from unittest import TestCase


class TestPicmiPseudoRandomLayout(TestCase):
    def test_basic(self):
        """simple translation"""
        layout = picmi.PseudoRandomLayout(n_macroparticles_per_cell=7)
        layout.ppc = 1
        layout.check()

    def test_invalid(self):
        """erros for invalid params entered"""
        with self.assertRaisesRegex(Exception, ".*per.*"):
            layout = picmi.PseudoRandomLayout(n_macroparticles=700)
            layout.ppc = 1
            layout.check()

        with self.assertRaises(AssertionError):
            layout = picmi.PseudoRandomLayout(n_macroparticles_per_cell=0)
            layout.ppc = 1
            layout.check()


class TestPicmiGriddedLayout(TestCase):
    def test_basic(self):
        """simple translation"""
        layout = picmi.GriddedLayout(n_macroparticle_per_cell=7)
        layout.ppc = 1
        layout.check()

    def test_invalid(self):
        """erros for invalid params entered"""
        with self.assertRaisesRegex(Exception, ".*per.*"):
            layout = picmi.GriddedLayout(n_macroparticles=700)
            layout.ppc = 1
            layout.check()

        with self.assertRaises(AssertionError):
            layout = picmi.GriddedLayout(n_macroparticle_per_cell=0)
            layout.check()
