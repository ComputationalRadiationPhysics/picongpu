"""
This file is part of the PIConGPU.
Copyright 2021-2023 PIConGPU contributors
Authors: Hannes Troepgen, Brian Edward Marre
License: GPLv3+
"""

from picongpu import picmi

import unittest


class TestPicmiPseudoRandomLayout(unittest.TestCase):
    def test_basic(self):
        """simple translation"""
        layout = picmi.PseudoRandomLayout(n_macroparticles_per_cell=7)
        layout.check()

    def test_not_translated(self):
        """pseudo random layout is not translated directly itself"""
        layout = picmi.PseudoRandomLayout(n_macroparticles_per_cell=7)
        with self.assertRaises(AttributeError):
            layout.get_as_pypicongpu()

    def test_invalid(self):
        """erros for invalid params entered"""
        with self.assertRaisesRegex(Exception, ".*per.*"):
            layout = picmi.PseudoRandomLayout(n_macroparticles=700)
            layout.check()

        with self.assertRaises(AssertionError):
            layout = picmi.PseudoRandomLayout(n_macroparticles_per_cell=0)
            layout.check()
