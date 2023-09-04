"""
This file is part of the PIConGPU.
Copyright 2021-2023 PIConGPU contributors
Authors: Hannes Troepgen, Brian Edward Marre
License: GPLv3+
"""

from picongpu.pypicongpu.output import Auto

import unittest


class TestAuto(unittest.TestCase):
    def test_empty(self):
        """empty args handled correctly"""
        a = Auto()
        # unset args
        with self.assertRaises(Exception):
            a.check()

        a.period = 1

        # ok:
        a.check()

    def test_types(self):
        """type safety is ensured"""
        a = Auto()

        invalid_periods = [13.2, [], "2", None, {}, (1)]
        for invalid_period in invalid_periods:
            with self.assertRaises(TypeError):
                a.period = invalid_periods
        # ok
        a.period = 17

    def test_period_invalid(self):
        """period must be positive, non-zero integer"""
        a = Auto()

        invalid_periods = [-1, 0, -1273]
        for invalid_period in invalid_periods:
            with self.assertRaises(ValueError):
                a.period = invalid_period
                a.check()

        # ok
        a.period = 1
        a.period = 2

    def test_rendering(self):
        """data transformed to template-consumable version"""
        a = Auto()
        a.period = 42

        # normal rendering
        context = a.get_rendering_context()
        self.assertEqual(42, context["period"])

        # refuses to render if check does not pass
        a.period = -1
        with self.assertRaises(ValueError):
            a.get_rendering_context()
