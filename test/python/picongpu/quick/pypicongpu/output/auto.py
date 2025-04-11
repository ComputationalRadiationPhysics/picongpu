"""
This file is part of PIConGPU.
Copyright 2021-2025 PIConGPU contributors
Authors: Hannes Troepgen, Brian Edward Marre, Julian Lenz
License: GPLv3+
"""

from picongpu.pypicongpu.output.timestepspec import TimeStepSpec
from picongpu.pypicongpu.output import Auto

import unittest
import typeguard


class TestAuto(unittest.TestCase):
    def test_types(self):
        """type safety is ensured"""
        a = Auto()

        invalid_periods = [13.2, [], "2", None, {}, (1)]
        for invalid_period in invalid_periods:
            with self.assertRaises(typeguard.TypeCheckError):
                a.period = invalid_periods
        # ok
        a.period = TimeStepSpec([slice(0, None, 17)])

    def test_rendering(self):
        """data transformed to template-consumable version"""
        a = Auto()
        a.period = TimeStepSpec([slice(0, None, 17)])

        # normal rendering
        context = a.get_rendering_context()
        self.assertTrue(context["typeID"]["auto"])
        context = context["data"]
        self.assertEqual(17, context["period"]["specs"][0]["step"])
