"""
This file is part of the PIConGPU.
Copyright 2021-2023 PIConGPU contributors
Authors: Hannes Troepgen, Brian Edward Marre
License: GPLv3+
"""

from picongpu.pypicongpu.species.operation.momentum import Temperature

import unittest


class TestTemperature(unittest.TestCase):
    def test_basic(self):
        """expected functions return something (valid)"""
        t = Temperature()
        t.temperature_kev = 17

        context = t.get_rendering_context()
        self.assertEqual(17, context["temperature_kev"])

    def test_invalid_values(self):
        """temperature must be >=0"""
        t = Temperature()

        for invalid in [-1, -47.1, -0.0000001]:
            t.temperature_kev = invalid
            with self.assertRaisesRegex(ValueError, ".*(0|zero).*"):
                t.check()

    def test_check_passthru(self):
        """rendering calls check"""
        t = Temperature()
        t.temperature_kev = -1

        # check does not pass, ...
        with self.assertRaises(ValueError):
            t.check()

        # ... hence can't be rendered
        with self.assertRaises(ValueError):
            t.get_rendering_context()

    def test_types(self):
        """invalid types are rejected"""
        t = Temperature()
        for invalid in [None, "asd", {}, []]:
            with self.assertRaises(TypeError):
                t.temperature_kev = invalid

        # work
        t.temperature_kev = 17.241
        t.temperature_kev = 0
