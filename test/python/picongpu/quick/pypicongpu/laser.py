"""
This file is part of the PIConGPU.
Copyright 2021-2022 PIConGPU contributors
Authors: Hannes Tröpgen, Brian Edward Marré, Alexander Debus
License: GPLv3+
"""

from picongpu.pypicongpu.laser import GaussianLaser

import unittest
import logging


class TestGaussianLaser(unittest.TestCase):
    def setUp(self):
        self.laser = GaussianLaser()
        self.laser.wavelength = 1.2
        self.laser.waist = 3.4
        self.laser.duration = 5.6
        self.laser.focus_pos = 7.8
        self.laser.phase = 2.9
        self.laser.E0 = 9.0
        self.laser.pulse_init = 1.3
        self.laser.init_plane_y = 1
        self.laser.polarization_type = GaussianLaser.PolarizationType.LINEAR_X
        self.laser.laguerre_modes = [1.0]
        self.laser.laguerre_phases = [0.0]

    def test_types(self):
        """invalid types are rejected"""
        laser = GaussianLaser()
        for not_float in [None, [], {}, "1"]:
            with self.assertRaises(TypeError):
                laser.wavelength = not_float
            with self.assertRaises(TypeError):
                laser.waist = not_float
            with self.assertRaises(TypeError):
                laser.duration = not_float
            with self.assertRaises(TypeError):
                laser.focus_pos = not_float
            with self.assertRaises(TypeError):
                laser.phase = not_float
            with self.assertRaises(TypeError):
                laser.E0 = not_float
            with self.assertRaises(TypeError):
                laser.pulse_init = not_float

        for not_int in [None, [], {}, "1", 1.3]:
            with self.assertRaises(TypeError):
                laser.init_plane_y = not_int

        for not_polarization_type in [1, 1.3, None, "", []]:
            with self.assertRaises(TypeError):
                laser.polarization_type = not_polarization_type

        for invalid_list in [None, 1.2, "1.2", ["string"]]:
            with self.assertRaises(TypeError):
                laser.laguerre_modes = invalid_list
            with self.assertRaises(TypeError):
                laser.laguerre_phases = invalid_list

    def test_polarization_type(self):
        """polarization type enum sanity checks"""
        lin_x = GaussianLaser.PolarizationType.LINEAR_X
        lin_z = GaussianLaser.PolarizationType.LINEAR_Z
        circular = GaussianLaser.PolarizationType.CIRCULAR

        self.assertNotEqual(lin_x, lin_z)
        self.assertNotEqual(lin_z, circular)
        self.assertNotEqual(circular, lin_x)

        self.assertNotEqual(lin_x.get_cpp_str(), lin_z.get_cpp_str())
        self.assertNotEqual(lin_z.get_cpp_str(), circular.get_cpp_str())
        self.assertNotEqual(circular.get_cpp_str(), lin_x.get_cpp_str())

        for polarization_type in [lin_x, lin_z, circular]:
            self.assertEqual(str, type(polarization_type.get_cpp_str()))

    def test_invalid_laguerre_modes_empty(self):
        """laguerre modes must be set non-empty"""
        laser = self.laser
        laser.laguerre_modes = []
        with self.assertRaisesRegex(ValueError, ".*mode.*empty.*"):
            laser.get_rendering_context()
        laser.laguerre_modes = [1.0]
        laser.laguerre_phases = []
        with self.assertRaisesRegex(ValueError, ".*phase.*empty.*"):
            laser.get_rendering_context()

    def test_invalid_laguerre_modes_invalid_length(self):
        """num of laguerre modes/phases must be equal"""
        laser = self.laser
        laser.laguerre_modes = [1.0]
        laser.laguerre_phases = [2, 3]

        with self.assertRaisesRegex(ValueError, ".*[Ll]aguerre.*length.*"):
            laser.get_rendering_context()

        laser.laguerre_modes = [1, 0]
        # no error anymore:
        self.assertNotEqual({}, laser.get_rendering_context())

    def test_positive_definite_laguerre_modes(self):
        """test whether laguerre modes are positive definite"""
        laser = self.laser
        laser.laguerre_modes = [-1.0]
        with self.assertLogs(level="WARNING") as caught_logs:
            # valid, but warns
            self.assertNotEqual({}, laser.get_rendering_context())
        self.assertEqual(1, len(caught_logs.output))
        self.assertTrue("positive" in caught_logs.output[0])

        # reverse: no warning if >=0
        laser.laguerre_modes = [0]
        with self.assertLogs(level="WARNING") as other_caught_logs:
            # no warning
            self.assertNotEqual({}, laser.get_rendering_context())
            # produce at least one warning, workaround for python <= 3.9
            logging.warning("TESTWARN")
        self.assertEqual(1, len(other_caught_logs.output))
        self.assertTrue("TESTWARN" in other_caught_logs.output[0])

    def test_translation(self):
        """is translated to context object"""
        # note: implicitly checks against schema
        context = self.laser.get_rendering_context()
        self.assertEqual(context["wave_length_si"], self.laser.wavelength)
        self.assertEqual(context["waist_si"], self.laser.waist)
        self.assertEqual(context["pulse_length_si"], self.laser.duration)
        self.assertEqual(context["focus_pos_si"], self.laser.focus_pos)
        self.assertEqual(context["phase"], self.laser.phase)
        self.assertEqual(context["polarization_type"],
                         self.laser.polarization_type.get_cpp_str())
        self.assertEqual(context["pulse_init"], self.laser.pulse_init)
        self.assertEqual(context["init_plane_y"], self.laser.init_plane_y)
        self.assertEqual(context["laguerre_modes"],
                         [{"single_laguerre_mode": 1.0}])
        self.assertEqual(context["laguerre_phases"],
                         [{"single_laguerre_phase": 0.0}])
        self.assertEqual(context["modenumber"], 0)
