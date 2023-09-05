"""
This file is part of the PIConGPU.
Copyright 2021-2022 PIConGPU contributors
Authors: Hannes Troepgen, Brian Edward Marre, Alexander Debus
License: GPLv3+
"""

from picongpu.pypicongpu.laser import GaussianLaser

import unittest
import logging
import copy

import typeguard

""" @file we only test for types here, test for values errors is done in the
   custom picmi-objects"""


class TestGaussianLaser(unittest.TestCase):
    def setUp(self):
        self.laser = GaussianLaser()
        self.laser.wavelength = 1.2
        self.laser.waist = 3.4
        self.laser.duration = 5.6
        self.laser.focus_pos = [0, 7.8, 0]
        self.laser.phase = 2.9
        self.laser.E0 = 9.0
        self.laser.pulse_init = 1.3
        self.laser.propagation_direction = [0., 1., 0.]
        self.laser.polarization_type = GaussianLaser.PolarizationType.LINEAR
        self.laser.polarization_direction = [0., 1., 0.]
        self.laser.laguerre_modes = [1.0]
        self.laser.laguerre_phases = [0.0]
        self.laser.huygens_surface_positions = [[1, -1], [1, -1], [1, -1]]

    def test_types(self):
        """invalid types are rejected"""
        laser = GaussianLaser()
        for not_float in [None, [], {}, "1"]:
            with self.assertRaises(typeguard.TypeCheckError):
                laser.wavelength = not_float
            with self.assertRaises(typeguard.TypeCheckError):
                laser.waist = not_float
            with self.assertRaises(typeguard.TypeCheckError):
                laser.duration = not_float
            with self.assertRaises(typeguard.TypeCheckError):
                laser.phase = not_float
            with self.assertRaises(typeguard.TypeCheckError):
                laser.E0 = not_float
            with self.assertRaises(typeguard.TypeCheckError):
                laser.pulse_init = not_float

        for not_position_vector in [1, 1., None, ["string"]]:
            with self.assertRaises(typeguard.TypeCheckError):
                laser.focus_pos = not_position_vector

        for not_polarization_type in [1, 1.3, None, "", []]:
            with self.assertRaises(typeguard.TypeCheckError):
                laser.polarization_type = not_polarization_type

        for not_direction_vector in [1, 1.3, None, "", ["string"]]:
            with self.assertRaises(typeguard.TypeCheckError):
                laser.polarization_direction = not_direction_vector
            with self.assertRaises(typeguard.TypeCheckError):
                laser.propagation_direction = not_direction_vector

        for invalid_list in [None, 1.2, "1.2", ["string"]]:
            with self.assertRaises(typeguard.TypeCheckError):
                laser.laguerre_modes = invalid_list
            with self.assertRaises(typeguard.TypeCheckError):
                laser.laguerre_phases = invalid_list
            with self.assertRaises(typeguard.TypeCheckError):
                laser.polarization_direction = invalid_list
            with self.assertRaises(typeguard.TypeCheckError):
                laser.propagation_direction = invalid_list
            with self.assertRaises(typeguard.TypeCheckError):
                laser.huygens_surface_positions = invalid_list

    def test_polarization_type(self):
        """polarization type enum sanity checks"""
        lin = GaussianLaser.PolarizationType.LINEAR
        circular = GaussianLaser.PolarizationType.CIRCULAR

        self.assertNotEqual(lin, circular)

        self.assertNotEqual(lin.get_cpp_str(), circular.get_cpp_str())

        for polarization_type in [lin, circular]:
            self.assertEqual(str, type(polarization_type.get_cpp_str()))

    def test_invalid_huygens_surface_description_types(self):
        """Huygens surfaces must be described as
           [[x_min:int, x_max:int], [y_min:int,y_max:int],
           [z_min:int, z_max:int]]"""
        laser = self.laser

        invalid_elements = [None, [], [1.2, 3.4]]
        valid_rump = [[5, 6], [7, 8]]

        invalid_descriptions = []
        for invalid_element in invalid_elements:
            for pos in range(3):
                base = copy.deepcopy(valid_rump)
                base.insert(pos, invalid_element)
                invalid_descriptions.append(base)

        for invalid_description in invalid_descriptions:
            with self.assertRaises(TypeError):
                laser.huygens_surface_positions(invalid_description)

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
        self.assertEqual(context["pulse_duration_si"], self.laser.duration)
        self.assertEqual(context["focus_pos_si"], [
            {"component": self.laser.focus_pos[0]},
            {"component": self.laser.focus_pos[1]},
            {"component": self.laser.focus_pos[2]}])
        self.assertEqual(context["phase"], self.laser.phase)
        self.assertEqual(context["E0_si"], self.laser.E0)
        self.assertEqual(context["pulse_init"], self.laser.pulse_init)
        self.assertEqual(context["propagation_direction"], [
            {"component": self.laser.propagation_direction[0]},
            {"component": self.laser.propagation_direction[1]},
            {"component": self.laser.propagation_direction[2]}])
        self.assertEqual(context["polarization_type"],
                         self.laser.polarization_type.get_cpp_str())
        self.assertEqual(context["polarization_direction"], [
            {"component": self.laser.polarization_direction[0]},
            {"component": self.laser.polarization_direction[1]},
            {"component": self.laser.polarization_direction[2]}])
        self.assertEqual(context["laguerre_modes"],
                         [{"single_laguerre_mode": 1.0}])
        self.assertEqual(context["laguerre_phases"],
                         [{"single_laguerre_phase": 0.0}])
        self.assertEqual(context["modenumber"], 0)
        self.assertEqual(context["huygens_surface_positions"], {
            "row_x": {
                "negative": self.laser.huygens_surface_positions[0][0],
                "positive": self.laser.huygens_surface_positions[0][1]},
            "row_y": {
                "negative": self.laser.huygens_surface_positions[1][0],
                "positive": self.laser.huygens_surface_positions[1][1]},
            "row_z": {
                "negative": self.laser.huygens_surface_positions[2][0],
                "positive": self.laser.huygens_surface_positions[2][1]},
            })
