"""
This file is part of the PIConGPU.
Copyright 2021-2023 PIConGPU contributors
Authors: Hannes Troepgen, Brian Edward Marre, Alexander Debus, Richard Pausch
License: GPLv3+
"""

from picongpu import picmi

import unittest
from picongpu import pypicongpu
from math import sqrt


class TestPicmiGaussianLaser(unittest.TestCase):
    def test_basic(self):
        """full laser example"""
        picmi_laser = picmi.GaussianLaser(
            wavelength=1,
            waist=2,
            duration=3,
            propagation_direction=[0, 1, 0],
            polarization_direction=[0, 0, 1],
            focal_position=[5, 4, 5],
            centroid_position=[0, 0, 0],
            E0=5,
            picongpu_laguerre_modes=[2.0, 3.0],
            picongpu_laguerre_phases=[4.0, 5.0],
            picongpu_phase=-2,
            picongpu_huygens_surface_positions=[[1, -1], [1, -1], [1, -1]])

        pypic_laser = picmi_laser.get_as_pypicongpu()
        # translated
        self.assertEqual(1, pypic_laser.wavelength)
        self.assertEqual(2, pypic_laser.waist)
        self.assertEqual(3, pypic_laser.duration)
        self.assertEqual([0, 1, 0], pypic_laser.propagation_direction)
        self.assertEqual([0, 0, 1], pypic_laser.polarization_direction)
        self.assertEqual([5, 4, 5], pypic_laser.focus_pos)
        # centroid is not a picongpu input
        self.assertEqual(5, pypic_laser.E0)
        self.assertEqual(
            pypicongpu.laser.GaussianLaser.PolarizationType.LINEAR,
            pypic_laser.polarization_type)
        self.assertEqual([2.0, 3.0], pypic_laser.laguerre_modes)
        self.assertEqual([4.0, 5.0], pypic_laser.laguerre_phases)
        self.assertEqual(-2, pypic_laser.phase)
        self.assertEqual([
            [1, -1], [1, -1], [1, -1]], pypic_laser.huygens_surface_positions)

        # computed values
        self.assertEqual(15, pypic_laser.pulse_init)

    def test_scalar_values_negative(self):
        """waist, duration and wavelelngth must be > 0"""
        with self.assertRaises(ValueError):
            picmi.GaussianLaser(-1, -2, -3,
                                focal_position=[0, 0, 0],
                                centroid_position=[0, -1, 0],
                                propagation_direction=[0, 1, 0],
                                polarization_direction=[1, 0, 0],
                                E0=1)

    def test_values_focal_pos(self):
        """only y of focal pos can be varied"""
        # x, z checked against centroid pos

        # all ok (difference in x)
        picmi_laser = picmi.GaussianLaser(
            1, 2, 3,
            focal_position=[1, 2, -5],
            centroid_position=[.5, 0, .5],
            propagation_direction=[0, 1, 0],
            polarization_direction=[1, 0, 0],
            E0=1)
        self.assertEqual(1, picmi_laser.get_as_pypicongpu().focus_pos[0])
        self.assertEqual(2, picmi_laser.get_as_pypicongpu().focus_pos[1])
        self.assertEqual(-5, picmi_laser.get_as_pypicongpu().focus_pos[2])

    def test_values_propagation_direction(self):
        """only propagation in y+ permitted"""
        invalid_propagation_vectors = [
            [1, 2, 3],
            [0, 0, 1],
            [1, 0, 0],
            [sqrt(2), sqrt(2), 0],
            [1, 0, -1],
            [0, 0, 0],
            [0, -1, 0],
            ]

        for invalid_propagation_vector in invalid_propagation_vectors:
            picmi_laser = picmi.GaussianLaser(
                1, 2, 3,
                focal_position=[.5, 0, .5],
                centroid_position=[.5, 0, .5],
                propagation_direction=invalid_propagation_vector,
                polarization_direction=[1, 0, 0],
                E0=1)
            with self.assertRaisesRegex(Exception, ".*propagation.*"):
                picmi_laser.get_as_pypicongpu()

        # positive direction works
        picmi_laser = picmi.GaussianLaser(
            1, 2, 3,
            focal_position=[.5, 0, .5],
            centroid_position=[.5, 0, .5],
            propagation_direction=[1/sqrt(3), 1/sqrt(3), 1/sqrt(3)],
            polarization_direction=[1, 0, 0],
            E0=1)

    def test_values_polarization_direction(self):
        """polarization_vector must be normalized"""
        invalid_polarizations = [
            [0, 0, 0],
            [1, 1, 1],
            [1, 0, -1],
            [sqrt(2), sqrt(2), 0],
        ]

        for invalid_polarization in invalid_polarizations:
            picmi_laser = picmi.GaussianLaser(
                1, 2, 3,
                focal_position=[0, 0, 0],
                centroid_position=[0, 0, 0],
                propagation_direction=[0, 1, 0],
                polarization_direction=invalid_polarization,
                E0=1)
            with self.assertRaisesRegex(Exception, ".*polarization.*"):
                picmi_laser.get_as_pypicongpu()

        # valid examples:
        valid_polarization_vectors = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]

        for valid_polarization_vector in valid_polarization_vectors:
            picmi_laser = picmi.GaussianLaser(
                1, 2, 3,
                focal_position=[0, 0, 0],
                centroid_position=[0, 0, 0],
                propagation_direction=[0, 1, 0],
                polarization_direction=valid_polarization_vector,
                E0=1)
            pypic_laser = picmi_laser.get_as_pypicongpu()
            self.assertEqual(
                valid_polarization_vector,
                pypic_laser.polarization_direction)

    def test_minimal(self):
        """mimimal possible initialization"""
        # does not throw, normal usage process works
        picmi_laser = picmi.GaussianLaser(1, 2, 3,
                                          focal_position=[0, 0, 0],
                                          centroid_position=[0, 0, 0],
                                          propagation_direction=[0, 1, 0],
                                          polarization_direction=[1, 0, 0],
                                          E0=1)
        pypic_laser = picmi_laser.get_as_pypicongpu()
        self.assertNotEqual({}, pypic_laser.get_rendering_context())

    def test_values_centroid_position_y_smaller_equal_zero(self):
        """centroid position must have y<=0"""

        with self.assertRaisesRegex(Exception, ".*centroid.*[yY].*(zero|0).*"):
            picmi.GaussianLaser(1, 2, 3,
                                centroid_position=[1, 1, 1],
                                focal_position=[1, 1, 1],
                                propagation_direction=[0, 1, 0],
                                polarization_direction=[1, 0, 0],
                                E0=1).get_as_pypicongpu()

        # valid example:
        self.assertNotEqual({},
                            picmi.GaussianLaser(
                                1, 2, 3,
                                centroid_position=[12, -3, 7],
                                focal_position=[12, 0, 7],
                                propagation_direction=[0, 1, 0],
                                polarization_direction=[1, 0, 0],
                                E0=1)
                            .get_as_pypicongpu().get_rendering_context())

    def test_laguerre_modes_types(self):
        """laguerre type-check before translation"""
        with self.assertRaises(TypeError):
            picmi.GaussianLaser(
                1, 2, 3,
                focal_position=[0, 0, 0],
                centroid_position=[0, 0, 0],
                propagation_direction=[0, 1, 0],
                E0=0,
                picongpu_laguerre_modes=["not float"])

        with self.assertRaises(TypeError):
            picmi.GaussianLaser(
                1, 2, 3,
                focal_position=[.5, 0, .5],
                centroid_position=[.5, 0, .5],
                propagation_direction=[0, 1, 0],
                E0=0,
                picongpu_laguerre_phases=set(2.0))

    def test_laguerre_modes_optional(self):
        """laguerre modes are optional"""
        # allowed: not given at all
        picmi_laser = picmi.GaussianLaser(
            wavelength=1,
            waist=2,
            duration=3,
            focal_position=[0, 0, 0],
            centroid_position=[0, 0, 0],
            E0=5,
            propagation_direction=[0, 1, 0],
            polarization_direction=[1, 0, 0])
        pypic_laser = picmi_laser.get_as_pypicongpu()
        self.assertEqual([1.0], pypic_laser.laguerre_modes)
        self.assertEqual([0.0], pypic_laser.laguerre_phases)

        # allowed: explicitly None
        picmi_laser = picmi.GaussianLaser(
            wavelength=1,
            waist=2,
            duration=3,
            focal_position=[0, 0, 0],
            centroid_position=[0, 0, 0],
            E0=5,
            propagation_direction=[0, 1, 0],
            polarization_direction=[1, 0, 0],
            picongpu_laguerre_modes=None,
            picongpu_laguerre_phases=None)
        pypic_laser = picmi_laser.get_as_pypicongpu()
        self.assertEqual([1.0], pypic_laser.laguerre_modes)
        self.assertEqual([0.0], pypic_laser.laguerre_phases)

        # not allowed: only phases (or only modes) given
        with self.assertRaisesRegex(Exception, ".*[Ll]aguerre.*"):
            picmi.GaussianLaser(
                wavelength=1,
                waist=2,
                duration=3,
                focal_position=[0, 0, 0],
                centroid_position=[0, 0, 0],
                polarization_direction=[1, 0, 0],
                E0=5,
                propagation_direction=[0, 1, 0],
                picongpu_laguerre_modes=[1.0, 2.0],
                picongpu_laguerre_phases=None)

        with self.assertRaisesRegex(Exception, ".*[Ll]aguerre.*"):
            picmi.GaussianLaser(
                wavelength=1,
                waist=2,
                duration=3,
                focal_position=[0, 0, 0],
                centroid_position=[0, 0, 0],
                polarization_direction=[1, 0, 0],
                E0=5,
                propagation_direction=[0, 1, 0],
                picongpu_laguerre_phases=[1.0, 2.0])

    def test_values_centroid_position_center(self):
        """centroid position is fixed for given bounding box"""
        # on its own, any centroid poisition with y=0 is permitted
        picmi_laser = picmi.GaussianLaser(
            1, 2, 3,
            centroid_position=[8.5, -3, 21],
            focal_position=[8.5, 2, 21],
            propagation_direction=[0, 1, 0],
            polarization_direction=[0, 0, 1],
            E0=1)
        self.assertNotEqual(
            {}, picmi_laser.get_as_pypicongpu().get_rendering_context())

        grid_valid = picmi.Cartesian3DGrid(
            number_of_cells=[128, 512, 256],
            lower_bound=[0, 0, 0],
            upper_bound=[17, 192, 42],
            lower_boundary_conditions=["periodic", "periodic", "open"],
            upper_boundary_conditions=["periodic", "periodic", "open"])

        # valid grid-laser combination working
        solver_valid = picmi.ElectromagneticSolver(method="Yee",
                                                   grid=grid_valid)
        sim_valid = picmi.Simulation(time_step_size=1,
                                     max_steps=2,
                                     solver=solver_valid)
        sim_valid.add_laser(picmi_laser, None)

        # translates without issue:
        self.assertNotEqual(
            {}, sim_valid.get_as_pypicongpu().get_rendering_context())
