# SPDX-FileCopyrightText: PIConGPU contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""
This file is part of PIConGPU.
Copyright 2021-2024 PIConGPU contributors
Authors: Hannes Troepgen, Brian Edward Marre, Alexander Debus, Richard Pausch
License: GPLv3+
"""

from picongpu import picmi

from unittest import TestCase
from math import sqrt
from scipy.constants import c

import pytest


class TestPicmiGaussianLaser(TestCase):
    def test_basic(self):
        """full laser example"""
        picmi_laser = picmi.GaussianLaser(
            wavelength=1,
            waist=2,
            duration=3,
            propagation_direction=[0, 1, 0],
            polarization_direction=[0, 0, 1],
            focal_position=[5, 4, 5],
            centroid_position=[5, -1.5, 5],
            E0=5,
            picongpu_laguerre_modes=[2.0, 3.0],
            picongpu_laguerre_phases=[4.0, 5.0],
            phi0=-2,
            picongpu_huygens_surface_positions=[[1, -1], [1, -1], [1, -1]],
        )

        pypic_laser = picmi_laser.get_as_pypicongpu()
        # translated
        assert pypic_laser.wave_length_si == 1
        assert pypic_laser.waist_si == 2
        assert pypic_laser.pulse_duration_si == 3
        assert pypic_laser.propagation_direction == (0, 1, 0)
        assert pypic_laser.polarization_direction == (0, 0, 1)
        assert pypic_laser.focus_pos_si == (5, 4, 5)
        # centroid is not a picongpu input
        assert pypic_laser.E0_si == 5
        assert picmi.lasers.PolarizationType.LINEAR.get_as_pypicongpu() == pypic_laser.polarization_type
        assert pypic_laser.laguerre_modes == [2.0, 3.0]
        assert pypic_laser.laguerre_phases == [4.0, 5.0]
        assert pypic_laser.phase == -2
        assert pypic_laser.huygens_surface_positions == [[1, -1], [1, -1], [1, -1]]

        # computed values
        assert (
            abs(
                -2.0
                * picmi_laser.centroid_position[1]
                / picmi_laser.propagation_direction[1]
                / c
                / picmi_laser.duration
                - pypic_laser.pulse_init
            )
            < 1e-10
        )

    def test_scalar_values_negative(self):
        """waist, duration and wavelelngth must be > 0"""
        with pytest.raises(ValueError):
            picmi.GaussianLaser(
                -1,
                -2,
                -3,
                focal_position=[0, 0, 0],
                centroid_position=[0, -1, 0],
                propagation_direction=[0, 1, 0],
                polarization_direction=[1, 0, 0],
                E0=1,
            )

    def test_values_focal_pos(self):
        """only y of focal pos can be varied"""
        # x, z checked against centroid pos

        # all ok (difference in x)
        picmi_laser = picmi.GaussianLaser(
            1,
            2,
            3,
            focal_position=[1, 2, -5],
            centroid_position=[1, 0, -5],
            propagation_direction=[0, 1, 0],
            polarization_direction=[1, 0, 0],
            E0=1,
        )
        assert picmi_laser.get_as_pypicongpu().focus_pos_si[0] == 1
        assert picmi_laser.get_as_pypicongpu().focus_pos_si[1] == 2
        assert picmi_laser.get_as_pypicongpu().focus_pos_si[2] == -5

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
            with pytest.raises(ValueError, match=".*propagation.*"):
                picmi.GaussianLaser(
                    1,
                    2,
                    3,
                    focal_position=[0.5, 0, 0.5],
                    centroid_position=[0.5, 0, 0.5],
                    propagation_direction=invalid_propagation_vector,
                    polarization_direction=[1, 0, 0],
                    E0=1,
                )

        # positive direction works
        picmi.GaussianLaser(
            1,
            2,
            3,
            focal_position=[0.5, 0, 0.5],
            centroid_position=[0.5, 0, 0.5],
            propagation_direction=[1 / sqrt(3), 1 / sqrt(3), 1 / sqrt(3)],
            polarization_direction=[1, 0, 0],
            E0=1,
        )

    def test_values_polarization_direction(self):
        """polarization_vector must be normalized"""
        invalid_polarizations = [
            [0, 0, 0],
            [1, 1, 1],
            [1, 0, -1],
            [sqrt(2), sqrt(2), 0],
        ]

        for invalid_polarization in invalid_polarizations:
            with pytest.raises(ValueError, match=".*polarization.*"):
                picmi.GaussianLaser(
                    1,
                    2,
                    3,
                    focal_position=[0, 0, 0],
                    centroid_position=[0, 0, 0],
                    propagation_direction=[0, 1, 0],
                    polarization_direction=invalid_polarization,
                    E0=1,
                )

        # valid examples:
        valid_polarization_vectors = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]

        for valid_polarization_vector in valid_polarization_vectors:
            picmi_laser = picmi.GaussianLaser(
                1,
                2,
                3,
                focal_position=[0, 0, 0],
                centroid_position=[0, 0, 0],
                propagation_direction=[0, 1, 0],
                polarization_direction=valid_polarization_vector,
                E0=1,
            )
            pypic_laser = picmi_laser.get_as_pypicongpu()
            assert pypic_laser.polarization_direction == valid_polarization_vector

    def test_minimal(self):
        """mimimal possible initialization"""
        # does not throw, normal usage process works
        picmi_laser = picmi.GaussianLaser(
            1,
            2,
            3,
            focal_position=[0, 0, 0],
            centroid_position=[0, -1, 0],
            propagation_direction=[0, 1, 0],
            polarization_direction=[1, 0, 0],
            E0=1,
        )
        pypic_laser = picmi_laser.get_as_pypicongpu()
        assert pypic_laser.model_dump() != {}

    def test_values_centroid_position_y_smaller_equal_zero(self):
        """centroid position must have y<=0"""

        with pytest.raises(ValueError, match=".*centroid.*[yY].*(zero|0).*"):
            picmi.GaussianLaser(
                1,
                2,
                3,
                centroid_position=[1, 1, 1],
                focal_position=[1, 1, 1],
                propagation_direction=[0, 1, 0],
                polarization_direction=[1, 0, 0],
                E0=1,
            ).get_as_pypicongpu()

        # valid example:
        assert (
            picmi.GaussianLaser(
                1,
                2,
                3,
                centroid_position=[12, -3, 7],
                focal_position=[12, 0, 7],
                propagation_direction=[0, 1, 0],
                polarization_direction=[1, 0, 0],
                E0=1,
            )
            .get_as_pypicongpu()
            .model_dump()
            != {}
        )

    def test_laguerre_modes_types(self):
        """laguerre type-check before translation"""
        with pytest.raises(TypeError):
            picmi.GaussianLaser(
                1,
                2,
                3,
                focal_position=[0, 0, 0],
                centroid_position=[0, 0, 0],
                propagation_direction=[0, 1, 0],
                E0=0,
                picongpu_laguerre_modes=["not float"],
            )

        with pytest.raises(TypeError):
            picmi.GaussianLaser(
                1,
                2,
                3,
                focal_position=[0.5, 0, 0.5],
                centroid_position=[0.5, 0, 0.5],
                propagation_direction=[0, 1, 0],
                E0=0,
                picongpu_laguerre_phases=set(2.0),
            )

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
            polarization_direction=[1, 0, 0],
        )
        pypic_laser = picmi_laser.get_as_pypicongpu()
        assert pypic_laser.laguerre_modes == [1.0]
        assert pypic_laser.laguerre_phases == [0.0]

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
            picongpu_laguerre_phases=None,
        )
        pypic_laser = picmi_laser.get_as_pypicongpu()
        assert pypic_laser.laguerre_modes == [1.0]
        assert pypic_laser.laguerre_phases == [0.0]

        # not allowed: only phases (or only modes) given
        with pytest.raises(Exception, match=".*[Ll]aguerre.*"):
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
                picongpu_laguerre_phases=None,
            )

        with pytest.raises(Exception, match=".*[Ll]aguerre.*"):
            picmi.GaussianLaser(
                wavelength=1,
                waist=2,
                duration=3,
                focal_position=[0, 0, 0],
                centroid_position=[0, 0, 0],
                polarization_direction=[1, 0, 0],
                E0=5,
                propagation_direction=[0, 1, 0],
                picongpu_laguerre_phases=[1.0, 2.0],
            )

    def test_values_centroid_position_center(self):
        """centroid position is fixed for given bounding box"""
        # on its own, any centroid poisition with y=0 is permitted
        picmi_laser = picmi.GaussianLaser(
            1,
            2,
            3,
            centroid_position=[8.5, -3, 21],
            focal_position=[8.5, 2, 21],
            propagation_direction=[0, 1, 0],
            polarization_direction=[0, 0, 1],
            E0=1,
        )
        assert picmi_laser.get_as_pypicongpu().model_dump() != {}

        grid_valid = picmi.Cartesian3DGrid(
            number_of_cells=[128, 512, 256],
            lower_bound=[0, 0, 0],
            upper_bound=[17, 192, 42],
            lower_boundary_conditions=["periodic", "periodic", "open"],
            upper_boundary_conditions=["periodic", "periodic", "open"],
        )

        # valid grid-laser combination working
        solver_valid = picmi.ElectromagneticSolver(method="Yee", grid=grid_valid)
        sim_valid = picmi.Simulation(time_step_size=1, max_steps=2, solver=solver_valid)
        sim_valid.add_laser(picmi_laser, None)

        # translates without issue:
        assert sim_valid.get_as_pypicongpu().model_dump() != {}

    def test_overdefinition_a0_E0(self):
        """only either a0 or E0 allowed to be set"""

        with pytest.raises(ValueError, match="Only one of E0 or a0 should be specified. You set both."):
            picmi.GaussianLaser(
                1,
                2,
                3,
                focal_position=[0.5, 0, 0.5],
                centroid_position=[0.5, 0, 0.5],
                propagation_direction=[0, 1, 0],
                polarization_direction=[1, 0, 0],
                E0=1,
                a0=1,
            )

    def test_no_a0_E0(self):
        """either a0 or E0 have to be set"""

        with pytest.raises(ValueError, match="Both E0 or a0 are None. You must specify exactly one."):
            picmi.GaussianLaser(
                1,
                2,
                3,
                focal_position=[0.5, 0, 0.5],
                centroid_position=[0.5, 0, 0.5],
                propagation_direction=[0, 1, 0],
                polarization_direction=[1, 0, 0],
            )
