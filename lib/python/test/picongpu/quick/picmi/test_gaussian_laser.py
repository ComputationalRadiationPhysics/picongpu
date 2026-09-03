"""
This file is part of PIConGPU.
Copyright 2021-2024 PIConGPU contributors
Authors: Hannes Troepgen, Brian Edward Marre, Alexander Debus, Richard Pausch
License: GPLv3+
"""

from math import sqrt
from unittest import TestCase

import os
import re
import tempfile
import pytest
from picongpu import picmi
from pydantic import ValidationError
from scipy.constants import c


def _pulse_duration(duration_picmi_si):
    """PIConGPU PULSE_DURATION (1 sigma of the intensity) from the PICMI-standard
    duration (1/e field width), i.e. PULSE_DURATION = duration / 2"""
    return duration_picmi_si / 2.0


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
        # picmi `duration` is the PICMI-standard 1/e field width (tau);
        # pypicongpu `pulse_duration_si` is PULSE_DURATION = tau / 2 (#5739)
        assert abs(pypic_laser.pulse_duration_si - _pulse_duration(3)) < 1e-15
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
        # pulse_init is counted in units of PULSE_DURATION (1 sigma of the intensity) (#5739)
        assert (
            abs(
                -2.0
                * picmi_laser.centroid_position[1]
                / picmi_laser.propagation_direction[1]
                / c
                / _pulse_duration(picmi_laser.duration)
                - pypic_laser.pulse_init
            )
            < 1e-10
        )

    def test_duration_converted_to_pulse_duration(self):
        """picmi `duration` is the PICMI-standard 1/e field width, pypicongpu expects PULSE_DURATION = duration / 2 (#5739)"""
        duration_picmi_si = 30e-15
        picmi_laser = picmi.GaussianLaser(
            wavelength=800e-9,
            waist=12e-6,
            duration=duration_picmi_si,
            focal_position=[0.0, 5e-6, 0.0],
            centroid_position=[0.0, -5e-6, 0.0],
            propagation_direction=[0, 1, 0],
            polarization_direction=[1, 0, 0],
            a0=1.0,
        )
        # the picmi object keeps the duration value as given (PICMI contract)
        assert picmi_laser.duration == duration_picmi_si

        pypic_laser = picmi_laser.get_as_pypicongpu()
        # the translated value is PULSE_DURATION = duration / 2 (1 sigma of the intensity)
        assert abs(pypic_laser.pulse_duration_si - _pulse_duration(duration_picmi_si)) < 1e-24
        # pulse_init is also counted in units of PULSE_DURATION
        assert (
            abs(
                -2.0
                * picmi_laser.centroid_position[1]
                / picmi_laser.propagation_direction[1]
                / c
                / _pulse_duration(picmi_laser.duration)
                - pypic_laser.pulse_init
            )
            < 1e-10
        )

    def test_values_focal_pos(self):
        """only y of focal pos can be varied"""
        # x, z checked against centroid pos

        # all ok (difference in x)
        picmi_laser = picmi.GaussianLaser(
            wavelength=1,
            waist=2,
            duration=3,
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
            with self.assertRaises(ValidationError):
                picmi.GaussianLaser(
                    wavelength=1,
                    waist=2,
                    duration=3,
                    focal_position=[0.5, 0, 0.5],
                    centroid_position=[0.5, 0, 0.5],
                    propagation_direction=invalid_propagation_vector,
                    polarization_direction=[1, 0, 0],
                    E0=1,
                )

        # positive direction works
        picmi.GaussianLaser(
            wavelength=1,
            waist=2,
            duration=3,
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
            with self.assertRaises(ValidationError):
                picmi.GaussianLaser(
                    wavelength=1,
                    waist=2,
                    duration=3,
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
                wavelength=1,
                waist=2,
                duration=3,
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
            wavelength=1,
            waist=2,
            duration=3,
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

        with self.assertRaises(ValidationError):
            picmi.GaussianLaser(
                wavelength=1,
                waist=2,
                duration=3,
                centroid_position=[1, 1, 1],
                focal_position=[1, 1, 1],
                propagation_direction=[0, 1, 0],
                polarization_direction=[1, 0, 0],
                E0=1,
            ).get_as_pypicongpu()

        # valid example:
        assert (
            picmi.GaussianLaser(
                wavelength=1,
                waist=2,
                duration=3,
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
        with self.assertRaises(ValidationError):
            picmi.GaussianLaser(
                wavelength=1,
                waist=2,
                duration=3,
                focal_position=[0, 0, 0],
                centroid_position=[0, 0, 0],
                propagation_direction=[0, 1, 0],
                E0=0,
                picongpu_laguerre_modes=["not float"],
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
            wavelength=1,
            waist=2,
            duration=3,
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

        with self.assertRaises(ValidationError):
            picmi.GaussianLaser(
                wavelength=1,
                waist=2,
                duration=3,
                focal_position=[0.5, 0, 0.5],
                centroid_position=[0.5, 0, 0.5],
                propagation_direction=[0, 1, 0],
                polarization_direction=[1, 0, 0],
                E0=1,
                a0=1,
            )

    def test_no_a0_E0(self):
        """either a0 or E0 have to be set"""

        with self.assertRaises(ValidationError):
            picmi.GaussianLaser(
                wavelength=1,
                waist=2,
                duration=3,
                focal_position=[0.5, 0, 0.5],
                centroid_position=[0.5, 0, 0.5],
                propagation_direction=[0, 1, 0],
                polarization_direction=[1, 0, 0],
            )


def test_duration_rendered_into_incident_field():
    """the PICMI-standard duration (1/e field width) must be rendered as PULSE_DURATION = duration / 2
    (1 sigma of the intensity) in incidentField.param (#5739)"""
    duration_picmi_si = 30e-15
    laser = picmi.GaussianLaser(
        wavelength=800e-9,
        waist=12e-6,
        duration=duration_picmi_si,
        focal_position=[8.5, 2, 21],
        centroid_position=[8.5, -3, 21],
        propagation_direction=[0, 1, 0],
        polarization_direction=[1, 0, 0],
        a0=1.0,
    )
    grid = picmi.Cartesian3DGrid(
        number_of_cells=[128, 512, 256],
        lower_bound=[0, 0, 0],
        upper_bound=[17, 192, 42],
        lower_boundary_conditions=["periodic", "periodic", "open"],
        upper_boundary_conditions=["periodic", "periodic", "open"],
    )
    solver = picmi.ElectromagneticSolver(method="Yee", grid=grid)
    sim = picmi.Simulation(time_step_size=1, max_steps=2, solver=solver)
    sim.add_laser(laser, None)

    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = os.path.join(tmpdir, "input")
        sim.write_input_file(output_dir)
        rendered_path = os.path.join(output_dir, "include", "picongpu", "param", "incidentField.param")
        with open(rendered_path) as rendered_file:
            rendered = rendered_file.read()

    match = re.search(r"PULSE_DURATION_SI = ([0-9eE.+-]+);", rendered)
    assert match is not None, "PULSE_DURATION_SI not found in rendered incidentField.param"
    assert abs(float(match.group(1)) - _pulse_duration(duration_picmi_si)) < 1e-24
    # the passed duration (tau) itself must not end up in PULSE_DURATION_SI
    assert abs(float(match.group(1)) - duration_picmi_si) > 1e-14


def test_dispersive_pulse_laser_duration_converted_to_pulse_duration():
    """DispersivePulseLaser inherits the PICMI-standard duration semantics of GaussianLaser (#5739)"""
    duration_picmi_si = 30e-15
    picmi_laser = picmi.DispersivePulseLaser(
        wavelength=800e-9,
        waist=12e-6,
        duration=duration_picmi_si,
        focal_position=[0.0, 5e-6, 0.0],
        centroid_position=[0.0, -5e-6, 0.0],
        propagation_direction=[0, 1, 0],
        polarization_direction=[1, 0, 0],
        a0=1.0,
    )
    pypic_laser = picmi_laser.get_as_pypicongpu()
    assert abs(pypic_laser.pulse_duration_si - _pulse_duration(duration_picmi_si)) < 1e-24
