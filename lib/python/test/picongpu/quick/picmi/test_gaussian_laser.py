"""
This file is part of PIConGPU.
Copyright 2021-2024 PIConGPU contributors
Authors: Hannes Troepgen, Brian Edward Marre, Alexander Debus, Richard Pausch
License: GPLv3+
"""

from math import pi, sqrt
from unittest import TestCase

import numpy as np
import os
import re
import tempfile
import pytest
from picongpu import picmi
from picongpu.picmi import Cartesian3DGrid, ElectromagneticSolver, GaussianLaser, Simulation
from pydantic import ValidationError
from scipy.constants import c


def _pulse_duration(duration_picmi_si):
    """PIConGPU PULSE_DURATION (1 sigma of the intensity) from the PICMI-standard
    duration (1/e field width), i.e. PULSE_DURATION = duration / 2"""
    return duration_picmi_si / 2.0


class TestPicmiGaussianLaser(TestCase):
    def test_basic(self):
        """full laser example"""
        picmi_laser = GaussianLaser(
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
        picmi_laser = GaussianLaser(
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
        picmi_laser = GaussianLaser(
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
        """standard lasers may enter through any coordinate face; the entry face is the
        one whose normal is the dominant propagation-direction component, and the
        centroid must be outside the box on that entry side."""

        # (propagation_direction, centroid_position, focal_position, expected entry face)
        valid_cases = [
            ([0, 1, 0], [0.5, -1, 0.5], [0.5, 2, 0.5], "YMin"),
            ([0, -1, 0], [0.5, 1, 0.5], [0.5, 2, 0.5], "YMax"),
            ([1, 0, 0], [-1, 0.5, 0.5], [2, 0.5, 0.5], "XMin"),
            ([-1, 0, 0], [1, 0.5, 0.5], [2, 0.5, 0.5], "XMax"),
            ([0, 0, 1], [0.5, 0.5, -1], [0.5, 0.5, 2], "ZMin"),
            ([0, 0, -1], [0.5, 0.5, 1], [0.5, 0.5, 2], "ZMax"),
            # non-axis direction with a clear dominant (+y) component
            (
                (np.array([1, 3, 1]) / sqrt(11)).tolist(),
                (np.array([-2, -6, -2]) / sqrt(11)).tolist(),
                [0, 0, 0],
                "YMin",
            ),
        ]

        for propagation_direction, centroid, focal, expected_face in valid_cases:
            laser = GaussianLaser(
                wavelength=1,
                waist=2,
                duration=3,
                focal_position=focal,
                centroid_position=centroid,
                propagation_direction=propagation_direction,
                polarization_direction=[1, 0, 0],
                E0=1,
            )
            self.assertEqual(laser.get_as_pypicongpu().entry_face, expected_face)

        # invalid propagation directions: unnormalized, zero, or centroid not outside
        # the box on the entry side.
        invalid_cases = [
            [1, 2, 3],  # unnormalized
            [0, 0, 0],  # zero
            [0, 0, 1],  # +z with centroid_z > 0 (below)
            [1, 0, 0],  # +x with centroid_x > 0 (below)
        ]
        for invalid_propagation_vector in invalid_cases:
            with self.assertRaises(ValidationError):
                GaussianLaser(
                    wavelength=1,
                    waist=2,
                    duration=3,
                    focal_position=[0.5, 0, 0.5],
                    centroid_position=[0.5, 0, 0.5],
                    propagation_direction=invalid_propagation_vector,
                    polarization_direction=[1, 0, 0],
                    E0=1,
                )

        # a valid direction can still be rejected when the centroid is on the wrong
        # side of the entry face (e.g. +y with centroid_y > 0).
        with self.assertRaises(ValidationError):
            GaussianLaser(
                wavelength=1,
                waist=2,
                duration=3,
                focal_position=[0.5, 2, 0.5],
                centroid_position=[0.5, 1, 0.5],
                propagation_direction=[0, 1, 0],
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
                GaussianLaser(
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
            picmi_laser = GaussianLaser(
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
        picmi_laser = GaussianLaser(
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
            GaussianLaser(
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
            GaussianLaser(
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
            GaussianLaser(
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
        picmi_laser = GaussianLaser(
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
            GaussianLaser(
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
            GaussianLaser(
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
        picmi_laser = GaussianLaser(
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

        grid_valid = Cartesian3DGrid(
            number_of_cells=[128, 512, 256],
            lower_bound=[0, 0, 0],
            upper_bound=[17, 192, 42],
            lower_boundary_conditions=["periodic", "periodic", "open"],
            upper_boundary_conditions=["periodic", "periodic", "open"],
        )

        # valid grid-laser combination working
        solver_valid = ElectromagneticSolver(method="Yee", grid=grid_valid)
        sim_valid = Simulation(time_step_size=1, max_steps=2, solver=solver_valid)
        sim_valid.add_laser(picmi_laser, None)

        # translates without issue:
        assert sim_valid.get_as_pypicongpu().model_dump() != {}

    def test_overdefinition_a0_E0(self):
        """only either a0 or E0 allowed to be set"""

        with self.assertRaises(ValidationError):
            GaussianLaser(
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
            GaussianLaser(
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
    laser = GaussianLaser(
        wavelength=800e-9,
        waist=12e-6,
        duration=duration_picmi_si,
        focal_position=[8.5, 2, 21],
        centroid_position=[8.5, -3, 21],
        propagation_direction=[0, 1, 0],
        polarization_direction=[1, 0, 0],
        a0=1.0,
    )
    grid = Cartesian3DGrid(
        number_of_cells=[128, 512, 256],
        lower_bound=[0, 0, 0],
        upper_bound=[17, 192, 42],
        lower_boundary_conditions=["periodic", "periodic", "open"],
        upper_boundary_conditions=["periodic", "periodic", "open"],
    )
    solver = ElectromagneticSolver(method="Yee", grid=grid)
    sim = Simulation(time_step_size=1, max_steps=2, solver=solver)
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


def _rendered_incident_field(laser):
    """Render a single-laser simulation and return the incidentField.param text."""
    grid = Cartesian3DGrid(
        number_of_cells=[128, 512, 256],
        lower_bound=[0, 0, 0],
        upper_bound=[17, 192, 42],
        lower_boundary_conditions=["periodic"] * 3,
        upper_boundary_conditions=["periodic"] * 3,
    )
    sim = Simulation(time_step_size=1, max_steps=2, solver=ElectromagneticSolver(method="Yee", grid=grid))
    sim.add_laser(laser, None)
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = os.path.join(tmpdir, "input")
        sim.write_input_file(output_dir)
        rendered_path = os.path.join(output_dir, "include", "picongpu", "param", "incidentField.param")
        with open(rendered_path) as rendered_file:
            return rendered_file.read()


def _enabled_faces(rendered):
    """Return the list of faces (of the six) that carry a LaserProfile."""
    faces = []
    for face in ["XMin", "XMax", "YMin", "YMax", "ZMin", "ZMax"]:
        if re.search(r"using %s = MakeSeq_t<[^>]*LaserProfile" % face, rendered, re.DOTALL):
            faces.append(face)
    return faces


def test_entry_face_plus_y_regression():
    """propagation along +y: entry face is YMin and pulse_init matches the legacy formula (#88)."""
    laser = GaussianLaser(
        wavelength=800e-9,
        waist=12e-6,
        duration=30e-15,
        focal_position=[0.5, 5e-6, 0.5],
        centroid_position=[0.5, -5e-6, 0.5],
        propagation_direction=[0, 1, 0],
        polarization_direction=[1, 0, 0],
        a0=1.0,
    )
    pypic = laser.get_as_pypicongpu()
    assert pypic.entry_face == "YMin"
    # legacy +y formula, in units of PULSE_DURATION (= duration / 2):
    expected = -2.0 * laser.centroid_position[1] / c / _pulse_duration(laser.duration)
    assert abs(pypic.pulse_init - expected) < 1e-9
    assert _enabled_faces(_rendered_incident_field(laser)) == ["YMin"]


def test_entry_face_plus_z_and_pulse_init():
    """propagation along +z: entry face is ZMin (not YMin) and pulse_init is valid (#88)."""
    laser = GaussianLaser(
        wavelength=800e-9,
        waist=12e-6,
        duration=30e-15,
        focal_position=[0.5, 0.5, 5e-6],
        centroid_position=[0.5, 0.5, -5e-6],
        propagation_direction=[0, 0, 1],
        polarization_direction=[1, 0, 0],
        a0=1.0,
    )
    pypic = laser.get_as_pypicongpu()
    assert pypic.entry_face == "ZMin"
    # generalized formula: -2 * dot(centroid, direction) / (c * PULSE_DURATION)
    expected = -2.0 * np.dot(laser.centroid_position, laser.propagation_direction) / c / _pulse_duration(laser.duration)
    assert abs(pypic.pulse_init - expected) < 1e-9
    assert pypic.pulse_init > 0.0
    assert _enabled_faces(_rendered_incident_field(laser)) == ["ZMin"]


def test_twts_laser_entry_behavior_unchanged():
    """TWTS keeps its dedicated YMin + ZMin/ZMax two-plane placement and the +y entry face (#88).

    With laserIncidenceAngle = 0 the propagation direction is [0, cos, sin] = +y and
    laserIncidenceAnglePositive is False, so the incident field is placed on YMin and
    ZMax (the pre-existing two-plane behavior that this change must not alter).
    """
    twts = picmi.TWTSLaser(
        wavelength=800e-9,
        waist=12e-6,
        duration=30e-15,
        laserIncidenceAngle=0.0,
        polarizationAngle=pi / 4,
        focal_position=[0.0, 5e-6, 0.0],
        centroid_position=[0.0, -5e-6, 0.0],
        a0=1.0,
    )
    pypic = twts.get_as_pypicongpu()
    # the propagation direction is [0, cos(angle), sin(angle)] = +y for angle 0
    assert pypic.entry_face == "YMin"
    # TWTS keeps its base pulse-duration convention (sigma = duration), and for +y
    # propagation the generalized pulse_init reduces to the original expression.
    assert abs(pypic.pulse_init - (-2.0 * twts.centroid_position[1] / c / twts.duration)) < 1e-9
    # incident field is placed on YMin and ZMax (angle not positive), not on any other face
    assert _enabled_faces(_rendered_incident_field(twts)) == ["YMin", "ZMax"]


class TestGaussianLaserFieldComputation(TestCase):
    """
    Check the analytic field computation against the properties the underlying
    formulas (docs/source/models/lasers.rst "GaussianPulse", mirrored by the C++
    GaussianPulseFunctorIncidentE) must satisfy.

    Note on coordinates: the PICMI laser interface only supports propagation with
    a positive y-component, so ``propagation_direction=[0, 1, 0]`` serves as the
    "standard conditions" reference here (internally the formulas always work in
    a frame with propagation along +z and polarization along +x).  "On axis"
    therefore means along the y-direction.
    """

    def setUp(self):
        # choosing quadratic to have some symmetry to exploit for easy transformations
        self.max_size = 50
        self.number_of_cells = 2 * self.max_size + 1
        self.grid = np.mgrid[: self.number_of_cells, : self.number_of_cells, : self.number_of_cells] - self.max_size
        self.reference_kwargs = dict(
            wavelength=4.0,
            waist=10.0,
            duration=10 / c,
            propagation_direction=[0, 1, 0],
            polarization_direction=[1, 0, 0],
            focal_position=[0, 0, 0],
            centroid_position=[0, 0, 0],
            a0=1.0,
        )

    def make_laser(self, **kwargs):
        return GaussianLaser(**(self.reference_kwargs | kwargs))

    def test_on_axis_focus_amplitude_is_E0(self):
        # At the focus and at the time the pulse peak reaches it (centroid == focus,
        # t=0) the on-axis, in-focus amplitude is exactly the user-provided E0.
        laser = self.make_laser(centroid_position=[0, 0, 0])
        focus = np.zeros((3, 1))
        np.testing.assert_allclose(laser.complex_amplitude(*focus, t=0.0), laser.E0, rtol=1e-6)

    def test_temporal_width_is_the_duration(self):
        # The GaussianPulse envelope is exp(-(t / (2 * PULSE_DURATION))^2) with
        # PULSE_DURATION = duration / 2 (the 1 sigma of the intensity; the PICMI
        # duration is the 1/e field width tau).  Hence the field decays as
        # exp(-(t / duration)^2), i.e. to 1/e of its peak at t = duration.
        laser = self.make_laser()
        focus = np.zeros((3, 1))
        for t, factor in ((1.0, np.exp(-1.0)), (2.0, np.exp(-4.0))):
            np.testing.assert_allclose(
                np.abs(laser.complex_amplitude(*focus, t=t * laser.duration))[0],
                laser.E0 * factor,
                rtol=1e-6,
            )

    def test_shift_centroid_complex_amplitude(self):
        # Shifting the (longitudinal) centroid by delta is equivalent to evaluating
        # the reference laser at a time offset delta/c.
        centroid = np.array([0, -self.max_size // 2, 0])
        found = self.make_laser(centroid_position=centroid.tolist()).complex_amplitude(*self.grid)
        expected = self.make_laser().complex_amplitude(*self.grid, t=centroid[1] / c)
        np.testing.assert_allclose(found, expected)

    def test_shift_focus_complex_amplitude(self):
        # Shifting focus (and centroid with it, so the peak still hits the focus
        # at t=0) is equivalent to evaluating the reference laser shifted in space.
        # (centroid_y must stay <= 0, hence the shift along -y.)
        focus = np.array([0, -self.max_size // 2, 0])
        found = self.make_laser(focal_position=focus.tolist(), centroid_position=focus.tolist()).complex_amplitude(
            *self.grid
        )
        expected = self.make_laser().complex_amplitude(*(self.grid - focus.reshape(-1, 1, 1, 1)))
        np.testing.assert_allclose(found, expected)

    def test_shift_centroid_and_focus_complex_amplitude(self):
        centroid = np.array([0, -self.max_size // 2, 0])
        focus = np.array([0, self.max_size // 2, 0])
        found = self.make_laser(focal_position=focus.tolist(), centroid_position=centroid.tolist()).complex_amplitude(
            *self.grid
        )
        expected = self.make_laser().complex_amplitude(
            *(self.grid - focus.reshape(-1, 1, 1, 1)), t=-(focus[1] - centroid[1]) / c
        )
        np.testing.assert_allclose(found, expected)

    def test_rotate_propagation_complex_amplitude(self):
        # Rotating the propagation direction rotates the whole field pattern.
        # B(r) = A(R^-1 r) with R aligning the reference frame to the rotated one
        # (polarization stays x so it is a rotation about the x-axis).
        from scipy.spatial.transform import Rotation

        rotated_propagation = [0, 1 / np.sqrt(2), 1 / np.sqrt(2)]
        rotation = Rotation.align_vectors([[1, 0, 0], [0, 0, 1]], [[1, 0, 0], rotated_propagation])[0]
        rotated_grid = np.moveaxis(rotation.inv().apply(np.moveaxis(self.grid, 0, -1)), -1, 0)
        found = self.make_laser(propagation_direction=rotated_propagation).complex_amplitude(*self.grid)
        expected = self.make_laser().complex_amplitude(*rotated_grid)
        np.testing.assert_allclose(found, expected)

    def test_polarization_vector_in_focus_plane(self):
        # In the focus plane (no wavefront tilt) the polarization vector is exactly
        # the polarization direction.
        focus_plane = self.grid[1] == 0
        found = self.make_laser().polarization_vector_at(*self.grid[:, focus_plane])
        expected = np.reshape(self.make_laser().polarization_direction, (-1, 1)) * np.ones_like(found)
        np.testing.assert_allclose(found, expected)

    def test_rotate_polarization_on_axis(self):
        # Rotating the polarization direction rotates the E field accordingly;
        # on axis at the focus this is an exact vector rotation.
        origin = np.zeros((3, 1))
        found = self.make_laser(polarization_direction=[0, 0, 1]).E(*origin)
        expected = np.array([[0.0], [0.0], [self.make_laser().E0]])
        np.testing.assert_allclose(found, expected)

    def test_E_has_component_first_layout(self):
        laser = self.make_laser()
        e = laser.E(*self.grid)
        self.assertEqual(e.shape, (3,) + self.grid.shape[1:])
        np.testing.assert_allclose(e, np.stack([laser.Ex(*self.grid), laser.Ey(*self.grid), laser.Ez(*self.grid)]))


def plot_half_box_slices(grid, data, field_components="xyz", title=""):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(len(field_components), 3, squeeze=False)
    fig.suptitle(title)
    for i, coord in enumerate("xyz"):
        ax_index = 0
        for field_component, d in zip("xyz", data):
            if field_component in field_components:
                a = ax[ax_index, i]
                ax_index += 1
                s = np.roll([d.shape[i] // 2, slice(None), slice(None)], shift=i)
                coordinates = sorted(set(range(3)) - {i})
                x, y = grid[coordinates, *s].reshape(2, -1)
                z = d[*s].reshape(-1)
                im = a.scatter(x, y, c=z)
                fig.colorbar(im)
                a.set_title(f"E{field_component}, slice: {coord}=Box/2")
                a.set_xlabel("xyz"[coordinates[0]])
                a.set_ylabel("xyz"[coordinates[1]])
    fig.tight_layout()
