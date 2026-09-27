"""
This file is part of PIConGPU.
Copyright 2021-2024 PIConGPU contributors
Authors: Hannes Troepgen, Brian Edward Marre, Alexander Debus, Richard Pausch
License: GPLv3+
"""

from unittest import TestCase

import numpy as np
from scipy.spatial.transform import Rotation
from picongpu.picmi.lasers import PlaneWaveLaser
from scipy.constants import c


class TestPlaneWaveLaserFieldComputation(TestCase):
    """
    Check the analytic plane-wave field computation against the properties it
    must satisfy when read in its "standard conditions" (linear polarization,
    propagation allowed by the interface, i.e. with positive y-component, and the
    pulse peak at the origin at t=0).

    The longitudinal model mirrors the C++ ``PlaneWaveFunctorIncidentE``:
    a Gaussian ramp towards the plateau (amplitude E0) around
    ``endUpramp = pulse_init * duration``, a plateau whose length is set by
    ``picongpu_plateau_duration``, and the carrier running as
    sin(Omega0 * T + phi0) with T = t - z/c in standard conditions.
    """

    def setUp(self):
        self.max_size = 50
        self.number_of_cells = 2 * self.max_size + 1
        self.grid = np.mgrid[: self.number_of_cells, : self.number_of_cells, : self.number_of_cells] - self.max_size
        self.reference_kwargs = dict(
            wavelength=10.0,
            duration=20 / c,
            propagation_direction=[0, 1, 0],
            polarization_direction=[1, 0, 0],
            centroid_position=[0, 0, 0],
            a0=1.0,
        )

    def make_laser(self, **kwargs):
        return PlaneWaveLaser(**(self.reference_kwargs | kwargs))

    def test_polarization_vector(self):
        found = self.make_laser().polarization_vector_at(*self.grid)
        expected = np.reshape(self.make_laser().polarization_direction, (-1,) + (1,) * 3) * np.ones_like(found)
        np.testing.assert_allclose(found, expected)

    def test_polarization_vector_rotated(self):
        direction = [0, 0, 1]
        found = self.make_laser(polarization_direction=direction).polarization_vector_at(*self.grid)
        expected = np.reshape(direction, (-1,) + (1,) * 3) * np.ones_like(found)
        np.testing.assert_allclose(found, expected)

    def test_amplitude_scale(self):
        # The plateau amplitude is the user-provided E0 (up to the tiny correction
        # due to the instantaneous ramp edges for a zero plateau length).
        laser = self.make_laser()
        times = np.linspace(0.0, 1.5 * laser.duration, 4001)
        max_field = max(np.abs(laser.E(np.zeros((1,)), np.zeros((1,)), np.zeros((1,)), t=t)[0, 0]) for t in times)
        self.assertGreater(max_field, 0.95 * laser.E0)
        self.assertLess(max_field, 1.01 * laser.E0)

    def test_rotate_polarization(self):
        # Rotating the polarization direction by 90 degrees about the propagation
        # axis swaps the field between the transverse components; since the plane
        # wave's amplitude depends only on the propagation axis this is a pure
        # component swap.
        found = self.make_laser(polarization_direction=[0, 0, 1]).E(*self.grid)
        expected = np.zeros_like(found)
        expected[2] = self.make_laser().E(*self.grid)[0]
        np.testing.assert_allclose(found, expected)

    def test_rotate_propagation(self):
        # Rotating the propagation direction rotates the field pattern (and vector
        # field) accordingly: B(r) = R A(R^-1 r).
        rotated_propagation = [0, 1 / np.sqrt(2), 1 / np.sqrt(2)]
        rotation = Rotation.align_vectors([[1, 0, 0], [0, 0, 1]], [[1, 0, 0], rotated_propagation])[0]
        rotated_grid = np.moveaxis(rotation.inv().apply(np.moveaxis(self.grid, 0, -1)), -1, 0)
        found = self.make_laser(propagation_direction=rotated_propagation).E(*self.grid)
        rotated_vectors = rotation.apply(np.moveaxis(self.make_laser().E(*rotated_grid), 0, -1))
        expected = np.moveaxis(rotated_vectors, -1, 0)
        # atol relative to the field scale: covers rounding differences in the
        # exponentially suppressed far tail of the pulse
        np.testing.assert_allclose(found, expected, atol=1e-9 * np.abs(expected).max())

    def test_ramp_does_not_depend_on_plateau_duration(self):
        # Before the plateau starts (T < endUpramp) the field only sees the (common)
        # upramp, so the plateau length must not matter there.
        t = -2.0 / c
        mask = self.grid[1] > t * c  # points with T = t - y/c < 0
        found = self.make_laser(picongpu_plateau_duration=100).E(*self.grid[:, mask], t=t)
        expected = self.make_laser().E(*self.grid[:, mask], t=t)
        np.testing.assert_allclose(found, expected)

    def test_single_components(self):
        laser = self.make_laser()
        found = np.stack([laser.Ex(*self.grid), laser.Ey(*self.grid), laser.Ez(*self.grid)])
        expected = laser.E(*self.grid)
        np.testing.assert_allclose(found, expected)

    def test_E_has_component_first_layout(self):
        laser = self.make_laser()
        e = laser.E(*self.grid)
        self.assertEqual(e.shape, (3,) + self.grid.shape[1:])
