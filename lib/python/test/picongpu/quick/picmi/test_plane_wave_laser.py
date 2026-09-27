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
    This test case is about comparing the effect of various parameters
    with the laser under standard conditions.
    This is useful because the latter formula can be found verbatim in the documentation,
    so there's a good chance we've got it right.
    """

    def setUp(self):
        self.max_size = 50
        self.number_of_cells = 2 * self.max_size + 1
        self.grid = np.mgrid[: self.number_of_cells, : self.number_of_cells, : self.number_of_cells] - self.max_size
        self.reference_kwargs = dict(
            wavelength=10.0,
            duration=20 / c,
            propagation_direction=[0, 0, 1],
            polarization_direction=[1, 0, 0],
            centroid_position=[0, 0, 0],
            a0=1.0,
        )

    def make_laser(self, **kwargs):
        return PlaneWaveLaser(**(self.reference_kwargs | kwargs))

    def test_rotate_propagation_complex_amplitude(self):
        found = self.make_laser(propagation_direction=[0, 1, 0]).complex_amplitude(*self.grid)
        expected = np.moveaxis(self.make_laser().complex_amplitude(*self.grid), 2, 1)
        np.testing.assert_allclose(found, expected)

    def test_shift_centroid_complex_amplitude(self):
        centroid = np.array([0, 0, self.max_size // 2])
        found = self.make_laser(centroid_position=centroid).complex_amplitude(*self.grid)
        expected = self.make_laser().complex_amplitude(*self.grid, t=centroid[2] / c)
        np.testing.assert_allclose(found, expected)

    def test_polarization_vector(self):
        found = self.make_laser().polarization_vector_at(*self.grid)
        expected = np.reshape(self.make_laser().polarization_direction, (1, 1, 1, -1)) * np.ones_like(found)
        np.testing.assert_allclose(found, expected)

    def test_polarization_vector_rotated(self):
        direction = [0, 1, 0]
        found = self.make_laser(polarization_direction=direction).polarization_vector_at(*self.grid)
        expected = np.reshape(direction, (1, 1, 1, -1)) * np.ones_like(found)
        np.testing.assert_allclose(found, expected)

    def test_shift_centroid(self):
        centroid = np.array([0, 0, self.max_size // 2])
        found = self.make_laser(centroid_position=centroid).E(*self.grid)
        expected = self.make_laser().E(*self.grid, t=centroid[2] / c)
        np.testing.assert_allclose(found, expected)

    def test_rotate_propagation(self):
        found = self.make_laser(propagation_direction=[0, 1, 0]).E(*self.grid)
        expected = np.moveaxis(self.make_laser().E(*self.grid), 2, 1)
        np.testing.assert_allclose(found, expected)

    def test_rotate_polarization(self):
        found = self.make_laser(polarization_direction=[0, 1, 0]).E(*self.grid)
        expected = Rotation.from_euler("z", 90, degrees=True).apply(self.make_laser().E(*self.grid))
        np.testing.assert_allclose(found, expected, atol=1.0e-10)

    def test_single_components(self):
        laser = self.make_laser()
        found = np.moveaxis([laser.Ex(*self.grid), laser.Ey(*self.grid), laser.Ez(*self.grid)], 0, -1)
        expected = self.make_laser().E(*self.grid)
        np.testing.assert_allclose(found, expected)

    def test_plateau_duration(self):
        found = self.make_laser(picongpu_plateau_duration=100).E(*self.grid)
        expected = self.make_laser().E(*self.grid)
        np.testing.assert_allclose(found, expected, atol=1.0e-10)
