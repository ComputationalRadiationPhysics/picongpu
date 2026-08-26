"""
This file is part of PIConGPU.
Copyright 2026 PIConGPU contributors
Authors: Julian Lenz
License: GPLv3+
"""

from unittest import TestCase

from pydantic import ValidationError

from picongpu import picmi
from picongpu.picmi.grid import Cartesian3DGrid


def _grid():
    return Cartesian3DGrid(
        lower_bound=[0, 0, 0],
        upper_bound=[1, 1, 1],
        number_of_cells=[1, 1, 1],
        lower_boundary_conditions=["periodic", "periodic", "periodic"],
        upper_boundary_conditions=["periodic", "periodic", "periodic"],
    )


class TestElectromagneticSolver(TestCase):
    def _make(self, **kwargs):
        params = {"method": "Yee", "grid": _grid()} | kwargs
        return picmi.ElectromagneticSolver(**params)

    def test_supported_methods(self):
        self.assertEqual(self._make(method="Yee").get_as_pypicongpu().__class__.__name__, "YeeSolver")
        self.assertEqual(self._make(method="Lehe").get_as_pypicongpu().__class__.__name__, "LeheSolver")

    def test_unsupported_method_rejected(self):
        with self.assertRaises(ValidationError):
            self._make(method="CKC")

    def test_cfl_and_source_smoother_accepted(self):
        # cfl is handled at simulation level, source_smoother switches on binomial
        # current deposition: both must be constructible
        solver = self._make(cfl=0.99, source_smoother=picmi.BinomialSmoother(n_pass=[2]))
        self.assertIsNotNone(solver.source_smoother)

    def test_unsupported_options_rejected(self):
        unsupported = {
            "stencil_order": [4],
            "field_smoother": picmi.BinomialSmoother(n_pass=[2]),
            "subcycling": 1,
            "galilean_velocity": [0.5, 0, 0],
            "divE_cleaning": True,
            "divB_cleaning": True,
            "pml_divE_cleaning": True,
            "pml_divB_cleaning": True,
        }
        for field, value in unsupported.items():
            with self.subTest(field=field):
                with self.assertRaises(ValidationError, msg=f"{field} must be rejected"):
                    self._make(**{field: value})

    def test_none_values_accepted(self):
        solver = self._make(stencil_order=None, field_smoother=None, subcycling=None)
        self.assertIsNone(solver.stencil_order)


class TestBinomialSmoother(TestCase):
    def test_n_pass_accepted(self):
        smoother = picmi.BinomialSmoother(n_pass=[2])
        self.assertEqual(smoother.n_pass, [2])

    def test_other_parameters_rejected(self):
        for field, value in {
            "compensation": [True],
            "stride": [1],
            "alpha": [1.0],
        }.items():
            with self.subTest(field=field):
                with self.assertRaises(ValidationError, msg=f"{field} must be rejected"):
                    picmi.BinomialSmoother(n_pass=[2], **{field: value})


class TestGaussianLaserUnsupportedOptions(TestCase):
    def _make(self, **kwargs):
        params = {
            "wavelength": 800e-9,
            "waist": 1e-5,
            "duration": 29e-15,
            "propagation_direction": [0, 1, 0],
            "polarization_direction": [1, 0, 0],
            "focal_position": [0, 0, 0],
            "centroid_position": [0, -1e-5, 0],
            "a0": 1.0,
        } | kwargs
        return picmi.GaussianLaser(**params)

    def test_plain_construction(self):
        laser = self._make()
        self.assertIsNotNone(laser.get_as_pypicongpu())

    def test_unsupported_options_rejected(self):
        for field, value in {"name": "my_laser", "zeta": 1.0, "beta": 0.5, "phi2": 0.1}.items():
            with self.subTest(field=field):
                with self.assertRaises(ValidationError, msg=f"{field} must be rejected"):
                    self._make(**{field: value})
