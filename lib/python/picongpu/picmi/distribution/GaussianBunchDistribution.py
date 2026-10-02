"""
This file is part of PIConGPU.
Copyright 2026 PIConGPU contributors
Authors: Julian Lenz
License: GPLv3+
"""

import math
from typing import Annotated

import numpy as np
import sympy
from picmistandard import PICMI_GaussianBunchDistribution
from pydantic import Field, model_validator

from ...pypicongpu import species, util
from .AnalyticDistribution import AnalyticDistribution

"""
note on rms_velocity:
---------------------
The rms_velocity is converted to a temperature in keV. This conversion requires the mass of the species to be known,
which is not the case inside the picmi density distribution.

As an abstraction, **every** PICMI density distribution implements `picongpu_get_rms_velocity_si()` which returns a
tuple (float, float, float) with the rms_velocity per axis in SI units (m/s).

In case the density profile does not have an rms_velocity, this method **MUST** return (0, 0, 0), which is translated to
"no temperature initialization" by the owning species.

note on drift:
--------------
The drift ("velocity") is represented using centroid_velocity (gamma*v) and
for the pypicongpu representation stored in a separate object (Drift).

To accommodate that, this separate Drift object can be requested by the method get_picongpu_drift(). In case of no drift,
this method returns None.
"""


class GaussianBunchDistribution(PICMI_GaussianBunchDistribution):
    """
    A finite Gaussian particle bunch as defined by the PICMI standard.

    The bunch is a 3D tri-Gaussian ellipsoid with peak number density

        n0 = n_physical_particles / ((2*pi)**1.5 * sx * sy * sz)

    centered at ``centroid_position`` with per-axis RMS sizes
    ``rms_bunch_size`` and an initial Gaussian velocity spread
    ``rms_velocity``.

    PIConGPU renders this profile through its analytic
    :class:`~picongpu.picmi.distribution.AnalyticDistribution` path, i.e. as a
    ``FreeFormulaImpl`` functor carrying the absolute density ``n0``. No C++
    core feature is required.

    Unsupported:
        * ``velocity_divergence``:
          PIConGPU has no correlated position-momentum initializer, so any
          non-zero value raises an ``UnsupportedFeatureError`` at construction.
        * non-3D grids:
          the profile is a 3D tri-Gaussian and would render a dead ``z`` term with
          a 3D-normalized ``n0`` on a 2D grid, so any non-3D grid raises an
          ``UnsupportedFeatureError`` at input-file generation.
    """

    velocity_divergence: Annotated[
        list[float], util.rejects_unsupported("velocity_divergence", default=[0.0, 0.0, 0.0])
    ] = Field(default_factory=lambda: [0.0, 0.0, 0.0], min_length=3, max_length=3)

    @model_validator(mode="after")
    def _validate(self):
        if any(sigma <= 0.0 for sigma in self.rms_bunch_size):
            raise ValueError(f"rms_bunch_size must be > 0 in every dimension, you gave {self.rms_bunch_size}.")
        return self

    def _peak_density(self) -> float:
        """Peak number density n0 [m^-3] of the normalized 3D Gaussian."""
        sx, sy, sz = self.rms_bunch_size
        return self.n_physical_particles / ((2.0 * math.pi) ** 1.5 * sx * sy * sz)

    def _derive_analytic_distribution_kwargs(self) -> dict:
        """Build the kwargs for the equivalent :class:`AnalyticDistribution`."""
        n0 = self._peak_density()
        sx, sy, sz = self.rms_bunch_size
        cx, cy, cz = self.centroid_position

        def density_function(x, y, z):
            return n0 * sympy.exp(-0.5 * (((x - cx) / sx) ** 2 + ((y - cy) / sy) ** 2 + ((z - cz) / sz) ** 2))

        return {"density_function": density_function}

    def get_as_pypicongpu(self, grid):
        # The standard GaussianBunchDistribution is a 3D tri-Gaussian: it always
        # carries a z term and normalizes n0 over three dimensions. Rendering it on
        # a 2D grid would emit a dead z term and a physically wrong 3D normalization,
        # so reject it just like the 2D z-guard rejects a z-dependent AnalyticDistribution.
        if grid.number_of_dimensions != 3:
            raise util.UnsupportedFeatureError("GaussianBunchDistribution on a non-3D grid", grid.number_of_dimensions)
        kwargs = self._derive_analytic_distribution_kwargs()
        return AnalyticDistribution(**kwargs).get_as_pypicongpu(grid)

    def picongpu_get_rms_velocity_si(self) -> tuple[float, float, float]:
        return tuple(self.rms_velocity)

    def get_picongpu_drift(self) -> species.operation.momentum.Drift | None:
        """
        Get drift for pypicongpu.

        The standard defines ``centroid_velocity`` as ``gamma*V``, so it is
        converted using :meth:`~picongpu.pypicongpu.species.operation.momentum.Drift.from_gamma_velocity`.
        """
        if all(velocity == 0.0 for velocity in self.centroid_velocity):
            return None
        return species.operation.momentum.Drift.from_gamma_velocity(tuple(self.centroid_velocity))

    def __call__(self, x, y, z):
        n0 = self._peak_density()
        sx, sy, sz = self.rms_bunch_size
        cx, cy, cz = self.centroid_position
        return n0 * np.exp(-0.5 * (((x - cx) / sx) ** 2 + ((y - cy) / sy) ** 2 + ((z - cz) / sz) ** 2))
