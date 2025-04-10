"""
This file is part of PIConGPU.
Copyright 2021-2024 PIConGPU contributors
Authors: Hannes Troepgen, Brian Edward Marre
License: GPLv3+
"""

from numpy import vectorize
from ...pypicongpu import species


import typeguard
import typing
import sympy

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
The drift ("velocity") is represented using either directed_velocity or centroid_velocity (v, gamma*v respectively) and
for the pypicongpu representation stored in a separate object (Drift).

To accommodate that, this separate Drift object can be requested by the method get_picongpu_drift(). In case of no drift,
this method returns None.
"""


@typeguard.typechecked
class AnalyticDistribution:
    """Analytic Particle Distribution as defined by PICMI @todo"""

    def __init__(self, density_expression, directed_velocity=(0.0, 0.0, 0.0)):
        self.density_expression = density_expression
        self.rms_velocity = (0.0, 0.0, 0.0)
        self.directed_velocity = tuple(float(v) for v in directed_velocity)

    def get_as_pypicongpu(self) -> species.operation.densityprofile.DensityProfile:
        return species.operation.densityprofile.FreeFormula(
            density_expression=self.density_expression(*sympy.symbols("x,y,z,dx,dy,dz"))
        )

    def picongpu_get_rms_velocity_si(self) -> typing.Tuple[float, float, float]:
        return self.rms_velocity

    def get_picongpu_drift(self) -> typing.Optional[species.operation.momentum.Drift]:
        """
        Get drift for pypicongpu
        :return: pypicongpu drift object or None
        """
        if all(v == 0 for v in self.directed_velocity):
            return None

        drift = species.operation.momentum.Drift()
        drift.fill_from_velocity(self.directed_velocity)
        return drift

    def __call__(self, *args, **kwargs):
        # we vectorize here, so you can use numpy arrays on your density
        return vectorize(self.density_expression)(*args, **kwargs)
