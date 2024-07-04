"""
This file is part of the PIConGPU.
Copyright 2021-2024 PIConGPU contributors
Authors: Hannes Troepgen, Brian Edward Marre
License: GPLv3+
"""

from ...pypicongpu import species
from ...pypicongpu import util

import picmistandard

import typeguard
import typing
import sympy
import math

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
class AnalyticDistribution(picmistandard.PICMI_AnalyticDistribution):
    """Analytic Particle Distribution as defined by PICMI @todo"""

    def picongpu_get_rms_velocity_si(self) -> typing.Tuple[float, float, float]:
        return tuple(self.rms_velocity)

    def get_as_pypicongpu(self) -> species.operation.densityprofile.DensityProfile:
        util.unsupported("momentum expressions", self.momentum_expressions)
        util.unsupported("fill in", self.fill_in)

        # TODO
        profile = object()
        profile.lower_bound = tuple(map(lambda x: -math.inf if x is None else x, self.lower_bound))
        profile.upper_bound = tuple(map(lambda x: math.inf if x is None else x, self.upper_bound))

        # final (more thorough) formula checking will be invoked inside
        # pypicongpu on translation to CPP
        sympy_density_expression = sympy.sympify(self.density_expression).subs(self.user_defined_kw)
        profile.expression = sympy_density_expression

        return profile

    def get_picongpu_drift(self) -> typing.Optional[species.operation.momentum.Drift]:
        """
        Get drift for pypicongpu
        :return: pypicongpu drift object or None
        """
        if [0, 0, 0] == self.directed_velocity:
            return None

        drift = species.operation.momentum.Drift()
        drift.fill_from_velocity(tuple(self.directed_velocity))
        return drift
