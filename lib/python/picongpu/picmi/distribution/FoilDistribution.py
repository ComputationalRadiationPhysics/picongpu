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
class FoilDistribution(picmistandard.PICMI_FoilDistribution):
    def picongpu_get_rms_velocity_si(self) -> typing.Tuple[float, float, float]:
        return tuple(self.rms_velocity)

    def get_as_pypicongpu(self) -> species.operation.densityprofile.DensityProfile:
        util.unsupported("fill in", self.fill_in)
        util.unsupported("lower bound", self.lower_bound, [None, None, None])
        util.unsupported("upper bound", self.upper_bound, [None, None, None])

        foilProfile = species.operation.densityprofile.Foil()
        foilProfile.density_si = self.density
        foilProfile.y_value_front_foil_si = self.front
        foilProfile.thickness_foil_si = self.thickness

        # create prePlasma ramp if indicated by settings
        prePlasma: bool = (self.exponential_pre_plasma_cutoff is not None) and (
            self.exponential_pre_plasma_length is not None
        )
        explicitlyNoPrePlasma: bool = (self.exponential_pre_plasma_cutoff is None) and (
            self.exponential_pre_plasma_length is None
        )

        if prePlasma:
            foilProfile.pre_foil_plasmaRamp = species.operation.densityprofile.plasmaramp.Exponential(
                self.exponential_pre_plasma_length,
                self.exponential_pre_plasma_cutoff,
            )
        elif explicitlyNoPrePlasma:
            foilProfile.pre_foil_plasmaRamp = species.operation.densityprofile.plasmaramp.None_()
        else:
            raise ValueError(
                "either both exponential_pre_plasma_length and"
                " exponential_pre_plasma_cutoff must be set to"
                " none or neither!"
            )

        # create postPlasma ramp if indicated by settings
        postPlasma: bool = (self.exponential_post_plasma_cutoff is not None) and (
            self.exponential_post_plasma_length is not None
        )
        explicitlyNoPostPlasma: bool = (self.exponential_post_plasma_cutoff is None) and (
            self.exponential_post_plasma_length is None
        )

        if postPlasma:
            foilProfile.post_foil_plasmaRamp = species.operation.densityprofile.plasmaramp.Exponential(
                self.exponential_post_plasma_length,
                self.exponential_post_plasma_cutoff,
            )
        elif explicitlyNoPostPlasma:
            foilProfile.post_foil_plasmaRamp = species.operation.densityprofile.plasmaramp.None_()
        else:
            raise ValueError(
                "either both exponential_post_plasma_length and"
                " exponential_post_plasma_cutoff must be set to"
                " none or neither!"
            )

        return foilProfile

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
