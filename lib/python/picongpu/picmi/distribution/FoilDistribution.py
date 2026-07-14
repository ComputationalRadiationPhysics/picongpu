"""
This file is part of PIConGPU.
Copyright 2021-2024 PIConGPU contributors
Authors: Hannes Troepgen, Brian Edward Marre
License: GPLv3+
"""

import typing

import numpy as np
from picmistandard import PICMI_FoilDistribution

from ...pypicongpu import species, util

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


class FoilDistribution(PICMI_FoilDistribution):
    def picongpu_get_rms_velocity_si(self) -> typing.Tuple[float, float, float]:
        return tuple(self.rms_velocity)

    def get_as_pypicongpu(self, _) -> species.operation.densityprofile.AnyDensityProfile:
        util.unsupported("fill in", self.fill_in)
        util.unsupported("lower bound", self.lower_bound, (None, None, None))
        util.unsupported("upper bound", self.upper_bound, (None, None, None))

        # create prePlasma ramp if indicated by settings
        prePlasma: bool = (self.exponential_pre_plasma_cutoff is not None) and (
            self.exponential_pre_plasma_length is not None
        )
        explicitlyNoPrePlasma: bool = (self.exponential_pre_plasma_cutoff is None) and (
            self.exponential_pre_plasma_length is None
        )

        if prePlasma:
            pre_foil_plasmaRamp = species.operation.densityprofile.plasmaramp.Exponential(
                PlasmaLength=self.exponential_pre_plasma_length,
                PlasmaCutoff=self.exponential_pre_plasma_cutoff,
            )
        elif explicitlyNoPrePlasma:
            pre_foil_plasmaRamp = species.operation.densityprofile.plasmaramp.None_()
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
            post_foil_plasmaRamp = species.operation.densityprofile.plasmaramp.Exponential(
                PlasmaLength=self.exponential_post_plasma_length,
                PlasmaCutoff=self.exponential_post_plasma_cutoff,
            )
        elif explicitlyNoPostPlasma:
            post_foil_plasmaRamp = species.operation.densityprofile.plasmaramp.None_()
        else:
            raise ValueError(
                "either both exponential_post_plasma_length and"
                " exponential_post_plasma_cutoff must be set to"
                " none or neither!"
            )

        return species.operation.densityprofile.Foil(
            density_si=self.density,
            y_value_front_foil_si=self.front,
            thickness_foil_si=self.thickness,
            pre_foil_plasmaRamp=pre_foil_plasmaRamp,
            post_foil_plasmaRamp=post_foil_plasmaRamp,
        )

    def get_picongpu_drift(self) -> species.operation.momentum.Drift | None:
        """
        Get drift for pypicongpu
        :return: pypicongpu drift object or None
        """
        if [0, 0, 0] == self.directed_velocity:
            return None
        return species.operation.momentum.Drift.from_velocity(tuple(self.directed_velocity))

    def __call__(self, x, y, z):
        # We do this to get the correct shape after broadcasting:
        result = 0.0 * (x + y + z)

        pre_plasma_ramp = (
            np.exp((y - self.front) / self.exponential_pre_plasma_length)
            if self.exponential_pre_plasma_length is not None
            else 0.0
        ) + result
        pre_plasma_mask = (y < self.front) * (y > self.front - self.exponential_pre_plasma_cutoff)

        post_plasma_ramp = (
            np.exp(((self.front + self.thickness) - y) / self.exponential_post_plasma_length)
            if self.exponential_post_plasma_length is not None
            else 0.0
        ) + result
        post_plasma_mask = (y > self.front + self.thickness) * (
            y < self.front + self.thickness + self.exponential_post_plasma_cutoff
        )

        result[pre_plasma_mask] = pre_plasma_ramp[pre_plasma_mask]
        result[post_plasma_mask] = post_plasma_ramp[post_plasma_mask]
        result[(y >= self.front) * (y <= self.front + self.thickness)] = 1.0

        return self.density * result
