"""
This file is part of the PIConGPU.
Copyright 2021-2023 PIConGPU contributors
Authors: Hannes Troepgen, Brian Edward Marre
License: GPLv3+
"""

from ..pypicongpu import species
from ..pypicongpu import util

import picmistandard

from typeguard import typechecked
import typing
import sympy
import math

# note on rms_velocity:
# The rms_velocity is converted to a temperature in keV.
# This conversion requires the mass of the species to be known, which is not
# the case inside the picmi density distribution.
# As an abstraction, **every** PICMI density distribution implements
# `picongpu_get_rms_velocity_si()` which returns a tuple (float, float, float)
# with the rms_velocity per axis in SI units (m/s). In case the density profile
# does not have an rms_velocity, this method **MUST** return (0, 0, 0), which
# is translated to "no temperature initialization" by the owning species.

# note on drift:
# The drift ("velocity") is represented using either directed_velocity or
# centroid_velocity (v, gamma*v respectively) and for the pypicongpu
# representation stored in a separate object (Drift).
# To accomodate that, this separate Drift object can be requested by the method
# get_picongpu_drift(). In case of no drift, this method returns None.


@typechecked
class UniformDistribution(picmistandard.PICMI_UniformDistribution):
    """Uniform Particle Distribution as defined by PICMI"""

    def picongpu_get_rms_velocity_si(self) -> typing.Tuple[float, float, float]:
        return tuple(self.rms_velocity)

    def get_as_pypicongpu(self) -> species.operation.densityprofile.DensityProfile:
        util.unsupported("fill in", self.fill_in)
        util.unsupported("lower bound", self.lower_bound, [None, None, None])
        util.unsupported("upper bound", self.upper_bound, [None, None, None])

        profile = species.operation.densityprofile.Uniform()
        profile.density_si = self.density
        # @todo respect bounding box, Brian Marre, 2023
        # profile.lower_bound = tuple(map(
        #   lambda x: -math.inf if x is None else x, self.lower_bound))
        # profile.upper_bound = tuple(map(
        #   lambda x: math.inf if x is None else x, self.upper_bound))
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


@typechecked
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


@typechecked
class GaussianBunchDistribution(picmistandard.PICMI_GaussianBunchDistribution):
    def picongpu_get_rms_velocity_si(self) -> typing.Tuple[float, float, float]:
        return tuple(self.rms_velocity)

    def get_as_pypicongpu(self) -> species.operation.densityprofile.DensityProfile:
        # @todo respect boundaries, Brian Marre, 2023
        profile = object()
        profile.lower_bound = (-math.inf, -math.inf, -math.inf)
        profile.upper_bound = (math.inf, math.inf, math.inf)
        profile.rms_bunch_size_si = self.rms_bunch_size
        profile.centroid_position_si = tuple(self.centroid_position)

        assert 0 != self.rms_bunch_size, "rms bunch size must not be zero"

        profile.max_density_si = self.n_physical_particles / ((2 * math.pi * self.rms_bunch_size**2) ** 1.5)

        return profile

    def get_picongpu_drift(self) -> typing.Optional[species.operation.momentum.Drift]:
        """
        Get drift for pypicongpu
        :return: pypicongpu drift object or None
        """
        if [0, 0, 0] == self.centroid_velocity:
            return None

        drift = species.operation.momentum.Drift()
        drift.fill_from_gamma_velocity(tuple(self.centroid_velocity))
        return drift


@typechecked
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
