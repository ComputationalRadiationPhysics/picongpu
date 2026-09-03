"""
This file is part of PIConGPU.
Copyright 2021-2024 PIConGPU contributors
Authors: Hannes Troepgen, Brian Edward Marre
License: GPLv3+
"""

from unittest import TestCase

import pytest
from picongpu import picmi
from picongpu.picmi.grid import Cartesian3DGrid
from picongpu.picmi.species import Species
from picongpu.picmi.species_requirements import SimpleMomentumOperation, run_construction
from picongpu.pypicongpu import species
from picongpu.pypicongpu.util import UnsupportedFeatureError
from pydantic import ValidationError

ARBITRARY_GRID = Cartesian3DGrid(
    lower_bound=[0, 0, 0],
    upper_bound=[1, 1, 1],
    number_of_cells=[1, 1, 1],
    lower_boundary_conditions=["periodic", "periodic", "periodic"],
    upper_boundary_conditions=["periodic", "periodic", "periodic"],
)


class HelperTestPicmiBoundaries:
    """
    provides test functions to check proper handling of boundaries

    expects a method self._get_distribution(lower_bound, upper_bound), which
    creates a distribution w/ lower & upper bound passed straight through.
    """

    def __init__(self):
        if type(self) is HelperTestPicmiBoundaries:
            raise RuntimeError("This class is abstract, inherit from it!")

    def _get_distribution(self, lower_bound, upper_bound):
        """
        helper to check against

        Must create the distribution to test with arbitrary params;
        must pass lower_bound and upper_bound straight through.

        :param lower_bound: any, passed through to PICMI
        :param upper_bound: any, passed through to PICMI
        :return: PICMI distribution
        """
        raise NotImplementedError("must be implemented in child classes")


class TestPicmiUniformDistribution(TestCase, HelperTestPicmiBoundaries):
    def _get_distribution(self, lower_bound, upper_bound):
        return picmi.UniformDistribution(density=1716273, lower_bound=lower_bound, upper_bound=upper_bound)

    def test_full(self):
        """full paramset"""
        uniform = picmi.UniformDistribution(density=42.42)
        pypic = uniform.get_as_pypicongpu(ARBITRARY_GRID)
        assert isinstance(pypic, species.operation.densityprofile.Uniform)

        assert pypic.density_si == 42.42

    def test_lower_upper_bound_not_supported(self):
        """the uniform profile has no bound support, so setting bounds must raise"""
        uniform = picmi.UniformDistribution(density=42.42, lower_bound=[111, 222, 333], upper_bound=[444, 555, 666])
        with pytest.raises(UnsupportedFeatureError, match="lower bound"):
            uniform.get_as_pypicongpu(ARBITRARY_GRID)

    def test_density_zero(self):
        """density set to zero is not accepted"""
        uniform = picmi.UniformDistribution(density=0)
        with pytest.raises(ValidationError):
            uniform.get_as_pypicongpu(ARBITRARY_GRID)

    def test_mandatory(self):
        """check that mandatory must be given"""
        # type of exception is not checked
        with pytest.raises(Exception):
            picmi.UniformDistribution().get_as_pypicongpu(ARBITRARY_GRID)

        # density is only required param
        picmi.UniformDistribution(density=3.14).get_as_pypicongpu(ARBITRARY_GRID)

    def test_drift(self):
        """drift is correctly translated"""
        # no drift
        uniform = picmi.UniformDistribution(density=1, directed_velocity=[0, 0, 0])
        drift = uniform.get_picongpu_drift()
        assert drift is None

        # some drift
        # uses velocity
        uniform = picmi.UniformDistribution(density=1, directed_velocity=[278487224.0, 103784563.0, 1283345.0])
        drift = uniform.get_picongpu_drift()
        assert drift is not None
        assert abs(drift.gamma - 7.6208808298928865) < 1e-10
        assert abs(drift.direction_normalized[0] - 0.9370354841199405) < 1e-10
        assert abs(drift.direction_normalized[1] - 0.34920746753855203) < 1e-10
        assert abs(drift.direction_normalized[2] - 0.004318114799291135) < 1e-10


class TestPicmiFoilDistribution(TestCase, HelperTestPicmiBoundaries):
    def _get_distribution(self, lower_bound, upper_bound):
        return picmi.FoilDistribution(
            density=1716273,
            front=1.0,
            thicknes=2.0,
            exponential_pre_plasma_length=3.0,
            exponential_pre_plasma_cutoff=4.0,
            exponential_post_plasma_length=5.0,
            exponential_post_plasma_cutoff=6.0,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
        )

    def test_full(self):
        """full paramset"""
        foil = picmi.FoilDistribution(
            density=42.42,
            front=1.0,
            thickness=2.0,
            exponential_pre_plasma_length=3.0,
            exponential_pre_plasma_cutoff=4.0,
            exponential_post_plasma_length=5.0,
            exponential_post_plasma_cutoff=6.0,
        )

        pypic = foil.get_as_pypicongpu(ARBITRARY_GRID)
        assert isinstance(pypic, species.operation.densityprofile.Foil)

        assert pypic.density_si == 42.42
        assert pypic.y_value_front_foil_si == 1.0
        assert pypic.thickness_foil_si == 2.0
        assert pypic.pre_foil_plasmaRamp.PlasmaLength == 3.0
        assert pypic.pre_foil_plasmaRamp.PlasmaCutoff == 4.0
        assert pypic.post_foil_plasmaRamp.PlasmaLength == 5.0
        assert pypic.post_foil_plasmaRamp.PlasmaCutoff == 6.0

    def test_lower_upper_bound_not_supported(self):
        """the foil profile has no bound support, so setting bounds must raise"""
        foil = picmi.FoilDistribution(
            density=42.42,
            front=1.0,
            thickness=2.0,
            lower_bound=[111, 222, 333],
            upper_bound=[444, 555, 666],
        )
        with pytest.raises(UnsupportedFeatureError, match="lower bound"):
            foil.get_as_pypicongpu(ARBITRARY_GRID)

    def _get_test_foils(self, cutoff, length):
        """
        helper function generating preRamp only, postRamp only
        and (pre+post ramp foil) with given cutoffs and lengths
        """
        foil_pre = picmi.FoilDistribution(
            density=1.0,
            thickness=2.0,
            front=3.0,
            exponential_pre_plasma_cutoff=cutoff,
            exponential_pre_plasma_length=length,
            exponential_post_plasma_cutoff=None,
            exponential_post_plasma_length=None,
        )

        foil_post = picmi.FoilDistribution(
            density=1.0,
            thickness=2.0,
            front=3.0,
            exponential_pre_plasma_cutoff=None,
            exponential_pre_plasma_length=None,
            exponential_post_plasma_cutoff=cutoff,
            exponential_post_plasma_length=length,
        )

        foil_both = picmi.FoilDistribution(
            density=1.0,
            thickness=2.0,
            front=3.0,
            exponential_pre_plasma_cutoff=cutoff,
            exponential_pre_plasma_length=length,
            exponential_post_plasma_cutoff=cutoff,
            exponential_post_plasma_length=length,
        )

        testFoils = [foil_pre, foil_post, foil_both]
        return testFoils

    def test_cutoff_zero(self):
        """cutoff set to zero is accepted"""
        testCases = self._get_test_foils(0, 1.0)

        for entry in testCases:
            pypic = entry.get_as_pypicongpu(ARBITRARY_GRID)
            # no error:
            assert pypic.density_si == 1.0
            assert pypic.thickness_foil_si == 2.0
            assert pypic.y_value_front_foil_si == 3.0

    def test_setting_noPlasmaRamps(self):
        testCases = self._get_test_foils(None, 1.0)

        for entry in testCases:
            with pytest.raises(
                ValueError,
                match="either both exponential_(pre|post)_plasma_"
                "length and exponential_(pre|post)_plasma_cutoff must be"
                " set to none or neither!",
            ):
                entry.get_as_pypicongpu(ARBITRARY_GRID)

        testCases = self._get_test_foils(1.0, None)
        for entry in testCases:
            with pytest.raises(
                ValueError,
                match="either both exponential_(pre|post)_plasma_"
                "length and exponential_(pre|post)_plasma_cutoff must be"
                " set to none or neither!",
            ):
                entry.get_as_pypicongpu(ARBITRARY_GRID)

    def test_mandatory(self):
        """check that mandatory must be given"""
        # type of exception is not checked
        with pytest.raises(Exception):
            picmi.FoilDistribution()

        # density, thickness and front are only required param
        picmi.FoilDistribution(density=3.14, thickness=1.0, front=3.0).get_as_pypicongpu(ARBITRARY_GRID)

    def test_drift(self):
        """drift is correctly translated"""
        # no drift
        foil = picmi.FoilDistribution(density=1.0, front=2.0, thickness=3.0, directed_velocity=[0, 0, 0])
        drift = foil.get_picongpu_drift()
        assert drift is None

        # some drift
        # uses velocity
        foil = picmi.FoilDistribution(
            density=1,
            front=2.0,
            thickness=3.0,
            directed_velocity=[278487224.0, 103784563.0, 1283345.0],
        )
        drift = foil.get_picongpu_drift()
        assert drift is not None
        assert abs(drift.gamma - 7.6208808298928865) < 1e-10
        assert abs(drift.direction_normalized[0] - 0.9370354841199405) < 1e-10
        assert abs(drift.direction_normalized[1] - 0.34920746753855203) < 1e-10
        assert abs(drift.direction_normalized[2] - 0.004318114799291135) < 1e-10


class TestPicmiGaussianDistribution(TestCase, HelperTestPicmiBoundaries):
    values = {
        "density": 42.42,
        "center_front": 1.0,
        "center_rear": 2.0,
        "sigma_front": 3.0,
        "sigma_rear": 4.0,
        "power": 5.0,
        "factor": -6.0,
        "vacuum_front": 50,
    }

    def _get_distribution(self, lower_bound=[None, None, None], upper_bound=[None, None, None], **kwargs):
        return picmi.GaussianDistribution(
            **dict(
                density=self.values["density"],
                center_front=self.values["center_front"],
                center_rear=self.values["center_rear"],
                sigma_front=self.values["sigma_front"],
                sigma_rear=self.values["sigma_rear"],
                power=self.values["power"],
                factor=self.values["factor"],
                vacuum_front=self.values["vacuum_front"],
                lower_bound=lower_bound,
                upper_bound=upper_bound,
            )
            | kwargs
        )

    def test_full(self):
        """full paramset"""
        gaussian = self._get_distribution()

        pypic = gaussian.get_as_pypicongpu(ARBITRARY_GRID)
        assert isinstance(pypic, species.operation.densityprofile.Gaussian)

        assert pypic.density == self.values["density"]
        assert pypic.gas_center_front == self.values["center_front"]
        assert pypic.gas_center_rear == self.values["center_rear"]
        assert pypic.gas_sigma_front == self.values["sigma_front"]
        assert pypic.gas_sigma_rear == self.values["sigma_rear"]
        assert pypic.gas_power == self.values["power"]
        assert pypic.gas_factor == self.values["factor"]
        assert pypic.vacuum_cells_front == self.values["vacuum_front"]

        # @todo repect bounding boxes, Brian Marre, 2024

    def test_density_zero(self):
        """density set to zero is not accepted"""
        gaussian = self._get_distribution(density=0.0)
        with pytest.raises(ValueError, match=".*density must be > 0.*"):
            gaussian.get_as_pypicongpu(ARBITRARY_GRID)

    def test_front_rear_swapped(self):
        """front and rear swapped is not accepted"""
        gaussian = self._get_distribution(
            center_front=self.values["center_rear"], center_rear=self.values["center_front"]
        )
        with pytest.raises(ValueError, match=".*center_front must be <= center_rear.*"):
            gaussian.get_as_pypicongpu(ARBITRARY_GRID)

    def test_sigma_zero(self):
        """sigma == 0 is not accepted"""
        gaussian = self._get_distribution(sigma_front=0.0)
        with pytest.raises(ValidationError):
            gaussian.get_as_pypicongpu(ARBITRARY_GRID).get_rendering_context()

        gaussian = self._get_distribution(sigma_rear=0.0)
        with pytest.raises(ValidationError):
            gaussian.get_as_pypicongpu(ARBITRARY_GRID).get_rendering_context()

    def test_drift(self):
        """drift is correctly translated"""
        # no drift
        gaussian = self._get_distribution(directed_velocity=[0, 0, 0])
        drift = gaussian.get_picongpu_drift()
        assert drift is None

        # some drift
        # uses velocity
        gaussian = self._get_distribution(directed_velocity=[278487224.0, 103784563.0, 1283345.0])

        drift = gaussian.get_picongpu_drift()
        assert drift is not None
        assert abs(drift.gamma - 7.6208808298928865) < 1e-10
        assert abs(drift.direction_normalized[0] - 0.9370354841199405) < 1e-10
        assert abs(drift.direction_normalized[1] - 0.34920746753855203) < 1e-10
        assert abs(drift.direction_normalized[2] - 0.004318114799291135) < 1e-10


class TestPicmiCylindricalDistribution(TestCase, HelperTestPicmiBoundaries):
    def _get_distribution(
        self,
        density=1.0,
        center_position=(0.0, 0.0, 0.0),
        radius=2.0,
        cylinder_axis=(0.0, 1.0, 0.0),
        exponential_pre_plasma_length=None,
        exponential_pre_plasma_cutoff=None,
    ):
        return picmi.CylindricalDistribution(
            density=density,
            center_position=center_position,
            radius=radius,
            cylinder_axis=cylinder_axis,
            exponential_pre_plasma_length=exponential_pre_plasma_length,
            exponential_pre_plasma_cutoff=exponential_pre_plasma_cutoff,
        )

    def test_full(self):
        """full paramset"""
        dist = self._get_distribution(
            density=42.42,
            center_position=(1.0, 2.0, 3.0),
            radius=4.0,
            cylinder_axis=(0.5, 0.5, 0.707),
            exponential_pre_plasma_length=0.1,
            exponential_pre_plasma_cutoff=0.2,
        )
        pypic = dist.get_as_pypicongpu(ARBITRARY_GRID)
        assert isinstance(pypic, species.operation.densityprofile.Cylinder)
        assert abs(pypic.density_si - 42.42) < 1e-10
        assert pypic.center_position_si == (1.0, 2.0, 3.0)
        assert abs(pypic.radius_si - 4.0) < 1e-10
        assert pypic.cylinder_axis == (0.5, 0.5, 0.707)

    def test_density_zero(self):
        """density set to zero is not accepted"""
        dist = self._get_distribution(density=0.0)
        with pytest.raises(ValueError, match=".*density must be > 0.*"):
            dist.get_as_pypicongpu(ARBITRARY_GRID).get_rendering_context()

    def test_radius_zero(self):
        """radius smaller sqrt(2)*preplasma_length is not axcepted"""
        dist = self._get_distribution(
            radius=0.05,
            exponential_pre_plasma_length=0.1,
            exponential_pre_plasma_cutoff=0.2,
        )
        with pytest.raises(ValueError, match=".*radius must be > sqrt(2)*"):
            dist.get_as_pypicongpu(ARBITRARY_GRID).get_rendering_context()

    def test_cutoff_zero(self):
        """cutoff set to zero is accepted"""
        dist = self._get_distribution(
            exponential_pre_plasma_length=1.0,
            exponential_pre_plasma_cutoff=0.0,
        )
        pypic = dist.get_as_pypicongpu(ARBITRARY_GRID)
        # no error
        assert abs(pypic.density_si - 1.0) < 1e-10
        assert abs(pypic.radius_si - 2.0) < 1e-10

    def test_cutoff_below_zero(self):
        """cutoff below zero is not accepted (depends on ramp checks)"""
        dist = self._get_distribution(
            exponential_pre_plasma_length=1.0,
            exponential_pre_plasma_cutoff=-0.5,
        )
        with pytest.raises(ValidationError):
            dist.get_as_pypicongpu(ARBITRARY_GRID).get_rendering_context()

    def test_length_zero(self):
        """length set to zero is not accepted"""
        dist = self._get_distribution(
            exponential_pre_plasma_length=0.0,
            exponential_pre_plasma_cutoff=1.0,
        )
        with pytest.raises(ValidationError):
            dist.get_as_pypicongpu(ARBITRARY_GRID).get_rendering_context()

    def test_length_below_zero(self):
        """length below zero is not accepted"""
        dist = self._get_distribution(
            exponential_pre_plasma_length=-1.0,
            exponential_pre_plasma_cutoff=1.0,
        )
        with pytest.raises(ValidationError):
            dist.get_as_pypicongpu(ARBITRARY_GRID).get_rendering_context()

    def test_setting_noPrePlasma(self):
        """must set either both cutoffs and length, or none"""
        # only one set
        dist = self._get_distribution(exponential_pre_plasma_length=1.0)
        with pytest.raises(
            ValueError,
            match="either both exponential_pre_plasma_length and exponential_pre_plasma_cutoff must be set.*",
        ):
            dist.get_as_pypicongpu(ARBITRARY_GRID).get_rendering_context()

        # partial other way
        dist = self._get_distribution(exponential_pre_plasma_cutoff=1.0)
        with pytest.raises(
            ValueError,
            match="either both exponential_pre_plasma_length and exponential_pre_plasma_cutoff must be set.*",
        ):
            dist.get_as_pypicongpu(ARBITRARY_GRID).get_rendering_context()

    def test_mandatory(self):
        """check that mandatory must be given"""
        with pytest.raises(Exception):
            picmi.CylindricalDistribution().get_as_pypicongpu(ARBITRARY_GRID)

        # minimal valid
        dist = self._get_distribution()
        dist.get_as_pypicongpu(ARBITRARY_GRID)


def _gaussian_distribution(rms_velocity):
    return picmi.GaussianDistribution(
        density=1.0,
        lower_bound=[0, 0, 0],
        upper_bound=[1, 1, 1],
        center_front=0.2,
        center_rear=0.8,
        sigma_front=0.01,
        sigma_rear=0.02,
        power=2.0,
        factor=-9.0,
        vacuum_front=0.0,
        vacuum_rear=0.0,
        rms_velocity=rms_velocity,
    )


def _momentum_of(rms_velocity):
    """translate the temperature of a distribution with the given rms_velocity"""
    species = Species(name="e", particle_type="electron", initial_distribution=_gaussian_distribution(rms_velocity))
    return run_construction(SimpleMomentumOperation(species)).temperature


class TestDirectionalTemperature(TestCase):
    """
    rms_velocity may be anisotropic; it is translated to a directional (per-component)
    temperature (see #5677) instead of requiring isotropic components.
    """

    def test_anisotropic_rms_velocity_accepted(self):
        distribution = _gaussian_distribution([1e5, 2e5, 3e5])
        assert distribution.rms_velocity == (1e5, 2e5, 3e5)

    def test_anisotropic_rms_velocity_gives_directional_temperature(self):
        temperature = _momentum_of([1e5, 2e5, 3e5])
        assert temperature is not None
        assert temperature.temperature_kev is None
        assert temperature.temperature_kev_directional is not None
        expected = (5.685630111285689e-05, 2.2742520445142756e-04, 5.11706710015712e-04)
        for given, want in zip(temperature.temperature_kev_directional, expected):
            assert abs(given - want) < 1e-12

    def test_isotropic_rms_velocity_gives_scalar_temperature(self):
        temperature = _momentum_of([1e5, 1e5, 1e5])
        assert temperature is not None
        assert temperature.temperature_kev_directional is None
        assert abs(temperature.temperature_kev - 5.685630111285689e-05) < 1e-12

    def test_zero_rms_velocity_gives_no_temperature(self):
        assert _momentum_of([0, 0, 0]) is None
