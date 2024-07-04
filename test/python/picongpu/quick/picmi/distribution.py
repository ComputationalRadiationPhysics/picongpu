"""
This file is part of the PIConGPU.
Copyright 2021-2023 PIConGPU contributors
Authors: Hannes Troepgen, Brian Edward Marre
License: GPLv3+
"""

from picongpu import picmi

import unittest

from picongpu.pypicongpu import species
import math
from typeguard import typechecked


@typechecked
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

    @unittest.skip("not implemented")
    def test_boundary_not_given_at_all(self):
        """no boundaries supplied at all"""
        picmi_dist = self._get_distribution([None, None, None], [None, None, None])
        pypic = picmi_dist.get_as_pypicongpu()
        self.assertEqual((math.inf, math.inf, math.inf), pypic.upper_bound)
        self.assertEqual((-math.inf, -math.inf, -math.inf), pypic.lower_bound)

    @unittest.skip("not implemented")
    def test_boundary_not_given_partial(self):
        """only some boundaries (components) are missing"""
        picmi_dist = self._get_distribution(lower_bound=[123, -569, None], upper_bound=[124, None, 17])
        pypic = picmi_dist.get_as_pypicongpu()
        self.assertEqual((123, -569, -math.inf), pypic.lower_bound)
        self.assertEqual((124, math.inf, 17), pypic.upper_bound)

    @unittest.skip("not implemented")
    def test_boundary_passthru(self):
        picmi_dist = self._get_distribution(lower_bound=[111, 222, 333], upper_bound=[444, 555, 666])
        pypic = picmi_dist.get_as_pypicongpu()
        self.assertTrue(isinstance(pypic, species.attributes.position.profile.Profile))
        self.assertEqual((111, 222, 333), pypic.lower_bound)
        self.assertEqual((444, 555, 666), pypic.upper_bound)


class TestPicmiUniformDistribution(unittest.TestCase, HelperTestPicmiBoundaries):
    def _get_distribution(self, lower_bound, upper_bound):
        return picmi.UniformDistribution(density=1716273, lower_bound=lower_bound, upper_bound=upper_bound)

    def test_full(self):
        """full paramset"""
        uniform = picmi.UniformDistribution(density=42.42, lower_bound=[111, 222, 333], upper_bound=[444, 555, 666])
        pypic = uniform.get_as_pypicongpu()
        self.assertTrue(isinstance(pypic, species.operation.densityprofile.Uniform))

        self.assertEqual(42.42, pypic.density_si)
        # TODO
        # self.assertEqual((111, 222, 333), pypic.lower_bound)
        # self.assertEqual((444, 555, 666), pypic.upper_bound)

    def test_density_zero(self):
        """density set to zero is accepted"""
        uniform = picmi.UniformDistribution(density=0)
        pypic = uniform.get_as_pypicongpu()
        # no error:
        self.assertEqual(0, pypic.density_si)

    def test_mandatory(self):
        """check that mandatory must be given"""
        # type of exception is not checked
        with self.assertRaises(Exception):
            picmi.UniformDistribution().get_as_pypicongpu()

        # density is only required param
        picmi.UniformDistribution(density=3.14).get_as_pypicongpu()

    def test_drift(self):
        """drift is correctly translated"""
        # no drift
        uniform = picmi.UniformDistribution(density=1, directed_velocity=[0, 0, 0])
        drift = uniform.get_picongpu_drift()
        self.assertEqual(None, drift)

        # some drift
        # uses velocity
        uniform = picmi.UniformDistribution(density=1, directed_velocity=[278487224.0, 103784563.0, 1283345.0])
        drift = uniform.get_picongpu_drift()
        self.assertNotEqual(None, drift)
        self.assertAlmostEqual(drift.gamma, 7.6208808298928865)
        self.assertAlmostEqual(drift.direction_normalized[0], 0.9370354841199405)
        self.assertAlmostEqual(drift.direction_normalized[1], 0.34920746753855203)
        self.assertAlmostEqual(drift.direction_normalized[2], 0.004318114799291135)


@unittest.skip("not implemented")
@typechecked
class TestPicmiAnalyticDistriution(unittest.TestCase, HelperTestPicmiBoundaries):
    def _get_distribution(self, lower_bound, upper_bound):
        return picmi.AnalyticDistribution(density_expression="x+y+z", lower_bound=lower_bound, upper_bound=upper_bound)

    def __get_profile_cpp_str(self, profile: species.operation.densityprofile.DensityProfile) -> str:
        cpp_str = " ".join(map(str, profile.get_cpp_preamble("species_t")))
        cpp_str += "\n" + str(profile.get_cpp_initexpr("species_t"))
        return cpp_str

    def test_full(self):
        """full paramset"""
        analytic = picmi.AnalyticDistribution(
            density_expression="x+y+z",
            lower_bound=[111, 222, 333],
            upper_bound=[444, 555, 666],
        )
        pypic = analytic.get_as_pypicongpu()
        self.assertTrue(isinstance(pypic, species.attributes.position.profile.Analytic))

        self.assertEqual((111, 222, 333), pypic.lower_bound)
        self.assertEqual((444, 555, 666), pypic.upper_bound)

        cpp_str = self.__get_profile_cpp_str(pypic)
        for expected_in_cpp_str in ["x", "y", "z", "+"]:
            self.assertTrue(expected_in_cpp_str in cpp_str)

    def test_substitution(self):
        """extra args are substituted inside expression(s)"""
        expr_str = "(x - x_offset) * z + my_extra_arg + constant_offset / y"
        analytic = picmi.AnalyticDistribution(
            density_expression=expr_str,
            my_extra_arg=1723,
            constant_offset=1,
            x_offset=123.3,
        )
        pypic = analytic.get_as_pypicongpu()

        # note: the conversion to cpp str serves to trigger pypicongpu-internal
        # checks
        cpp_str = self.__get_profile_cpp_str(pypic)
        # note: do not check for floats, they might be translated imprecisely
        expected = ["x", "y", "*", "1723", "1", "123"]
        for expected_in_cpp_str in expected:
            self.assertTrue(expected_in_cpp_str in cpp_str)
        # note: do not check for + AND -, as they migh be joined together
        self.assertTrue("+" in cpp_str or "-" in cpp_str)

        not_expected = ["my_extra_arg", "x_offset", "constant_offset"]
        for not_expected_in_cpp_str in not_expected:
            self.assertTrue(not_expected_in_cpp_str not in cpp_str)

    def test_mandatory(self):
        """check that mandatory must be given"""
        # type of exception is not checked
        with self.assertRaises(Exception):
            picmi.AnalyticDistribution().get_as_pypicongpu()

        # density is only required param
        picmi.AnalyticDistribution(density_expression="3.14").get_as_pypicongpu()

    def test_drift(self):
        """drift is correctly translated"""
        # no drift
        analytic = picmi.AnalyticDistribution(density_expression="1", directed_velocity=[0, 0, 0])
        drift = analytic.get_picongpu_drift()
        self.assertEqual(None, drift)

        # some drift
        # uses velocity
        analytic = picmi.AnalyticDistribution(
            density_expression="1", directed_velocity=[111111111, 111111111, 111111111]
        )
        drift = analytic.get_picongpu_drift()
        self.assertNotEqual(None, drift)
        self.assertAlmostEqual(drift.gamma, 1.3042040403107296)
        self.assertAlmostEqual(drift.direction_normalized[0], 0.5773502691896257)
        self.assertAlmostEqual(drift.direction_normalized[1], 0.5773502691896257)
        self.assertAlmostEqual(drift.direction_normalized[2], 0.5773502691896257)


@unittest.skip("not implemented")
@typechecked
class TestPicmiGaussianBunchDistribution(unittest.TestCase):
    def test_full(self):
        """check for all possible params"""
        gb = picmi.GaussianBunchDistribution(1337, 0.05, centroid_position=[111, 222, 333])
        pypic = gb.get_as_pypicongpu()
        self.assertTrue(isinstance(pypic, species.attributes.position.profile.GaussianCloud))
        self.assertEqual((111, 222, 333), pypic.centroid_position_si)
        self.assertAlmostEqual(0.05, pypic.rms_bunch_size_si)
        self.assertAlmostEqual(679127.9299526414, pypic.max_density_si)

    def test_defaults(self):
        """mandatory params are enforced"""
        gb = picmi.GaussianBunchDistribution(1, 1)
        pypic = gb.get_as_pypicongpu()
        self.assertEqual((0, 0, 0), pypic.centroid_position_si)

    def test_conversion(self):
        """params are correctly transformed"""
        max_density_by_n_particles_and_rms_bunch_size = {
            (1337, 0.05): 679127.9299526414,
            (1, 1): 0.06349363593424097,
            (10, 1): 0.6349363593424097,
            (1968.7012432153024, 5): 1,
        }

        for param_tuple in max_density_by_n_particles_and_rms_bunch_size:
            n_particles, rms_bunch_size = param_tuple
            gb = picmi.GaussianBunchDistribution(n_particles, rms_bunch_size)
            pypic = gb.get_as_pypicongpu()
            self.assertAlmostEqual(
                pypic.max_density_si,
                max_density_by_n_particles_and_rms_bunch_size[param_tuple],
            )

    def test_drift(self):
        """drift is correctly translated"""
        # no drift
        gb = picmi.GaussianBunchDistribution(1, 1, centroid_velocity=[0, 0, 0])
        drift = gb.get_picongpu_drift()
        self.assertEqual(None, drift)

        # some drift
        # uses velocity * gamma as input
        gb = picmi.GaussianBunchDistribution(
            1,
            1,
            centroid_velocity=[
                17694711.860033844,
                1666140.8815973825,
                63366940.8605043,
            ],
        )
        drift = gb.get_picongpu_drift()
        self.assertNotEqual(None, drift)
        self.assertAlmostEqual(drift.gamma, 1.0238123040019211)
        self.assertAlmostEqual(drift.direction_normalized[0], 0.2688666691231957)
        self.assertAlmostEqual(drift.direction_normalized[1], 0.025316589084272104)
        self.assertAlmostEqual(drift.direction_normalized[2], 0.9628446315744488)


class TestPicmiFoilDistribution(unittest.TestCase, HelperTestPicmiBoundaries):
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
            lower_bound=[111, 222, 333],
            upper_bound=[444, 555, 666],
        )

        pypic = foil.get_as_pypicongpu()
        self.assertTrue(isinstance(pypic, species.operation.densityprofile.Foil))

        self.assertEqual(42.42, pypic.density_si)
        self.assertEqual(1.0, pypic.y_value_front_foil_si)
        self.assertEqual(2.0, pypic.thickness_foil_si)
        self.assertEqual(3.0, pypic.pre_foil_plasmaRamp.PlasmaLength)
        self.assertEqual(4.0, pypic.pre_foil_plasmaRamp.PlasmaCutoff)
        self.assertEqual(5.0, pypic.post_foil_plasmaRamp.PlasmaLength)
        self.assertEqual(6.0, pypic.post_foil_plasmaRamp.PlasmaCutoff)

        # @todo repect bounding boxes, Brian Marre, 2023
        # self.assertEqual((111, 222, 333), pypic.lower_bound)
        # self.assertEqual((444, 555, 666), pypic.upper_bound)

    def test_density_zero(self):
        """density set to zero is not accepted"""
        foil = picmi.FoilDistribution(density=0, thickness=1.0, front=2.0)
        with self.assertRaisesRegex(ValueError, ".*density must be > 0.*"):
            foil.get_as_pypicongpu().get_generic_profile_rendering_context()

    def test_front_zero(self):
        """front set to zero is accepted"""
        foil = picmi.FoilDistribution(density=1.0, thickness=2.0, front=0)
        pypic = foil.get_as_pypicongpu()
        # no error:
        self.assertEqual(0, pypic.y_value_front_foil_si)

    def test_thickness_zero(self):
        """thickness set to zero is accepted"""
        foil = picmi.FoilDistribution(density=1.0, thickness=0, front=2.0)
        pypic = foil.get_as_pypicongpu()
        # no error
        self.assertEqual(0, pypic.thickness_foil_si)

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
            pypic = entry.get_as_pypicongpu()
            # no error:
            self.assertEqual(1.0, pypic.density_si)
            self.assertEqual(2.0, pypic.thickness_foil_si)
            self.assertEqual(3.0, pypic.y_value_front_foil_si)

    def test_cutoff_below_zero(self):
        """length below zero is not accepted"""

        testCases = self._get_test_foils(-1.0, 1.0)

        for i, entry in enumerate(testCases):
            with self.assertRaisesRegex(ValueError, ".*PlasmaCutoff must be >=0.*"):
                entry.get_as_pypicongpu().get_generic_profile_rendering_context()

    def test_length_zero(self):
        """length set to zero is not accepted"""
        testCases = self._get_test_foils(1.0, 0)

        for entry in testCases:
            with self.assertRaisesRegex(ValueError, ".*PlasmaLength must be >0.*"):
                entry.get_as_pypicongpu().get_generic_profile_rendering_context()

    def test_length_below_zero(self):
        """length below zero is not accepted"""

        testCases = self._get_test_foils(1.0, -1.0)

        for entry in testCases:
            with self.assertRaisesRegex(ValueError, ".*PlasmaLength must be >0.*"):
                entry.get_as_pypicongpu().get_generic_profile_rendering_context()

    def test_setting_noPlasmaRamps(self):
        testCases = self._get_test_foils(None, 1.0)

        for entry in testCases:
            with self.assertRaisesRegex(
                ValueError,
                "either both exponential_(pre|post)_plasma_"
                "length and exponential_(pre|post)_plasma_cutoff must be"
                " set to none or neither!",
            ):
                entry.get_as_pypicongpu().get_generic_profile_rendering_context()

        testCases = self._get_test_foils(1.0, None)
        for entry in testCases:
            with self.assertRaisesRegex(
                ValueError,
                "either both exponential_(pre|post)_plasma_"
                "length and exponential_(pre|post)_plasma_cutoff must be"
                " set to none or neither!",
            ):
                entry.get_as_pypicongpu().get_generic_profile_rendering_context()

    def test_mandatory(self):
        """check that mandatory must be given"""
        # type of exception is not checked
        with self.assertRaises(Exception):
            picmi.FoilDistribution().get_as_pypicongpu()

        # density, thickness and front are only required param
        picmi.FoilDistribution(density=3.14, thickness=1.0, front=3.0).get_as_pypicongpu()

    def test_drift(self):
        """drift is correctly translated"""
        # no drift
        foil = picmi.FoilDistribution(density=1.0, front=2.0, thickness=3.0, directed_velocity=[0, 0, 0])
        drift = foil.get_picongpu_drift()
        self.assertEqual(None, drift)

        # some drift
        # uses velocity
        foil = picmi.FoilDistribution(
            density=1,
            front=2.0,
            thickness=3.0,
            directed_velocity=[278487224.0, 103784563.0, 1283345.0],
        )
        drift = foil.get_picongpu_drift()
        self.assertNotEqual(None, drift)
        self.assertAlmostEqual(drift.gamma, 7.6208808298928865)
        self.assertAlmostEqual(drift.direction_normalized[0], 0.9370354841199405)
        self.assertAlmostEqual(drift.direction_normalized[1], 0.34920746753855203)
        self.assertAlmostEqual(drift.direction_normalized[2], 0.004318114799291135)


class TestPicmiGaussianDistribution(unittest.TestCase, HelperTestPicmiBoundaries):
    def _get_distribution(self, lower_bound=[None, None, None], upper_bound=[None, None, None]):
        return picmi.GaussianDistribution(
            density=42.42,
            center_front=1.0,
            center_rear=2.0,
            sigma_front=3.0,
            sigma_rear=4.0,
            power=5.0,
            factor=-6.0,
            vacuum_cells_front=50,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
        )

    def test_full(self):
        """full paramset"""
        gaussian = self._get_distribution()

        pypic = gaussian.get_as_pypicongpu()
        self.assertTrue(isinstance(pypic, species.operation.densityprofile.Gaussian))

        self.assertEqual(42.42, pypic.density)
        self.assertEqual(1.0, pypic.gas_center_front)
        self.assertEqual(2.0, pypic.gas_center_rear)
        self.assertEqual(3.0, pypic.gas_sigma_front)
        self.assertEqual(4.0, pypic.gas_sigma_rear)
        self.assertEqual(5.0, pypic.gas_power)
        self.assertEqual(-6.0, pypic.gas_factor)
        self.assertEqual(50, pypic.vacuum_cells_front)

        # @todo repect bounding boxes, Brian Marre, 2024

    def test_density_zero(self):
        """density set to zero is not accepted"""
        gaussian = self._get_distribution()
        gaussian.density = 0.0
        with self.assertRaisesRegex(ValueError, ".*density must be > 0.*"):
            gaussian.get_as_pypicongpu().get_generic_profile_rendering_context()

    def test_front_rear_swapped(self):
        """front and rear swapped is not accepted"""
        gaussian = self._get_distribution()
        gaussian.center_front = 3.0
        gaussian.center_rear = 2.0
        with self.assertRaisesRegex(ValueError, ".*center_front must be <= center_rear.*"):
            gaussian.get_as_pypicongpu().get_generic_profile_rendering_context()

    def test_sigma_zero(self):
        """sigma == 0 is not accepted"""
        gaussian = self._get_distribution()
        gaussian.sigma_front = 0.0
        with self.assertRaisesRegex(ValueError, ".*sigma_front must be != 0.*"):
            gaussian.get_as_pypicongpu().get_generic_profile_rendering_context()

        gaussian = self._get_distribution()
        gaussian.sigma_rear = 0.0
        with self.assertRaisesRegex(ValueError, ".*sigma_rear must be != 0.*"):
            gaussian.get_as_pypicongpu().get_generic_profile_rendering_context()

    def test_drift(self):
        """drift is correctly translated"""
        # no drift
        gaussian = self._get_distribution()
        gaussian.directed_velocity = [0, 0, 0]
        drift = gaussian.get_picongpu_drift()
        self.assertEqual(None, drift)

        # some drift
        # uses velocity
        gaussian = self._get_distribution()
        gaussian.directed_velocity = [278487224.0, 103784563.0, 1283345.0]

        drift = gaussian.get_picongpu_drift()
        self.assertNotEqual(None, drift)
        self.assertAlmostEqual(drift.gamma, 7.6208808298928865)
        self.assertAlmostEqual(drift.direction_normalized[0], 0.9370354841199405)
        self.assertAlmostEqual(drift.direction_normalized[1], 0.34920746753855203)
        self.assertAlmostEqual(drift.direction_normalized[2], 0.004318114799291135)
