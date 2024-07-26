"""
This file is part of PIConGPU.
Copyright 2021-2023 PIConGPU contributors
Authors: Hannes Troepgen, Brian Edward Marre
License: GPLv3+
"""

from picongpu import picmi

import unittest
import typeguard

from picongpu.pypicongpu import species
from picongpu.picmi.interaction import Interaction
from picongpu.picmi.interaction.ionization.fieldionization import ADK, ADKVariant
from copy import deepcopy
import re
import logging


class TestPicmiSpecies(unittest.TestCase):
    def setUp(self):
        self.profile_uniform = picmi.UniformDistribution(density=42, rms_velocity=[1, 1, 1])

        self.species_electron = picmi.Species(
            name="e",
            density_scale=3,
            particle_type="electron",
            initial_distribution=self.profile_uniform,
        )
        self.species_nitrogen = picmi.Species(
            name="nitrogen",
            charge_state=+3,
            particle_type="N",
            picongpu_fixed_charge=True,
            initial_distribution=self.profile_uniform,
        )

    def __helper_get_distributions_with_rms_velocity(self):
        """
        helper to get a list of (all) profiles (PICMI distributions) that have
        an rms_velocity attribute.

        intended to run tests against this temperature
        """
        return [
            # TODO add profiles after implementation
            # picmi.GaussianBunchDistribution(4e10, 4e-15),
            picmi.UniformDistribution(8e24),
            # picmi.AnalyticDistribution("x+y+z"),
        ]

    def test_basic(self):
        """check that all params are translated"""
        # check that translation works
        for s in [self.species_electron, self.species_nitrogen]:
            pypic, rest = s.get_as_pypicongpu(None)
            del rest
            self.assertEqual(pypic.name, s.name)

    def test_mandatory(self):
        """mandatory params are enforced with a somewhat reasonable message"""
        # required: name, particle type
        species_no_name = picmi.Species(particle_type="N")
        species_empty = picmi.Species()
        species_invalid_list = [species_no_name, species_empty]

        for invalid_species in species_invalid_list:
            with self.assertRaises(AssertionError):
                invalid_species.get_as_pypicongpu(None)

        # (everything else is optional)

    def test_mass_charge(self):
        """mass & charge are passed through"""
        picmi_s = picmi.Species(name="any", mass=17, charge=-4)
        pypicongpu_s, rest = picmi_s.get_as_pypicongpu(None)

        mass_const = pypicongpu_s.get_constant_by_type(species.constant.Mass)
        self.assertEqual(17, mass_const.mass_si)

        charge_const = pypicongpu_s.get_constant_by_type(species.constant.Charge)
        self.assertEqual(-4, charge_const.charge_si)

    def test_density_scale(self):
        """density scale is correctly transformed"""
        # simple example
        picmi_s = picmi.Species(name="any", density_scale=37.2)
        pypicongpu_s, rest = picmi_s.get_as_pypicongpu(None)

        ratio_const = pypicongpu_s.get_constant_by_type(species.constant.DensityRatio)
        self.assertAlmostEqual(37.2, ratio_const.ratio)

        # no density scale
        picmi_s = picmi.Species(name="any")
        pypicongpu_s, rest = picmi_s.get_as_pypicongpu(None)
        self.assertTrue(not pypicongpu_s.has_constant_of_type(species.constant.DensityRatio))

    def test_get_independent_operations(self):
        """operations which can be set without external dependencies work"""
        picmi_s = picmi.Species(name="any", mass=1, charge=2)
        pypicongpu_s, rest = picmi_s.get_as_pypicongpu(None)

        # note: placement is not considered independent (it depends on also
        # having no layout)
        self.assertNotEqual(None, picmi_s.get_independent_operations(pypicongpu_s, None))

    def test_get_independent_operations_type(self):
        """arg type is checked"""
        picmi_s = picmi.Species(name="any", mass=1, charge=2)
        for invalid_species in [[], None, picmi_s, "name"]:
            with self.assertRaises(typeguard.TypeCheckError):
                picmi_s.get_independent_operations(invalid_species, None)

    def test_get_independent_operations_different_name(self):
        """only generate operations for pypicongpu species of same name"""
        picmi_s = picmi.Species(name="any", mass=1, charge=2)
        pypicongpu_s, rest = picmi_s.get_as_pypicongpu(None)

        pypicongpu_s.name = "different"
        with self.assertRaisesRegex(AssertionError, ".*name.*"):
            picmi_s.get_independent_operations(pypicongpu_s, None)

        # same name is okay:
        pypicongpu_s.name = "any"
        self.assertNotEqual(None, picmi_s.get_independent_operations(pypicongpu_s, None))

    def test_get_independent_operations_ionization_set_bound_electrons(self):
        """SetBoundElectrons is properly generated"""
        picmi_species = picmi.Species(name="nitrogen", particle_type="N", charge_state=2)
        e = picmi.Species(name="e", particle_type="electron")
        interaction = Interaction(
            ground_state_ionization_model_list=[
                ADK(
                    ion_species=picmi_species,
                    ionization_current=None,
                    ionization_electron_species=e,
                    ADK_variant=ADKVariant.LinearPolarization,
                )
            ]
        )

        pypic_species, rest = picmi_species.get_as_pypicongpu(interaction)
        ops = picmi_species.get_independent_operations(pypic_species, interaction)
        ops_types = list(map(lambda op: type(op), ops))
        self.assertEqual(1, ops_types.count(species.operation.SetBoundElectrons))
        self.assertEqual(0, ops_types.count(species.operation.NoBoundElectrons))

        for op in ops:
            if not isinstance(op, species.operation.SetBoundElectrons):
                continue

            self.assertEqual(pypic_species, op.species)
            self.assertEqual(5, op.bound_electrons)

    def test_get_independent_operations_ionization_not_ionizable(self):
        """ionization operation is not returned if there is no ionization"""
        picmi_species = picmi.Species(name="hydrogen", particle_type="H", picongpu_fixed_charge=True)
        pypic_species, rest = picmi_species.get_as_pypicongpu(None)

        ops = picmi_species.get_independent_operations(pypic_species, None)
        ops_types = list(map(lambda op: type(op), ops))
        self.assertEqual(0, ops_types.count(species.operation.NoBoundElectrons))
        self.assertEqual(0, ops_types.count(species.operation.SetBoundElectrons))

    def test_get_independent_operations_momentum(self):
        """momentum is correctly translated"""
        for set_drift in [False, True]:
            for set_temperature in [False, True]:
                for dist in self.__helper_get_distributions_with_rms_velocity():
                    if set_temperature:
                        dist.rms_velocity = 3 * [42]

                    if set_drift:
                        # note: same velocity, different representations
                        if isinstance(dist, picmi.UniformDistribution) or isinstance(dist, picmi.AnalyticDistribution):
                            # v (as is)
                            dist.directed_velocity = [41363723.0, 8212468.0, 68174325.0]
                        elif isinstance(dist, picmi.GaussianBunchDistribution):
                            # v * gamma
                            dist.centroid_velocity = [
                                42926825.65008125,
                                8522810.724577945,
                                70750579.27176853,
                            ]
                        else:
                            # fail: unkown distribution type
                            assert False, "unkown distribution type in " "test: {}".format(type(dist))

                    picmi_s = picmi.Species(name="name", mass=1, initial_distribution=dist)

                    pypicongpu_s, rest = picmi_s.get_as_pypicongpu(None)
                    ops = picmi_s.get_independent_operations(pypicongpu_s, None)

                    momentum_ops = list(
                        filter(
                            lambda op: isinstance(op, species.operation.SimpleMomentum),
                            ops,
                        )
                    )

                    self.assertEqual(1, len(momentum_ops))
                    # must pass silently
                    momentum_ops[0].check_preconditions()
                    self.assertEqual(pypicongpu_s, momentum_ops[0].species)

                    if set_drift:
                        self.assertEqual(
                            momentum_ops[0].drift.direction_normalized,
                            (
                                0.5159938229615939,
                                0.10244684114313779,
                                0.8504440130927325,
                            ),
                        )
                        self.assertAlmostEqual(momentum_ops[0].drift.gamma, 1.0377892156874091)
                    else:
                        self.assertEqual(None, momentum_ops[0].drift)

                    if set_temperature:
                        self.assertAlmostEqual(
                            momentum_ops[0].temperature.temperature_kev,
                            1.10100221e19,
                            delta=1e13,
                        )
                    else:
                        self.assertEqual(None, momentum_ops[0].temperature)

    def test_temperature_invalid(self):
        """check that invalid rms_velocities are not converted"""
        for dist in self.__helper_get_distributions_with_rms_velocity():

            def get_rms_species(rms_velocity):
                dist_copy = deepcopy(dist)
                dist_copy.rms_velocity = rms_velocity
                new_species = picmi.Species(name="name", mass=1, initial_distribution=dist_copy)
                return new_species

            # all components must be equal
            invalid_rms_vectors = [[0, 1, 1], [1, 0, 1], [1, 1, 0], [1, 2, 3]]
            for invalid_rms_vector in invalid_rms_vectors:
                rms_species = get_rms_species(invalid_rms_vector)
                with self.assertRaisesRegex(Exception, ".*(equal|same).*"):
                    pypicongpu_species, rest = rms_species.get_as_pypicongpu(None)
                    rms_species.get_independent_operations(pypicongpu_species, None)

    def test_from_speciestype(self):
        """mass & charge will be derived from species type"""
        picmi_species = picmi.Species(name="nitrogen", particle_type="N", charge_state=5)
        e = picmi.Species(name="e", particle_type="electron")

        interaction = Interaction(
            ground_state_ionization_model_list=[
                ADK(
                    ion_species=picmi_species,
                    ionization_current=None,
                    ionization_electron_species=e,
                    ADK_variant=ADKVariant.LinearPolarization,
                )
            ]
        )

        pypic_species, rest = picmi_species.get_as_pypicongpu(interaction)

        # mass & charge derived
        self.assertTrue(pypic_species.has_constant_of_type(species.constant.Mass))
        self.assertTrue(pypic_species.has_constant_of_type(species.constant.Charge))

        mass_const = pypic_species.get_constant_by_type(species.constant.Mass)
        charge_const = pypic_species.get_constant_by_type(species.constant.Charge)

        nitrogen = species.util.Element("N")
        self.assertAlmostEqual(mass_const.mass_si, nitrogen.get_mass_si())
        self.assertAlmostEqual(charge_const.charge_si, nitrogen.get_charge_si())

        # element properties are available
        self.assertTrue(pypic_species.has_constant_of_type(species.constant.ElementProperties))

    def test_charge_state_without_element_forbidden(self):
        """charge state is not allowed without element name"""
        with self.assertRaisesRegex(Exception, ".*particle_type.*"):
            picmi.Species(name="abc", charge=1, mass=1, charge_state=-1, picongpu_fixed_charge=True).get_as_pypicongpu(
                None
            )

        # allowed with particle species
        # (actual charge state is inserted by )
        picmi.Species(name="abc", particle_type="H", charge_state=+1, picongpu_fixed_charge=True).get_as_pypicongpu(
            None
        )

    def test_has_ionizers(self):
        """generated species gets ionizers when appropriate"""
        # only mass & charge: no ionizers
        no_ionizers_picmi = picmi.Species(name="simple", mass=1, charge=2)
        no_ionizers_pypic, rest = no_ionizers_picmi.get_as_pypicongpu(None)
        self.assertTrue(not no_ionizers_pypic.has_constant_of_type(species.constant.GroundStateIonization))

        # no charge state, but (theoretically) ionization levels known (as
        # particle type is given):
        with self.assertLogs(level=logging.WARNING) as implicit_logs:
            with_warn_picmi = picmi.Species(name="HELIUM", particle_type="He", picongpu_fixed_charge=True)

            with_warn_pypic, rest = with_warn_picmi.get_as_pypicongpu(None)
            self.assertTrue(not with_warn_pypic.has_constant_of_type(species.constant.GroundStateIonization))

        self.assertEqual(1, len(implicit_logs.output))
        self.assertTrue(
            re.match(
                ".*HELIUM.*fixed charge state.*",
                implicit_logs.output[0],
            )
        )

        with self.assertLogs(level=logging.WARNING) as explicit_logs:
            # workaround b/c self.assertNoLogs() is not available yet
            logging.warning("TESTWARN")
            no_warn_picmi = picmi.Species(name="HELIUM", particle_type="He", picongpu_fixed_charge=True)
            no_warn_pypic, rest = no_warn_picmi.get_as_pypicongpu(None)
            self.assertTrue(not no_warn_pypic.has_constant_of_type(species.constant.GroundStateIonization))

        self.assertTrue(1 <= len(explicit_logs.output))
        self.assertTrue("TESTWARN" in explicit_logs.output[0])

    def test_fully_ionized_warning_electrons(self):
        """electrons will not have the fully ionized warning"""
        with self.assertLogs(level=logging.WARNING) as explicit_logs:
            # workaround b/c self.assertNoLogs() is not available yet
            logging.warning("TESTWARN")
            no_warn_picmi = picmi.Species(name="ELECTRON", particle_type="electron")

            no_warn_pypic, rest = no_warn_picmi.get_as_pypicongpu(None)
            self.assertTrue(not no_warn_pypic.has_constant_of_type(species.constant.GroundStateIonization))

        self.assertEqual(1, len(explicit_logs.output))
        self.assertTrue("TESTWARN" in explicit_logs.output[0])

    def test_ionize_non_elements(self):
        """non-elements may not have a charge_state"""
        with self.assertRaisesRegex(Exception, ".*charge_state may only be set for ions.*"):
            picmi.Species(name="e", particle_type="electron", charge_state=-1).get_as_pypicongpu(None)

    def test_electron_from_particle_type(self):
        """electron is correctly constructed from particle_type"""
        picmi_e = picmi.Species(name="e", particle_type="electron")
        pypic_e, rest = picmi_e.get_as_pypicongpu(None)
        self.assertTrue(not pypic_e.has_constant_of_type(species.constant.GroundStateIonization))
        self.assertTrue(not pypic_e.has_constant_of_type(species.constant.ElementProperties))

        mass_const = pypic_e.get_constant_by_type(species.constant.Mass)
        charge_const = pypic_e.get_constant_by_type(species.constant.Charge)

        self.assertAlmostEqual(mass_const.mass_si, picmi.constants.m_e)
        self.assertAlmostEqual(charge_const.charge_si, -picmi.constants.q_e)

    def test_fully_ionized_typesafety(self):
        """picongpu_fully_ioinized is type safe"""
        for invalid in [1, "yes", [], {}]:
            with self.assertRaises(typeguard.TypeCheckError):
                picmi.Species(name="x", picongpu_fixed_charge=invalid)

        # works:
        picmi_species = picmi.Species(name="x", particle_type="He", picongpu_fixed_charge=True)

        for invalid in [0, "no", [], {}]:
            with self.assertRaises(typeguard.TypeCheckError):
                picmi_species.picongpu_fixed_charge = invalid

        # None is allowed as value in general (but not in constructor)
        picmi_species.picongpu_fixed_charge = None

    def test_particle_type_invalid(self):
        """unkown particle type rejects"""
        for invalid in ["", "elektron", "e", "e-", "Uux"]:
            with self.assertRaisesRegex(ValueError, ".*not a valid openPMD particle type.*"):
                picmi.Species(name="x", particle_type=invalid).get_as_pypicongpu(None)

    def test_ionization_charge_state_too_large(self):
        """charge state must be <= number of protons"""
        with self.assertRaises(AssertionError):
            picmi.Species(name="x", particle_type="N", charge_state=8).get_as_pypicongpu(None)
