"""
This file is part of the PIConGPU.
Copyright 2021-2023 PIConGPU contributors
Authors: Hannes Troepgen, Brian Edward Marre
License: GPLv3+
"""

from picongpu.pypicongpu.species.operation import SimpleDensity, Operation

import unittest

from picongpu.pypicongpu.species import Species
from picongpu.pypicongpu.species.operation import densityprofile
from picongpu.pypicongpu.species.attribute import \
    Position, Weighting, Momentum
from picongpu.pypicongpu.species.constant import DensityRatio


class TestSimpleDensity(unittest.TestCase):
    def setUp(self):
        self.species1 = Species()
        self.species1.name = "species1"
        self.species1_density_ratio = DensityRatio()
        self.species1_density_ratio.ratio = 0.8
        self.species1.constants = [self.species1_density_ratio]

        self.species2 = Species()
        self.species2.name = "species2"
        self.species2_density_ratio = DensityRatio()
        self.species2_density_ratio.ratio = 1
        self.species2.constants = [self.species2_density_ratio]

        self.species3 = Species()
        self.species3.name = "species3"
        self.species3_density_ratio = DensityRatio()
        self.species3_density_ratio.ratio = 5
        self.species3.constants = [self.species3_density_ratio]

        self.species4 = Species()
        self.species4.name = "species4"
        # note: no explicit density ratio (should be assumed 1)
        self.species4.constants = []

        self.profile = densityprofile.Uniform()
        self.profile.density_si = 42

        self.sd = SimpleDensity()
        self.sd.ppc = 2
        self.sd.profile = self.profile
        self.sd.species = {
            self.species1,
            self.species3,
            self.species2,
            self.species4,
        }

    def test_basic(self):
        """simple scenario"""
        # passes silently
        self.sd.check_preconditions()
        self.sd.prebook_species_attributes()

    def test_inheritance(self):
        """is an operation"""
        self.assertTrue(issubclass(SimpleDensity, Operation))

    def test_check_passthru(self):
        """passes check through to profile & density ratios"""
        # break profile -> check fails
        self.sd.profile.density_si = -2

        # direct check fails
        with self.assertRaises(ValueError):
            self.sd.profile.check()

        # ... as well as check of entire object
        with self.assertRaises(ValueError):
            self.sd.check_preconditions()

        # but now ok:
        self.sd.profile.density_si = 1
        self.sd.check_preconditions()

        # ratios are checked too:
        self.assertTrue(self.species3 in self.sd.species)
        self.assertNotEqual([], self.species3.constants)
        density_ratio_const = self.species3.constants[0]

        self.assertTrue(isinstance(density_ratio_const, DensityRatio))

        # update ratio s.t. it now violates checks
        density_ratio_const.ratio = -1
        with self.assertRaises(ValueError):
            self.sd.check_preconditions()

    def test_typesafety(self):
        """typesafety enforced"""
        for invalid_pcc in [None, [], {}]:
            with self.assertRaises(TypeError):
                self.sd.ppc = invalid_pcc

        for invalid_profile in [None, [], {}, 1, "3"]:
            with self.assertRaises(TypeError):
                self.sd.profile = invalid_profile

        for invalid_set in [None, {self.species1: 2}, 1, {1, 2}, "3"]:
            with self.assertRaises(TypeError):
                self.sd.species = invalid_set

    def test_empty(self):
        """empty object raises"""
        sd = SimpleDensity()

        # non-assigned attributes
        with self.assertRaises(Exception):
            sd.check_preconditions()

        sd.species = {
            self.species1,
        }
        with self.assertRaises(Exception):
            sd.check_preconditions()

        sd.profile = self.profile
        with self.assertRaises(Exception):
            sd.check_preconditions()

        sd.ppc = 1

        # now all attributes present -> ok
        sd.check_preconditions()

        # test with empty set of species -> must break
        sd.species = set()

        # along the lines of "at least one species"
        with self.assertRaisesRegex(ValueError, ".*species.*"):
            sd.check_preconditions()
        # also does not render
        with self.assertRaisesRegex(ValueError, ".*species.*"):
            sd.get_rendering_context()

    def test_check(self):
        """enforces value ranges"""
        # is valid on its own
        sd = self.sd
        sd.check_preconditions()

        # pcc non-negative
        for invalid_ppc in [0, -1, -1000]:
            sd.ppc = invalid_ppc
            with self.assertRaisesRegex(ValueError, ".*particle.*per.*cell.*"):
                sd.check_preconditions()
        sd.ppc = 1
        sd.check_preconditions()

        # (profile check passthru tested above in test_check_passthru())
        # (density ratio check passthru tested above in test_check_passthru())
        # (at least one species required is tested above in test_empty())

    def test_prebooking(self):
        """prebooks correct attributes"""
        sd = self.sd
        self.assertEqual(4, len(sd.species))
        sd.check_preconditions()
        sd.prebook_species_attributes()

        self.assertEqual(4, len(sd.attributes_by_species))
        self.assertTrue(self.species1 in sd.attributes_by_species)
        self.assertTrue(self.species2 in sd.attributes_by_species)
        self.assertTrue(self.species3 in sd.attributes_by_species)
        self.assertTrue(self.species4 in sd.attributes_by_species)

        # for each species the same
        for species, attributes in sd.attributes_by_species.items():
            # assign position & weighting
            attribute_names = list(map(lambda attr: attr.PICONGPU_NAME,
                                       attributes))
            self.assertEqual(2, len(attribute_names))
            self.assertTrue(Position.PICONGPU_NAME in attribute_names)
            self.assertTrue(Weighting.PICONGPU_NAME in attribute_names)

    def test_rendering_context(self):
        """rendering works & passes values through"""
        # initialized as would be performed by initmanager (s.t. attributes are
        # defined)
        for species in self.sd.species:
            species.attributes = [Momentum()]
        self.sd.check_preconditions()
        self.sd.prebook_species_attributes()
        self.sd.bake_species_attributes()

        context = self.sd.get_rendering_context()

        self.assertEqual(2, context["ppc"])
        self.assertEqual(
            context["profile"],
            self.sd.profile.get_generic_profile_rendering_context())

        # species with lowest ratio must be placed as first, which is species1
        self.assertEqual(context["placed_species_initial"],
                         self.species1.get_rendering_context())

        copied_species_names = list(map(lambda d: d["name"],
                                        context["placed_species_copied"]))
        self.assertEqual({"species2", "species3", "species4"},
                         set(copied_species_names))

    def test_rendering_minimal(self):
        """minimal example for rendering"""
        species = Species()
        species.name = "species1"
        species.constants = []

        sd = SimpleDensity()
        sd.profile = self.profile
        sd.ppc = 1
        sd.species = {species}

        # would normally be performed by init manager:
        species.attributes = [Momentum()]
        sd.check_preconditions()
        sd.prebook_species_attributes()
        sd.bake_species_attributes()

        # actual checks
        context = sd.get_rendering_context()

        self.assertEqual(1, context["ppc"])
        self.assertEqual(
            context["profile"],
            sd.profile.get_generic_profile_rendering_context())

        self.assertEqual(context["placed_species_initial"],
                         species.get_rendering_context())
        self.assertEqual(context["placed_species_copied"], [])
