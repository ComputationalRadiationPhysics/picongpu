"""
This file is part of PIConGPU.
Copyright 2021-2023 PIConGPU contributors
Authors: Hannes Troepgen, Brian Edward Marre
License: GPLv3+
"""

from picongpu.pypicongpu.species import Species

from picongpu.pypicongpu.species.attribute import Position, Weighting, Momentum
from picongpu.pypicongpu.species.constant import Mass, Charge, DensityRatio, ElementProperties, Constant
from picongpu.pypicongpu.species.util import Element

from .attribute import DummyAttribute

import itertools
import unittest
import typeguard


class TestSpecies(unittest.TestCase):
    def setUp(self):
        self.pos = Position()
        self.mom = Momentum()
        self.species = Species()
        self.species.attributes = [self.pos, self.mom]
        self.species.constants = []
        self.species.name = "valid"

        self.electron = Species()
        self.electron.name = "electron"
        self.electron.attributes = [Position(), Momentum()]
        self.electron.constants = []

        self.const = Constant()
        self.const_charge = Charge()
        self.const_charge.charge_si = 1
        self.const_mass = Mass()
        self.const_mass.mass_si = 2
        self.const_density_ratio = DensityRatio()
        self.const_density_ratio.ratio = 4.2
        self.const_ionizers = None
        self.const_ionizers.electron_species = self.electron

        self.const_element_properties = ElementProperties()
        self.const_element_properties.element = Element.H

    def test_basic(self):
        """setup provides working species"""
        # does not throw
        self.species.check()

    def test_empty(self):
        """attrs have no defaults"""
        s = Species()

        with self.assertRaises(Exception):
            # must not pass unless attrs are all explicitly set
            s.check()

        s.attributes = [Position(), Momentum()]
        s.constants = []
        # note: non-empty not allowed
        s.name = "x"
        s.check()

    def test_types(self):
        """typesafety is ensured"""
        species = self.species

        for invalid_name in [None, [], {}, 123]:
            with self.assertRaises(typeguard.TypeCheckError):
                species.name = invalid_name

        invalid_attr_lists = [None, {}, set(), [Constant()], DummyAttribute()]
        for invalid_attr_list in invalid_attr_lists:
            with self.assertRaises(typeguard.TypeCheckError):
                species.attributes = invalid_attr_list

        invalid_const_lists = [None, {}, set(), [DummyAttribute()], Constant()]
        for invalid_const_list in invalid_const_lists:
            with self.assertRaises(typeguard.TypeCheckError):
                species.constants = invalid_const_list

    def test_mandatory_attribute_position(self):
        """test position present"""
        self.assertNotEqual([], self.species.attributes)
        self.species.check()

        self.species.attributes = [self.mom]
        with self.assertRaisesRegex(ValueError, ".*position.*"):
            self.species.check()

    def test_mandatory_attribute_momentum(self):
        """test momentum present"""
        self.assertNotEqual([], self.species.attributes)
        self.species.check()

        self.species.attributes = [self.pos]
        with self.assertRaisesRegex(ValueError, ".*momentum.*"):
            self.species.check()

    def test_attributes_unique(self):
        """all defined PIConGPU particle attributes must be uniquely named"""
        species = self.species
        nattr1 = DummyAttribute()
        nattr1.PICONGPU_NAME = "test_attr"
        nattr2 = DummyAttribute()
        nattr2.PICONGPU_NAME = "test_attr"
        other_nattr = DummyAttribute()
        other_nattr.PICONGPU_NAME = "other_attr"

        species.attributes = [self.pos, self.mom, nattr1, nattr2, other_nattr]

        # duplicates -> throw (require violiating attr to be mentioned)
        with self.assertRaisesRegex(ValueError, ".*test_attr.*"):
            species.check()

        for single_attr in [nattr1, nattr2]:
            # no duplicates -> pass silently
            species.attributes = [self.pos, self.mom, single_attr, other_nattr]
            species.check()

    def test_constants_unique(self):
        """all defined PIConGPU particle flags must be uniquely named"""
        species = self.species
        const1 = Charge()
        const1.charge_si = 17
        const2 = Charge()
        const2.charge_si = 18
        other_const = Constant()

        species.constants = [const1, const2, other_const]

        # duplicates -> throw (require violiating const to be mentioned)
        with self.assertRaisesRegex(ValueError, ".*charge.*"):
            species.check()

        for single_const in [const1, const2]:
            # no duplicates -> pass silently
            species.constants = [single_const, other_const]
            species.check()

    def test_check_constant_passthhru(self):
        """species check also calls constants check"""

        class ConstantFail(Constant):
            ERROR_STR: str = "IDSTRING_XKCD_927_BEST"

            def check(self):
                raise ValueError(self.ERROR_STR)

        # passes
        self.species.check()

        # add raising constant
        self.species.constants.append(ConstantFail())
        with self.assertRaisesRegex(ValueError, ConstantFail.ERROR_STR):
            self.species.check()

    def test_get_cxx_typename(self):
        """c++ typenames make sense"""
        # 1. is mandatory
        tmp = Species()
        tmp.constants = []
        tmp.attributes = []
        with self.assertRaises(Exception):
            tmp.check()
        with self.assertRaises(Exception):
            tmp.get_cxx_typename()

        # 2. is (somewhat) human-readable
        def get_typename(name):
            self.species.name = name
            return self.species.get_cxx_typename()

        for txt in ["H", "h", "electron", "e", "1", "lulal"]:
            self.assertTrue(txt in get_typename(txt))

        # 3. reject invalid strings (not alphanum)
        for invalid in ["", "\n", " ", "var\n", "abc sad", ".", "-"]:
            with self.assertRaises(ValueError):
                get_typename(invalid)

    def test_get_constant_by_type(self):
        """constant by type works as specifed"""
        species = self.species
        self.assertEqual([], species.constants)

        with self.assertRaises(typeguard.TypeCheckError):
            species.get_constant_by_type("string")
        with self.assertRaises(typeguard.TypeCheckError):
            # only children of Constant() are accepted
            species.get_constant_by_type(object)

        with self.assertRaises(RuntimeError):
            species.get_constant_by_type(type(self.const))

        species.constants = [self.const, self.const_mass, self.const_charge]
        # note: check for *identity* with is (instead of pure equality)
        self.assertTrue(self.const is species.get_constant_by_type(Constant))
        self.assertTrue(self.const_charge is species.get_constant_by_type(Charge))
        self.assertTrue(self.const_mass is species.get_constant_by_type(Mass))

    def test_has_constant_of_type(self):
        """check for constant existance is valid"""
        species = self.species
        self.assertEqual([], species.constants)

        with self.assertRaises(typeguard.TypeCheckError):
            species.has_constant_of_type("density")
        with self.assertRaises(typeguard.TypeCheckError):
            # only children of Constant() are accepted
            species.has_constant_of_type(object)

        self.assertTrue(not species.has_constant_of_type(type(self.const)))
        self.assertTrue(not species.has_constant_of_type(Mass))
        self.assertTrue(not species.has_constant_of_type(Charge))

        species.constants = [self.const, self.const_mass]

        self.assertTrue(species.has_constant_of_type(type(self.const)))
        self.assertTrue(species.has_constant_of_type(Mass))
        self.assertTrue(not species.has_constant_of_type(Charge))

    def test_rendering_simple(self):
        """passes information from rendering through"""
        species = Species()
        species.name = "myname"
        species.attributes = [Position(), Momentum(), Weighting()]
        # note: no charge
        species.constants = [
            self.const_mass,
            self.const_density_ratio,
        ]

        context = species.get_rendering_context()

        self.assertEqual("myname", context["name"])
        self.assertEqual(species.get_cxx_typename(), context["typename"])
        self.assertEqual(3, len(context["attributes"]))
        attribute_names = list(map(lambda attr_obj: attr_obj["picongpu_name"], context["attributes"]))
        self.assertEqual(
            [Position.PICONGPU_NAME, Momentum.PICONGPU_NAME, Weighting.PICONGPU_NAME],
            attribute_names,
        )

        self.assertEqual(self.const_mass.get_rendering_context(), context["constants"]["mass"])
        self.assertEqual(
            self.const_density_ratio.get_rendering_context(),
            context["constants"]["density_ratio"],
        )

        # no charge set -> key still exists
        self.assertEqual(None, context["constants"]["charge"])

    def test_rendering_constants(self):
        """constants are rendered as expected"""
        # constants are passed as a dictionary which *always* has *all* keys,
        # but those undefined are (explicitly) set to "null"
        # (rationale: prevent mistyping/false-positive "unkown var" warnings)
        #
        # This test ensures that for *all* permutations of keys being
        # defined/undefined the passthru is as expected
        # (note: might be overengineered)

        expected_const_by_name = {
            "density_ratio": self.const_density_ratio,
            "charge": self.const_charge,
            "mass": self.const_mass,
            "ionizers": self.const_ionizers,
            "element_properties": self.const_element_properties,
        }

        for enabled_vector in itertools.product((0, 1), repeat=len(expected_const_by_name)):
            species = Species()
            species.name = "myname"
            species.attributes = [Position(), Momentum()]
            name_enabled_pairs = zip(expected_const_by_name.keys(), enabled_vector)
            enabled_by_name = {}

            species.constants = []
            for const_name, enabled in name_enabled_pairs:
                enabled_by_name[const_name] = enabled
                if enabled:
                    species.constants.append(expected_const_by_name[const_name])

            context = species.get_rendering_context()
            self.assertEqual(set(expected_const_by_name.keys()), set(context["constants"].keys()))

            for const_name, enabled in enabled_by_name.items():
                self.assertTrue(const_name in context["constants"])
                if enabled:
                    self.assertEqual(
                        expected_const_by_name[const_name].get_rendering_context(),
                        context["constants"][const_name],
                    )
                else:
                    self.assertEqual(None, context["constants"][const_name])

    def test_rendering_checks(self):
        """retrieving rendering context enforces checks"""
        species = self.species
        species.name = ""

        with self.assertRaisesRegex(ValueError, ".*name.*"):
            species.check()

        with self.assertRaisesRegex(ValueError, ".*name.*"):
            species.get_rendering_context()
