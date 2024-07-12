"""
This file is part of PIConGPU.
Copyright 2021-2023 PIConGPU contributors
Authors: Hannes Troepgen, Brian Edward Marre
License: GPLv3+
"""

from picongpu.pypicongpu.species import InitManager

import unittest

from .attribute import DummyAttribute

from picongpu.pypicongpu import species
from picongpu.pypicongpu.species.attribute import Position, Momentum
from picongpu.pypicongpu.species.constant import Mass, Charge
from picongpu.pypicongpu.species.operation import (
    SimpleDensity,
    SimpleMomentum,
    NotPlaced,
    densityprofile,
)

import typing
import typeguard


class TestInitManager(unittest.TestCase):
    @typeguard.typechecked
    class ConstantWithDependencies(species.constant.Constant):
        def __init__(self, dependencies=[]):
            if type(dependencies) is not list:
                self.dependencies = [dependencies]
            else:
                self.dependencies = dependencies

            self.attribute_dependencies = []
            self.constant_dependencies = []
            self.constant_dependencies_called = 0
            self.attribute_dependencies_called = 0

        def check(self):
            pass

        def get_species_dependencies(self) -> typing.List[species.Species]:
            return self.dependencies

        def get_attribute_dependencies(self) -> typing.List[type]:
            self.attribute_dependencies_called += 1
            return self.attribute_dependencies

        def get_constant_dependencies(self) -> typing.List[type]:
            self.constant_dependencies_called += 1
            return self.constant_dependencies

    class OperationInvalidBehavior(species.operation.Operation):
        def __init__(self, species_list=[]):
            self.species_list = species_list
            self.check_adds_attribute = False
            self.prebook_adds_attribute = False

        def __species_add_attr(self):
            for spec in self.species_list:
                attr = DummyAttribute()
                # -> search for "IDSTRINGDZ" in error message
                attr.PICONGPU_NAME = "this_attr_is_added_too_early__IDSTRINGDZ"
                spec.attributes = [attr]

        def check_preconditions(self):
            if self.check_adds_attribute:
                self.__species_add_attr()

        def prebook_species_attributes(self):
            if self.prebook_adds_attribute:
                self.__species_add_attr()

    class OperationCallTracer(species.operation.Operation):
        def __init__(self, unique_id="", species_list=[]):
            # record order of function calls
            self.calls = ["init"]
            self.abort_check = False
            self.abort_prebook = False
            self.species_list = species_list
            self.unique_id = unique_id

        def get_attr_name(self):
            return "tracer_attr_" + self.unique_id

        def check_preconditions(self):
            self.calls.append("check")
            assert not self.abort_check, "IDSTRING_FROM_CHECK"

        def prebook_species_attributes(self):
            self.calls.append("prebook")
            assert not self.abort_prebook, "IDSTRING_FROM_PREBOOK"

            self.attributes_by_species = {}
            for spec in self.species_list:
                attr = DummyAttribute()
                attr.PICONGPU_NAME = self.get_attr_name()
                self.attributes_by_species[spec] = [attr]

    class OperationAddMandatoryAttributes(species.operation.Operation):
        def __init__(self, species_list=[]):
            self.species_list = species_list

        def check_preconditions(self):
            pass

        def prebook_species_attributes(self):
            self.attributes_by_species = {}
            for spec in self.species_list:
                self.attributes_by_species[spec] = [Position(), Momentum()]

    def __get_five_simple_species(self):
        """helper to build dependency graphs"""
        all_species = []

        for i in range(5):
            new_species = species.Species()
            new_species.name = "species{}".format(i)
            new_species.constants = []
            # note: attributes left intentionally undefined, as they would be
            # when generating from picongpu.pypicongpu

            all_species.append(new_species)

        return tuple(all_species)

    def setUp(self):
        self.species1 = species.Species()
        self.species1.name = "species1"
        self.species1.constants = []
        self.species1_copy = species.Species()
        self.species1_copy.name = "species1"
        self.species1_copy.constants = []
        self.species2 = species.Species()
        self.species2.name = "species2"
        self.species2.constants = []
        self.attribute1 = DummyAttribute()
        self.attribute1.PICONGPU_NAME = "attribute1"

        self.initmgr = InitManager()

    def test_setup(self):
        """setUp() provides a working initmanager"""
        # passes silently
        self.initmgr.bake()
        # implicitly calls checks
        self.initmgr.get_rendering_context()

    def test_nameconflicts(self):
        """species names must be unique"""
        initmgr = self.initmgr
        initmgr.all_species = [self.species1, self.species1_copy]
        initmgr.all_operations = [self.OperationAddMandatoryAttributes(initmgr.all_species)]
        self.assertTrue(initmgr.all_species[0] is not initmgr.all_species[1])
        self.assertEqual(initmgr.all_species[0].name, initmgr.all_species[1].name)

        with self.assertRaisesRegex(ValueError, ".*unique.*species1.*"):
            initmgr.bake()

    def test_check_passthru_species(self):
        """species.check() is called"""
        spec = self.species1
        # trigger check by setting name invalid
        spec.name = "-"
        with self.assertRaisesRegex(ValueError, ".*c[+][+].*"):
            # message must say sth like "c++ incompatible"
            spec.check()

        initmgr = self.initmgr
        initmgr.all_species = [spec]

        with self.assertRaisesRegex(ValueError, ".*c[+][+].*"):
            initmgr.bake()

    def test_shared_consts(self):
        """multiple species can share the same const object"""
        s1 = self.species1
        s2 = self.species2
        const = self.ConstantWithDependencies()
        const.PICONGPU_NAME = "a single const"

        s1.constants = [const]
        s2.constants = [const]

        # they use **the same** object
        self.assertTrue(s1.constants[0] is s2.constants[0])

        initmgr = self.initmgr
        initmgr.all_species = [s1, s2]
        initmgr.all_operations = [self.OperationAddMandatoryAttributes(initmgr.all_species)]

        # simply works
        initmgr.bake()

    def test_operation_lifecycle_normal(self):
        """operation methods are called in the right order"""
        initmgr = self.initmgr
        initmgr.all_species = [self.species1]

        # create multiple tracer operations
        tracer1 = self.OperationCallTracer("1", initmgr.all_species)
        tracer2 = self.OperationCallTracer("2", initmgr.all_species)
        tracer3 = self.OperationCallTracer("3", initmgr.all_species)

        initmgr.all_operations = [
            tracer1,
            tracer3,
            tracer2,
            self.OperationAddMandatoryAttributes(initmgr.all_species),
        ]

        initmgr.bake()

        expected_callchain = ["init", "check", "prebook"]
        self.assertEqual(expected_callchain, tracer1.calls)
        self.assertEqual(expected_callchain, tracer2.calls)
        self.assertEqual(expected_callchain, tracer3.calls)

        # actually added the attributes
        all_attr_names = list(map(lambda attr: attr.PICONGPU_NAME, self.species1.attributes))
        self.assertEqual(5, len(all_attr_names))
        self.assertTrue(tracer1.get_attr_name() in all_attr_names)
        self.assertTrue(tracer2.get_attr_name() in all_attr_names)
        self.assertTrue(tracer3.get_attr_name() in all_attr_names)

    def test_operation_lifecycle_aborted_check(self):
        """no prebook call to any operation made if one check fails"""
        initmgr = self.initmgr
        initmgr.all_species = [self.species1]

        tracer_passing = self.OperationCallTracer("1", initmgr.all_species)
        tracer_throwing = self.OperationCallTracer("2", initmgr.all_species)
        tracer_throwing.abort_check = True
        initmgr.all_operations = [tracer_passing, tracer_throwing]

        with self.assertRaisesRegex(AssertionError, "IDSTRING_FROM_CHECK"):
            initmgr.bake()

        # prebook not called even for passing
        self.assertTrue("prebook" not in tracer_passing.calls)
        self.assertTrue("prebook" not in tracer_throwing.calls)

    def test_operation_lifecycle_aborted_prebook(self):
        """no bake call to operation made if one prebook fails"""
        initmgr = self.initmgr
        initmgr.all_species = [self.species1]

        tracer_passing = self.OperationCallTracer("1", initmgr.all_species)
        tracer_throwing = self.OperationCallTracer("2", initmgr.all_species)
        tracer_throwing.abort_prebook = True
        initmgr.all_operations = [tracer_passing, tracer_throwing]

        with self.assertRaisesRegex(AssertionError, "IDSTRING_FROM_PREBOOK"):
            initmgr.bake()

        # check has been called...
        self.assertTrue("check" in tracer_passing.calls)
        self.assertTrue("check" in tracer_throwing.calls)
        # at least one prebook() has been called,
        # tracer_throwing failed
        # -> no bake() *at all* has been called
        # -> no species have been assigned *at all*
        #    (i.e. even if one prebook passed)
        self.assertEqual([], self.species1.attributes)

    def test_operation_invalid_behavior_str(self):
        """check string representation of OperationInvalidBehavior"""
        # rationale: sometimes the operation name must be in an error message
        # -> ensure regex-able string representation of offending operation
        with self.assertRaisesRegex(ValueError, ".*OperationInvalidBehavior.*"):
            raise ValueError(str(self.OperationInvalidBehavior([])))

    def test_operation_invalid_behavior_check(self):
        """if a operation check adds attributes it is reported"""
        initmgr = self.initmgr
        initmgr.all_species = [self.species1]
        invalid = self.OperationInvalidBehavior(initmgr.all_species)
        invalid.check_adds_attribute = True
        initmgr.all_operations = [invalid]

        with self.assertRaisesRegex(AssertionError, ".*check.*OperationInvalidBehavior.*IDSTRINGDZ.*"):
            initmgr.bake()

    def test_operation_invalid_behavior_prebook(self):
        """if a operation prebook adds attributes it is reported"""
        initmgr = self.initmgr
        initmgr.all_species = [self.species1]
        invalid = self.OperationInvalidBehavior(initmgr.all_species)
        invalid.prebook_adds_attribute = True
        initmgr.all_operations = [invalid]

        with self.assertRaisesRegex(AssertionError, ".*prebook.*OperationInvalidBehavior.*IDSTRINGDZ.*"):
            initmgr.bake()

    def test_multiple_assigned_species(self):
        """each species object may only be added once"""
        initmgr = InitManager()
        initmgr.all_species = [self.species1, self.species1]
        initmgr.all_operations = [self.OperationAddMandatoryAttributes([self.species1])]

        with self.assertRaisesRegex(ValueError, ".*[Ss]pecies.*once.*species1.*"):
            # duplicate species
            initmgr.bake()

        # but works if deduplicated
        initmgr.all_species = list(set(initmgr.all_species))
        initmgr.bake()

    def test_multiple_assigned_operation(self):
        """each operation object may only be added once"""
        initmgr = InitManager()
        initmgr.all_species = [self.species1]
        tracer = self.OperationCallTracer("", initmgr.all_species)
        initmgr.all_operations = [
            self.OperationAddMandatoryAttributes([self.species1]),
            tracer,
            tracer,
        ]

        with self.assertRaisesRegex(ValueError, ".*[Oo]peration.*once.*"):
            # duplicate operation
            initmgr.bake()

        # but works with deduplicated operations
        initmgr.all_operations = list(set(initmgr.all_operations))
        self.assertEqual(2, len(initmgr.all_operations))
        initmgr.bake()

    def test_exclusiveness_checked(self):
        """attributes must be exclusively owned by species"""

        class NonExclusiveOp(species.operation.Operation):
            attr_for_all = DummyAttribute()

            def __init__(self, species_list=[]):
                # record order of function calls
                self.species_list = species_list
                NonExclusiveOp.attr_for_all.PICONGPU_NAME = "exists_only_once"

            def check_preconditions(self):
                pass

            def prebook_species_attributes(self):
                self.attributes_by_species = {}
                for spec in self.species_list:
                    # note: uses **global** (static) attribute object
                    self.attributes_by_species[spec] = [NonExclusiveOp.attr_for_all]

        initmgr = self.initmgr
        initmgr.all_species = [self.species1, self.species2]
        # idea: assign *the same* attribute object using *two different*
        # operation objects to *two different* species
        # -> operation-local checks (inside Operation.bake()) will not catch
        #    this
        op1 = NonExclusiveOp([self.species1])
        op2 = NonExclusiveOp([self.species2])
        initmgr.all_operations = [op1, op2]

        with self.assertRaisesRegex(ValueError, ".*exclusive.*"):
            initmgr.bake()

    def test_types(self):
        """lists have typechecks"""
        initmgr = self.initmgr

        invalid_specieslists = []
        for invalid_specieslist in invalid_specieslists:
            with self.assertRaises(TypeError):
                initmgr.all_species = invalid_specieslist

        invalid_oplists = []
        for invalid_oplist in invalid_oplists:
            with self.assertRaises(TypeError):
                initmgr.all_operations = invalid_oplist

    def test_empty(self):
        """works by default"""
        initmgr = InitManager()
        # just works:
        initmgr.bake()

        # produces empty lists for rendering
        context = initmgr.get_rendering_context()
        self.assertEqual([], context["species"])

        # types of operations are listed, but the lists are empty
        self.assertNotEqual(0, len(context["operations"]))
        self.assertEqual([], context["operations"]["simple_density"])

    def test_basic(self):
        """valid scenario"""
        initmgr = self.initmgr
        initmgr.all_species = [self.species1, self.species2]

        op1 = self.OperationCallTracer("a", [self.species1])
        op2 = self.OperationCallTracer("xkcd927", [self.species2])
        op3 = self.OperationCallTracer("1337", [self.species1, self.species2])
        initmgr.all_operations = [
            op1,
            op2,
            op3,
            self.OperationAddMandatoryAttributes([self.species1, self.species2]),
        ]

        initmgr.bake()

        species1_attr_names = list(map(lambda attr: attr.PICONGPU_NAME, self.species1.attributes))
        species2_attr_names = list(map(lambda attr: attr.PICONGPU_NAME, self.species2.attributes))

        self.assertEqual(4, len(species1_attr_names))
        self.assertTrue(op1.get_attr_name(), species1_attr_names)
        self.assertTrue(op3.get_attr_name(), species1_attr_names)

        self.assertEqual(4, len(species2_attr_names))
        self.assertTrue(op2.get_attr_name(), species2_attr_names)
        self.assertTrue(op3.get_attr_name(), species2_attr_names)

    def test_species_check(self):
        """species incorrectly created -> species check raises"""
        # mandatory "position" attribute missing
        initmgr = self.initmgr
        initmgr.all_species = [self.species1]
        initmgr.all_operations = []

        with self.assertRaisesRegex(ValueError, ".*[Pp]osition.*"):
            initmgr.bake()

    def test_operation_conflict(self):
        """conflicting operations reject"""
        initmgr = self.initmgr
        initmgr.all_species = [self.species1]

        # will create identical attributes
        tracer1 = self.OperationCallTracer("1", initmgr.all_species)
        tracer1_copy = self.OperationCallTracer("1", initmgr.all_species)

        initmgr.all_operations = [tracer1, tracer1_copy]

        attr_name = tracer1.get_attr_name()

        with self.assertRaisesRegex(ValueError, ".*conflict.*{}.*".format(attr_name)):
            initmgr.bake()

    def test_unregistered_species(self):
        """all species used by any operator must be known to the InitManager"""
        initmgr = self.initmgr

        # species2 is not known to initmgr:
        # attr list will not be initialized to [],
        # might provoke "unset attribute" error
        # -> circumvent this "unset attribute" error by manual initialization
        #    (DO NOT DO THIS AT HOME)
        self.species2.attributes = []

        initmgr.all_species = [self.species1]

        # assigns more species than known to the init manager
        op = self.OperationAddMandatoryAttributes([self.species1, self.species2])
        initmgr.all_operations = [op]

        with self.assertRaisesRegex(ValueError, ".*register.*species2.*"):
            initmgr.bake()

    def test_bake_twice(self):
        # can only bake once
        self.initmgr.bake()

        with self.assertRaisesRegex(AssertionError, ".*once.*"):
            self.initmgr.bake()

    def test_rendering_implictly_bakes(self):
        """rendering calls bake(), unless already called"""
        initmgr = InitManager()
        # directly works:
        res_no_bake = initmgr.get_rendering_context()

        # also works if bake has been called previously
        initmgr = InitManager()
        initmgr.bake()
        res_with_bake = initmgr.get_rendering_context()

        # must be **equivalent**
        self.assertEqual(res_no_bake, res_with_bake)

    def test_rendering_context_passthru(self):
        """renderer passes information through"""
        initmgr = self.initmgr
        initmgr.all_species = [self.species1, self.species2]

        simple_density = SimpleDensity()
        simple_density.ppc = 927
        simple_density.profile = densityprofile.Uniform()
        simple_density.profile.density_si = 1337
        simple_density.species = {
            self.species1,
            self.species2,
        }
        momentum_ops = []
        for single_species in initmgr.all_species:
            simple_momentum = SimpleMomentum()
            simple_momentum.species = single_species
            simple_momentum.drift = None
            simple_momentum.temperature = None
            momentum_ops.append(simple_momentum)

        initmgr.all_operations = [simple_density] + momentum_ops

        # (implicitly bakes)
        context = initmgr.get_rendering_context()

        self.assertEqual(2, len(context["species"]))
        self.assertEqual(context["species"][0], self.species1.get_rendering_context())
        self.assertEqual(context["species"][1], self.species2.get_rendering_context())

        self.assertEqual(1, len(context["operations"]["simple_density"]))
        self.assertEqual(
            context["operations"]["simple_density"][0],
            simple_density.get_rendering_context(),
        )

    def test_rendering_context_passthru_ops(self):
        """operations are passed through into their respective locations"""
        initmgr = self.initmgr
        initmgr.all_species = [self.species1, self.species2]
        initmgr.all_operations = []

        simple_density = SimpleDensity()
        simple_density.ppc = 927
        simple_density.profile = densityprofile.Uniform()
        simple_density.profile.density_si = 1337
        simple_density.species = {
            self.species1,
        }
        initmgr.all_operations.append(simple_density)

        # store momentum ops separately for assertion later
        momentum_ops = []
        for single_species in initmgr.all_species:
            simple_momentum = SimpleMomentum()
            simple_momentum.species = single_species
            simple_momentum.drift = None
            simple_momentum.temperature = None
            momentum_ops.append(simple_momentum)
        initmgr.all_operations += momentum_ops

        not_placed = NotPlaced()
        not_placed.species = self.species2
        initmgr.all_operations.append(not_placed)

        # note: further operation passthrough tests in other methods

        # implicitly bakes
        context = initmgr.get_rendering_context()

        # note: NotPlaced only adds attr and no data, hence is not in context
        self.assertEqual(
            simple_density.get_rendering_context(),
            context["operations"]["simple_density"][0],
        )

        for momentum_op in momentum_ops:
            self.assertTrue(momentum_op.get_rendering_context() in context["operations"]["simple_momentum"])

    def test_constants_dependencies_outside(self):
        """species dependencies outside of initmanager are detected"""
        a, b, c, d, e = self.__get_five_simple_species()

        # a -> b -> c, but c is not registered with initmanager
        a.constants = [self.ConstantWithDependencies(b)]
        b.constants = [self.ConstantWithDependencies(c)]

        initmgr = InitManager()
        initmgr.all_species = [a, b, d, e]

        with self.assertRaisesRegex(ReferenceError, ".*unkown.*"):
            initmgr.bake()

    def test_constant_species_dependencies_circle_detection_complicated(self):
        """circular dependencies are detected and caught"""
        a, b, c, d, e = self.__get_five_simple_species()

        # circle a -> b -> c -> d -> a
        # additionally: e -> a, e -> c
        a.constants = [self.ConstantWithDependencies(b)]
        b.constants = [self.ConstantWithDependencies(c)]
        c.constants = [self.ConstantWithDependencies(d)]
        d.constants = [self.ConstantWithDependencies(a)]

        e.constants = [self.ConstantWithDependencies([a, c])]

        initmgr = InitManager()
        initmgr.all_species = [a, b, c, d, e]

        with self.assertRaisesRegex(RecursionError, ".*circular.*"):
            initmgr.bake()

    def test_constant_species_dependencies_circle_detection_simple(self):
        """simple circular dependencies are also caugth"""
        a, b, c, d, e = self.__get_five_simple_species()
        a.constants = [self.ConstantWithDependencies([a, b])]
        initmgr = InitManager()
        initmgr.all_species = [a, b]

        with self.assertRaisesRegex(RecursionError, ".*circular.*"):
            initmgr.bake()

    def test_constant_species_dependencies_order(self):
        """order between dependencies is created"""
        a, b, c, d, e = self.__get_five_simple_species()

        a.constants = [self.ConstantWithDependencies(d)]
        b.constants = [self.ConstantWithDependencies([e, c])]
        c.constants = [self.ConstantWithDependencies(a)]
        d.constants = []
        e.constants = [self.ConstantWithDependencies(d)]

        initmgr = InitManager()
        initmgr.all_species = [a, b, c, d, e]
        # associate ops for required attrs (to make checks pass)
        initmgr.all_operations = [self.OperationAddMandatoryAttributes(initmgr.all_species)]

        # bake reorders dependencies
        initmgr.bake()
        baked_names_in_order = list(map(lambda species: species.name, initmgr.all_species))

        # must be equal to context order
        context = initmgr.get_rendering_context()
        context_names_in_order = list(map(lambda species: species["name"], context["species"]))
        self.assertEqual(baked_names_in_order, context_names_in_order)

        index_by_name = dict(
            map(
                lambda species_name: (
                    species_name,
                    context_names_in_order.index(species_name),
                ),
                context_names_in_order,
            )
        )

        # a->0, b->1, c->2, d->3, e->4
        # expected order: d < e = a < c < b
        #                 3 < 4 = 0 < 2 < 1
        self.assertEqual(0, index_by_name["species3"])
        self.assertEqual({1, 2}, {index_by_name["species0"], index_by_name["species4"]})
        self.assertEqual(3, index_by_name["species2"])
        self.assertEqual(4, index_by_name["species1"])

    def test_constant_attribute_dependencies_ok(self):
        """a constant may require an attribute to be present"""

        class DummyOperation(species.operation.Operation):
            def __init__(self):
                pass

            def check_preconditions(self):
                pass

            def prebook_species_attributes(self):
                self.attributes_by_species = {self.species: [self.attr]}

        attr = DummyAttribute()

        op = DummyOperation()
        op.species = self.species1
        op.attr = attr

        const = self.ConstantWithDependencies()
        const.attribute_dependencies = [type(attr)]
        self.species1.constants.append(const)

        initmgr = InitManager()
        initmgr.all_species = [self.species1, self.species2]
        initmgr.all_operations = [
            self.OperationAddMandatoryAttributes(initmgr.all_species),
            op,
        ]

        # species1 required "attr" to be present after generation,
        # which is provided by op

        # works
        initmgr.bake()
        self.assertNotEqual(0, const.attribute_dependencies_called)

    def test_constant_attribute_dependencies_missing(self):
        """constant requires an attribute, but it is not present"""
        const = self.ConstantWithDependencies()
        const.attribute_dependencies = [DummyAttribute]
        self.species1.constants.append(const)

        initmgr = InitManager()
        initmgr.all_species = [self.species1, self.species2]
        initmgr.all_operations = [self.OperationAddMandatoryAttributes(initmgr.all_species)]

        # DummyAttribute is required for species1, but not assigned
        with self.assertRaisesRegex(AssertionError, ".*species1.*"):
            initmgr.bake()

        # ...but works without dependency
        const.attribute_dependencies = []

        initmgr = InitManager()
        initmgr.all_species = [self.species1, self.species2]
        initmgr.all_operations = [self.OperationAddMandatoryAttributes(initmgr.all_species)]
        initmgr.bake()

    def test_constant_attribute_dependencies_typechecked(self):
        """constant must return correct dependency list"""
        invalid_attr_dependency_lists = [
            set(),
            None,
            [None],
            ["DUMMYATTR"],
            "dummyattr",
            [int, str],
            [Mass],
        ]
        for invalid_list in invalid_attr_dependency_lists:
            const = self.ConstantWithDependencies()
            const.attribute_dependencies = invalid_list
            self.species1.constants = [const]

            initmgr = InitManager()
            initmgr.all_species = [self.species1, self.species2]
            initmgr.all_operations = [self.OperationAddMandatoryAttributes(initmgr.all_species)]

            with self.assertRaises(typeguard.TypeCheckError):
                initmgr.bake()

    def test_constant_constant_dependencies_ok(self):
        """constants requires other constant and it is present"""
        mass = Mass()
        mass.mass_si = 1
        charge = Charge()
        charge.charge_si = 1

        const_dep = self.ConstantWithDependencies()
        const_dep.constant_dependencies = [Mass, Charge]

        self.species1.constants = [
            mass,
            charge,
            const_dep,
        ]

        # has no constants (serves as distraction)
        self.assertEqual([], self.species2.constants)

        initmgr = InitManager()
        initmgr.all_species = [self.species1, self.species2]
        initmgr.all_operations = [self.OperationAddMandatoryAttributes(initmgr.all_species)]

        # passes silently, all checks ok
        initmgr.bake()
        self.assertNotEqual(0, const_dep.constant_dependencies_called)

    def test_constant_constant_dependencies_missing(self):
        """constants requires other constant and it is missing"""
        charge = Charge()
        charge.charge_si = 1

        const_dep = self.ConstantWithDependencies()
        const_dep.constant_dependencies = [Mass, Charge]

        self.species1.constants = [
            const_dep,
            charge,
        ]

        initmgr = InitManager()
        initmgr.all_species = [self.species1, self.species2]
        initmgr.all_operations = [self.OperationAddMandatoryAttributes(initmgr.all_species)]

        with self.assertRaisesRegex(AssertionError, ".*species1.*Mass.*"):
            initmgr.bake()

    def test_constant_constant_dependencies_circular(self):
        """circular dependencies are allowed, (self references not)"""

        class OtherConstWithDeps(species.constant.Constant):
            def __init__(self):
                pass

            def check(self):
                pass

            def get_species_dependencies(self):
                return []

            def get_attribute_dependencies(self):
                return []

            def get_constant_dependencies(self):
                return self.constant_dependencies

        const1_dep = self.ConstantWithDependencies()
        const1_dep.constant_dependencies = [OtherConstWithDeps]

        const2_dep = OtherConstWithDeps()
        const2_dep.constant_dependencies = [self.ConstantWithDependencies]

        self.species1.constants = [
            const1_dep,
            const2_dep,
        ]

        initmgr = InitManager()
        initmgr.all_species = [self.species1, self.species2]
        initmgr.all_operations = [self.OperationAddMandatoryAttributes(initmgr.all_species)]

        # works
        initmgr.bake()
        self.assertNotEqual(0, const1_dep.constant_dependencies_called)

    def test_constant_constant_dependencies_self(self):
        """self-references in dependencies are not allowed (circular are)"""
        const_dep = self.ConstantWithDependencies()
        const_dep.constant_dependencies = [self.ConstantWithDependencies]

        self.species1.constants = [
            const_dep,
        ]

        initmgr = InitManager()
        initmgr.all_species = [self.species1]
        initmgr.all_operations = [self.OperationAddMandatoryAttributes(initmgr.all_species)]

        # self-reference is detected and caught
        with self.assertRaisesRegex(ReferenceError, ".*(self|selve).*"):
            initmgr.bake()

    def test_constant_constant_dependencies_typechecked(self):
        """the constant-constant dependency interface is typchecked"""
        invalid_constant_dependency_lists = [
            set(),
            None,
            [None],
            ["DUMMYCONST"],
            "dummyconst",
            [int, str],
            [Position],
        ]
        for invalid_list in invalid_constant_dependency_lists:
            const = self.ConstantWithDependencies()
            const.constant_dependencies = invalid_list
            self.species1.constants = [const]

            initmgr = InitManager()
            initmgr.all_species = [self.species1, self.species2]
            initmgr.all_operations = [self.OperationAddMandatoryAttributes(initmgr.all_species)]

            with self.assertRaises(typeguard.TypeCheckError):
                initmgr.bake()

    def test_set_bound_electrons_passthrough(self):
        """bound electrons operation is included in rendering context"""
        # create full electron species
        electron = species.Species()
        electron.name = "e"
        electron.constants = []

        ion = species.Species()
        ion.name = "ion"
        ionizers_const = species.constant.Ionizers()
        ionizers_const.electron_species = electron
        element_const = species.constant.ElementProperties()
        element_const.element = species.util.Element.N
        ion.constants = [ionizers_const, element_const]

        ion_op = species.operation.SetBoundElectrons()
        ion_op.species = ion
        ion_op.bound_electrons = 2

        initmgr = InitManager()
        initmgr.all_species = [electron, ion]
        initmgr.all_operations = [
            self.OperationAddMandatoryAttributes(initmgr.all_species),
            ion_op,
        ]

        context = initmgr.get_rendering_context()

        self.assertEqual(
            [ion_op.get_rendering_context()],
            context["operations"]["set_bound_electrons"],
        )
