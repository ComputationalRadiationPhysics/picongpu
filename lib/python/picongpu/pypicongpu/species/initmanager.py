"""
This file is part of PIConGPU.
Copyright 2021-2023 PIConGPU contributors
Authors: Hannes Troepgen, Brian Edward Marre
License: GPLv3+
"""

import typeguard
import typing
from .. import util
from .species import Species
from .operation import (
    Operation,
    DensityOperation,
    SimpleDensity,
    SimpleMomentum,
    SetBoundElectrons,
)
from .attribute import Attribute
from .constant import Constant
from functools import reduce
from ..rendering import RenderedObject


@typeguard.typechecked
class InitManager(RenderedObject):
    """
    Helper to manage species translation to PIConGPU

    Collects all species to be initialized and the operations initializing
    them.

    Invokes the methods of the Operation lifecycle (check, prebook, bake).

    Workflow:

    1. Fill InitManager (from outside) with

       - Species and their constants (no attributes!)
       - Operations, fully initialized (all params set)

    2. invoke InitManager.bake(), which

       - checks the Species for conflicts (name...)
       - performs dependency checks, possibly reorders species
       - invokes the Operation lifecycle (check, prebook, bake)
       - sanity-checks the results

    3. retrieve rendering context

       - organizes operations into lists

    Note: The InitManager manages a lifecycle, it does not perform deep checks
          -> most errors have to be caught by delegated checks.
    """

    all_species = util.build_typesafe_property(typing.List[Species])
    """all species to be initialized"""

    all_operations = util.build_typesafe_property(typing.List[Operation])
    """all species to be initialized"""

    __baked = util.build_typesafe_property(bool)
    """if bake() has already been called"""

    def __init__(self) -> None:
        self.all_species = []
        self.all_operations = []
        self.__baked = False

    def __get_all_attributes(self):
        """
        accumulate *all* attributes currently assigned to any species

        Note: does not filter duplicates

        :return: list of all attributes
        """
        return list(
            reduce(
                lambda list_a, list_b: list_a + list_b,
                map(lambda species: species.attributes, self.all_species),
                [],
            )
        )

    def __precheck_species_conflicts(self) -> None:
        """
        checks for conflicts between species, if ok passes silently

        intended to verify input (i.e. before any operation is performed)

        conflict types:

        - same object twice in self.all_species
        - name not unique in self.all_species
        """
        # (1) check object uniqueness
        duplicate_species = set([species.name for species in self.all_species if self.all_species.count(species) > 1])
        if 0 != len(duplicate_species):
            raise ValueError(
                "every species object may only be added once, offending: {}".format(", ".join(duplicate_species))
            )

        # (2) check name conflicts
        species_names = [species.name for species in self.all_species]
        duplicate_names = set([name for name in species_names if species_names.count(name) > 1])
        if 0 != len(duplicate_names):
            raise ValueError("species names must be unique, offending: {}".format(", ".join(duplicate_names)))

    def __precheck_operation_conflicts(self) -> None:
        """
        checks for conflicts between operations, ik ok passes silently

        intended to verify input (i.e. before any operation is performed)

        conflict types:

        - same object twice in self.all_operations
        """
        duplicate_operations = set(
            [operation for operation in self.all_operations if self.all_operations.count(operation) > 1]
        )
        if 0 != len(duplicate_operations):
            raise ValueError(
                "every operation object may only be added once, offending: {}".format(
                    ", ".join(map(str, duplicate_operations))
                )
            )

    def __check_operation_phase_left_attributes_untouched(self, phase_name: str, operation: Operation) -> None:
        """
        ensures that no attributes have been added to any species

        if ok passes silently

        intended to be run after an operation phase

        parameters are used for error message generation
        :param phase_name: name of operation phase for error (prebook/check)
        :param operation: operation that is being checked
        """
        # use assertion instead of ValueError()
        # rationale: assertions check self (be unfriendly),
        #            ValueError()s user input (be more friendly)
        assert 0 == len(self.__get_all_attributes()), "phase {} of operation {} added attributes: {}".format(
            phase_name,
            str(operation),
            ", ".join(map(lambda attr: attr.PICONGPU_NAME, self.__get_all_attributes())),
        )

    def __check_operation_prebook_only_known_species(self, operation: Operation) -> None:
        """
        ensure that only registered species are prebooked

        passes silently if ok

        :param operation: checked operation
        """
        # ensure only registered species are added
        prebooked_species = set(operation.attributes_by_species.keys())
        unknown_species = prebooked_species - set(self.all_species)
        if 0 != len(unknown_species):
            unknown_species_names = list(map(lambda species: species.name, unknown_species))
            raise ValueError(
                "operation {} initialized species, but they are not "
                "registered in InitManager.all_species: {}".format(str(operation), ", ".join(unknown_species_names))
            )

    def __check_attributes_species_exclusive(self) -> None:
        """
        ensure attributes are exclusively owned by exactly one species

        if ok passes silently
        """
        all_attributes = list(
            reduce(
                lambda list_a, list_b: list_a + list_b,
                map(lambda species: species.attributes, self.all_species),
                [],
            )
        )
        if len(all_attributes) != len(set(all_attributes)):
            raise ValueError("attributes must be exclusively owned by exactly one species")

    def __check_species_dependencies_registered(self) -> None:
        """
        check that all dependencies of species are also in self.all_species

        passes silently if ok, else raises

        Note: only parses constants (operations must NOT have inter-species
        dependencies)
        """
        for species in self.all_species:
            for constant in species.constants:
                for dependency in constant.get_species_dependencies():
                    if dependency not in self.all_species:
                        raise ReferenceError(
                            "species {} is dependency (is required by) {}, but unkown to the init manager".format(
                                dependency.name, species.name
                            )
                        )

    def __check_species_dependencies_circular(self) -> None:
        """
        ensure that there are no circular dependencies

        passes silently if ok, else raises

        assumes that all dependencies are inside of self.all_species;
        see __check_species_dependencies_registered()

        Note: only parses constants
        """
        # approach:
        # 1) build "closure" (all dependencies including transitive
        #    dependencies)
        # 2) check if self is in closure

        for species in self.all_species:
            dependency_closure = set()

            # initialize closure with immediate dependencies
            for constant in species.constants:
                dependency_closure = dependency_closure.union(constant.get_species_dependencies())

            # compute transitive dependencies
            is_closure_final = False
            while not is_closure_final:
                closure_size_before = len(dependency_closure)
                for dependency_species in dependency_closure:
                    for constant in dependency_species.constants:
                        dependency_closure = dependency_closure.union(constant.get_species_dependencies())
                closure_size_after = len(dependency_closure)
                is_closure_final = closure_size_after == closure_size_before

            print(dependency_closure)

            # check: self in dependency closure?
            if species in dependency_closure:
                raise RecursionError(
                    "species {} is in circular dependency, all dependencies are: {}".format(
                        species.name,
                        ", ".join(map(lambda species: species.name, dependency_closure)),
                    )
                )

    def __reorder_species_dependencies(self) -> None:
        """
        reorder self.all_species to respect dependencies of constants

        Constants may depend on other species to be present, i.e. ionizers may
        require an electron species.

        This method reorders self.all_species to ensure that the dependency is
        defined first, and the dependent species follows later.
        Transitive dependencies are supported.

        performs additional checks:

        1. all dependencies registered with init manager
        2. no circular dependencies
        """
        # check that all dependencies are registered with initmanager
        self.__check_species_dependencies_registered()

        # check for circular dependencies
        self.__check_species_dependencies_circular()

        # compute correct inclusion order
        # approach:
        # 1. assign each species a number
        # 2. iterate over species, increasing their
        #    number to 1 + maximum of their dependencies
        # 3. order by this number
        #    (note: the same ordering number is allowed multiple times)

        # initialize each species with its current index
        # -> if possible, the order will be preserved
        ordernumber_by_species = dict(
            map(
                lambda species: (species, self.all_species.index(species)),
                self.all_species,
            )
        )
        assert 0 == len(ordernumber_by_species) or 0 <= min(ordernumber_by_species.values())

        is_ordering_final = False
        while not is_ordering_final:
            # stop working, unless a change is introduced
            is_ordering_final = True

            for species in ordernumber_by_species:
                # accumulate max order number of dependencies
                dependencies_max_ordernumber = -1
                for constant in species.constants:
                    for dependency in constant.get_species_dependencies():
                        dependencies_max_ordernumber = max(
                            dependencies_max_ordernumber,
                            ordernumber_by_species[dependency],
                        )

                # ensure self comes *after* all dependencies
                self_ordernumber = ordernumber_by_species[species]
                if dependencies_max_ordernumber >= self_ordernumber:
                    is_ordering_final = False
                    ordernumber_by_species[species] = 1 + dependencies_max_ordernumber

        # actually reorder species
        self.all_species = sorted(self.all_species, key=lambda species: ordernumber_by_species[species])

    def __check_constant_attribute_dependencies(self) -> None:
        """
        ensure that attributes required by constants are present

        Intended to be run after operations sucessfully executed.

        Silently passes if okay, raises on error.

        Constants may require certain attributes to be present
        (e.g. a pusher requires a momentum).
        After all attributes have been assigned in a final state, these
        dependencies can be checked with this method.
        """
        for species in self.all_species:
            species_attr_names = set(map(lambda attr: attr.PICONGPU_NAME, species.attributes))

            for constant in species.constants:
                required_attrs = constant.get_attribute_dependencies()

                # perform more rigerous typechecks than typeguard
                typeguard.check_type(required_attrs, list[type])
                for required_attr in required_attrs:
                    if not issubclass(required_attr, Attribute):
                        raise typeguard.TypeCheckError(
                            "required attribute must be attribute type, got: {}".format(required_attr)
                        )

                    # actual check:
                    assert (
                        required_attr.PICONGPU_NAME in species_attr_names
                    ), "constant {} of species {} requires attribute {} to be present, but it is not".format(
                        constant, species.name, required_attr
                    )

    def __check_constant_constant_dependencies(self):
        """
        ensure that constants required by other constants are present

        Passes silently if okay, raises on error.

        Constants may require the existance of other constants (e.g. Ionizers
        may require Atomic Numbers).
        Notably the value of these constants can't be checked, only that a
        constant of the given type is present.

        A constant may **NOT** depend on itself, circular dependencies are
        allowed though.
        """
        for species in self.all_species:
            for constant in species.constants:
                required_constants = constant.get_constant_dependencies()
                typeguard.check_type(required_constants, list[type])

                for required_constant in required_constants:
                    if not issubclass(required_constant, Constant):
                        raise typeguard.TypeCheckError(
                            "required constants must be of Constant type, got: {}".format(required_constant)
                        )

                    # self-references are not allowed
                    if type(constant) is required_constant:
                        raise ReferenceError("constants may not depend on themselves")

                    # check if constant exists
                    assert species.has_constant_of_type(
                        required_constant
                    ), "species {}: required constant {} not found, (required by constant {})".format(
                        species.name, required_constant, constant
                    )

    def bake(self) -> None:
        """
        apply operations to species
        """
        assert not self.__baked, "can only bake once"

        # check that constants required by other constants are present
        self.__check_constant_constant_dependencies()

        # check & resolve dependency of constants on other species
        # Note: "resolve" here means to reorder
        self.__reorder_species_dependencies()

        # check species & operation conflicts
        self.__precheck_species_conflicts()
        self.__precheck_operation_conflicts()

        # prepare species: set attrs to empty list
        for species in self.all_species:
            species.attributes = []

        # apply operation: check
        for operation in self.all_operations:
            operation.check_preconditions()
            self.__check_operation_phase_left_attributes_untouched("check", operation)

        # sync across operations:
        # all checks must pass before the first prebook is called

        # apply operation: prebook
        for operation in self.all_operations:
            operation.prebook_species_attributes()
            self.__check_operation_phase_left_attributes_untouched("prebook", operation)
            self.__check_operation_prebook_only_known_species(operation)

        # sync across operations:
        # We now enter the "species writing" phase (bake() modifies species,
        # check & prebook do not write to species).
        # This ensures that there are no read-write conflicts between adding
        # attributes and running checks/prebooking.
        # (this is essentially the naive approach to transaction handling)

        # -> this is non-reversable, set baking flag
        self.__baked = True

        # apply operation: bake
        for operation in self.all_operations:
            # implicitly checks attribute type is unique per species
            operation.bake_species_attributes()

        # check attribute objects exclusive to species
        self.__check_attributes_species_exclusive()

        # check that attributes required by constants are present
        self.__check_constant_attribute_dependencies()

        # check species themselves
        for species in self.all_species:
            species.check()

    def get_typical_particle_per_cell(self) -> int:
        """
        get typical number of macro particles per cell(ppc) of simulation

        @returns middle value of ppc-range of all operations, minimum 1
        """
        ppcs = []

        for operation in self.all_operations:
            if isinstance(operation, DensityOperation):
                ppcs.append(operation.ppc)

        if len(ppcs) == 0:
            return 1

        max_ppc = max(ppcs)
        min_ppc = min(ppcs)

        if max_ppc < 1:
            max_ppc = 1
        if min_ppc < 1:
            min_ppc = 1

        return (max_ppc - min_ppc) // 2 + min_ppc

    def _get_serialized(self) -> dict:
        """
        retrieve representation for rendering

        The initmanager pass mainly *a set of lists* to the templating engine.
        This set of lists is well defined an *always the same*, while only
        content of the lists varies.

        To enable direct access to specific types of operations all operations
        are split into separate lists containing only operations of this type,
        e.g. sth along the lines of:

        .. code::

            {
                species: [species1, species2, ...],
                operations: {
                    simple_density: [density1, density2],
                    momentum: [],
                    preionization: [ionization1],
                }
            }

        (Note: This also make schema description much simpler, as there is no
        need for a generic "any operation" schema.)
        """
        # note: implicitly runs checks
        if not self.__baked:
            self.bake()

        operation_types_by_name = {
            "simple_density": SimpleDensity,
            "simple_momentum": SimpleMomentum,
            "set_bound_electrons": SetBoundElectrons,
            # note: NotPlaced is not rendered (as it provides no data & does
            # nothing anyways) -> it is not in this list
            # same as NoBoundElectrons
        }

        # note: this will create lists for every name (which is intented), they
        # might be empty
        operations_context = {}
        for op_name, op_type in operation_types_by_name.items():
            operations_context[op_name] = list(
                map(
                    lambda op: op.get_rendering_context(),
                    filter(lambda op: type(op) is op_type, self.all_operations),
                )
            )

        return {
            "species": list(map(lambda species: species.get_rendering_context(), self.all_species)),
            "operations": operations_context,
        }
