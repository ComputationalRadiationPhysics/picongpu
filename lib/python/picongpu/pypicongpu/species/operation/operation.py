"""
This file is part of the PIConGPU.
Copyright 2021-2023 PIConGPU contributors
Authors: Hannes Troepgen, Brian Edward Marre
License: GPLv3+
"""

from ...rendering import RenderedObject
from ... import util
from ..attribute import Attribute
from ..species import Species

import typeguard
import typing
from functools import reduce


@typeguard.typechecked
class Operation(RenderedObject):
    """
    Defines the initialization of a set of attributes across multiple species

    One attribute of one species is initialized by exactly one Operation.

    May require the species to have certain constants.

    Operations are treated strictly independently from one another.
    Dependencies are thus handled inside of a single Operation.
    (Note: The code generation may impose a custom order between the types of
    operations.)

    All Operations combined generate the PIConGPU init pipeline.

    This leads to some notable cases:

    - Species that are to be placed together (ion+electrons) must be
      initialized by the **same** Operation.
    - Dependent attributes must be initialized by the **same** Operation.

    The typical lifecycle of an Operation object is as follows:

    1. Get created (i.e. __init__() is called)

       - performed from outside (PICMI interface)
       - affected species & parameters passed from outside (store as class
         attribute)

    2. check preconditions for passed species

       - params set correctly
       - required constants present in species
       - At the point of checking the given species object are not fully
         defined yet:
         **DO NOT CHECK FOR OTHER ATTRIBUTES, THEY WILL NOT BE THERE!!!**
       - operation-implemented in self.check_preconditions()

    3. prebook attributes (for each species)

       - generate attribute objects
       - "bookmark" that this operation will add them to the species,
         but does **NOT** assign the attributes (see 4.)
       - stored inside self.attributes_by_species
       - operation-implemented in self.prebook_species_attributes()

    4. associate generated attributes with their respective species

       - invoked by the initmanager (do **not** do this in this class)
       - based on self.attributes_by_species
       - (performed by pre-defined self.bake_species_attributes())

    So to define your own operator:

    - inherit from this class
    - accept species list & additional params from users
      (and store in attributes of self)
      -> no pre-defined interface, define your own
    - implement self.check_preconditions()
      -> check for required constants etc.
    - implement self.prebook_species_attributes()
      -> fill self.attributes_by_species

    Then use your operator somewhere and register it with the
    InitializationManager, which will make calls to the functions above, check
    for incompatibilities to other operators and finally associates the
    generated attributes to their species.
    """

    attributes_by_species = util.build_typesafe_property(typing.Dict[Species, typing.List[Attribute]])
    """attributes (exclusively) initialized by this operation"""

    def check_preconditions(self) -> None:
        """
        check own parameters and species this operator will be applied to

        Must be implemented (overwritten) by the operator.

        Throws if a precondition is not met, and passes silently if everything
        is okay.

        If this operator relies on certain constants of classes to be set,
        (e.g. mass, charge) this must be checked here.

        All parameters, including the species this Operator is to be applied to
        must be passed beforehand.
        Note: there is no unified way of passing parameters, please define your
        own. (Rationale: The structure of parameters is very heterogeneous.)
        """
        raise NotImplementedError()

    def prebook_species_attributes(self) -> None:
        """
        fills attributes_by_species

        Must be implemented (overwritten) by the operator.

        Generates the Attribute objects and pre-books them together with the
        species they are to be initialized for in self.attributes_by_species.

        Will only be run after self.check_preconditions() has passed.

        Note: Additional checks are not required, compatibility to other
        operations will be ensured from outside.

        MUST **NOT** MODIFY THE SPECIES.
        """
        raise NotImplementedError()

    def bake_species_attributes(self) -> None:
        """
        applies content of attributes_by_species to species

        For each species in attributes_by_species.keys() assigns
        the attributes to their respective species,
        precisely appends to the list Species.attributes

        Expects check_preconditions() and prebook_species_attributes() to have
        passed previously (without error).

        Additionally, performs the following sanity checks:

        - at least one attribute is assigned
        - the species does not already have an attribute of the same type
        - every attribute is assigned exclusively to one species

        Intended usage:

        1. check for dependencies in used species
        2. fill self.attributes_by_species (with newly generated objects)
           must be performed by self.prebook_species_attributes()
        3. call self.bake_species_attributes() (this method) to set
           species.attributes accordingly
        """
        # part A: sanity check self (i.e. pre-booked attributes)
        # (1) definition is not empty
        if 0 == len(self.attributes_by_species):
            raise ValueError("must pre-book for at least one species")

        attributes_cnt_by_species = dict(
            map(
                lambda kv_pair: (kv_pair[0], len(kv_pair[1])),
                self.attributes_by_species.items(),
            )
        )
        if 0 in attributes_cnt_by_species.values():
            raise ValueError("must assign at least one attribute to species")

        # (2) every object is exclusive to its species
        # extract all attribute lists, then join them
        all_attributes = list(reduce(lambda a, b: a + b, self.attributes_by_species.values()))
        duplicate_attribute_names = [attr.PICONGPU_NAME for attr in all_attributes if all_attributes.count(attr) > 1]
        if 0 != len(duplicate_attribute_names):
            raise ValueError(
                "attribute objects must be exclusive to species, offending: {}".format(
                    ", ".join(duplicate_attribute_names)
                )
            )

        # (3) each species only gets one attribute of each type (==name)
        for species, attributes in self.attributes_by_species.items():
            attr_names = list(map(lambda attr: attr.PICONGPU_NAME, attributes))
            duplicate_names = [name for name in attr_names if attr_names.count(name) > 1]
            if 0 != len(duplicate_names):
                raise ValueError(
                    "only on attribute per type is allowed per species, offending: {}".format(
                        ", ".join(duplicate_names)
                    )
                )

        # part B: check species to be assigned to
        # is a pre-booked attribute already defined?
        for species, attributes in self.attributes_by_species.items():
            present_attr_names = list(map(lambda attr: attr.PICONGPU_NAME, species.attributes))
            prebooked_attr_names = list(map(lambda attr: attr.PICONGPU_NAME, attributes))
            conflicting_attr_names = set(present_attr_names).intersection(prebooked_attr_names)
            if 0 != len(conflicting_attr_names):
                raise ValueError(
                    "conflict: species {} already has the attributes {}".format(
                        species.name, ", ".join(conflicting_attr_names)
                    )
                )

        # part C: actually assign the attributes
        for species, attributes in self.attributes_by_species.items():
            species.attributes += attributes

    def __init__(self):
        """
        constructor (abstract)

        Must be overwritten by implementation.

        Note: An inheriting operator must implement self.check_preconditions()
        and self.prebook_species_attributes(); while overwriting __init__() to
        do nothing is valid.
        """
        raise NotImplementedError()
