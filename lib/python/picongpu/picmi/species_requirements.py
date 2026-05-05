"""
This file is part of PIConGPU.
Copyright 2025 PIConGPU contributors
Authors: Julian Lenz
License: GPLv3+
"""

from types import UnionType
from typing import Any, Callable
from scipy.constants import electron_volt

import numpy as np
from pydantic import BaseModel, Field

from picongpu.pypicongpu.species.attribute.attribute import Attribute
from picongpu.pypicongpu.species.constant.constant import Constant
from picongpu.pypicongpu.species.constant.groundstateionization import GroundStateIonization
from picongpu.pypicongpu.species.constant.synchrotron import SynchrotronConstant
from picongpu.pypicongpu.species.operation.momentum.temperature import Temperature
from picongpu.pypicongpu.species.operation.setchargestate import SetChargeState
from picongpu.pypicongpu.species.operation.simpledensity import SimpleDensity
from picongpu.pypicongpu.species.operation.simplemomentum import SimpleMomentum


def resolving_add(new, to):
    # safeguard against exhaustible iterators, this is multi-pass
    to = list(to)

    for rhs in to:
        check_for_conflict(new, rhs)

    if must_be_unique(new) and any(can_be_dropped_due_to_uniqueness(new, rhs) for rhs in to):
        return to

    for into_instance in to:
        if try_update_with(into_instance, new):
            return to

    return to + [new]


def get_as_pypicongpu(obj, *args, **kwargs):
    if hasattr(obj, "get_as_pypicongpu"):
        return obj.get_as_pypicongpu(*args, **kwargs)
    return obj


def must_be_unique(requirement):
    return (hasattr(requirement, "must_be_unique") and requirement.must_be_unique) or (
        isinstance(requirement, Constant) or isinstance(requirement, Attribute)
    )


def can_be_dropped_due_to_uniqueness(lhs, rhs):
    if hasattr(lhs, "can_be_dropped_due_to_uniqueness") and lhs.can_be_dropped_due_to_uniqueness(rhs):
        return True
    if hasattr(rhs, "can_be_dropped_due_to_uniqueness") and rhs.can_be_dropped_due_to_uniqueness(lhs):
        return True
    try:
        # These might well be apples and oranges and the comparison might fail.
        if lhs == rhs:
            return True
    except Exception:
        pass
    return False


def try_update_with(into_instance, from_instance):
    if hasattr(into_instance, "try_update_with"):
        return into_instance.try_update_with(from_instance)
    return False


class RequirementConflict(Exception):
    pass


def check_for_conflict(obj1, obj2):
    try:
        if hasattr(obj1, "check_for_conflict"):
            obj1.check_for_conflict(obj2)
        if hasattr(obj2, "check_for_conflict"):
            obj2.check_for_conflict(obj1)
        if isinstance(obj1, Constant) and (isinstance(obj1, type(obj2)) or isinstance(obj2, type(obj1))):
            if obj1 != obj2:
                raise RequirementConflict(f"Conflicting constants {obj1=} and {obj2=} required.")
    except Exception as err:
        raise RequirementConflict(
            f"A conflict in requirements between {obj1=} and {obj2=} has been detected."
            "See above error message for details."
        ) from err


def run_construction(obj):
    return obj.run_construction() if hasattr(obj, "run_construction") else obj


def evaluate_requirements(requirements, Types):
    if isinstance(Types, type) or isinstance(Types, UnionType):
        return next(evaluate_requirements(requirements, [Types]))
    return (
        filter(
            lambda req: isinstance(req, Type)
            or (isinstance(req, DelayedConstruction) and issubclass(req.metadata.Type, Type)),
            requirements,
        )
        for Type in Types
    )


class _Operators(BaseModel):
    # More precisely, this returns self.metadata.Type
    # but it is kinda hard to express this in type hints
    # and I've got more urgent matters to deal with.
    constructor: Callable[[Any], Any] = lambda self: self.metadata.Type(
        *map(get_as_pypicongpu, self.metadata.args),
        **dict(map(lambda kv: (kv[0], get_as_pypicongpu(kv[1])), self.metadata.kwargs.items())),
    )
    try_update_with: Callable[[Any, Any], bool] = lambda self, other: False
    is_same_as: Callable[[Any, Any], bool] = lambda self, other: (
        isinstance(other, DelayedConstruction) and (self.metadata == other.metadata)
    )
    # This is supposed to raise in case of conflict:
    check_for_conflict: Callable[[Any, Any], None] = lambda self, other: None


class _Metadata(BaseModel):
    Type: type
    args: tuple[Any, ...] = tuple()
    kwargs: dict[str, Any] = Field(default_factory=dict)

    # This might be necessary to distinguish
    # processes with identical (kw)args but different operatos.
    misc: dict[str, Any] = Field(default_factory=dict)


class DelayedConstruction(BaseModel):
    """
    This class models the delayed construction of an object.

    While the user composes their simulation,
    the individual components will register requirements with our PICMI species
    in the form of what PyPIConGPU objects it needs the PyPIConGPU species to contain.
    But any individual registration cannot assume that
    the PICMI species has already obtained all knowledge
    necessary to construct its PyPIConGPU counterpart,
    so the registering object can only express its intent
    but not actually perform the construction in most cases.

    This class models such an intent.
    The metadata is supposed to contain
    a faithful and complete representation of what is constructed.
    It should be sufficient to
        - perform the construction and
        - compare intents (Is this other DelayedConstruction encoding the same action?)
    The operators are customisable actions to take under specific circumstances:
        - constructor: How to perform the construction.
        - update_with: How to merge another object into this one (returns if successful or not)
        - can_be_dropped_due_to_uniqueness: How to compare with another object
    The constructor is allowed to assume that at the time of its execution
        - all information is available and all objects can be constructed
        - the handled objects are stateless,
          i.e., the process is repeatable and the order of execution doesn't matter.
    A custom constructor should obviously follow these principles as well.
    The other operators cannot make such assumptions; they operate on mutable snapshots.

    A word of warning:
    You've got arbitrary functions for customisation at your disposal.
    Use them wisely!
    In particular, the construction should perform the obvious and minimal action
    required to fulfill the intent declared in the metadata.
    """

    metadata: _Metadata
    operators: _Operators = _Operators()
    must_be_unique: bool = False

    def run_construction(self):
        # This is a member variable not a member function,
        # so we gotta hand it the `self` argument explicitly.
        return self.operators.constructor(self)

    def try_update_with(self, other):
        return self.operators.try_update_with(self, other)

    def can_be_dropped_due_to_uniqueness(self, other):
        return self.operators.is_same_as(self, other)

    def check_for_conflict(self, other):
        return self.operators.check_for_conflict(self, other)

    def __eq__(self, other):
        # We do not check for operator equality here because we can't possibly inspect that.
        return (
            isinstance(other, DelayedConstruction)
            and (self.metadata == other.metadata)
            and (self.must_be_unique == other.must_be_unique)
        )


class GroundStateIonizationConstruction(DelayedConstruction):
    def __init__(self, /, ionization_model):
        def constructor(self):
            return GroundStateIonization(
                ionization_model_list=[m.get_as_pypicongpu() for m in self.metadata.kwargs["ionization_model_list"]]
            )

        def try_update_with(self, other):
            if not isinstance(other, GroundStateIonizationConstruction):
                return False
            for model in other.metadata.kwargs["ionization_model_list"]:
                if model not in self.metadata.kwargs["ionization_model_list"]:
                    self.metadata.kwargs["ionization_model_list"].append(model)
            return True

        operators = {"constructor": constructor, "try_update_with": try_update_with}
        metadata = {"Type": GroundStateIonization, "kwargs": {"ionization_model_list": [ionization_model]}}

        return super().__init__(operators=operators, metadata=metadata, must_be_unique=True)


class SetChargeStateOperation(DelayedConstruction):
    def __init__(self, /, species):
        metadata = {"Type": SetChargeState, "kwargs": {"species": species, "charge_state": species.charge_state}}
        return super().__init__(metadata=metadata, must_be_unique=True)


class SimpleDensityOperation(DelayedConstruction):
    def __init__(self, /, species, grid, layout):
        def constructor(self):
            kwargs = self.metadata.kwargs
            return self.metadata.Type(
                species=[s.get_as_pypicongpu() for s in sorted(kwargs["species"])],
                profile=kwargs["profile"].get_as_pypicongpu(kwargs["grid"]),
                layout=kwargs["layout"].get_as_pypicongpu(),
            )

        def try_update_with(self, other):
            return (
                isinstance(other, SimpleDensityOperation)
                and other.metadata.kwargs["profile"] == self.metadata.kwargs["profile"]
                and other.metadata.kwargs["layout"] == self.metadata.kwargs["layout"]
                and (self.metadata.kwargs["species"].extend(other.metadata.kwargs["species"]) or True)
            )

        metadata = {
            "Type": SimpleDensity,
            "kwargs": {
                "species": [species],
                "profile": species.initial_distribution,
                "layout": layout,
                "grid": grid,
            },
        }
        operators = {"constructor": constructor, "try_update_with": try_update_with}

        return super().__init__(metadata=metadata, operators=operators)


class SimpleMomentumOperation(DelayedConstruction):
    def __init__(self, /, species):
        def constructor(self):
            species = self.metadata.kwargs["species"].get_as_pypicongpu()
            particle_mass_si = species.constants.mass.mass_si
            rms_velocity = np.asarray(self.metadata.kwargs["rms_velocity"])
            kev_factor = (particle_mass_si / electron_volt) / 1000.0
            if np.allclose(rms_velocity, rms_velocity[0]):
                # Isotropic case: compute scalar temperature
                rms_velocity_si_squared = np.linalg.norm(rms_velocity) ** 2
                temperature_kev_scalar = kev_factor * rms_velocity_si_squared / 3.0
                temperature = (
                    Temperature(temperature_kev=temperature_kev_scalar) if temperature_kev_scalar > 0 else None
                )
            else:
                # Anisotropic case: compute directional temperatures
                directional_kev = kev_factor * (rms_velocity**2)
                if np.any(directional_kev > 0):
                    temperature = Temperature(temperature_kev_directional=tuple(directional_kev))
                else:
                    temperature = None
            return SimpleMomentum(species=species, drift=self.metadata.kwargs["drift"], temperature=temperature)

        metadata = {
            "Type": SimpleMomentum,
            "kwargs": {
                "species": species,
                "drift": species.initial_distribution.get_picongpu_drift(),
                "rms_velocity": species.initial_distribution.picongpu_get_rms_velocity_si(),
            },
        }
        operators = {"constructor": constructor}

        return super().__init__(metadata=metadata, operators=operators)


class SynchrotronConstantConstruction(DelayedConstruction):
    def __init__(self, /, photon_species):
        return super().__init__(metadata={"Type": SynchrotronConstant, "kwargs": {"photon_species": photon_species}})
