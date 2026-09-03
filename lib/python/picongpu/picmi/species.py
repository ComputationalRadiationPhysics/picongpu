"""
This file is part of PIConGPU.
Copyright 2021-2024 PIConGPU contributors
Authors: Hannes Troepgen, Brian Edward Marre, Julian Lenz
License: GPLv3+
"""

import re
from collections.abc import Mapping
from types import MappingProxyType
from typing import Any

from picmistandard import PICMI_Species
from pydantic import (
    BaseModel,
    ConfigDict,
    PrivateAttr,
    computed_field,
    field_validator,
    model_validator,
)

from picongpu.picmi.species_requirements import evaluate_requirements, resolving_add, run_construction
from picongpu.pypicongpu.species.attribute import Momentum, Position
from picongpu.pypicongpu.species.attribute.attribute import Attribute
from picongpu.pypicongpu.species.attribute.weighting import Weighting
from picongpu.pypicongpu.species.constant.charge import Charge
from picongpu.pypicongpu.species.constant.constant import Constant
from picongpu.pypicongpu.species.constant.densityratio import DensityRatio
from picongpu.pypicongpu.species.constant.mass import Mass
from picongpu.pypicongpu.species.operation import AnyOperation
from picongpu.pypicongpu.species.species import Pusher, Shape
from picongpu.pypicongpu.species.species import Species as PyPIConGPUSpecies

from .. import pypicongpu
from ..pypicongpu.species.util.element import Element
from .predefinedparticletypeproperties import PredefinedParticleTypeProperties


# Accepted particle-shape terms: the PICMI-standard names plus PIConGPU-only
# extensions, which (following the PICMI "other:" extension convention) are
# prefixed with "other:".
_SHAPE_BY_NAME: Mapping[str, Shape] = MappingProxyType(
    {
        "NGP": Shape.NGP,
        "linear": Shape.linear,
        "quadratic": Shape.quadratic,
        "cubic": Shape.cubic,
        "other:quartic": Shape.quartic,
        "other:counter": Shape.counter,
    }
)

# Accepted pusher-method terms: the PICMI-standard names plus PIConGPU-only
# extensions ("other:"-prefixed). Standard methods without a PIConGPU
# implementation (e.g. "Li") and unknown "other:*" terms are accepted at
# construction time (code-specific escape hatch) but are rejected with a clear
# message when the species is translated.
_PUSHER_BY_NAME: Mapping[str, Pusher] = MappingProxyType(
    {
        "Boris": Pusher.Boris,
        "Vay": Pusher.Vay,
        "Higuera-Cary": Pusher.Higuera,
        "free-streaming": Pusher.Free,
        "LLRK4": Pusher.ReducedLandauLifshitz,
        "other:Acceleration": Pusher.Acceleration,
        "other:Photon": Pusher.Photon,
        "other:Probe": Pusher.Probe,
        "other:Axel": Pusher.Axel,
    }
)

_STANDARD_SHAPES = ("NGP", "linear", "quadratic", "cubic")
_STANDARD_METHODS = ("Boris", "Vay", "Higuera-Cary", "Li", "free-streaming", "LLRK4")


def _lookup(kind: str, table: Mapping[str, Any], key: str):
    try:
        return table[key]
    except KeyError:
        raise ValueError(f"PIConGPU does not support {kind} {key!r}. Supported: {', '.join(table)}.") from None


class Species(PICMI_Species):
    """
    PICMI Species with PIConGPU-specific shape and pusher-method support.

    `particle_shape` accepts the PICMI-standard shapes ('NGP', 'linear',
    'quadratic', 'cubic') and PIConGPU extensions prefixed with 'other:'
    (e.g. 'other:quartic', 'other:counter').

    `method` accepts the PICMI-standard pusher methods ('Boris', 'Vay',
    'Higuera-Cary', 'Li', 'free-streaming', 'LLRK4') and PIConGPU-specific
    pushers prefixed with 'other:' (e.g. 'other:Acceleration',
    'other:Photon', 'other:Probe', 'other:Axel').
    """

    picongpu_fixed_charge: bool = False
    particle_shape: str | None = "quadratic"
    method: str | None = "Boris"

    # Theoretically, Position(), Momentum() and Weighting() are also requirements imposed from the outside,
    # e.g., by the current deposition, pusher, ..., but these concepts are not separately modelled in PICMI
    # particularly not as being applied to a particular species.
    # For now, we add them to all species. Refinements might be necessary in the future.
    _requirements: list[Any] = PrivateAttr(default_factory=lambda: [Position(), Weighting(), Momentum()])

    @field_validator("method")
    @classmethod
    def _validate_method(cls, value):
        # Note: this shadows picmistandard.PICMI_Species._validate_method, whose
        # access to PICMI_Species.methods_list crashes with an AttributeError.
        if value is not None and value not in _STANDARD_METHODS and not value.startswith("other:"):
            raise ValueError(
                f"Unsupported pusher method {value!r}. Must be one of "
                f"{', '.join(_STANDARD_METHODS)} or be prefixed with 'other:'."
            )
        return value

    @field_validator("particle_shape")
    @classmethod
    def _validate_particle_shape(cls, value):
        if value is not None and value not in _STANDARD_SHAPES and not value.startswith("other:"):
            raise ValueError(
                f"Unsupported particle shape {value!r}. Must be one of "
                f"{', '.join(_STANDARD_SHAPES)} or be prefixed with 'other:'."
            )
        return value

    @field_validator("name", mode="before")
    @classmethod
    def _validate_name(cls, value, values):
        if value is None:
            if values["particle_type"] is None:
                raise ValueError(
                    "Can't come up with a proper name for your species because neither name nor particle type are given."
                )
            value = values["particle_type"]
        return value

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @model_validator(mode="after")
    def check(self):
        if self.particle_type is None:
            assert self.charge_state is None, (
                f"Species {self.name} specified initial charge state via charge_state without also specifying particle "
                "type, must either set particle_type explicitly or only use charge instead"
            )
            assert self.picongpu_fixed_charge is False, (
                f"Species {self.name} specified fixed charge without also specifying particle_type"
            )
        # Returns None if it is not an element, so is False-y in those cases, and True-y otherwise:
        elif not self.picongpu_element:
            assert self.charge_state is None, "charge_state may only be set for ions"
            assert self.picongpu_fixed_charge is False, (
                f"Species {self.name} configured with fixed charge state but particle_type indicates non ion"
            )
        return self

    @computed_field
    def picongpu_element(self) -> Element | None:
        if self.particle_type is None:
            return None
        try:
            return (
                pypicongpu.species.util.Element(self.particle_type) if Element.is_element(self.particle_type) else None
            )
        except ValueError:
            return None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._register_initial_requirements()

    def _register_initial_requirements(self):
        constants = (
            ([DensityRatio(ratio=self.density_scale)] if self.density_scale is not None else [])
            + ([Mass(mass_si=self.mass)] if self.mass is not None else [])
            + ([Charge(charge_si=self.charge)] if self.charge is not None else [])
        )
        self.register_requirements(particle_type_requirements(self.particle_type) + constants)

    def _shape(self) -> Shape:
        return _lookup("particle shape", _SHAPE_BY_NAME, self.particle_shape or "quadratic")

    def _pusher(self) -> Pusher:
        return _lookup("pusher method", _PUSHER_BY_NAME, self.method or "Boris")

    def get_as_pypicongpu(self, *args, **kwargs):
        return PyPIConGPUSpecies(
            name=self.name,
            **self._evaluate_species_requirements(),
            shape=self._shape(),
            pusher=self._pusher(),
        )

    def get_operation_requirements(self):
        return evaluate_requirements(self._requirements, AnyOperation)

    def _evaluate_species_requirements(self):
        return {
            key: [run_construction(value) for value in values]
            for key, values in zip(
                ("constants", "attributes"), evaluate_requirements(self._requirements, [Constant, Attribute])
            )
        }

    def __gt__(self, other):
        # This defines a partial ordering on all species.
        # This is necessary to determine the definition order inside of the C++ header.
        if not isinstance(other, Species):
            raise ValueError(f"Unknown comparison between {self=} and {other=}.")
        return any(isinstance(req, DependsOn) and req.species == other for req in self._requirements)

    def register_requirements(self, requirements):
        for requirement in requirements:
            self._requirements = resolving_add(requirement, self._requirements)


def particle_type_requirements(particle_type):
    if (particle_type is None) or re.match(r"other:.*", particle_type):
        # no particle or custom particle type set
        return []
    if particle_type in (props := PredefinedParticleTypeProperties()).get_known_particle_types():
        mass, charge = props.get_mass_and_charge_of_non_element(particle_type)
    elif Element.is_element(particle_type):
        element = pypicongpu.species.util.Element(particle_type)
        mass = element.get_mass_si()
        charge = element.get_charge_si()
    else:
        # unknown particle type
        raise ValueError(f"Species has unknown particle type {particle_type}")
    return [Mass(mass_si=mass or 0.0), Charge(charge_si=charge or 0.0)]


class DependsOn(BaseModel):
    species: Species
