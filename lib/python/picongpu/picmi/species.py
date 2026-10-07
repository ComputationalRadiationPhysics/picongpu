"""
This file is part of PIConGPU.
Copyright 2021-2024 PIConGPU contributors
Authors: Hannes Troepgen, Brian Edward Marre, Julian Lenz
License: GPLv3+
"""

import re
import warnings
from collections.abc import Mapping
from types import MappingProxyType
from typing import Any, ClassVar

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
from picongpu.pypicongpu.species.pusherschedule import build_pusher_schedule
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
    }
)

_STANDARD_SHAPES = ("NGP", "linear", "quadratic", "cubic")
_STANDARD_METHODS = ("Boris", "Vay", "Higuera-Cary", "Li", "free-streaming", "LLRK4")


def _lookup(kind: str, table: Mapping[str, Any], key: str):
    try:
        return table[key]
    except KeyError:
        raise ValueError(f"PIConGPU does not support {kind} {key!r}. Supported: {', '.join(table)}.") from None


class CompositePusher(BaseModel):
    """A step-dependent (composite) particle pusher.

    Maps **pre-called** :class:`~picongpu.picmi.diagnostics.TimeStepSpec`
    instances to pusher names.  The first matching specification wins, so the
    order matters::

        CompositePusher({
            TimeStepSpec[::3]("steps"): "Boris",
            TimeStepSpec[:5]("steps"): "Vay",
            TimeStepSpec[6:]("steps"): "free-streaming",
        })

    expands, step by step, to Boris at 0, 3, 6, 9, 12; Vay at 1, 2, 4, 5; and
    free-streaming elsewhere.

    The specification keys must be given in the ``"steps"`` unit (a
    ``"seconds"`` unit cannot be resolved without -- and would silently change
    with -- the time step size, so it is rejected).  Accepted pusher names are
    the same as for the scalar ``method`` string, including the ``other:``
    extension prefix.

    Every step in ``[0, max_steps)`` must be claimed by exactly one
    specification; a gap is a hard error (there is no silent fallback).  A
    composite with a single entry is allowed and behaves like the plain scalar
    pusher.  A ``CompositePusher`` is given through the same ``method`` field as
    the scalar pusher name, so the two cannot be set at once: when a
    ``CompositePusher`` is given, the scalar ``method`` string is unused (and a
    :class:`CompositePusher` is never accepted on the scalar string path).

    Validation needs the simulation's step count, so it happens when the owning
    :class:`~picongpu.picmi.simulation.Simulation` is translated.
    """

    _items: list[tuple[Any, str]] = PrivateAttr(default_factory=list)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __init__(self, schedule: Mapping | None = None, **kwargs):
        if schedule is None:
            schedule = kwargs.pop("schedule", None)
        if schedule is None:
            raise ValueError("CompositePusher requires a mapping of TimeStepSpec to pusher name.")
        super().__init__(**kwargs)
        # TimeStepSpec keys are not valid pydantic dict-key types, so we accept
        # the raw mapping ourselves (bypassing validation) and keep the ordered
        # items for ordered, first-match resolution.
        self._items = list(schedule.items())

    @computed_field
    def schedule(self) -> list[dict]:
        """JSON-serialisable view of the schedule (ordered, first-match)."""
        return [{"spec": repr(spec.specs), "pusher": name} for spec, name in self._items]

    @property
    def items(self) -> list[tuple[Any, str]]:
        return list(self._items)

    def _resolved_entries(self, num_steps: int):
        """Resolve the ordered specifications to ``(slices, Pusher)`` stage entries.

        ``slices`` are ``(start, stop, step)`` tuples in steps, inclusive on both
        ends (``stop == -1`` meaning open).  The pusher is looked up by its PICMI
        name.
        """
        entries = []
        for spec, name in self._items:
            if getattr(spec, "unit_system", None) != "steps":
                raise ValueError(
                    "CompositePusher keys must be TimeStepSpec instances given in the 'steps' unit, "
                    f'i.e. TimeStepSpec[...]("steps"); you gave {spec!r} with unit_system='
                    f"{getattr(spec, 'unit_system', None)!r}. Seconds cannot be used here because they "
                    "depend on the time step size."
                )
            slices = [
                (s.start, s.stop, s.step)
                for s in (spec._interpret_negatives(spec._interpret_nones(s), num_steps) for s in spec.specs)
            ]
            entries.append((slices, _lookup("pusher method", _PUSHER_BY_NAME, name)))
        return entries

    def validate(self, num_steps: int) -> None:
        """Check that every step in ``[0, num_steps)`` is claimed by exactly one entry."""
        entries = self._resolved_entries(num_steps)
        for step in range(num_steps):
            if not any(any(_slice_contains(s, step) for s in slices) for slices, _ in entries):
                raise ValueError(
                    f"CompositePusher does not cover time step {step} (of {num_steps}). "
                    "Every step in [0, max_steps) must be mapped to a pusher; add a catch-all "
                    'specification such as TimeStepSpec[::1]("steps").'
                )
        _warn_on_adjacent_end_overlap(entries, num_steps)


def _slice_contains(one_slice, step: int) -> bool:
    start, stop, slicestep = one_slice
    if step < start:
        return False
    if stop != -1 and step > stop:
        return False
    return (step - start) % slicestep == 0


def _warn_on_adjacent_end_overlap(entries, num_steps: int) -> None:
    """Warn when two adjacent entries both claim a shared interval end.

    Overlaps that are *not* at a shared end are a deliberate first-match
    feature and stay silent; a shared end is almost always an off-by-one in the
    intended disjoint split and gets a message that says how to fix it.
    """
    for stage in range(len(entries) - 1):
        earlier_stops = {stop for _start, stop, _step in entries[stage][0] if stop != -1}
        later_starts = {start for start, _stop, _step in entries[stage + 1][0]}
        for step in sorted(earlier_stops & later_starts):
            if not (0 <= step < num_steps):
                continue
            if not any(_slice_contains(s, step) for s in entries[stage][0]):
                continue
            if not any(_slice_contains(s, step) for s in entries[stage + 1][0]):
                continue
            warnings.warn(
                f"CompositePusher specifications {stage} and {stage + 1} both claim step {step} "
                "(one interval ends where the next begins). The earlier specification wins, but you "
                f'probably meant disjoint intervals; make them disjoint, e.g. TimeStepSpec[:{step}]("steps") '
                f'and TimeStepSpec[{step + 1}:]("steps").',
                UserWarning,
                stacklevel=3,
            )


class Species(PICMI_Species):
    """
    PICMI Species with PIConGPU-specific shape and pusher-method support.

    `particle_shape` accepts the PICMI-standard shapes ('NGP', 'linear',
    'quadratic', 'cubic') and PIConGPU extensions prefixed with 'other:'
    (e.g. 'other:quartic', 'other:counter'). If left unset it is inherited
    from the owning Simulation (see ``Simulation.particle_shape``) and, if
    that too is unset, falls back to the PIConGPU default 'quadratic' (TSC).

    `method` accepts the PICMI-standard pusher methods ('Boris', 'Vay',
    'Higuera-Cary', 'Li', 'free-streaming', 'LLRK4') and PIConGPU-specific
    pushers prefixed with 'other:' (e.g. 'other:Acceleration',
    'other:Photon', 'other:Probe'). If left unset it falls back
    to the PIConGPU default 'Boris'.

    `method` additionally accepts a :class:`CompositePusher`, a step-dependent
    schedule mapping pre-called :class:`~picongpu.picmi.diagnostics.TimeStepSpec`
    instances to pusher names (first match wins); see the class docstring of
    :class:`CompositePusher` for the supported syntax.
    """

    # PIConGPU's native default particle shape (TSC). This is the code-level
    # fallback used whenever neither the species nor its owning Simulation sets
    # a shape. Declared here (rather than buried in a resolution method) so the
    # intention is discoverable on the class that actually resolves the shape.
    DEFAULT_PARTICLE_SHAPE: ClassVar[str] = "quadratic"

    picongpu_fixed_charge: bool = False
    particle_shape: str | None = None
    method: "str | CompositePusher | None" = None

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
        if isinstance(value, CompositePusher):
            return value
        if value is not None and value not in _STANDARD_METHODS and not value.startswith("other:"):
            raise ValueError(
                f"Unsupported pusher method {value!r}. Must be one of "
                f"{', '.join(_STANDARD_METHODS)}, a CompositePusher, or be prefixed with 'other:'."
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

    @model_validator(mode="before")
    @classmethod
    def _default_name_from_particle_type(cls, data):
        # A before-field validator would never run when `name` is left at its
        # default (None) without `validate_default=True`, and would need
        # `ValidationInfo` to reach `particle_type`. A model-level before
        # validator runs on the raw input regardless of defaults.
        if isinstance(data, dict) and data.get("name") is None:
            particle_type = data.get("particle_type")
            if particle_type is None:
                raise ValueError(
                    "Can't come up with a proper name for your species because neither name nor particle type are given."
                )
            if isinstance(particle_type, str) and particle_type.startswith("other:"):
                # "other:..." particle types are code-specific custom types and are
                # not valid C++ identifiers, so they cannot serve as the species name
                # (the rendered C++ header name must match r"^[A-Za-z0-9_]+$"). Require
                # an explicit, valid name instead of auto-naming from the particle type.
                raise ValueError(
                    f"particle_type {particle_type!r} is a custom 'other:' type and cannot be used as a species name. "
                    "Provide an explicit `name` (a valid identifier, e.g. name='myType')."
                )
            data["name"] = particle_type
        return data

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

    def _shape(self, default_particle_shape: str | None = None) -> Shape:
        return _lookup("particle shape", _SHAPE_BY_NAME, self._resolved_particle_shape(default_particle_shape))

    def _pusher(self) -> Pusher:
        if isinstance(self.method, CompositePusher):
            # The scalar pusher field is unused for a step-dependent schedule;
            # keep a valid default so the backend model stays well-formed.
            return Pusher.Boris
        return _lookup("pusher method", _PUSHER_BY_NAME, self.method or "Boris")

    def _pusher_schedule(self, time_step_size: float | None, num_steps: int | None):
        """Render the step-dependent pusher for a ``CompositePusher`` ``method``.

        Returns ``None`` for the scalar pusher path.  ``num_steps`` is only
        known at Simulation translation; other translation sites (diagnostics,
        collisions) translate the species for their own, pusher-independent
        purposes and pass ``None``, so no schedule is produced there.  The
        authoritative step-coverage validation therefore runs when the
        Simulation translates its own species list.
        """
        if not isinstance(self.method, CompositePusher) or num_steps is None:
            return None
        self.method.validate(num_steps)
        entries = self.method._resolved_entries(num_steps)
        return build_pusher_schedule(
            [slices for slices, _ in entries],
            [pusher.cpp_name for _, pusher in entries],
        )

    def _resolved_particle_shape(self, default_particle_shape: str | None = None) -> str:
        # An unset species shape falls back to the shape supplied by the owning
        # Simulation (passed in at translation time), then to the PIConGPU
        # default declared on the class.
        if self.particle_shape is not None:
            return self.particle_shape
        if default_particle_shape is not None:
            return default_particle_shape
        return self.DEFAULT_PARTICLE_SHAPE

    def get_as_pypicongpu(
        self,
        *args,
        default_particle_shape: str | None = None,
        time_step_size: float | None = None,
        num_steps: int | None = None,
        **kwargs,
    ):
        pusher = self._pusher()
        return PyPIConGPUSpecies(
            name=self.name,
            **self._evaluate_species_requirements(),
            shape=self._shape(default_particle_shape),
            pusher=pusher,
            pusher_schedule=self._pusher_schedule(time_step_size, num_steps),
        )

    def picongpu_get_mass_si(self) -> float:
        """
        Mass of the physical particle of this species in kg (SI units).

        Resolved from the explicit ``mass`` if given, otherwise from the particle type.
        """
        for requirement in self._requirements:
            if isinstance(requirement, Mass):
                if requirement.mass_si <= 0:
                    raise ValueError(f"Species {self.name} has no positive mass defined.")
                return requirement.mass_si
        raise ValueError(f"Species {self.name} has no resolvable mass. Specify a mass or a known particle type.")

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
