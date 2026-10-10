"""
This file is part of PIConGPU.
Copyright 2021-2025 PIConGPU contributors
Authors: Hannes Troepgen, Brian Edward Marre, Julian Lenz
License: GPLv3+
"""

import math
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import (
    AfterValidator,
    BaseModel,
    BeforeValidator,
    Field,
    computed_field,
    field_serializer,
    field_validator,
)

from picongpu.pypicongpu.collisions import CollisionalPhysicsSetup
from picongpu.pypicongpu.output.optical_imaging import (
    COMPILE_TIME_FIELDS,
    MASK_FIELDS,
    OpticalImaging,
    default_optical_imaging_params,
)
from picongpu.pypicongpu.output.radiation import RadiationPlugin
from picongpu.pypicongpu.output.timestepspec import TimeStepSpec
from picongpu.pypicongpu.particle_functor.particle_functor import ParticleFunctor
from picongpu.pypicongpu.species.constant.synchrotron import SynchrotronParams
from picongpu.pypicongpu.species.operation import AnyOperation
from picongpu.pypicongpu.species.species import Species

from .backgroundfield import BackgroundField
from .customuserinput import CustomUserInput
from .field_solver import AnySolver
from .grid import AnyGrid
from .laser import AnyLaser
from .memory import MemoryConfig
from .movingwindow import MovingWindow
from .output import AnyPlugin, OpenPMDPlugin
from .precision_config import PrecisionConfig
from .rendering import RenderedObject
from .walltime import Walltime


def _default_min_weighting(value: float | None) -> float:
    """Fall back to PIConGPU's C++ default (10.0) when no weighting is given (unit: none)."""
    return 10.0 if value is None else value


def _validate_min_weighting(value: float) -> float:
    """Reject non-positive and non-finite weightings; mirrors the C++ MIN_WEIGHTING assumption (unit: none)."""
    if not (math.isfinite(value) and value > 0):
        raise ValueError(f"Minimum weighting must be finite and > 0, not {value=}.")
    return value


def _optical_imaging_params(outputs) -> dict:
    """The compile-time ``params::`` values shared by all optical-imaging instances.

    The plugin's ``params`` namespace (and its three mask functions) is compiled
    in, so every ``OpticalImaging`` of a simulation must agree on it. Without an
    imaging diagnostic the C++ defaults with constant mask functions are
    returned, so ``shadowgraphy.param`` is always renderable.

    The mask functions are compared as their **rendered C++ strings**: two
    callables that are semantically equal but spelled differently (e.g.
    ``x + x`` vs ``2 * x``) render differently and are rejected. That is
    intentional -- the plugin compiles exactly one of them -- but callers
    should share the callable rather than re-deriving an equal expression.
    """
    instances = [entry for entry in outputs or [] if isinstance(entry, OpticalImaging)]
    if not instances:
        return default_optical_imaging_params()
    first = instances[0]
    values = {name: getattr(first, name) for name in COMPILE_TIME_FIELDS + MASK_FIELDS}
    for other in instances[1:]:
        for name in COMPILE_TIME_FIELDS + MASK_FIELDS:
            if getattr(other, name) != values[name]:
                raise ValueError(
                    "All OpticalImaging instances share the compile-time shadowgraphy parameters, but they "
                    f"disagree on {name!r}: {values[name]!r} vs {getattr(other, name)!r}. Keep the "
                    "compile-time values (t_res, x_res, y_res, numerical_aperture, central_lambda, d_lambda, "
                    "t_wf_buffer, pos_wf_size, numerical_aperture_wf_size, d_lambda_wf and the three mask "
                    "functions) identical across instances; only the .cfg options may differ."
                )
    return values


def _uniquify_optical_imaging_files(outputs) -> None:
    """Give every optical-imaging instance a distinct output ``file`` prefix.

    The plugin writes ``<file>_%T.<ext>`` per instance, so two instances sharing
    a prefix (e.g. two default-named ``Shadowgraphy``) would overwrite each
    other's output. Explicitly distinct prefixes are left untouched; a duplicate
    gets the smallest free numeric suffix. Idempotent, so repeated validation
    (e.g. a second ``get_as_pypicongpu`` call) does not keep appending.
    """
    used: set[str] = set()
    for instance in outputs or []:
        if not isinstance(instance, OpticalImaging):
            continue
        name = instance.file
        if name not in used:
            used.add(name)
            continue
        suffix = 2
        while f"{name}_{suffix}" in used:
            suffix += 1
        instance.file = f"{name}_{suffix}"
        used.add(instance.file)


class Simulation(RenderedObject, BaseModel):
    """
    Represents all parameters required to build & run a PIConGPU simulation.

    Most of the individual parameters are delegated to other objects held as
    attributes.

    To run a Simulation object pass it to the Runner (for details see there).
    """

    base_density: float
    """value to normalise densities"""

    delta_t_si: float
    """Width of a single timestep, given in seconds."""

    time_steps: int
    """Total number of time steps to be executed."""

    grid: AnyGrid
    """Used grid Object"""

    laser: list[AnyLaser] | None
    """List of laser objects to use in the simulation, or None to disable lasers"""

    background_field: BackgroundField | None = None
    """
    Background field applied to the grid E and B fields (see BackgroundField),
    or None to disable the field background.

    A background field is added to the fields around the particle push, i.e.
    it affects the particles but is not evolved by the field solver itself.
    """

    solver: AnySolver
    """Used Solver"""

    typical_ppc: int
    """
    typical number of macro particles spawned per cell, >=1

    used for normalization of units
    """

    customuserinput: list[CustomUserInput] | None
    """
    object that contains additional user specified input parameters to be used in custom templates

    @attention custom user input is global to the simulation
    """

    moving_window: MovingWindow | None
    """used moving Window, set to None to disable"""

    walltime: Walltime
    """time limit of the simulation run"""

    binomial_current_interpolation: bool
    """switch on a binomial current interpolation"""

    output: list[AnyPlugin] | None
    species: list[Species]
    init_operations: list[AnyOperation]
    synchrotron_params: SynchrotronParams = SynchrotronParams()
    collisional_physics: CollisionalPhysicsSetup = CollisionalPhysicsSetup()
    particle_filters: list[ParticleFunctor] = Field(default_factory=list)

    min_weighting: Annotated[
        float, BeforeValidator(_default_min_weighting), AfterValidator(_validate_min_weighting)
    ] = 10.0
    """
    minimum macro-particle weighting below which particles are not created / are deleted, unit: none

    rendered as ``MIN_WEIGHTING`` into ``include/picongpu/param/particle.param``;
    defaults to PIConGPU's C++ default of 10.0
    """

    precision: Literal[32, 64] = 32
    """
    floating point precision of the simulation core (see ``precision.param``)

    32 -> ``precision32Bit`` (single precision, default), 64 -> ``precision64Bit``
    (double precision). Controls ``namespace precisionPIConGPU`` in the generated
    ``include/picongpu/param/precision.param``.
    """

    precision_overrides: PrecisionConfig = PrecisionConfig()
    """per-namespace precision overrides rendered into ``precision.param`` (see ``PrecisionConfig``)."""

    memory_config: MemoryConfig = MemoryConfig()
    """memory / exchange-buffer knobs rendered into ``memory.param`` (see ``MemoryConfig``)."""

    @computed_field
    def precisionSqrt(self) -> str:
        return (
            "precisionPIConGPU"
            if self.precision_overrides.sqrt == "core"
            else f"precision{self.precision_overrides.sqrt}Bit"
        )

    @computed_field
    def precisionExp(self) -> str:
        return (
            "precisionPIConGPU"
            if self.precision_overrides.exp == "core"
            else f"precision{self.precision_overrides.exp}Bit"
        )

    @computed_field
    def precisionTrigonometric(self) -> str:
        return (
            "precisionPIConGPU"
            if self.precision_overrides.trig == "core"
            else f"precision{self.precision_overrides.trig}Bit"
        )

    @computed_field
    def optical_imaging_params(self) -> dict:
        """The compile-time ``params::`` values for ``shadowgraphy.param``.

        The plugin's ``params`` (and its three mask functions) are compiled in,
        so all instances must agree on them. With no imaging diagnostic the C++
        defaults with constant mask functions are rendered so the param file is
        always present.
        """
        return _optical_imaging_params(self.output)

    @field_validator("output", mode="after")
    @classmethod
    def _output_validation(cls, outputs):
        # The radiation plugin expects to always have content in its param file,
        # so we'll always add a RadiationPlugin to make them appear.
        default = [
            RadiationPlugin(
                species=[],
                period=TimeStepSpec([]),
                config={"observer": {"N_observer": 1, "index_to_direction": lambda _: [1, 0, 0]}},
            )
        ]
        if outputs is None:
            outputs = default
        elif not any(isinstance(o, RadiationPlugin) for o in outputs):
            outputs = outputs + default
        # validate the (compile-time) consistency of the optical-imaging
        # instances eagerly, so a mismatch is reported at simulation
        # construction rather than only when the params are accessed.
        _optical_imaging_params(outputs)
        # two instances with the default prefix would clobber each other's
        # ``<file>_%T`` output; ensure every prefix is distinct.
        _uniquify_optical_imaging_files(outputs)
        return outputs

    @field_serializer("customuserinput")
    def _render_custom_user_input_list(self, value) -> dict[str, Any] | None:
        if value is None:
            return None
        custom_rendering_context = {"tags": []}

        for entry in value:
            add_context = entry.get_rendering_context()
            tags = entry.get_tags()

            entry.check_does_not_change_existing_key_values(custom_rendering_context, add_context)
            entry.check_tags(custom_rendering_context["tags"], tags)

            custom_rendering_context.update(add_context)
            custom_rendering_context["tags"].extend(tags)

        return custom_rendering_context

    def spread_directory_information(self, setup_dir):
        for plugin in self.output or []:
            if isinstance(plugin, OpenPMDPlugin):
                plugin.setup_dir = Path(setup_dir)
