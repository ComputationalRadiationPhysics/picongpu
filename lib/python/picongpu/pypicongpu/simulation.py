"""
This file is part of PIConGPU.
Copyright 2021-2025 PIConGPU contributors
Authors: Hannes Troepgen, Brian Edward Marre, Julian Lenz
License: GPLv3+
"""

from pathlib import Path
from typing import Annotated, Any

from pydantic import BaseModel, PlainSerializer, Field, field_serializer, field_validator

from picongpu.pypicongpu.collisions import CollisionalPhysicsSetup
from picongpu.pypicongpu.particle_functor.particle_functor import ParticleFunctor
from picongpu.pypicongpu.output.radiation import RadiationPlugin
from picongpu.pypicongpu.output.timestepspec import TimeStepSpec
from picongpu.pypicongpu.species.constant.synchrotron import SynchrotronParams
from picongpu.pypicongpu.species.operation.operation import Operation
from picongpu.pypicongpu.species.species import Species

from .customuserinput import CustomUserInput
from .field_solver import AnySolver
from .grid import Grid3D
from .laser import AnyLaser
from .movingwindow import MovingWindow
from .output import OpenPMDPlugin, Plugin
from .rendering import RenderedObject
from .walltime import Walltime


# The annotation with -> Any is a temporary workaround.
# Pydantic cannot know about our custom schema,
# so it fails to validate correctly.
def _serialize(value) -> Any:
    if isinstance(value, list):
        return [_serialize(v) for v in value]
    return value.get_rendering_context() if value is not None else None


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

    grid: Grid3D
    """Used grid Object"""

    laser: Annotated[list[AnyLaser] | None, PlainSerializer(_serialize)]
    """List of laser objects to use in the simulation, or None to disable lasers"""

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

    output: Annotated[list[Plugin] | None, PlainSerializer(_serialize)]
    species: list[Species]
    init_operations: Annotated[list[Operation], PlainSerializer(_serialize)]
    synchrotron_params: SynchrotronParams = SynchrotronParams()
    collisional_physics: CollisionalPhysicsSetup = CollisionalPhysicsSetup()
    particle_filters: list[ParticleFunctor] = Field(default_factory=list)

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
            return default
        if not any(isinstance(o, RadiationPlugin) for o in outputs):
            return outputs + default
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
