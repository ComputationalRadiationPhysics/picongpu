"""
This file is part of PIConGPU.
Copyright 2021-2023 PIConGPU contributors
Authors: Hannes Troepgen, Brian Edward Marre
License: GPLv3+
"""

from .grid import Grid3D
from .laser import GaussianLaser
from .movingwindow import MovingWindow
from .solver import Solver
from . import species
from . import util
from . import output
from .rendering import RenderedObject
from .customuserinput import InterfaceCustomUserInput

import typing
import typeguard
import logging


@typeguard.typechecked
class Simulation(RenderedObject):
    """
    Represents all parameters required to build & run a PIConGPU simulation.

    Most of the individual parameters are delegated to other objects held as
    attributes.

    To run a Simulation object pass it to the Runner (for details see there).
    """

    delta_t_si = util.build_typesafe_property(float)
    """Width of a single timestep, given in seconds."""

    time_steps = util.build_typesafe_property(int)
    """Total number of time steps to be executed."""

    grid = util.build_typesafe_property(typing.Union[Grid3D])
    """Used grid Object"""

    laser = util.build_typesafe_property(typing.Optional[GaussianLaser])
    """Used (gaussian) Laser"""

    solver = util.build_typesafe_property(Solver)
    """Used Solver"""

    init_manager = util.build_typesafe_property(species.InitManager)
    """init manager holding all species & their information"""

    typical_ppc = util.build_typesafe_property(int)
    """
    typical number of macro particles spawned per cell, >=1

    used for normalization of units
    """

    custom_user_input = util.build_typesafe_property(typing.Optional[list[InterfaceCustomUserInput]])
    """
    object that contains additional user specified input parameters to be used in custom templates

    @attention custom user input is global to the simulation
    """

    moving_window = util.build_typesafe_property(typing.Optional[MovingWindow])
    """used moving Window, set to None to disable"""

    def __get_output_context(self) -> dict:
        """retrieve all output objects"""
        auto = output.Auto()
        auto.period = max(1, int(self.time_steps / 100))

        return {
            "auto": auto.get_rendering_context(),
        }

    def __render_custom_user_input_list(self) -> dict:
        custom_rendering_context = {"tags": []}

        for entry in self.custom_user_input:
            add_context = entry.get_generic_rendering_context()
            tags = entry.get_tags()

            entry.check_does_not_change_existing_key_values(custom_rendering_context, add_context)
            entry.check_tags(custom_rendering_context["tags"], tags)

            custom_rendering_context.update(add_context)
            custom_rendering_context["tags"].extend(tags)

        return custom_rendering_context

    def __found_custom_input(self, serialized: dict):
        logging.info(
            "found custom user input with tags: "
            + str(serialized["customuserinput"]["tags"])
            + "\n"
            + "\t WARNING: custom input is not checked, it is the user's responsibility to check inputs and generated input.\n"
            + "\t WARNING: custom templates are required if using custom user input.\n"
        )

    def _get_serialized(self) -> dict:
        serialized = {
            "delta_t_si": self.delta_t_si,
            "time_steps": self.time_steps,
            "typical_ppc": self.typical_ppc,
            "solver": self.solver.get_rendering_context(),
            "grid": self.grid.get_rendering_context(),
            "species_initmanager": self.init_manager.get_rendering_context(),
            "output": self.__get_output_context(),
        }

        if self.laser is not None:
            serialized["laser"] = self.laser.get_rendering_context()
        else:
            serialized["laser"] = None

        if self.moving_window is not None:
            serialized["moving_window"] = self.moving_window.get_rendering_context()
        else:
            serialized["moving_window"] = None

        if self.custom_user_input is not None:
            serialized["customuserinput"] = self.__render_custom_user_input_list()
            self.__found_custom_input(serialized)
        else:
            serialized["customuserinput"] = None

        return serialized
