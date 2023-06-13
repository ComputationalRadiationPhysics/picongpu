"""
This file is part of the PIConGPU.
Copyright 2021-2023 PIConGPU contributors
Authors: Hannes Troepgen, Brian Edward Marre
License: GPLv3+
"""

from .grid import Grid3D
from .laser import GaussianLaser
from .solver import Solver
from . import species
from . import util
from . import output
from .rendering import RenderedObject

import typing
from typeguard import typechecked


@typechecked
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

    def __get_output_context(self) -> dict:
        """retrieve all output objects"""
        auto = output.Auto()
        auto.period = max(1, int(self.time_steps / 100))

        return {
            "auto": auto.get_rendering_context(),
        }

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

        return serialized
