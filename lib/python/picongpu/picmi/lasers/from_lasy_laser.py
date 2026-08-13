"""
This file is part of PIConGPU.
Copyright 2025 PIConGPU contributors
Authors: Julian Lenz
License: GPLv3+
"""

import typeguard

from ...pypicongpu import laser
from ..copy_attributes import default_converts_to


@default_converts_to(laser.FromLasyLaser)
@typeguard.typechecked
class FromLasyLaser:
    """PICMI object for FromLasyLaser via FromOpenPMDPulseLaser"""

    def __init__(
        self,
        propagation_direction,
        polarization_direction,
        time_offset_si,
        file_path,
        lasyLaser,
        iteration=0,
        Nt=None,
        Nx=None,
        Ny=None,
        points_between_r=1.0,
        forced_dt=None,
        data_step=1,
        append=False,
        # make sure to always place Huygens-surface inside PML-boundaries,
        # default is valid for standard PMLs
        # @todo create check for insufficient dimension
        # @todo create check in simulation for conflict between PMLs and
        # Huygens-surfaces
        picongpu_huygens_surface_positions: list[list[int]] = [
            [16, -16],
            [16, -16],
            [16, -16],
        ],
    ):
        self.propagation_direction = propagation_direction
        self.polarization_direction = polarization_direction
        self.file_path = file_path
        self.iteration = iteration
        self.time_offset_si = time_offset_si
        self.picongpu_huygens_surface_positions = picongpu_huygens_surface_positions
        self.lasyLaser = lasyLaser
        self.Nt = Nt
        self.Nx = Nx
        self.Ny = Ny
        self.points_between_r = points_between_r
        self.forced_dt = forced_dt
        self.data_step = data_step
        self.append = append
