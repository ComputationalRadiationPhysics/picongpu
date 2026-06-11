"""
# SPDX-FileCopyrightText: Julian Lenz
#
# SPDX-License-Identifier: GPL-3.0-or-later
"""

import typeguard

from ...pypicongpu import laser
from ..copy_attributes import default_converts_to


@default_converts_to(laser.FromOpenPMDPulseLaser)
@typeguard.typechecked
class FromOpenPMDPulseLaser:
    """PICMI object for FromOpenPMDPulseLaser"""

    def __init__(
        self,
        propagation_direction,
        polarization_direction,
        time_offset_si,
        file_path,
        iteration,
        dataset_name,
        datatype,
        polarisationAxisOpenPMD,
        propagationAxisOpenPMD,
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
        self.dataset_name = dataset_name
        self.datatype = datatype
        self.time_offset_si = time_offset_si
        self.polarisationAxisOpenPMD = polarisationAxisOpenPMD
        self.propagationAxisOpenPMD = propagationAxisOpenPMD
        self.picongpu_huygens_surface_positions = picongpu_huygens_surface_positions
