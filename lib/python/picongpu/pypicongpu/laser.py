"""
This file is part of the PIConGPU.
Copyright 2021-2022 PIConGPU contributors
Authors: Hannes Tröpgen, Brian Edward Marré, Alexander Debus
License: GPLv3+
"""

from . import util
from typeguard import typechecked
from enum import Enum
from .rendering import RenderedObject
import typing
import logging


@typechecked
class GaussianLaser(RenderedObject):
    """
    PIConGPU Gaussian Laser

    Holds Parameters to specify a gaussian laser
    """

    class PolarizationType(Enum):
        """represents a polarization of a laser (for PIConGPU)"""
        LINEAR_X = 1
        LINEAR_Z = 2
        CIRCULAR = 3

        def get_cpp_str(self) -> str:
            """retrieve name as used in c++ param files"""
            cpp_by_ptype = {
                GaussianLaser.PolarizationType.LINEAR_X: "LINEAR_X",
                GaussianLaser.PolarizationType.LINEAR_Z: "LINEAR_Z",
                GaussianLaser.PolarizationType.CIRCULAR: "CIRCULAR",
            }
            return cpp_by_ptype[self]

    wavelength = util.build_typesafe_property(float)
    """wave length in m"""
    waist = util.build_typesafe_property(float)
    """beam waist in m"""
    duration = util.build_typesafe_property(float)
    """length in seconds (1 sigma)"""
    focus_pos = util.build_typesafe_property(float)
    """y coordinate of focus in m"""
    phase = util.build_typesafe_property(float)
    """phase in rad, periodic in 2*pi"""
    E0 = util.build_typesafe_property(float)
    """E0 in V/m"""
    pulse_init = util.build_typesafe_property(float)
    """laser will be initialized pulse_init times of duration (unitless)"""
    init_plane_y = util.build_typesafe_property(int)
    """absorber cells in neg. y direction, 0 to disable (number of cells)"""
    polarization_type = util.build_typesafe_property(PolarizationType)
    """laser polarization"""
    laguerre_modes = util.build_typesafe_property(typing.List[float])
    """array containing the magnitudes of radial Laguerre-modes"""
    laguerre_phases = util.build_typesafe_property(typing.List[float])
    """array containing the phases of radial Laguerre-modes"""

    def _get_serialized(self) -> dict:
        if [] == self.laguerre_modes:
            raise ValueError("Laguerre modes MUST NOT be empty.")
        if [] == self.laguerre_phases:
            raise ValueError("Laguerre phases MUST NOT be empty.")
        if len(self.laguerre_phases) != len(self.laguerre_modes):
            raise ValueError("Laguerre modes and Laguerre phases MUST BE "
                             "arrays of equal length.")
        if len(list(filter(lambda x: x < 0, self.laguerre_modes))) > 0:
            logging.warning(
                "Laguerre mode magnitudes SHOULD BE positive definite.")
        return {
            "wave_length_si": self.wavelength,
            "waist_si": self.waist,
            "pulse_length_si": self.duration,
            "focus_pos_si": self.focus_pos,
            "phase": self.phase,
            "E0_si": self.E0,
            "pulse_init": self.pulse_init,
            "init_plane_y": self.init_plane_y,
            "polarization_type": self.polarization_type.get_cpp_str(),
            "laguerre_modes": list(map(lambda x: {"single_laguerre_mode": x},
                                       self.laguerre_modes)),
            "laguerre_phases": list(map(lambda x: {"single_laguerre_phase": x},
                                        self.laguerre_phases)),
            "modenumber": len(self.laguerre_modes)-1,
        }
