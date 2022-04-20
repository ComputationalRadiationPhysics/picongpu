"""
This file is part of the PIConGPU.
Copyright 2021-2022 PIConGPU contributors
Authors: Hannes Tröpgen, Brian Edward Marré, Alexander Debus
License: GPLv3+
"""

from ..pypicongpu import util, laser

import picmistandard

from typeguard import typechecked
import typing


@typechecked
class GaussianLaser(picmistandard.PICMI_GaussianLaser):
    """PICMI object for Gaussian Laser"""

    def __init__(
            self, wavelength, waist, duration,
            picongpu_laguerre_modes: typing.Optional[
                typing.List[float]] = None,
            picongpu_laguerre_phases: typing.Optional[
                typing.List[float]] = None,
            **kw):
        assert (picongpu_laguerre_modes is None
                and picongpu_laguerre_phases is None) or \
               (picongpu_laguerre_modes is not None
                and picongpu_laguerre_phases is not None), \
               "laguerre_modes and laguerre_phases MUST BE both set or " \
               "both unset"
        self.picongpu_laguerre_modes = picongpu_laguerre_modes
        self.picongpu_laguerre_phases = picongpu_laguerre_phases
        super().__init__(wavelength, waist, duration, **kw)

    def get_as_pypicongpu(self) -> laser.GaussianLaser:
        util.unsupported("laser name", self.name)
        util.unsupported("laser zeta", self.zeta)
        util.unsupported("laser beta", self.beta)
        util.unsupported("laser phi2", self.phi2)
        # unsupported: fill_in (do not warn, b/c we don't know if it has been
        # set explicitly, and always warning is bad)

        assert self.centroid_position[0] == self.focal_position[0] \
            and self.centroid_position[2] == self.focal_position[2], \
            "focal position x and z MUST be equal to centroid x and z"
        assert [0, 1, 0] == self.propagation_direction, \
            "only support propagation along Y axis"
        assert 0 == self.centroid_position[1], "centroid MUST have y=0"

        polarization_by_normal = {
            (1, 0, 0): laser.GaussianLaser.PolarizationType.LINEAR_X,
            (0, 0, 1): laser.GaussianLaser.PolarizationType.LINEAR_Z,
        }
        assert tuple(self.polarization_direction) in polarization_by_normal, \
            "only laser polarization [1, 0, 0] and [0, 0, 1] supported"

        pypicongpu_laser = laser.GaussianLaser()
        pypicongpu_laser.wavelength = self.wavelength
        pypicongpu_laser.waist = self.waist
        pypicongpu_laser.duration = self.duration
        pypicongpu_laser.focus_pos = self.focal_position[1]
        pypicongpu_laser.E0 = self.E0
        pypicongpu_laser.phase = 0
        pypicongpu_laser.pulse_init = 15
        pypicongpu_laser.init_plane_y = 0
        pypicongpu_laser.polarization_type = polarization_by_normal[
            tuple(self.polarization_direction)]

        if self.picongpu_laguerre_modes is None:
            pypicongpu_laser.laguerre_modes = [1.0]
        else:
            pypicongpu_laser.laguerre_modes = self.picongpu_laguerre_modes

        if self.picongpu_laguerre_phases is None:
            pypicongpu_laser.laguerre_phases = [0.0]
        else:
            pypicongpu_laser.laguerre_phases = self.picongpu_laguerre_phases

        return pypicongpu_laser
