"""
This file is part of the PIConGPU.
Copyright 2023 PIConGPU contributors
Authors: Kristin Tippey, Brian Edward Marre
License: GPLv3+
"""

from .densityprofile import DensityProfile
from .... import util
from typeguard import typechecked

from .plasmaramp import PlasmaRamp


@typechecked
class Foil(DensityProfile):
    """
    Directional density profile with thickness and pre- and
    post-plasma lengths and cutoffs
    """

    density_si = util.build_typesafe_property(float)
    """density at every point in space (kg * m^-3)"""

    y_value_front_foil_si = util.build_typesafe_property(float)
    """position of the front of the foil plateau (m)"""

    thickness_foil_si = util.build_typesafe_property(float)
    """thickness of the foil plateau (m)"""

    pre_foil_plasmaRamp = util.build_typesafe_property(PlasmaRamp)
    """pre(lower y) foil-plateau ramp of density"""
    post_foil_plasmaRamp = util.build_typesafe_property(PlasmaRamp)
    """post(higher y) foil-plateau ramp of density"""

    def __init__(self):
        # (nothing to do, overwrite from abstract parent)
        pass

    def check(self) -> None:
        if self.density_si <= 0:
            raise ValueError("density must be > 0")
        if self.y_value_front_foil_si < 0:
            raise ValueError("y-value_front must be >= 0")
        if self.thickness_foil_si < 0:
            raise ValueError("thickness must be >= 0")
        self.pre_foil_plasmaRamp.check()
        self.post_foil_plasmaRamp.check()

    def _get_serialized(self) -> dict:
        self.check()

        return {
            "density_si": self.density_si,
            "y_value_front_foil_si": self.y_value_front_foil_si,
            "thickness_foil_si": self.thickness_foil_si,
            "pre_foil_plasmaRamp":
                self.pre_foil_plasmaRamp
                    .get_generic_profile_rendering_context(),
            "post_foil_plasmaRamp":
                self.post_foil_plasmaRamp
                    .get_generic_profile_rendering_context()
            }
