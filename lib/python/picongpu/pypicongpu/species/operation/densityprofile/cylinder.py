"""
# SPDX-FileCopyrightText: Kristin Tippey, Brian Edward Marre
#
# SPDX-License-Identifier: GPL-3.0-or-later
"""

from math import sqrt
from typing import Annotated, Literal

from pydantic import BaseModel, BeforeValidator, Field, model_validator

from .plasmaramp import AllPlasmaRamps, None_


class _Component(BaseModel):
    component: float

    def __eq__(self, other):
        if isinstance(other, float) or isinstance(other, int):
            return self.component == other
        return super().__eq__(other)


def validate_component_vector(value):
    try:
        return [_Component(component=c) for c in value]
    except Exception:
        return value


class Cylinder(BaseModel):
    """
    Describes a cylindrical density distribution of particles with gaussian up-ramp
    with a constant density region in between. It can have an arbitrary orientation
    and position in space.

    Will create the following profile:
      n = density if r < reduced_radius
      n is 0 or follows the exponential ramp if r > reduced_radius
      n is 0 if r > reduced_radius + prePlasmaCutoff
      the reduced_radius is equal = @f[\\sqrt{R^2 -L^2} -L @f]
      with R - cylinder_radius and L - prePlasmaLength (scale length of the ramp)
      the reduced radius ensures mass conservation
    """

    type_cylinder: Literal[True] = True

    density_si: float = Field(gt=0.0)
    """particle number density at at the foil plateau (m^-3)"""

    center_position_si: Annotated[tuple[_Component, _Component, _Component], BeforeValidator(validate_component_vector)]
    """center of the cylinder [x, y, z], [m]"""

    radius_si: float
    """cylinder radius, [m]"""

    cylinder_axis: Annotated[tuple[_Component, _Component, _Component], BeforeValidator(validate_component_vector)]
    """cylinder axis [x, y, z], [unitless]"""

    # This still relies on some magic to insert the typeID.
    # We'll handle it another time:
    pre_plasma_ramp: AllPlasmaRamps = None_()
    """pre plasma ramp"""

    @model_validator(mode="after")
    def check(self):
        min_radius = sqrt(2.0) * self.pre_plasma_ramp.PlasmaLength if type(self.pre_plasma_ramp) is not None_ else 0.0
        if self.radius_si < min_radius:
            raise ValueError(
                f"radius must be > sqrt(2)*pre_plasma_length = {min_radius}, so that the reduced radius stays non negative. In case of no preplasma radius must be >= 0.0."
            )
        return self
