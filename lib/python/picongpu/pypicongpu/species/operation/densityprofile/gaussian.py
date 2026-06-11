"""
# SPDX-FileCopyrightText: Brian Edward Marre, Masoud Afshari, Julian Lenz
#
# SPDX-License-Identifier: GPL-3.0-or-later
"""

from typing import Annotated, Literal
from pydantic import AfterValidator, BaseModel, Field, model_validator


def neq_0(value):
    if value == 0:
        raise ValueError("value is not allowed to be 0.")
    return value


class Gaussian(BaseModel):
    """
    gaussian density profile

    density=
    - for y < gasCenterFront;   density * exp(gasFactor * (abs( (y - gasCenterFront) / gasSigmaFront))^gasPower)
    - for gasCenterFront >= y >= gasCenterRear; density
    - for gasCenterRear < y;    density * exp(gasFactor * (abs( (y - gasCenterRear) / gasSigmaRear))^gasPower)
    """

    type_gaussian: Literal[True] = True

    gas_center_front: float = Field(ge=0.0, alias="center_front")
    """position of the front edge of the constant middle of the density profile, [m]"""

    gas_center_rear: float = Field(ge=0.0, alias="center_rear")
    """position of the rear edge of the constant middle of the density profile, [m]"""

    gas_sigma_front: Annotated[float, AfterValidator(neq_0)] = Field(alias="sigma_front")
    """distance from gasCenterFront until the gas density decreases to its 1/e-th part, [m]"""

    gas_sigma_rear: Annotated[float, AfterValidator(neq_0)] = Field(alias="sigma_rear")
    """distance from gasCenterRear until the gas density decreases to its 1/e-th part, [m]"""

    gas_factor: float = Field(lt=0.0, alias="factor")
    """exponential scaling factor, see formula above"""

    gas_power: Annotated[float, AfterValidator(neq_0)] = Field(alias="power")
    """power-exponent in exponent of density function"""

    vacuum_cells_front: int = Field(ge=0)
    """number of vacuum cells in front of foil for laser init"""

    density: float = Field(gt=0.0)
    """particle number density in m^-3"""

    @model_validator(mode="after")
    def check(self):
        if self.gas_center_rear < self.gas_center_front:
            raise ValueError("gas_center_rear must be >= gas_center_front")
        return self
