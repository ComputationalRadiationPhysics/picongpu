# SPDX-FileCopyrightText: PIConGPU contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""
This file is part of PIConGPU.
Copyright 2021-2024 PIConGPU contributors
Authors: Hannes Troepgen, Brian Edward Marre, Masoud Afshari
License: GPLv3+
"""

from typing import Literal

from pydantic import BaseModel, Field

from ..species import Species


class SetChargeState(BaseModel):
    """
    assigns boundElectrons attribute and sets it to the initial charge state

    used for ionization of ions
    """

    species: Species
    """species which will have boundElectrons set"""

    charge_state: int = Field(ge=0)
    """initial ion charge state"""

    type_setchargestate: Literal[True] = True
