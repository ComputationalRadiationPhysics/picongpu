"""
# SPDX-FileCopyrightText: Brian Edward Marre
#
# SPDX-License-Identifier: GPL-3.0-or-later
"""

from ..groundstateionizationmodel import GroundStateIonizationModel
from ..... import pypicongpu

import typeguard


@typeguard.typechecked
class ThomasFermi(GroundStateIonizationModel):
    """thomas fermi ionization model"""

    MODEL_NAME: str = "ThomasFermi"

    def get_as_pypicongpu(self) -> pypicongpu.species.constant.ionizationmodel.IonizationModel:
        self.check()

        return pypicongpu.species.constant.ionizationmodel.ThomasFermi()
