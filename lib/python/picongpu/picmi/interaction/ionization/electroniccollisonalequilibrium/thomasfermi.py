"""
This file is part of PIConGPU.
Copyright 2024 PIConGPU contributors
Authors: Brian Edward Marre
License: GPLv3+
"""

from ..ionizationmodel import IonizationModel


class ThomasFermi(IonizationModel):
    """thomas fermi ionization model"""

    MODEL_NAME: str = "ThomasFermi"
