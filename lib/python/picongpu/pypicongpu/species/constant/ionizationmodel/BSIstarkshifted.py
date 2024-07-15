"""
This file is part of PIConGPU.
Copyright 2024 PIConGPU contributors
Authors: Brian Edward Marre
License: GPLv3+
"""

from .ionizationmodel import IonizationModel
from ..ionizationcurrent import IonizationCurrent


class BSIStarkShifted(IonizationModel):
    """
    Barrier Suppression Ionization for hydrogen-like ions, accounting for stark upshift of ionization energies

    see BSI.py for further information

    Variant of the BSI ionization model accounting for the Stark upshift of ionization energies.
    """

    PICONGPU_NAME: str = "BSI"
    """C++ Code type name of ionizer"""

    ionization_current: IonizationCurrent
    """ionization current implementation to use"""
