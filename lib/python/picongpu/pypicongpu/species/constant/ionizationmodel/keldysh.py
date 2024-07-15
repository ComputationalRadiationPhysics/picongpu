"""
This file is part of PIConGPU.
Copyright 2024 PIConGPU contributors
Authors: Brian Edward Marre
License: GPLv3+
"""

from .ionizationmodel import IonizationModel
from ..ionizationcurrent import IonizationCurrent


class Keldysh(IonizationModel):
    """
    Keldysh multi photon ionization

    see for example: D. Bauer and P. Mulser(1999)
      "Exact field ionization rates in the barrier-suppression regime from numerical time-dependent
      Schroedinger-equation calculations"
      Physical Review A, 59(1):569+, January 1999

    @attention this model is derived for near constant fields and may give erroneous predictions for rapidly changing
        high intensity laser fields.
    """

    PICONGPU_NAME: str = "Keldysh"
    """C++ Code type name of ionizer"""

    ionization_current: IonizationCurrent
    """ionization current implementation to use"""
