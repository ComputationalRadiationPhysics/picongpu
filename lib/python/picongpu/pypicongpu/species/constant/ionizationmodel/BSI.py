"""
This file is part of PIConGPU.
Copyright 2024 PIConGPU contributors
Authors: Brian Edward Marre
License: GPLv3+
"""

from .ionizationmodel import IonizationModel
from ..ionizationcurrent import IonizationCurrent


class BSI(IonizationModel):
    """
    Barrier Suppression Ionization for hydrogen-like ions

    see for example: Delone, N. B.; Krainov, V. P. (1998).
      "Tunneling and barrier-suppression ionization of atoms and ions in a laser radiation field"
      doi:10.1070/PU1998v041n05ABEH000393

    Calculates the electric field strength limit necessary to overcome the binding energy of the electron to the
    core. If this limit exceed by the local electric field strength of an ion the ion is ionized.

    This model uses for naive inner electron charge shielding, assumes that the charge the electron 'feels' is equal to
    `proton number - number of inner shell electrons`, but neglects the Stark upshift of ionization energies.
    """

    PICONGPU_NAME: str = "BSI"
    """C++ Code type name of ionizer"""

    ionization_current: IonizationCurrent
    """ionization current implementation to use"""
