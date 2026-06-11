"""
# SPDX-FileCopyrightText: Brian Edward Marre
#
# SPDX-License-Identifier: GPL-3.0-or-later
"""

from .ionizationmodel import IonizationModel
from ..ionizationcurrent import IonizationCurrent


class BSIStarkShifted(IonizationModel):
    """
    Barrier Suppression Ionization for hydrogen-like ions, accounting for stark upshift of ionization energies

    see BSI.py for further information

    Variant of the BSI ionization model accounting for the Stark upshift of ionization energies.
    """

    ionizer_picongpu_name: str = "BSIStarkShifted"
    """C++ Code type name of ionizer"""

    ionization_current: IonizationCurrent
    """ionization current implementation to use"""
