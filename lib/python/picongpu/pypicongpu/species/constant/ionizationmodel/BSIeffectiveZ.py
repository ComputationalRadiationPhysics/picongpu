"""
# SPDX-FileCopyrightText: Brian Edward Marre
#
# SPDX-License-Identifier: GPL-3.0-or-later
"""

from .ionizationmodel import IonizationModel
from ..ionizationcurrent import IonizationCurrent


class BSIEffectiveZ(IonizationModel):
    """
    Barrier Suppression Ionization for hydrogen-like ions, using tabulated Z_effective values

    see BSI.py for further information

    Variant of the BSI ionization model using tabulated Z_effective values instead of the naive inner electron charge
    shielding, but still neglecting the Stark upshift of ionization energies.
    """

    ionizer_picongpu_name: str = "BSIEffectiveZ"
    """C++ Code type name of ionizer"""

    ionization_current: IonizationCurrent
    """ionization current implementation to use"""
