"""
# SPDX-FileCopyrightText: Brian Edward Marre
#
# SPDX-License-Identifier: GPL-3.0-or-later
"""

from .ionizationmodel import IonizationModel
from ..ionizationcurrent import IonizationCurrent


class ADKCircularPolarization(IonizationModel):
    """
    Ammosov-Delone-Krainov tunnelling ionization for hydrogenlike atoms model -- circular polarization

     see for example: Delone, N. B.; Krainov, V. P. (1998).
       "Tunneling and barrier-suppression ionization of atoms and ions in a laser radiation field"
       doi:10.1070/PU1998v041n05ABEH000393

    @attention this model is derived for near constant fields and may give erroneous predictions for rapidly changing
        high intensity laser fields.
    """

    ionizer_picongpu_name: str = "ADKCircPol"
    """C++ Code type name of ionizer"""

    ionization_current: IonizationCurrent
    """ionization current implementation to use"""
