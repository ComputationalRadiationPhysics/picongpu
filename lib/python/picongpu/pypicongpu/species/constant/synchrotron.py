"""
# SPDX-FileCopyrightText: Julian Lenz
#
# SPDX-License-Identifier: GPL-3.0-or-later
"""

from typing import Any
from pydantic import BaseModel

from picongpu.pypicongpu.species.constant.constant import Constant


class FirstSynchrotronFunctionParams(BaseModel):
    """
    Parameters for computing the first synchrotron function.

    Corresponds to FirstSynchrotronFunctionParams struct in C++.
    """

    log_end: float = 7.0
    """
    log2(100.0), arbitrary cutoff for 2nd kind cyclic
    Bessel function -> function close enough to zero
    """

    num_sample_points: int = 8096
    """
    Number of sample points to use in integration in firstSynchrotronFunction.
    """


class InterpolationParams(BaseModel):
    """
    Parameters for precomputation of interpolation table.

    Corresponds to InterpolationParams struct in C++.
    """

    number_table_entries: int = 512
    """
    Number of synchrotron function values to precompute and store in table.
    """

    min_Zq_exponent: float = -50.0
    """
    In log2: -50 means minimum Zq that is still not 0 is 2^-50 ~ 10^-15.
    """

    max_Zq_exponent: float = 10.0
    """
    In log2: 10 means maximum Zq that is still not 0 is 2^10 ~ 10^+3.
    If set to larger value than 10: that can result in runtime error
    in precomputing cyclic Bessel function.
    """


class SynchrotronParams(BaseModel):
    """
    Synchrotron radiation.
    """

    electron_recoil: bool = True
    """
    Turn off or turn on the electron recoil from electrons generated.
    """

    min_energy: float | None = None
    """
    Energy high-pass filter: accept only photons with energy higher than this value.
    """

    first_synchrotron_function_params: FirstSynchrotronFunctionParams = FirstSynchrotronFunctionParams()
    """
    Parameters for computing the first synchrotron function.
    """

    interpolation_params: InterpolationParams = InterpolationParams()
    """
    Parameters for precomputation of interpolation table.
    """

    supress_requirement_warning: bool = False
    """
    If true, the warning for requirement 1 and 2 is suppressed.

    This may speed the simulation a little bit because there is no call to global memory.

    This warning means that the probability of generating a photon is high for given dt
    (higher than 10%) - this means we generate photons possibly every timestep
    (numerical artefacts) and the radiation is underestimated if probability is greater than 1.
    The timestep should be reduced.
    """


class SynchrotronConstant(Constant):
    photon_species: Any
