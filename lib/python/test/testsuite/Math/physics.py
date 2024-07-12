"""
This file is part of PIConGPU.

Copyright 2022-2023 PIConGPU contributors
Authors: Mika Soren Voss
License: GPLv3+

This module provides frequently used functions and laws of physics.
Please note that this version is not complete and will be
expanded over time.

Routines in this module:

calculateV_O(gamma = None):
    calculate the speed v and use
    beta = v/c = sqrt(1-1/gamma**2)

calculateTimeFreq(frequency, steps=None, deltaT:float = None, **kwargs):
    calculate the time in [frequency ** -1]

calculateTime(steps = None, deltaT: float = None):
    calculate the time in s

calculateBeta(v_0 = None)

plasmafrequence(density = None, gamma:float = None,
                    relativistic:bool = True):

    calculate the plasmafrequency  using the formula

        omega = sqrt(density * e ** 2 / ( eps_0 * m_e))

        e...electron charge
        eps_0...electric field constant
        m_e... mass electron

    with parameter relativistic = True:
        omega = sqrt((density * e ** 2) /(eps_0 * gamma * m_e)
"""

__all__ = [
    "calculateV_O",
    "calculateTime",
    "calculateTimeFreq",
    "plasmafrequence",
    "calculateBeta",
]

import numpy as np
from scipy.constants import c, epsilon_0, e, m_e
from . import _searchData as sD


def calculateV_O(gamma=None):
    """
    calculate the speed v and use

    gamma  = 1 / (1 - beta**2)**0.5

    Input:
    -------
    gamma :   float or numpy.ndarray, optional
              Lorentz factor
              If None, gamma is searched for in the parameter files.

    Return:
    -------
    out :    float or numpy.ndarray
             speed given by gamma
    """

    if gamma is None:
        gamma = sD.searchParameter("gamma", directiontype="param")

    return np.sqrt((1 - 1 / gamma**2) * c**2)


def calculateTimeFreq(frequency, steps=None, deltaT: float = None, **kwargs):
    """
    calculate the time in [frequency ** -1]

    Input:
    -------
    steps :     list, optional
                list of number of timesteps
                if None, steps are searched for in the .dat files

    deltaT :    float, optional
                time step of the simulation
                if None, deltaT is searched for in the parameter files

    frequency : float
                plasma frequency

    Return:
    -------
    out : list
          time
    """

    if steps is None:
        steps = sD.searchParameter("step", directiontype="dat", **kwargs)

    if deltaT is None:
        deltaT = sD.searchParameter("DELTA_T_SI", directiontype="param")
    return steps * deltaT * frequency


def calculateTime(steps=None, deltaT: float = None):
    """
    calculate the time in s

    Input:
    -------
    steps :     list
                list of number of timesteps
                if None, steps are searched for in the .dat files

    deltaT :    float
                time step of the simulation
                if None, deltaT is searched for in the parameter files

    Return:
    -------
    out : list
          time
    """
    if steps is None:
        steps = sD.searchParameter("step", directiontype="dat")

    if deltaT is None:
        deltaT = sD.searchParameter("DELTA_T_SI", directiontype="param")

    return steps * deltaT


def calculateBeta(v_0=None, gamma=None):
    """
    calculate beta with

        beta = v_0 / c

    or

        beta = sqrt(1-1 / gamma**2)

    Input:
    -------
    v_0:    float or numpy.ndarray
            speed
    gamma:  float
            Lorentz factor

    Use:
    -------
    calculateV_O() if v_0 is None

    Return:
    -------
    beta:    float or numpy.ndarray
    """

    if gamma is not None:
        beta = np.sqrt(1 - 1 / gamma**2)
    elif v_0 is not None:
        beta = v_0 / c
    elif gamma is None:
        gamma_n = sD.searchParameter("gamma", directiontype="param")
        beta = np.sqrt(1 - 1 / gamma_n**2)
    else:
        v_0 = calculateV_O()
        beta = v_0 / c

    return beta


def plasmafrequence(density=None, gamma: float = None, relativistic: bool = True):
    """
    calculate the plasmafrequency  using the formula

        omega = sqrt(density * e ** 2 / ( eps_0 * m_e))

        e...electron charge
        eps_0...electric field constant
        m_e... mass electron

    with parameter relativistic = True:
        omega = sqrt((density * e ** 2) /(eps_0 * gamma * m_e)

    Input:
    -------
    density :      float, optional
                   If None, density is searched for in the parameter files.

    gamma :        float, optional
                   Lorentz factor
                   If None, gamma is searched for in the parameter files.

    relativistic : bool, optional
                   default : True
                   Determines whether the plasma frequency should be
                   calculated relativistically corrected (if True)

    Return:
    -------
    omega:    float
              plasmafrequency for the MI

    """

    if density is None:
        density = sD.searchParameter("BASE_DENSITY_SI", directiontype="param")

    if relativistic:
        if gamma is None:
            gamma = sD.searchParameter("gamma", directiontype="param")
        return np.sqrt((density * e**2) / (epsilon_0 * gamma * m_e))
    else:
        return np.sqrt((density * e**2) / (epsilon_0 * m_e))
