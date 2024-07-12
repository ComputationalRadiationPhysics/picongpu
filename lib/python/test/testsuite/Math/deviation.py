"""
This file is part of PIConGPU.

Copyright 2022-2023 PIConGPU contributors
Authors: Mika Soren Voss
License: GPLv3+

This module contains some auxiliary functions to determine the
deviation between theory and simulation. It is not yet complete
in the present version and will be expanded over time.

Routines in this module:

getDifference(theory, simulation)->float or array:
    calculates the difference between the theory value and
    the simulation value

getMaxDifference(theory, simulation):
    calculates the maximum deviation between theory and simulation

getDifferenceInPercentage(theory, simulation):
    calculates the difference between the theory value and
    the simulation value in percentage

getAcceptanceRange(theory, acceptance:float = None):
    represents the range around the theoretical value,
    which as a limit leads to the acceptance of the test

getTestResult(theory, simulation, acceptance:float = None) -> bool:
    determines whether a test passes or fails based on the
    acceptance range
"""

__all__ = [
    "getDifference",
    "getMaxDifference",
    "getDifferenceInPercentage",
    "getAcceptanceRange",
    "getTestResult",
]

import testsuite._checkData as cD
import numpy as np


def getMaxDifference(theory, simulation):
    """
    calculates the maximum deviation between theory and simulation

    Calculates:
    --------
    max(abs(theory -simulation))

    Input:
    -------
    theory :     float or array
                 theoretical value
                 Note: if list and the values
                 of the simulations are also a list, both must have
                 the same length

    simulation : float or array
                 Values from the simulation
                 Note: if the list and the values
                 of the theory are also a list,
                 both must have the same length

    Return:
    -------
    out : The maximum in the deviation
    """

    difference = np.abs(theory - simulation)

    return max(difference)


def getMinDifference(theory, simulation):
    """
    calculates the minimum deviation between theory and simulation

    Calculates:
    --------
    min(abs(theory - simulation))

    Input:
    -------
    theory :     float or array
                 theoretical value
                 Note: if list and the values
                 of the simulations are also a list, both must have
                 the same length

    simulation : float or array
                 Values from the simulation
                 Note: if the list and the values
                 of the theory are also a list,
                 both must have the same length

    Return:
    -------
    out : The minimum in the deviation
    """

    difference = np.abs(min(theory - simulation))

    return difference


def getDifference(theory, simulation):
    """
    calculates the absolute difference between the theory value and
    the simulation value

    Calculates:
    --------
    abs(theory -simulation)

    Input:
    -------
    theory :     float or array
                 theoretical value
                 Note: if list and the values
                 of the simulations are also a list, both must have
                 the same length

    simulation : float or array
                 Values from the simulation
                 Note: if the list and the values
                 of the theory are also a list,
                 both must have the same length

    Return:
    -------
    out : Array or float
          the absolute difference between the two values
    """

    return np.abs(theory - simulation)


def getDifferenceInPercentage(theory, simulation):
    """
    calculates the difference between the theory value and
    the simulation's maximum value in percentage

    Calculates:
    --------
    (theory - max(simulation)) / simulation * 100

    Input:
    -------
    theory :     float or array
                 theoretical value
                 Note: if list and the values
                 of the simulations are also a list, both must have
                 the same length

    simulation : float or array
                 Values from the simulation
                 Note: if the list and the values
                 of the theory are also a list,
                 both must have the same length

    Return:
    -------
    out : Array or float
          the difference between the two values in percentage

    """

    simMax = max(simulation)
    return (theory - simMax) / theory * 100


def getAcceptanceRange(theory, acceptance: float = None):
    """
    represents the range around the theoretical value,
    which as a limit leads to the acceptance of the test

    Input:
    --------

    theory :     float or array
                 theoretical value

    acceptance : float, optional
                 acceptance percentage. If None, the value must have
                 been specified in Data.py

    Return:
    -------
    out : float or array
    """

    acceptance = cD.checkVariables(variable="acceptance", parameter=acceptance)

    return theory * (1 - acceptance), theory * (1 + acceptance)


def getTestResult(theory, simulation, acceptance: float = None) -> bool:
    """
    determines whether a test passes or fails based on the acceptance range

    Input:
    -------

    theory :     float or array
                 theoretical value
                 Note: if list and the values
                 of the simulations are also a list, both must have
                 the same length

    simulation : float or array
                 Values from the simulation
                 Note: if the list and the values
                 of the theory are also a list,
                 both must have the same length

    acceptance : float, optional
                 acceptance percentage. If None, the value must have
                 been specified in Data.py

    Use:
    -------
    getDifferenceInPercentage(theory, simulation)

    Return:
    -------
    out : bool
          true if the test passed
    """

    acceptance = cD.checkVariables(variable="acceptance", parameter=acceptance)

    if abs(getDifferenceInPercentage(theory, simulation)) <= acceptance * 100:
        return True
    else:
        return False
