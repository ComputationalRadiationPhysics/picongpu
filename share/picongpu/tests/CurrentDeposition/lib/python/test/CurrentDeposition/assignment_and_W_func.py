"""
This file is part of PIConGPU.

Copyright 2023 PIConGPU contributors
Authors: Hannes Wolf
License: GPLv3+

Implements the assignment functions of order 0 to 3 as well as a function to
calculate the current deposition vector.

"""


# functions for single values
def NGP(x):
    """Generates hat function with width 1 (symmetrical around x=0) -
    assignment function of order 0 for single values x.

    Parameters:
    x (float): distance between particle and evaluation point

    Returns:
    y (float): value of the hat function NGP(x)

    """

    if abs(x) < 1 / 2:
        y = 1
    elif abs(x) == 1 / 2:
        y = 1 / 2
    else:
        y = 0
    return y


def CIC(x):
    """Generates triangle with width 2 (symmetrical around x=0) -
    assignment function of order 1 for single values x.

    Parameters:
    x (float): distance between particle and evaluation point

    Returns:
    y (float): value of the triangle function CIC(x)

    """

    if abs(x) < 1:
        y = 1 - abs(x)
    else:
        y = 0
    return y


def TSC(x):
    """assignment function of order 2 for single values x.

    Parameters:
    x (float): distance between particle and evaluation point

    Returns:
    y (float): value of the function TSC(x)

    """

    if abs(x) < 3 / 2 and abs(x) >= 1 / 2:
        y = ((3 / 2 - abs(x)) ** 2) / 2
    elif abs(x) < 1 / 2:
        y = -(x**2) + 3 / 4
    else:
        y = 0
    return y


def PQS(x):
    """assignment function of order 3 for single values x.

    Parameters:
    x (float): distance between particle and evaluation point

    Returns:
    y (float): value of the function PQS(x)

    """
    if abs(x) < 2 and abs(x) >= 1:
        y = 1 / 6 * (2 - abs(x)) ** 3
    elif abs(x) < 1:
        y = 2 / 3 - x**2 + (abs(x) ** 3) / 2
    else:
        y = 0
    return y


def W(s1, s2, s3, s4, s5, s6):
    """Calculation of the current deposition (one component).
    The Parameters s1..s6 are just placeholders in this case. The values
    to calculate the respective W_x, W_y, W_z are inserted in grid_class.py
    appropriately.
    For the explicit defenition see e.g. : EZ: An Efficient, Charge
    Conserving Current Deposition Algorithm for Electromagnetic
    Particle-In-Cell Simulations (https://doi.org/10.1016/j.cpc.2023.108849.
    ;https://www.sciencedirect.com/science/article/pii/S0010465523001947)

    Parameters:
    s1, s2, ..., s6(float): values of the assignment function at the old and
                            the new coordinates

    Returns:
    W(float): component of the current deposition vector with respect to the
              choice of the si

    """

    W = 1 / 3 * (s4 * s5 * s6 - s1 * s5 * s6 + s4 * s2 * s3 - s1 * s2 * s3) + 1 / 6 * (
        s4 * s2 * s6 - s1 * s2 * s6 + s4 * s5 * s3 - s1 * s5 * s3
    )
    return W
