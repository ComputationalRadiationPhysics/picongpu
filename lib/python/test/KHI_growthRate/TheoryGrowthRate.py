"""
This file is part of PIConGPU.

Copyright 2022 PIConGPU contributors
Authors: Mika Soren Voss
License: GPLv3+
"""

import numpy as np
from scipy.constants import c


class TheoryGrowthRate:
    """
    this class provides functions for calculating and estimating the growth
    rate according to the theory

    functions:
    -------
    getTheoryESKHI_max(gamma):
        calculate the maximum growth rate of the electron scaled
        KHI(ESKHI)

    getTheoryMIGR_max(gamma):
        calculate the maximum growth rate of the mushroom
        instability(MI)

    estimationPredominatesGrowthRate(gamma, fields = None):
        estimates the predominant factor in the growth rate
    """

    def __init__(self):
        pass

    @staticmethod
    def getTheoryESKHI_max(gamma):
        """
        calculate the maximum growth rate of the electron scaled
        KHI(ESKHI)

        input:
        -------
        gamma:  float
                lorentz factor

        return:
        -------
        Lambda: float
                maximum growth rate of the ESKHI
        """

        return 1 / (np.sqrt(8) * gamma)

    @staticmethod
    def getTheoryMIGR_max(gamma):
        """
        calculate the maximum growth rate of the mushroom
        instability(MI)

        input:
        -------
        gamma:  float
                lorentz factor

        return:
        -------
        Lambda:  float
                 maximum growth rate of the MI
        """

        v = np.sqrt((1 - 1 / gamma ** 2) * c ** 2)

        return (v / (c * np.sqrt(gamma)))

    @staticmethod
    def estimationPredominatesGrowthRate(gamma, fields=None):
        """
        estimates the predominant factor in the growth rate

        input:
        -------
        gamma:  float

        fields: dictionary, optional, None
                dictionary of all field data with the structure(example)
                {"B_x": None, "B_y": [...], ...}(None if direction is
                negligible, key = (total, B_x, B_y, B_z, E_x, E_y, E_z))

                if fields = None, a 3d simulation is assumed

        return:
        -------
        predominateGrowthrate: tuple (str, float)
                               tuple contains a string which effect
                               predominates and the effect factor as
                               a float
        """

        mi = TheoryGrowthRate.getTheoryMIGR_max(gamma)
        eskhi = TheoryGrowthRate.getTheoryESKHI_max(gamma)

        if fields is not None and fields.get("E_z") is None:
            return "The ESKHI predominates with {}".format(eskhi), eskhi

        elif fields is not None and (fields.get("E_x") is None or
                                     fields.get("E_y") is None):
            return "The MI predominates with {}".format(mi), mi

        elif mi < eskhi:
            return "The ESKHI predominates with {}".format(eskhi), eskhi

        else:
            return "The MI predominates with {}".format(mi), mi
