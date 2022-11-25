"""
This file is part of PIConGPU.

Copyright 2022 PIConGPU contributors.
Authors: Mika Soren Voss
License: GPLv3+
"""

import numpy as np


class DifferenceKHIGrowthRate:
    """
    this class contains functions for calculating deviations
    betwenn the theoretical prediction and the values from the
    simulation

    functions:
    -------
    getMaxDifference(growthRate, maxgrowthRate)
        calculate the max difference of a field or value and the
        theory

    getDifference(theory, simulation)
        calculate  the difference of a field or value and the
        theory

    getDifferenceInPercentage(theoryMax: float, simMax)
        calculate the max difference in percentage
    """

    def __init__(self):
        pass

    @staticmethod
    def getMaxDifference(growthRate, maxGrowthRate):
        """
        return the difference of a field or value and the theory

        input:
        -------
        growthRate:     float, array
        maxGrowthRate:  float
                        theory value

        return:
        -------
        difference: float
        """

        difference = np.abs(growthRate - maxGrowthRate)

        return min(difference)

    @staticmethod
    def getDifference(theory, simulation):
        """
        calculate the difference between a given value or field and
        the theoretical prediction

        input:
        -------
        theory: float
                   theory value

        simulation:    float or array
                   value/field from simulation

        return:
        -------
        difference: float
        """

        return np.abs(theory - simulation)

    @staticmethod
    def getDifferenceInPercentage(theoryMax: float, simMax):
        """
        input:
        -------
        theoryMax:      float
                        theory value
        simMax:            float
                        maximum from simulation

        return:
        -------
        diff_perc:      float
                        difference in percentage between theory and simulation
        """

        simMax = max(simMax)
        return (theoryMax - simMax) / theoryMax * 100
