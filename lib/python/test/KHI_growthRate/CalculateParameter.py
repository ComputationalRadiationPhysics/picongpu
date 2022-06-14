"""
This file is part of the PIConGPU.

Copyright 2022 PIConGPU contributors
Authors: Mika Soren Voss
License: GPLv3+
"""

import numpy as np


class CalculateParameter:
    """
    This class calculates essential quantities, such as the speed or the
    growth rate, for the KHI

    functions:
    -------
    calculateV_O(gamma):
        calculate the speed v

    calculateBeta(v_0)

    theoryPlasmafrequence_MI(density):
        calculate the plasmafrequency for the MI

    theoryPlasmafrequence_ESKHI(density, gamma):
        calculate the plasmafrequency for the ESKHI

    growthRate(f, time, interval = None):
        this function calculate the growthrate of a given quantity f(t)
    """

    # fixe parameter
    __c = 299792458
    __m_e = 9.1093837015e-31
    __e = -1.602176634e-19
    __e_0 = 8.8541878128e-12

    def __init__(self):
        pass

    @staticmethod
    def calculateV_O(gamma):
        """
        calculate the speed v and use

            gamma  = 1 / (1 - beta**2)**0.5

        input:
        -------
        gamma:   float or numpy.ndarray
                 lorentz factor

        return:
        -------
        v:       float or numpy.ndarray
                 speed given by gamma
        """

        return np.sqrt((1 - 1 / gamma**2) * CalculateParameter.__c ** 2)

    @staticmethod
    def calculateTime(steps, deltaT: float, frequency):
        """
        calculate the time in [frequency ** -1]

        input:
        -------
        steps:     list
                   list of number of timesteps

        deltaT:    float
                   time step of the simulation

        frequency: float

        return:
        -------
        time:      list
        """

        return steps * deltaT * frequency

    @staticmethod
    def calculateBeta(v_0):
        """
        calculate beta with

            beta = v_0 / c

        input:
        -------
        v_0:    float or numpy.ndarray
                speed

        return:
        -------
        beta:    float or numpy.ndarray

        """

        return v_0 / CalculateParameter.__c

    @staticmethod
    def theoryPlasmafrequence_MI(density):
        """
        calculate the plasmafrequency for the MI using the formula

            omega = sqrt(density * e ** 2 / ( eps_0 * m_e))

            e...electron charge
            eps_0...electric field constant
            m_e... mass electron

        input:
        -------
        density:  float

        return:
        -------
        omega:    float
                  plasmafrequency for the MI

        """

        return np.sqrt((density * CalculateParameter.__e**2) / (
               CalculateParameter.__e_0 * CalculateParameter.__m_e))

    @staticmethod
    def theoryPlasmafrequence_ESKHI(density, gamma):
        """
        calculate the plasmafrequency for the ESKHI using the formula

            omega = sqrt(density * e ** 2 / ( eps_0 * gamma * m_e))

            e...electron charge
            eps_0...electric field constant
            gamma...lorentz factor
            m_e... mass electron

        input:
        -------
        density:  float

        gamma:    float
                  lorentz factor

        return:
        -------
        omega:    float
                  plasmafrequency for the ESKHI

        """

        return np.sqrt((density * CalculateParameter.__e ** 2) /
                       (CalculateParameter.__e_0 * gamma *
                        CalculateParameter.__m_e))

    @staticmethod
    def growthRate(f, time, interval=None):
        """
        This function calculate the growthrate of a given quantity f(t)

        Gamma_f(t_k)= log(f(t_(k+1))/f(t_(k-1)))/t_(k+1)- t_(k-1)

        input:
        -------
        f:         array
                   quantity whose growth rate is to be determined

        time:      array
                   the array of the timesteps with the same size
                   as the quantity f

        interval:  [starttime, endtime], optional
                   determines the time interval in which the growth rate
                   is to be determined

        return:
        -------
        growth rate: array if interval = None
                     growth rate calculated using the above formula

                     tuple of (growthrate, start, stop)
                     growth rate calculated using the above formula and
                     the start and end of the interval
        """

        if interval is None:
            Gamma = 0.5 * np.log(f[3:]/f[1:-2])/(time[3:] - time[1:-2])
            return Gamma

        else:
            # calculate lower limit
            start = np.where(time < interval[0])[0][-1] + 1

            # calculate upper limit
            stop = np.where(time > interval[1])[0][0] - 1

            Gamma = 0.5 * np.log(f[start + 1:stop + 1] / f[start - 1:stop - 1]
                                 ) / (time[start + 1: stop + 1] -
                                      time[start - 1: stop - 1])

            return Gamma, start, stop
