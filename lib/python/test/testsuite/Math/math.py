"""
This file is part of PIConGPU.

Copyright 2022-2023 PIConGPU contributors
Authors: Mika Soren Voss
License: GPLv3+

This module includes some commonly used math functions.
This module is not yet complete with this version.

Routines in this module:

growthRate(f, time, interval=None):
    This function calculate the growthrate of a given quantity f(t)
"""

__all__ = ["growthRate"]

import numpy as np


def growthRate(f, time, interval=None):
    """
    This function calculate the growthrate of a given quantity f(t)

    Gamma_f(t_k)= log(f(t_(k+1))/f(t_(k-1)))/(t_(k+1)- t_(k-1))

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
        # The initial value is zero, to avoid noise it is removed
        Gamma = 0.5 * np.log(f[3:] / f[1:-2]) / (time[3:] - time[1:-2])
        return Gamma

    else:
        # calculate lower limit
        start = np.where(time < interval[0])[0][-1] + 1

        # calculate upper limit
        stop = np.where(time > interval[1])[0][0] - 1

        Gamma = (
            0.5
            * np.log(f[start + 1 : stop + 1] / f[start - 1 : stop - 1])
            / (time[start + 1 : stop + 1] - time[start - 1 : stop - 1])
        )

    return Gamma, start, stop
