"""
This file is part of PIConGPU.

Copyright 2022-2023 PIConGPU contributors
Authors: Mika Soren Voss
License: GPLv3+

This test allows verification of the MI. It is irrelevant whether
it is a 2D or 3D simulation. With a 3D simulation, however, attention
must be paid to the Lorentz factor (for more information, see the
bachelor thesis, for example, submitted by Mika Soren Voss).

For information:
    according to the documentation, the case distinction will
    only be incorporated in a later version, which means that
    two cases of KHI currently have to be treated differently.
"""

from scipy.signal import argrelextrema
from scipy.constants import c
import numpy as np
import os
import sys

sys.path.insert(1, os.path.dirname(os.path.abspath(__file__)) + "/../../")
import testsuite as ts  # noqa

# general information about the test
title = "KHI Growthrate (2D MI)"
author = "Mika Soren Voss"

# directory specified
direction = None
resultDirection = None
paramDirection = None
dataDirection = None
cmakeDirection = None

# parameter information found in .param files
param_Parameter = ["DIMENSION", "gamma", "BASE_DENSITY_SI", "DELTA_T_SI"]
data_Parameter = ["Bx", "step"]

# acceptance, ratio to 1
acceptance = 0.2

# plot data
plot_xlabel = r"$t[\omega_{pe}^ {-1}]$"
plot_ylabel = r"$\Gamma_\mathrm{Fi}$"
# if None or not defined the standard type will be used, see documentation
plot_type = None
# if None or not defined the time will be used
plot_xaxis = None
# for more values see the documentation (e.g. 2D plot needs zaxis and yaxis)


def theory(gamma, **kwargs):
    """
    this function indicates how the theoretical values
    can be calculated from the data. Please complete this
    function and use only the theoretical values
    as the return value.

    All parameters that are read from the test-suite must
    be given the same names as in the parameter lists.

    Return:
    -------
    out : theoretical values!
    """
    # gamma is calculated automatically, does not have to be passed
    v = ts.Math.physics.calculateV_O(gamma)
    return v / (c * np.sqrt(gamma))


def simData(Bx, **kwargs):
    """
    this function indicates how the values from the simulation
    can be calculated from the data. Please complete this
    function and use only the values from the simulation
    as the return value.

    All parameters that are read from the test-suite must
    be given the same names as in the parameter lists.

    Return:
    -------
    out : values from the simulation!
    """
    frequency = ts.Math.physics.plasmafrequence(relativistic=False)
    time = ts.Math.physics.calculateTimeFreq(frequency, step_direction="fields_energy.dat")

    sim_values = ts.Math.math.growthRate(Bx, time)
    sim_values = sim_values[argrelextrema(sim_values, np.less)[0][0] :]

    return sim_values
