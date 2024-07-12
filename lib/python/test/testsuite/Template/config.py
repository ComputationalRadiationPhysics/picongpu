"""
This file is part of PIConGPU.

Copyright 2022-2023 PIConGPU contributors
Authors: Mika Soren Voss
License: GPLv3+

/ToDo: Description of the test case
"""

# uncomment if needed
# import sys
# sys.path.insert(1, "../../")
# import testsuite as ts

# general information about the test
title = "Title"
author = "Author"

# directory specified
resultDirection = None
dataDirection = None
paramDirection = None
openPMDDirection = None
jsonDirection = None  # if None paramDirection will be used
step_direction = None

# parameter information found in .param files
param_Parameter = []
data_Parameter = []
json_Parameter = []

resultTitle = None

# acceptance in percentage
acceptance = None

# plot data
plot_xlabel = None
plot_ylabel = None
plot_log = None  # should be x,y,xy,None or not defined
# if None or not defined the standard type will be used, see documentation
plot_type = None
# if None or not defined the time will be used
plot_xaxis = None
# for more values see the documentation (e.g. 2D plot needs zaxis and yaxis)


def theory(**kwargs):
    """
    this function indicates how the theoretical values
    can be calculated from the data. Please complete this
    function and use only the theoretical values
    as the return value.

    All parameters that are read from the test-suite must
    be given the same names as in the parameter lists.

    Return:
    -------
    out : theoretical values
    """

    # \ToDo: Fill in the calculation of the theory
    return None


def simData(**kwargs):
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

    # \ToDo: Fill in the Calculation

    return None
