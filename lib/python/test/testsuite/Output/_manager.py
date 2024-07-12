"""
This file is part of PIConGPU.

Copyright 2022-2023 PIConGPU contributors
Authors: Mika Soren Voss
License: GPLv3+

This module can only be used in conjunction with Data.py.
Independent use is not intended.
Plots the given case and generates the log files.

ToDo: So far only the 1D plot is integrated in the plot.
"""

from . import Log
from . import Viewer
import config
import sys
import testsuite._checkData as cD
import testsuite.Math.physics as ph
from inspect import getmembers, isfunction


def __calculate(axis, parameter, *args):
    """
    calls possible functions for calculating the axis
    values and tries to calculate them
    """
    try:
        if axis == "plot_xaxis":
            return config.plot_xaxis(*args)
        elif axis == "plot_yaxis":
            return config.plot_yaxis(*args)
        else:
            return config.plot_zaxis(*args)

    except Exception:
        error1 = str(sys.exc_info()[1])
        for var in parameter:
            if var in error1.split(":")[-1]:
                args = args + (parameter[var],)
                return __calculate(axis, parameter, *args)


def getaxisvalues(parameter, axis: str = "plot_xaxis"):
    # check if variable has an allowed value
    if axis not in ["plot_xaxis", "plot_yaxis", "plot_zaxis"]:
        raise ValueError("variable has to be plot_xaxis, plot_yaxis" " or plot_zaxis")

    # check if a function is given
    if axis in [func[0] for func in getmembers(config, isfunction)]:
        values = __calculate(axis=axis, parameter=parameter)

    # otherwise it has to be a list
    else:
        values = cD.checkVariables(variable=axis, default="")

        # check if the value is a parameter
        if values in parameter.keys():
            values = parameter[values]

        # else a default value should be defined
        if values == "":
            frequency = ph.plasmafrequence()
            time = ph.calculateTimeFreq(frequency, step_direction="fields_energy.dat")
            values = time

    return values


def getInputparameter(parameter: dict) -> dict:
    param_Parameter = cD.checkExistVariables("param_Parameter")
    if param_Parameter:
        return {key: parameter[key] for key in config.param_Parameter}
    else:
        return None


def _output(direction, theory, simulation, max_diff, perc, acc_range, result, parameter):
    """
    generates the output. For this purpose, the plot is created
    first and then the log file is written
    """

    acceptance = cD.checkVariables(variable="acceptance")

    # plotter
    plot_type = cD.checkVariables(variable="plot_type", default="1D")
    inputparameter = getInputparameter(parameter)
    if plot_type == "1D":
        # get the values for the axis of the plot
        x_value = getaxisvalues(parameter)
        plot_log = cD.checkVariables(variable="plot_log", default="")
        if plot_log == "":
            plot_log = None

        # check that both have the same size
        if len(x_value) < len(simulation):
            simulation = simulation[: len(x_value)]
        elif len(x_value) > len(simulation):
            x_value = x_value[: len(simulation)]

        Viewer.plot_1D(
            theory,
            simulation,
            x_value,
            plotLog=plot_log,
            acceptance=acceptance,
            direction=direction,
        )
    else:
        Viewer.plot_2D()
    print(inputparameter)
    # resultlog
    value_sim = max(simulation)
    Log.resultLog(
        theory,
        value_sim,
        acceptance,
        perc,
        result,
        max_diff,
        direction=direction,
        inputparameter=inputparameter,
    )
