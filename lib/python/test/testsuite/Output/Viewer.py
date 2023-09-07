"""
This file is part of PIConGPU.

Copyright 2022-2023 PIConGPU contributors
Authors: Mika Soren Voss
License: GPLv3+
"""

__all__ = ["plot_1D", "plot_2D"]

import matplotlib.pyplot as plt
import testsuite._checkData as cD
import matplotlib
matplotlib.use("Agg")


def plot_1D(theory,
            simulation,
            x_value,
            xlabel: str = None,
            ylabel: str = None,
            title: str = None,
            plotLog: str = None,
            acceptance: float = None,
            savefig: bool = True,
            direction: str = None):
    """
    Input:
    -------
    theory :    array, tuple or value
                describes the theoretical value.
                If tuple: (x value, y value), plot only that value
                          as a point.
                If float: The theoretical value is plotted as a line.
                If array: must have the same length as x_value.

    simulation : array, tuple or value
                describes the value from the simulation.
                If tuple: (x value, y value), plot only that value
                          as a point.
                If float: The value from the simulation is plotted
                          as a line.
                If array: must have the same length as x_value.

    savefig :   bool, optional
                if True the generated plot is saved as testresult_1D.png,
                otherwise only displayed.
                Default: True

    direction : str, optional
                The directory in which the plot should be saved.
                If None or set in config.py, the value from config.py is
                used.
                Default: None

    xlabel/ylabel : str, optional
                Axis labeling of the x or y axis.
                If None or set in config.py, the value from config.py is
                used.
                Default: None ("No Title" is displayed)

    title :     str, optional
                title of the Plot
                If None or set in config.py, the value from config.py is
                used.
                Default: None ("No Title" is displayed)

    acceptance : float, optional
                 maximum deviation from the theoretical value in
                 percent
                 Default: None

    plotLog :   str, optional
                plots individual axes logarithmically ("y", "x" or "yx")
                Default: None
    """
    direction = cD.checkDirection(variable="resultDirection",
                                  direction=direction)
    # If only one value is given, a horizontal line is plotted
    # as the theoretical value
    if isinstance(theory, int) or isinstance(theory, float):
        plt.axhline(theory, color="k", label="theory")
        if acceptance is not None:
            offset = theory * acceptance
            plt.axhline(theory + offset, color="g")
            plt.axhline(theory - offset, color="g")
            plt.fill_between(x_value,
                             theory + offset,
                             theory - offset,
                             color='green',
                             alpha=.5)
    # if there are two values, one point is assumed to be the test object.
    elif len(theory) == 2:
        plt.plot(theory[0], theory[1], "k.", label="theory")
        if acceptance is not None:
            offset = theory[1] * acceptance
            plt.errorbar(theory[0], theory[1], offset, color="g")

    # a function as theory
    else:
        plt.plot(x_value, theory, label="theory")
        if acceptance is not None:
            offset = theory * acceptance
            plt.plot(x_value, theory + offset, color="g")
            plt.plot(x_value, theory - offset, color="g")
            plt.fill_between(x_value,
                             theory + offset,
                             theory - offset,
                             color='green',
                             alpha=.5)

    if isinstance(simulation, int) or isinstance(simulation, float):
        plt.axhline(simulation, label="simulation")

    elif len(simulation) == 2:
        plt.plot(simulation[0], simulation[1], label="simulation")
    else:
        plt.plot(x_value, simulation, label="simulation")

    if plotLog == "y":
        plt.yscale("log")
    elif plotLog == "x":
        plt.xscale("log")
    elif plotLog == "xy":
        plt.loglog()

    plt.xlim(x_value[0], x_value[-1])

    title = cD.checkVariables(variable="plot_title",
                              default="No title",
                              parameter=title)
    if title == "No title":
        title = cD.checkVariables(variable="title",
                                  default="No title",
                                  parameter=title)

    xlabel = cD.checkVariables(variable="plot_xlabel",
                               default="No title",
                               parameter=xlabel)

    ylabel = cD.checkVariables(variable="plot_ylabel",
                               default="No title",
                               parameter=ylabel)
    plt.title(title)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid()

    plt.legend()

    if savefig:
        plt.savefig(direction + "/testresult_1D.png")
    else:
        plt.show()


def plot_2D():
    """
    """
    pass
