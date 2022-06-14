"""
This file is part of the PIConGPU.

Copyright 2022 PIConGPU contributors
Authors: Mika Soren Voss
License: GPLv3+
"""

import numpy as np
from typing import Tuple
from TestKHIManager import TestKHIManager
from TheoryGrowthRate import TheoryGrowthRate
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib
matplotlib.use("Agg")


class ViewerKHI:
    """
    A class to plot the result from the KHI testsuite

    Attributes:
    -------
    No attributes

    functions:
    -------
    plotTheoryGrowthRate_MaxGamma:
        plots the theoretical course of the maximum growth rate for MI and
        ESKHI as a function of gamma

    plotField:
        plots the fields passed to the function

    plotGrowthRate:
        plots the growth rate of the simulation and, if necessary,
        a reference line for the theoretical maximum value
    """

    def __init__(self, paramDir: str, simDir: str):
        """
        constructor

        input:
        -------

        """
        self.__simDir = simDir

        self.__testKHIManager = TestKHIManager(paramDir, simDir)

    def plotTheoryGrowthRate_MaxGamma(self,
                                      interval: Tuple[float, float] = None,
                                      currentGamma: bool = False,
                                      simulationGrowthRate: bool = False,
                                      field: str = None,
                                      plotLog: bool = True):
        """
        plots the theoretical course of the maximum growth rate for MI and
        ESKHI as a function of gamma

        input:
        -------
        interval:     Tuple[float, float], None, optional
                      this sets the gamma interval where the first value is
                      the starting value and the last is the end value

        currentGamma: bool, False, optional
                      if true, it sets a point for the theoretical prediction
                      for the gamma of the simulation

        simulationGrowthRate: bool, False, optional
                      if true, a point is set for the actual growth rate of
                      the simulation of field

        field:        str or list, optional, None
                      sets the fields to be viewed for simulationGrowthRate,
                      if None the standard value
                      (B_x for 3D and B_z for ESKHI)is used
                      (total, B_x, B_y, B_z, E_x, E_y, E_z)

        plotlog:      bool, True, optional
                      plots the chart logarithmically if True

        """

        if interval is not None:
            gamma = np.linspace(interval[0], interval[1], 500)
        else:
            gamma = np.linspace(1, 10, 500)

        lambdaMI_max = TheoryGrowthRate.getTheoryMIGR_max(gamma)
        lambdaESKHI_max = TheoryGrowthRate.getTheoryESKHI_max(gamma)

        plt.plot(gamma, lambdaMI_max, "b", label="max GrowtRate MI")
        plt.plot(gamma, lambdaESKHI_max, "g", label="max GrowthRate ESKHI")

        # plot of the theoretical value for the gamma from the simulation
        if currentGamma:

            gamma = self.__testKHIManager.getKHIData().gamma
            plt.plot(gamma,
                     self.__testKHIManager.getMaxGrowthRate(),
                     ".r",
                     label="theoretical prediction")

        # plot of the actual value from the simulation for field
        if field is not None:
            growthrate_all = self.__testKHIManager.getKHIData(
                                ).getGrowthRateByKey(field)
        else:
            growthrate_all = self.__testKHIManager.getKHIData(
                             ).getGrowthRateByKey(
                self.__testKHIManager.getKHIData().field)

        if simulationGrowthRate and growthrate_all is not None:
            gamma = self.__testKHIManager.getKHIData().gamma
            growthRate = max(growthrate_all)
            plt.plot(gamma, growthRate, ".k", label="value from simulation")

        if plotLog:
            plt.loglog()

        plt.xlabel(r"$\gamma$")
        plt.ylabel(r"$\Gamma[\omega_{pe}]$")

        plt.legend()
        plt.show()

    def plotField(self, field, linestyle=None):
        """
        plots the fields passed to the function

        input:
        -------
        field:       str or list
                     sets the fields to be drawn
                     (total, B_x, B_y, B_z, E_x, E_y, E_z)
        """

        # reading of field and time
        field_sum_ene = self.__testKHIManager.getFieldsSumByName(field)
        time = self.__testKHIManager.getTime()

        # plot for single field
        if isinstance(field, str) and field_sum_ene is not None:
            plt.plot(time, field_sum_ene, label="$" + field + "$")

        # plot for multiple fields
        elif(not isinstance(field, str)):
            for lab, number in zip(field, list(range(len(field)))):
                if field_sum_ene[number] is not None:
                    plt.plot(time, field_sum_ene[number],
                             label="$" + lab + "$")

        plt.legend()
        plt.xlabel(r"$t[\omega_{pe}^ {-1}]$")
        plt.ylabel(r"$\varepsilon_\mathrm{Fi}$")
        plt.semilogy()
        plt.grid()

        plt.show()

    def plotGrowthRate(self,
                       field,
                       theoryline: bool = False,
                       plotUsedValue: bool = True,
                       save: bool = False):
        """
        plots the growth rate of the simulation and, if necessary,
        a reference line for the theoretical maximum value

        input:
        -------
        field:       str or list
                     sets the fields to be drawn
                     (total, B_x, B_y, B_z, E_x, E_y, E_z)

        theoryline:  bool, optional, false
                     if true, a theoretical reference line of the maximum
                     growth rate is drawn

        plotUsedValue: bool, optional, true
                       if true, a point at the value used for the test is
                       plotted

        save:        bool, optional, false
                     if true the picture will be saved in the direction of
                     simDir as result.png
        """

        # reading of growthrate and time
        growthrate = self.__testKHIManager.getGrowthRateByName(field)
        time = self.__testKHIManager.getTime()[2:-1]

        fig, ax = plt.subplots()

        # plot for multiple growthrates
        if(not isinstance(field, str)):
            for lab, rate in zip(field, growthrate):
                if rate is not None:
                    ax.plot(time, rate, label="$" + lab + "$")

        elif(isinstance(field, str) and growthrate is not None):
            ax.plot(time, growthrate, label="$" + field + "$")

        # plotting the theoretical reference line
        if theoryline:
            ax.axhline(y=self.__testKHIManager.getMaxGrowthRate(),
                       linestyle=":",
                       color="black",
                       label="Theory max")

            width = time[-1] - time[0]
            acceptance = self.__testKHIManager.getAcceptanceRange()
            height = acceptance[1] - acceptance[0]

            ax.add_patch(Rectangle((time[0], acceptance[0]), width, height,
                                   edgecolor="none", facecolor="grey",
                                   fill=True, alpha=0.5))

        if plotUsedValue:
            growthrate_all = self.__testKHIManager.getKHIData(
                                         ).getGrowthRateByKey(
                     self.__testKHIManager.getKHIData().field)

            growthRate = max(growthrate_all)
            ax.plot(time[np.where(growthrate_all == growthRate)[0]],
                    growthRate,
                    ".k",
                    label="value from simulation")

        ax.legend()
        ax.set_xlabel(r"$t[\omega_{pe}^ {-1}]$")
        ax.set_ylabel(r"$\Gamma_\mathrm{Fi}$")
        ax.grid()

        if save:
            plt.savefig(self.__simDir + "/result.png")
        else:
            plt.show()
