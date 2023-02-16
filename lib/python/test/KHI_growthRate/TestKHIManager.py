"""
This file is part of PIConGPU.

Copyright 2022 PIConGPU contributors
Authors: Mika Soren Voss
License: GPLv3+
"""

from scipy.signal import argrelextrema
import numpy as np
import collections
from DifferenceKHIGrowthRate import DifferenceKHIGrowthRate as diffGR
from TheoryGrowthRate import TheoryGrowthRate as theoGR
from KHIData import KHIData
from ParamReader import ParamReader
from KHIParams import KHIParams
from CalculateParameter import CalculateParameter as calPar


class TestKHIManager:
    """

    attributes:
    -------

    path:       str
                specifies the path to the .param files

    simDir:     str
                specifies the path to the file simOutput/

    functions:
    -------
    getTime():
        returns time in [w_pe ** -1]

    getFieldsSumByName(key):
        returns field data for the passed fields

    getGrowthRateByName(key):
        returns growth rate for the passed fields

    getAcceptanceLevel() -> float:
        returns the acceptance level on which the assessment of the
        test is based

    get AcceptanceRange() -> tuple(float, float):
        returns the range around the max growthrate, that is accepted

    getMaxDifference() -> float:
        indicates the maximum difference in the growth rate between the
        transferred field from the simulation and the theoretical value

    getMaxGrowthRate() -> float:
        returns the maximum growth rate based on the type of simulation

    getKHIData():
        returns the object that contains the data of the KHI

    getMaxGrowthRate_Sim(field) -> float:
        return the max growth rate of the
        transferred field from the simulation
    """

    # The level of acceptance results from the scan over several Lorentz
    # factors. The maximum deviation for the 3D case of KHI was 10% and
    # for the 2D case of KHI in the MI regimen of 20%.
    __acceptanceLevel = 0.2

    def __init__(self, paramDir, simDir):
        """
        constructor

        input:
        -------
        paramDir:   str
                    specifies the path to the .param files

        simDir:     str
                    specifies the path to the file simOutput/
        """

        self.paramDir = paramDir
        self.simDir = simDir
        self.__khiData = KHIData()

        # initialize all Data
        self._initializeDataFromParams()
        self._initializeDataFromSimOutput()
        self.__calculateValues()

    def getTime(self):
        """
        returns time in [w_pe ** -1]

        return:
        -------
        time:   list
                List of all time steps of the simulation
        """

        return self.__khiData.time

    def getFieldsSumByName(self, key):
        """
        returns field data for the passed fields

        input:
        -------
        key:  str or list
              sets the fields
              (total, B_x, B_y, B_z, E_x, E_y, E_z)

        return:
        -------
        fieldData: list or array
                   list or array of field Data
        """

        # Single key returns list
        if (isinstance(key, str)):
            return self.__khiData.getFieldSumByKey(key)

        # List of keys returns array
        else:
            return [self.__khiData.getFieldSumByKey(name) for name in key]

    def getGrowthRateByName(self, key):
        """
        returns growth rate for the passed fields

        input:
        -------
        key:  str or list
              sets the fields
              (total, B_x, B_y, B_z, E_x, E_y, E_z)

        return:
        -------
        fieldData: list or array
                   list or array of growth rate
        """

        # Single key returns list
        if (isinstance(key, str)):
            return self.__khiData.getGrowthRateByKey(key)

        # List of keys returns array
        else:
            return [self.__khiData.getGrowthRateByKey(name) for name in key]

    def getAcceptanceLevel(self) -> float:
        """
        returns the acceptance level on which the assessment of the test
        is based

        return:
        -------
        acceptanceLevel:  float
        """

        return self.__acceptanceLevel

    def getAcceptanceRange(self):
        """
        returns:
        -------
        acceptanceRange: tuple(start:float, end:float)
        """

        maxValue = self.getMaxGrowthRate()

        return maxValue - (maxValue * self.__acceptanceLevel), maxValue + (
                           maxValue * self.__acceptanceLevel)

    def getTestResult(self) -> bool:
        """
        returns the result of the test, and thus whether the simulation
        yields growth rates within the acceptance level

        return:
        -------
        result: bool
        """
        if (abs(self.getMaxDifference(percentage=True)) <=
                self.__acceptanceLevel * 100):
            return True
        else:
            return False

    def getMaxDifference(self, field: str = None, percentage=False) -> float:
        """
        indicates the difference in the growth rate between the
        transferred field from the simulation and the theoretical value

        input:
        -------
        field:   str, optional, None
                 ("total", "B_x", "B_y", "B_z", "E_x", "E_y", "E_z")
                 if None the calculated field from the testsuite is used
                 ("B_z" only for ESKHI, else "B_x")

        percentage: bool, optional, False
                    if true, the function returns the difference as a
                    percentage

        return:
        -------
        difference: float
        """

        if field is None:
            field = self.__khiData.field

        if self.__khiData.regime == "ESKHI":
            maxValue = theoGR.getTheoryESKHI_max(
                                           self.__khiData.gamma)
        else:
            maxValue = theoGR.getTheoryMIGR_max(
                                           self.__khiData.gamma)

        growthRate = self.__khiData.getGrowthRateByKey(field)

        if percentage:
            return diffGR.getDifferenceInPercentage(maxValue, growthRate)
        else:
            return diffGR.getMaxDifference(maxValue, growthRate)

    def getMaxGrowthRate_Sim(self, field: str = None) -> float:
        """
        return the max growth rate of the
        transferred field from the simulation

        input:
        -------
        field:      str, optional, None
                    ("total", "B_x", "B_y", "B_z", "E_x", "E_y", "E_z")
                    if None the calculated field from the testsuite is used
                    ("B_z" only for ESKHI, else "B_x")

        return:
        -------
        growthrate: float
        """

        if field is None:
            field = self.__khiData.field

        return max(self.__khiData.getGrowthRateByKey(field))

    def getMaxGrowthRate(self, onlyValue: bool = True) -> float:
        """
        returns the maximum growth rate based on the type of simulation
        (ESKHI, MI)

        input:
        -------
        onlyValue:  bool, optional, False
                    if true only a float value is returned, otherwise a tuple
                    (string, float) with the predominates regime and the value

        return:
        -------
        maxGrowthRate:   float
        """
        self.__khiData.getFieldSum().values()

        if onlyValue:
            if self.__khiData.regime == "ESKHI":
                return theoGR.getTheoryESKHI_max(
                       self.__khiData.gamma)
            else:
                return theoGR.getTheoryMIGR_max(
                       self.__khiData.gamma)
        else:
            return theoGR.estimationPredominatesGrowthRate(
                 self.__khiData.gamma, self.__khiData.getFieldSum())

    def _initializeDataFromParams(self):
        """
        initializes data from the .param files
        """

        paramReader = ParamReader(self.paramDir)

        allParameter = collections.deque()

        for entry in KHIParams().getAllParamNames():
            paramReader.setSearchParameter(entry)

            allParameter.append(paramReader.getParameterValue())

        self.__khiData.setAllData(allParameter)

        try:
            self.__dim = paramReader.getDimension()
            self.__drift = paramReader.getDriftVector()

        except Exception:
            self.__dim, self.__drift = None, None

    def _initializeDataFromSimOutput(self):
        """
        initializes the field data
        """

        self.values = np.loadtxt(self.simDir + "fields_energy.dat")

        # check is self.values is empty
        if not self.values.any():
            raise ValueError(" fields_energy.dat is empty")

        keys = list(self.__khiData.getFieldSum())
        for i in range(0, len(keys)):

            # check if 2D
            if (self.__dim is not None and self.__drift is not None and
                    "DIM2" in self.__dim and self.__drift == [1.0, 0.0, 0.0]):
                if keys[i] not in ["B_x", "B_y", "E_z"]:
                    self.__khiData.setFieldSumByKey(keys[i],
                                                    self.values[:, i + 1])
                    self.__khiData.regime = "ESKHI"

            elif (self.__dim is not None and self.__drift is not None and
                  "DIM2" in self.__dim and self.__drift == [0.0, 0.0, 1.0]):
                self.__khiData.regime = "MI"
                self.__khiData.setFieldSumByKey(key=keys[i],
                                                value=self.values[:, i + 1])

            elif (max(self.values[:, i + 1]) >= max(self.values[:, 1]) *
                  10 ** (-7)):
                self.__khiData.setFieldSumByKey(key=keys[i],
                                                value=self.values[:, i + 1])

    def __calculateValues(self):
        """
        coordinates the calculation of all data that must be generated
        from the field data and .param file data
        """

        # calculate the speed
        self.__khiData.v_0 = calPar.calculateV_O(self.__khiData.gamma)

        # different frequencies must be calculated for ESKHI and MI
        if self.__khiData is not None and self.__khiData.regime == "ESKHI":
            self.__khiData.field = "B_z"
            self.__khiData.w_pe = calPar.theoryPlasmafrequence_ESKHI(
                                               self.__khiData.density,
                                               self.__khiData.gamma)

        elif (self.__khiData.regime is not None and
              self.__khiData.regime == "MI"):

            self.__khiData.w_pe = calPar.theoryPlasmafrequence_MI(
                                           self.__khiData.density)

        elif "ESKHI" in theoGR.estimationPredominatesGrowthRate(
             self.__khiData.gamma, self.__khiData.getFieldSum())[0]:

            # calculate the frequency
            self.__khiData.w_pe = calPar.theoryPlasmafrequence_ESKHI(
                self.__khiData.density, self.__khiData.gamma)
            if self.__khiData.getFieldSumByKey("B_x") is None:
                self.__khiData.field = "B_z"
            self.__khiData.regime = "ESKHI"

        else:
            # calculate the frequency
            self.__khiData.w_pe = calPar.theoryPlasmafrequence_MI(
                self.__khiData.density)
            self.__khiData.regime = "MI"

        # calculate the time
        self.__khiData.time = calPar.calculateTime(self.values[:, 0],
                                                   self.__khiData.deltaT,
                                                   self.__khiData.w_pe)

        # calculate the growthrate
        keys = list(self.__khiData.getGrowthRate())
        cut = 28  # Auxiliary variable to slice the data

        for key in keys:

            fieldValues = self.__khiData.getFieldSumByKey(key)

            if fieldValues is not None:

                value = calPar.growthRate(fieldValues, self.getTime())

                if cut < argrelextrema(value, np.less)[0][0]:
                    cut = argrelextrema(value, np.less)[0][0]

                self.__khiData.setGrowthRateByKey(key, value)
        print(cut)
        # cut the field data and the growth rate
        if cut != 0:

            # time
            self.__khiData.time = self.__khiData.time[cut:]

            for key in keys:
                if self.__khiData.getFieldSumByKey(key) is not None:
                    # field data
                    cut_field = self.__khiData.getFieldSumByKey(key)[cut:]
                    self.__khiData.setFieldSumByKey(key, cut_field)

                    # growth rate
                    cut_growth = self.__khiData.getGrowthRateByKey(key)[cut:]
                    self.__khiData.setGrowthRateByKey(key, cut_growth)

    def getKHIData(self):
        """
        returns the object that contains the data of the KHI

        return:
        -------
        khiData:  KHIData
        """

        return self.__khiData
