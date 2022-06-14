"""
This file is part of PIConGPU.

Copyright 2022 PIConGPU contributors
Authors: Mika Soren Voss
License: GPLv3+
"""


class KHIData:
    """
    This class holds any data required by the test-suite

    attributes:
    -------
    gamma:  float

    v_o:    float

    density:float

    w_pe:   float
            plasma frequency

    deltaT: float
            time steps of the simulation

    time:   list
            list of the time of the simulation

    field:  str
            contains the name of the main field

    regime: str
            contains the predominate regime (ESKHI/MI)

    functions:
    -------
    setAllData(allParameter):
        sets the data for gamma, density and delta t

    getFieldSum() -> dict:
        return the field data

    getFieldSumByKey(key: str):
        return the field data of a specific field given by key

    setFieldSumByKey(key: str, value):
        set the field data of a specific field given by key

    getGrowthRate() -> dict:
        return the growth rates for each field

    getGrowthRateByKey(key: str):
        return the growth rate of a specific field given by key

    setGrowthRateByKey(key: str, value):
        set the growth rate of a specific field given by key

    """

    def __init__(self):
        """constructor"""

        self.gamma = None
        self.v_0 = None
        self.density = None
        self.deltaT = None
        self.w_pe = None
        self.time = None
        self.field = "B_x"
        self.regime = None

        self.__fieldsum = {"total": None, "B_x": None, "B_y": None,
                           "B_z": None, "E_x": None, "E_y": None,
                           "E_z": None}

        self.__growthrate = {"total": None, "B_x": None, "B_y": None,
                             "B_z": None, "E_x": None, "E_y": None,
                             "E_z": None}

    def setAllData(self, allParameter):
        """
        sets the data for gamma, density and delta t

        input:
        -------
        allParameter:  list
                       list of all data,
                       structure: [gamma, density, delta t]
        """
        self.gamma = allParameter[0]
        self.density = allParameter[1]
        self.deltaT = allParameter[2]

    def getFieldSum(self) -> dict:
        """
        return:
        -------
        fieldsum:  dict
                   dictionary of all field data
        """

        return self.__fieldsum

    def getFieldSumByKey(self, key: str):
        """
        return the field data of a specific field given by key

        input:
        -------
        key:    str
                designation of a field
                (total, E_x, E_y, E_z, B_x, B_y, B_z)

        raise:
        -------
        KeyError:
            if an incorrect key was passed to the function
        """

        if (key in self.__fieldsum.keys()):
            return self.__fieldsum.get(key)
        else:
            raise KeyError("Use a valid key(total, E_x, E_y, E_z,"
                           " B_x, B_y, B_z)")

    def setFieldSumByKey(self, key: str, value):
        """
        set the field data of a specific field given by key

        input:
        -------
        key:    str
                designation of a field
                (total, E_x, E_y, E_z, B_x, B_y, B_z)

        value:  list
                field data

        raise:
        -------
        KeyError:
            if an incorrect key was passed to the function
        """
        if (key in self.__fieldsum.keys()):
            self.__fieldsum[key] = value
        else:
            raise KeyError("Use a valid key(total, E_x, E_y, E_z,"
                           " B_x, B_y, B_z)")

    def getGrowthRate(self) -> dict:
        """
        return:
        -------
        growthrate: dict
                    dictionary of all growth rates of each field
        """
        return self.__growthrate

    def getGrowthRateByKey(self, key: str):
        """
        return the growth rate of a specific field given by key

        input:
        -------
        key:    str
                designation of a field
                (total, E_x, E_y, E_z, B_x, B_y, B_z)

        raise:
        -------
        KeyError:
            if an incorrect key was passed to the function
        """

        if (key in self.__growthrate.keys()):
            return self.__growthrate.get(key)
        else:
            raise KeyError("Use a valid key(total, E_x, E_y, E_z,"
                           " B_x, B_y, B_z)")

    def setGrowthRateByKey(self, key: str, value):
        """
        set the growth rate of a specific field given by key

        input:
        -------
        key:    str
                designation of a field
                (total, E_x, E_y, E_z, B_x, B_y, B_z)

        value:  list
                growth rate belongs to the key

        raise:
        -------
        KeyError:
            if an incorrect key was passed to the function
        """

        if (key in self.__growthrate.keys()):
            self.__growthrate[key] = value
        else:
            raise KeyError("Use a valid key(total, E_x, E_y, E_z,"
                           " B_x, B_y, B_z)")
