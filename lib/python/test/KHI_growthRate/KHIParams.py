"""
This file is part of the PIConGPU.

Copyright 2022 PIConGPU contributors
Authors: Mika Soren Voss
License: GPLv3+
"""

import collections


class KHIParams:
    """
    this class contains all parameters names which have to be
    loaded from the .param files

    functions:
    -------
    getAllParamNames():
        returns all parameters as a collection

    getParamNameByIndex(index: int) -> str:
        return a special parameter name given by the index

    addNewParamName(paramName: str):
        adds another parameter to the end of the collection
    """

    # list of all parameters that have to be read from the .param files
    __allParamNames = collections.deque(["gamma",
                                         "BASE_DENSITY_SI",
                                         "DELTA_T_SI"])

    def __init__(self):
        pass

    def getAllParamNames(self):
        """
        returns all parameters as a collection
        """

        return self.__allParamNames

    def getParamNameByIndex(self, index: int) -> str:
        """
        return a special parameter name given by the index

        input:
        -------
        index:      int
                    indicates the position of the parameter
        return:
        -------
        parameter:  str
                    parameter name at the index
        """

        return self.__allParamNames[index]

    def addNewParamName(self, paramName: str):
        """
        adds another parameter to the end of the collection

        input:
        -------
        paramName:    str
                      name of the parameters
        """

        self.__allParamNames.append(paramName)
