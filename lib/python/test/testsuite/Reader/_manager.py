"""
This file is part of the PIConGPU.

Copyright 2022 PIConGPU contributors
Authors: Mika Soren Voss
License: GPLv3+
"""

from . import dataReader
from . import jsonReader
from . import paramReader
# from . import cmakeflagReader
import testsuite._checkData as cD


def mainsearch(dataDirection: str = None,
               paramDirection: str = None,
               jsonDirection: str = None):

    json = {}
    param = {}
    data = {}

    # first read jsonParameter
    if (cD.checkExistVariables(variable="jsonDirection") or
            jsonDirection is not None):

        jsonDirection = cD.checkDirection(variable="jsonDirection",
                                          direction=jsonDirection)

        params = cD.checkVariables(variable="json_Parameter")
        for parameter in params:
            json[parameter] = jsonReader.getValue(parameter,
                                                  direction=jsonDirection)

    # read param Parameter
    if (cD.checkExistVariables(variable="paramDirection") or
            paramDirection is not None):

        paramDirection = cD.checkDirection(variable="paramDirection",
                                           direction=paramDirection)

        params = cD.checkVariables(variable="param_Parameter")
        for parameter in params:
            param[parameter] = paramReader.getValue(parameter,
                                                    direction=paramDirection)

    # read .data values
    if (cD.checkExistVariables(variable="dataDirection") or
            dataDirection is not None):
        params = cD.checkVariables(variable="data_Parameter")
        dataDirection = cD.checkDirection(variable="dataDirection",
                                          direction=dataDirection)
        step_dir = cD.checkVariables(variable="step_direction",
                                     default="")
        if step_dir == "":
            step_dir = None

        for parameter in params:
            data[parameter] = dataReader.getValue(parameter,
                                                  direction=dataDirection,
                                                  step_direction=step_dir)
    return json, param, data
