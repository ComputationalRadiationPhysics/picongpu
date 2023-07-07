"""
This file is part of the PIConGPU.

Copyright 2023 PIConGPU contributors
Authors: Mika Soren Voss
License: GPLv3+
"""

from . import dataReader
from . import jsonReader
from . import paramReader
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

        params = cD.checkVariables(variable="json_Parameter")

        jR = jsonReader.JSONReader(direction=jsonDirection,
                                   directiontype="jsonDirection")
        for parameter in params:
            json[parameter] = jR.getValue(parameter)

    # read param Parameter
    if (cD.checkExistVariables(variable="paramDirection") or
            paramDirection is not None):

        params = cD.checkVariables(variable="param_Parameter")

        pR = paramReader.ParamReader(direction=paramDirection,
                                     directiontype="paramDirection")
        for parameter in params:
            param[parameter] = pR.getValue(parameter)

    # read .data values
    if (cD.checkExistVariables(variable="dataDirection") or
            dataDirection is not None):
        params = cD.checkVariables(variable="data_Parameter")
        step_dir = cD.checkVariables(variable="step_direction",
                                     default="")
        if step_dir == "":
            step_dir = None

        dR = dataReader.DataReader(direction=dataDirection,
                                   directiontype="dataDirection")

        for parameter in params:
            data[parameter] = dR.getValue(parameter, step_direction=step_dir)

    return json, param, data
