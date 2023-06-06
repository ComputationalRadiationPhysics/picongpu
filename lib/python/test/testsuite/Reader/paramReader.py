"""
This file is part of the PIConGPU.

Copyright 2022 PIConGPU contributors
Authors: Mika Soren Voss
License: GPLv3+

Contains functions to read .param files.

Routines in this module:

checkParamFilesInDir(direction:str = None) -> bool

getAllParam(direction:str = None) -> list

getParam(parameter:str, direction:str = None) -> list

paramInLine(parameter:str, filename, direction:str = None) -> dict
"""

__all__ = ["paramInLine",
           "checkParamFilesInDir",
           "getAllParam",
           "getParam",
           "getValue"]

import os
import testsuite._checkData as cD
from . import jsonReader as js
import warnings


# private functions -> no documentation
def __checkIfDefination(line: str, parameter) -> bool:
    if "=" in line:
        if parameter in line.partition("=")[0]:
            return True
        else:
            return False
    else:
        return False


def __calculateIfNoDef(parameter: str, direction: str) -> float:
    if "PARAM_" in parameter:
        search_u = parameter.upper()
        parameter = parameter.split("_")[-1]
    else:
        search_u = "PARAM_" + parameter.upper()
    try:
        # first search in json
        if js.getAllJSONFiles():
            if (js.getJSONwithParam(parameter.lower()) or
                    js.getJSONwithParam(parameter.upper())):
                return js.getValue(parameter)

    except Exception:
        # search for if not defined
        direction = cD.checkDirection(variable="paramDirection",
                                      direction=direction)

        all_paramFiles = getParam(parameter, direction)

        if len(all_paramFiles) > 1:
            warnings.warn("Multiple files could be found"
                          " with an \"undefined block\" for"
                          " the same parameter.")

        parameter = None

        for filename in all_paramFiles:
            all_Lines = paramInLine(search_u,
                                    filename,
                                    direction).values()

            for line in all_Lines:
                if "define" + search_u in line and parameter is None:
                    parameter = float(line[line.find(search_u) +
                                      len(search_u) + 1: -1])
        return parameter


def __calculateOperations(line: str, direction: str, value=0) -> float:
    if "+" in line:
        split = line.partition("+")

        value = float(split[0])
        value_2 = __calculateResult(split[2], direction)

        return value + value_2

    elif "*" in line:
        split = line.partition("*")

        value = float(split[0])
        value_2 = __calculateResult(split[2], direction)
        return value * value_2

    elif "-" in line:
        split = line.partition("-")

        value = float(split[0])
        value_2 = __calculateResult(split[2], direction)
        return value - value_2

    elif "/" in line:
        split = line.partition("/")

        value = float(split[0])
        value_2 = __calculateResult(split[2], direction)
        return value / value_2


def __calculateResult(line: str, direction: str) -> float:
    result = None

    if ";" in line:
        line = line[:-2]

    # it is assumed to be a defining line
    if ("=" not in line and ("+" in line or "*" in line or
                             "/" in line or "-" in line)):
        result = __calculateOperations(line, direction)

    elif ("=" not in line):
        try:
            result = float(line)
        except Exception:

            # check if there is a if no defined block in the .param files
            ifno = getParam(line, direction)

            if ifno and "PARAM_" in line:
                result = __calculateIfNoDef(line, direction)

            if result is None:
                result = getValue(line, direction)

    if "=" in line:
        result = __calculateResult(line.partition("=")[2], direction)

    return result


def paramInLine(parameter: str, filename, direction: str = None) -> dict:
    """
    Returns the lines in which the parameter is located

    Input:
    -------
    parameter : str

    filename :  str
                name of the .param file in which to search
                for the parameter

    direction : str
                path to the .param file

    Raise:
    -------
    ValueError:
        the parameter could not be found

    Return:
    -------
    out : dict
          Line number in which the parameter was found
          and the context of the line
    """

    direction = cD.checkDirection(variable="paramDirection",
                                  direction=direction)

    result = {}

    lines = open(direction + filename, 'r')

    allLines = lines.readlines()

    number = 0

    for line in allLines:
        number += 1

        if parameter in line:
            result[number] = line

    if not result:
        raise ValueError("The parameter {}"
                         " could not be found".format(parameter))
    else:
        return result


def checkParamFilesInDir(direction: str = None) -> bool:
    """
    checks if there are .param files in the directory

    Input:
    -------
    direction : str, optional
                Directory of the .param files, the value from
                Data.py is used. Must only be set if Data.py is
                not used or paramDirectory is not set there

    Return:
    -------
    out : bool
          True if there are .param files in the specified directory,
          False otherwise
    """

    direction = cD.checkDirection(variable="paramDirection",
                                  direction=direction)

    # fixed value, just search for .param
    fileExt = r".param"

    all_files = [_ for _ in os.listdir(direction) if _.endswith(fileExt)]

    if all_files:
        return True
    else:
        return False


def getAllParam(direction: str = None) -> list:
    """
    returns all .param files from the directory

    Input:
    -------
    direction : str, optional
                Directory of the .param files, the value from
                Data.py is used. Must only be set if Data.py is
                not used or paramDirectory is not set there

    Return:
    -------
    out : list
          List of all names of .param files
    """

    direction = cD.checkDirection(variable="paramDirection",
                                  direction=direction)

    # fixed value, just search for .param
    fileExt = r".param"

    return [_ for _ in os.listdir(direction) if _.endswith(fileExt)]


def getParam(parameter: str, direction: str = None) -> list:
    """
    returns all .param files in which the parameter is present

    Input:
    -------
    parameter : str
                Name of the value to be searched for

    direction : str, optional
                Directory of the .param files, the value from
                Data.py is used. Must only be set if Data.py is
                not used or paramDirectory is not set there

    Use:
    -------
    checkParamFilesInDir

    getAllParam

    Raise:
    -------
    ValueError:
        If no .param files could be found in the specified directory

    Return:
    -------
    out : list
          List with the names of all .param files in which the
          parameter could be found.
    """

    searchResult = []

    direction = cD.checkDirection(variable="paramDirection",
                                  direction=direction)

    # check if there are .param Files
    if not checkParamFilesInDir(direction):
        raise ValueError("No .param files could be found in the"
                         " specified directory. Note: The directory"
                         " from Data.py may have been used (if Data.py"
                         " defined).")

    for file in getAllParam(direction):
        fileParam = open(direction + file, 'r')

        if (fileParam.read().find(parameter) != -1):

            searchResult.append(file)

        fileParam.close()

    return searchResult


def getValue(parameter: str, direction: str = None):
    """
    returns the value of the searched parameter

    Input:
    ------
    parameter : str
                Name of the value to be searched for

    direction : str, optional
                Directory of the .param files, the value from
                Data.py is used. Must only be set if Data.py is
                not used or paramDirectory is not set there

    Use:
    -------
    getParam

    paramInLine

    Raise:
    -------
    ValueError:
        If the parameter could not be found or more than one value
        was found

    Return:
    -------
    out : float
          value of the parameter, None if it is not possible
          to read the value
    """

    direction = cD.checkDirection(variable="paramDirection",
                                  direction=direction)

    if not getParam(parameter, direction):
        raise ValueError("No .param file containing the"
                         " parameter.")

    all_paramFiles = getParam(parameter, direction)
    for filename in all_paramFiles:
        all_Lines = paramInLine(parameter, filename, direction).values()
        for line in paramInLine(parameter, filename, direction).values():
            if __checkIfDefination(line, parameter):
                value = __calculateResult(line, direction)

    if "value" not in locals():
        raise ValueError("The parameter searched for could not be read."
                         " If Data.py is used, please use the interfaces"
                         " provided there to transfer the parameter.")

    return value
