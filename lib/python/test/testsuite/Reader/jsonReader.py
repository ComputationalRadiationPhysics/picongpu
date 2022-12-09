"""
This file is part of the PIConGPU.

Copyright 2022 PIConGPU contributors
Authors: Mika Soren Voss
License: GPLv3+

This module provides functions for reading JSON files

Routines in this module:

checkJSONFilesInDir(direction:str = None) -> bool
    checks if there are .json files in the directory

getAllJSONFiles(direction:str = None) -> list
    returns all .json files from the directory

getJSONwithParam(parameter:str, direction:str = None) -> list
    returns all .json files in which the parameter is present

getValue(parameter:str, direction:str = None)
    returns the value of the searched parameter
"""

__all__ = ['checkJSONFilesInDir',
           'getAllJSONFiles',
           'getJSONwithParam',
           'getValue']

import os
import json
import testsuite._checkData as cD


def checkJSONFilesInDir(direction: str = None) -> bool:
    """
    checks if there are .json files in the directory

    Input:
    -------
    direction : str, optional
                Directory of the .json files, the value from
                Data.py is used. Must only be set if Data.py is
                not used or jsonDirectory is not set there

    Return:
    -------
    out : bool
          True if there are .json files in the specified directory,
          False otherwise
    """

    direction = cD.checkDirection(variable="jsonDirection",
                                  direction=direction)

    # fixed value, just search for .json
    fileExt = r".json"

    all_files = [_ for _ in os.listdir(direction) if _.endswith(fileExt)]

    if all_files:
        return True
    else:
        return False


def getAllJSONFiles(direction: str = None) -> list:
    """
    returns all .json files from the directory

    Input:
    -------
    direction : str, optional
                Directory of the .json files, the value from
                Data.py is used. Must only be set if Data.py is
                not used or jsonDirectory is not set there

    Return:
    -------
    out : list
          List of all names of .json files
    """

    direction = cD.checkDirection(variable="jsonDirection",
                                  direction=direction)

    # fixed value, just search for .json
    fileExt = r".json"

    return [_ for _ in os.listdir(direction) if _.endswith(fileExt)]


def getJSONwithParam(parameter: str, direction: str = None) -> list:
    """
    returns all .json files in which the parameter is present

    Input:
    -------
    parameter : str
                Name of the value to be searched for

    direction : str, optional
                Directory of the .json files, the value from
                Data.py is used. Must only be set if Data.py is
                not used or paramDirectory is not set there

    Use:
    -------
    checkJSONFilesInDir

    getAllJSONFiles

    Raise:
    -------
    ValueError:
        If no .json files could be found in the specified directory

    Return:
    -------
    out : list
          List with the names of all .json files in which the
          parameter could be found.
    """

    searchResult = []

    direction = cD.checkDirection(variable="jsonDirection",
                                  direction=direction)

    # check if there are .param Files
    if not checkJSONFilesInDir(direction):
        raise FileNotFoundError("No .json files could be found in the"
                                " specified directory. Note: The directory"
                                " from Data.py may have been used"
                                " (if Data.py defined).")

    for file in getAllJSONFiles(direction):

        with open(direction + file) as json_file:

            data = json.load(json_file)

            if parameter in data:
                searchResult.append(file)

    return searchResult


def getValue(parameter: str, direction: str = None):
    """
    returns the value of the searched parameter

    Input:
    -------
    parameter : str
                Name of the value to be searched for

    direction : str, optional
                Directory of the .json files, the value from
                Data.py is used. Must only be set if Data.py is
                not used or jsonDirectory is not set there

    Use:
    -------

    getJSONwithParam

    Raise:
    -------
    ValueError:
        If the parameter could not be found or more
        than one value could be found

    Return:
    -------
    out :
          value from the parameter
    """

    direction = cD.checkDirection(variable="jsonDirection",
                                  direction=direction)

    if not getJSONwithParam(parameter, direction):
        raise ValueError("The parameter could not be found in"
                         " the .json Files")

    all_files = getJSONwithParam(parameter, direction)

    for file in all_files:
        with open(direction + file) as json_file:

            data = json.load(json_file)

            if "value" not in locals():
                value = data[parameter]
            elif "value" in locals() and value != data[parameter]:
                raise ValueError("More than one value could be found"
                                 " for {}".format(parameter))

    return value
