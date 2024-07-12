"""
This file is part of PIConGPU.

Copyright 2022-2023 PIConGPU contributors
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


__all__ = ["JSONReader"]

import json
from . import readFiles as rF


class JSONReader(rF.ReadFiles):
    def __init__(
        self,
        fileExtension: str = r".json",
        direction: str = None,
        directiontype: str = None,
    ):
        """
        constructor

        Input:
        -------
        fileExtension : str, optional
                        The file extension to search for
                        (e.g. .dat, .param, .json,...)
                        Default: r".json"

        direction :     str, optional
                        Directory of the files, the value from
                        config.py is used. Must only be set if config.py is
                        not used or directiontype is not set there
                        Default: None

        directiontype : str, optional
                        Is the designation of the variable in
                        config.py for the directory
                        (e.g dataDirection, jsonDirection)
                        Default: None

        Raise:
        -------
        TypeError: If neither a directory nor a directiontype was passed
        """

        super().__init__(fileExtension, direction, directiontype)

    def getJSONwithParam(self, parameter: str) -> list:
        """
        returns all .json files in which the parameter is present

        Input:
        -------
        parameter : str
                    Name of the value to be searched for

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

        # check if there are .param Files
        if not self.checkFilesInDir():
            raise FileNotFoundError(
                "No .json files could be found in the"
                " specified directory. Note: "
                "The directory from config.py may"
                " have been used (if config.py defined)."
            )

        for file in self.getAllFiles():
            with open(self._direction + file) as json_file:
                data = json.load(json_file)

                if parameter in data:
                    searchResult.append(file)

        return searchResult

    def getValue(self, parameter: str):
        """
        returns the value of the searched parameter

        Input:
        -------
        parameter : str
                    Name of the value to be searched for

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

        if not self.getJSONwithParam(parameter):
            raise ValueError("The parameter could not be found in" " the .json Files")

        all_files = self.getJSONwithParam(parameter)

        for file in all_files:
            with open(self._direction + file) as json_file:
                data = json.load(json_file)

                if "value" not in locals():
                    value = data[parameter]
                elif "value" in locals() and value != data[parameter]:
                    raise ValueError("More than one value could be found" " for {}".format(parameter))

        return value["values"]
