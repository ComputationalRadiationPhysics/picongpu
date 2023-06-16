"""
This file is part of the PIConGPU.

Copyright 2023 PIConGPU contributors
Authors: Mika Soren Voss
License: GPLv3+

Contains functions to read .param files.

Routines in this module:

checkParamFilesInDir(direction:str = None) -> bool

getAllParam(direction:str = None) -> list

getParam(parameter:str, direction:str = None) -> list

paramInLine(parameter:str, filename, direction:str = None) -> dict
"""

__all__ = ["ParamReader"]

from . import jsonReader
import warnings
from . import readFiles as rF


class ParamReader(rF.ReadFiles):

    def __init__(self, fileExtension: str = r".param",
                 direction: str = None,
                 directiontype: str = None):
        """
        constructor

        Input:
        -------
        fileExtension : str, optional
                        The file extension to search for
                        (e.g. .dat, .param, .json,...)
                        Default: r".param"

        direction :     str, optional
                        Directory of the files, the value from
                        config.py is used. Must only be set if config.py
                        is not used or directiontype is not set there
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

    # private functions -> no documentation
    def __checkifDefination(self, line: str, parameter) -> bool:
        if "=" in line:
            if parameter in line.partition("=")[0]:
                return True
            else:
                return False
        else:
            return False

    def __calculateIfNoDef(self, parameter: str) -> float:
        if "PARAM_" in parameter:
            search_u = parameter.upper()
            parameter = parameter.split("_")[-1]
        else:
            search_u = "PARAM_" + parameter.upper()
        try:
            jReader = jsonReader.JSONReader()
            # first search in json
            if jReader.getAllFiles():
                if (jReader.getJSONwithParam(parameter.lower()) or
                        jReader.getJSONwithParam(parameter.upper())):
                    return jReader.getValue(parameter)

        except Exception:
            # search for if not defined

            all_paramFiles = self.getParam(parameter)

            if len(all_paramFiles) > 1:
                warnings.warn("Multiple files could be found"
                              " with an \"undefined block\" for"
                              " the same parameter.")

            parameter = None

            for filename in all_paramFiles:
                all_Lines = self.paramInLine(search_u,
                                             filename).values()

                for line in all_Lines:
                    if "define" + search_u in line and parameter is None:
                        parameter = float(line[line.find(search_u) +
                                          len(search_u) + 1: -1])
            return parameter

    def __calculateOperations(self, line: str, value=0) -> float:

        if "+" in line:
            split = line.partition("+")

            value = float(split[0])
            value_2 = self.__calculateResult(split[2])

            return value + value_2

        elif "*" in line:
            split = line.partition("*")

            value = float(split[0])
            value_2 = self.__calculateResult(split[2])
            return value * value_2

        elif "-" in line:
            split = line.partition("-")

            value = float(split[0])
            value_2 = self.__calculateResult(split[2])
            return value - value_2

        elif "/" in line:
            split = line.partition("/")

            value = float(split[0])
            value_2 = self.__calculateResult(split[2])
            return value / value_2

    def __calculateResult(self, line: str) -> float:
        result = None

        if ";" in line:
            line = line[:-2]

        # it is assumed to be a defining line
        if ("=" not in line and "e" in line and "-" in line):
            try:
                result = float(line)
            except Exception:
                result = self.__calculateOperations(line)
        elif ("=" not in line and ("+" in line or "*" in line or
                                   "/" in line or "-" in line)):
            result = self.__calculateOperations(line)

        elif ("=" not in line):
            try:
                result = float(line)
            except Exception:

                # check if there is a if no defined block in the .param files
                ifno = self.getParam(line)

                if ifno and "PARAM_" in line:
                    result = self.__calculateIfNoDef(line)

                if result is None:
                    result = self.getValue(line)

        if "=" in line:
            result = self.__calculateResult(line.partition("=")[2])
        return result

    def paramInLine(self, parameter: str, filename) -> dict:
        """
        Returns the lines in which the parameter is located

        Input:
        -------
        parameter : str

        filename :  str
                    name of the .param file in which to search
                    for the parameter

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

        result = {}

        lines = open(self._direction + filename, 'r')

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

    def getParam(self, parameter: str) -> list:
        """
        returns all .param files in which the parameter is present

        Input:
        -------
        parameter : str
                    Name of the value to be searched for

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

        # check if there are .param Files
        if not self.checkFilesInDir():
            raise ValueError("No .param files could be found in the"
                             " specified directory. Note: The directory"
                             " from config.py may have been used"
                             " (if config.py defined).")

        for file in self.getAllFiles():
            fileParam = open(self._direction + file, 'r')

            if (fileParam.read().find(parameter) != -1):

                searchResult.append(file)

            fileParam.close()

        return searchResult

    def getValue(self, parameter: str):
        """
        returns the value of the searched parameter

        Input:
        ------
        parameter : str
                    Name of the value to be searched for

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

        if not self.getParam(parameter):
            raise ValueError("No .param file containing the"
                             " parameter.")

        all_paramFiles = self.getParam(parameter)
        for filename in all_paramFiles:
            all_Lines = self.paramInLine(parameter, filename).values()
            for line in self.paramInLine(parameter, filename).values():
                if self.__checkifDefination(line, parameter):
                    value = self.__calculateResult(line)

        if "value" not in locals():
            raise ValueError("The parameter searched for could not be read."
                             " If config.py is used, please use the"
                             " interfaces provided there to transfer"
                             " the parameter.")

        return value
