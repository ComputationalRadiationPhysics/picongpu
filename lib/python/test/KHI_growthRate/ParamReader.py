"""
This file is part of PIConGPU.

Copyright 2022 PIConGPU contributors
Authors: Mika Soren Voss
License: GPLv3+
"""

import os
import json


class ParamReader:
    """
    Class includes functions to search .param files for a parameter
    or params.json

    functions:
    -------
    getPath -> str:
        returns the specified path to .param files

    setPath:
        resets  the path

    setSearchParameter:
        sets the variable to be searched for

    getSearchParameter -> str:
        returns search variable

    getAllData -> list:
        searches for all .param files in the specified path and returns them
        as a list

    getParameterValue -> float:
        return the value of the searched parameter
    """

    def __init__(self, path):
        """
        constructor

        input:
        -------
        path: str
              path to files (filename extension: .param)
              structure: r"path/to/params/"
        """

        # path cannot be empty
        assert (path), "You have to enter a path"

        self.__path = path

    def getPath(self):

        return self.__path

    def setPath(self, path):
        """
        resets  the path

        input:
        -------
        path: str
              path to files (filename extension: .param)
              structure: r"path/to/params/"
        """

        # path cannot be empty
        assert (path), "You have to enter a path"

        self.__path = path

    def setSearchParameter(self, parameter):
        """
        set the variable

        input:
        -------
        parameter: String
                   variablename
        """

        assert (parameter), "No empty parameter"

        self.__searchParameter = parameter

    def getSearchParameter(self) -> str:
        """
        returns search variable

        raise:
        -----
        AssertionError:
            if no search parameters were set
        """

        assert (self.__searchParameter not in globals()), ("There is no"
                                                           " searchparameter")

        return self.__searchParameter

    def getAllData(self):
        """
        searches for all .param files in the specified path and returns them
        as a list

        return:
        -------
        listOfFiles: list
                     all files with the extension .param
        """

        # fixed value, just search for .param
        fileExt = r".param"

        return [_ for _ in os.listdir(self.__path) if _.endswith(fileExt)]

    def __searchForString(self):
        """
        searches all .param files for those containing the search parameter

        use:
        -------
        searchParameter: str
                         to set over setSearchParameter, cannot be empty

        return:
        -------
        listOfFiles: list
                     all files (.param) in the path that contain the
                     search parameter
        """

        searchResult = []

        for file in self.getAllData():
            fileParam = open(self.__path + file, 'r')

            if (fileParam.read().find(self.__searchParameter) != -1):

                searchResult.append(file)

            fileParam.close()

        return searchResult

    def getParameterValue(self) -> float:
        """
        return the value of the searched parameter

        use:
        -------
        searchParameter: str
                         to set over setSearchParameter, cannot be empty

        return:
        -------
        parameter: float
                   value of the searched parameter

        raise:
        -------
        AssertionError:
            if there are no .param files in the specified path

        ValueError:
            if the paramter cannot be found
        """

        parameter = None

        files = self.getAllData()

        # check that files not empty
        assert (files), "There is no .param file in the path"
        filesWhereParam = self.__searchForString()

        for file in filesWhereParam:
            lines = open(self.__path + file, 'r')
            allLines = lines.readlines()

            for line in allLines:
                if self.__searchParameter + " =" in line:

                    # calculate the parameter if addition is included
                    if "+" in line[line.find("=") + 1:line.find(";")]:
                        parameter = 0
                        s = line[line.find("=") + 1:
                                 line.find(";")].split("+")

                        for entry in s:
                            parameter += float(entry)

                    # calculate the parameter if multiplication is included
                    elif "*" in line[line.find("=") + 1:line.find(";")]:
                        s = line[line.find("=") + 1:line.find(";")].split("*")
                        for entry in s:
                            if parameter is None:
                                parameter = float(entry)
                            else:
                                parameter *= float(entry)
                    else:
                        try:
                            parameter = float(line[line.find("=") + 1:
                                                   line.find(";")])

                        except Exception:
                            new_path = self.__path.rstrip(
                                       "input/include/picongpu/param/")

                            try:
                                with open(new_path +
                                          '/params.json') as json_file:

                                    data = json.load(json_file)
                                try:
                                    index = self.__searchParameter.upper()
                                    parameter = data[index]

                                except Exception:
                                    parameter = data[self.__searchParameter]

                                parameter = parameter["values"][0]

                            except Exception:
                                lines.close()

                                parameter = self.__findifndefParams(file)

            lines.close()

        if parameter is None:
            raise ValueError("The parameter could not be found")

        return parameter

    def __findifndefParams(self, file: str) -> float:
        """
        """
        parameter = None

        lines = open(self.__path + file, 'r')
        allLines = lines.readlines()

        if "PARAM" in self.__searchParameter:
            search = self.__searchParameter

        else:
            search = "PARAM_" + self.__searchParameter.upper()

        for line in allLines:
            if "define " + search in line:
                parameter = float(line[line.find(search) +
                                       len(search) + 1: -2])

        return parameter

    def getDimension(self) -> str:
        """
        looks for the parameter PARAM_DIMENSION in dimension.param
        and returns the dimension

        return:
        -------
        dimension:   str
                     DIM3 or DIM2
        """

        search_par = "PARAM_DIMENSION"

        file = open(self.__path + "dimension.param", "r")
        allLines = file.readlines()

        for line in allLines:
            if search_par in line and "DIM" in line:
                parameter = str(line[line.find(search_par) +
                                     len(search_par) + 1:])
                if "DIM" in parameter:
                    dim = parameter

        return dim

    def getDriftVector(self):
        """
        finds the drift vector and returns it

        return:
        -------
        driftVector: list[float, float, float]
        """

        search_par = "DriftParamPositive_direction"

        file = open(self.__path + "particle.param", "r")
        allLines = file.readlines()

        for line in allLines:
            if search_par in line and "CONST_VEC" in line:
                z_dir = line.split(",")[-1]
                direction = [float(line.split(",")[-3]),
                             float(line.split(",")[-2]),
                             float(z_dir.split(")")[0])]

        return direction
