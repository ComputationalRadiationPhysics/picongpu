"""
This file is part of PIConGPU.

Copyright 2023-2023 PIConGPU contributors
Authors: Mika Soren Voss
License: GPLv3+

"""

__all__ = ["CMAKEFlagReader"]

from . import readFiles as rF
import os


class CMAKEFlagReader(rF.ReadFiles):
    def __init__(
        self,
        fileExtension: str = None,
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
                        Default: None

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

    def checkFilesInDir(self) -> bool:
        """
        Checks whether files(cmakeFlags and cmakeFlagsSetup) can be
        found in the transferred directory

        Return:
        -------
        out : bool
              True if there are cmake files in the specified directory,
              False otherwise
        """

        files = os.listdir(self._direction)

        if "cmakeFlags" in files and "cmakeFlagsSetup" in files:
            return True
        else:
            return False

    def usedSetup(self) -> int:
        """
        Indicates which simulation setup was used.

        Return:
        -------
        out : int
              Index of list flag from cmakeflag. (simulation setup)
        """

        with open(self._direction + "cmakeFlagsSetup") as file:
            line = file.readlines()[0].split(":")

        return int(line[-1])

    def getAllSetups(self) -> list:
        """
        Returns the flag list from cmakeflag

        Return:
        out : list
               The entire flag list from cmakeflag,
               and thus all simulation setups
        """

        flags = []

        file = open(self._direction + "cmakeFlags", "r")
        lines = file.readlines()

        i = 0
        for line in lines:
            if "flags[" + str(i) + "]=" in line:
                flags.append(line.split('"')[1])
                i += 1

        return flags

    def getValue(self, parameter: str):
        """
        determines the value of the parameter from the setup used

        Input:
        -------
        parameter : str
                    name of the Parameter
                    (for the search to run stable you should use
                    the whole name as in cmakeflag and not just
                    parts of it)

        Return:
        -------
        out : int, float or str
              Value of the parameter, note the function tries to
              convert the values to int or float, if this is not
              possible it outputs str
        """

        # check which setup was used
        setup = self.usedSetup()

        # get all parameter from the setup
        allparameters = self.getAllSetups()[setup]

        if (parameter and parameter.upper()) not in allparameters:
            raise ValueError("The parameter {} could not be" " found.".format(parameter))

        allparameters = allparameters.split(";")

        for para in allparameters:
            if (parameter in para) or (parameter.upper() in para):
                value = para.split("=")[-1]
                try:
                    value = int(value)
                except Exception:
                    try:
                        value = float(value)
                    except Exception:
                        value = value

        return value
