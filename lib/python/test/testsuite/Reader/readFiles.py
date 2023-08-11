"""
This file is part of the PIConGPU.

Copyright 2023 PIConGPU contributors
Authors: Mika Soren Voss
License: GPLv3+

Class in this module:
ReadFiles

with following funktions:

checkFilesInDir() -> bool
    Checks whether files(with fileExtension) can be found in the
    transferred directory

getAllFiles() -> List
    returns all files from the directory with the fileExtension
"""

__all__ = ["ReadFiles"]

import os
import testsuite._checkData as cD


class ReadFiles():
    """
    superclass for all Reader
    """

    def __init__(self, fileExtension: str,
                 direction: str = None,
                 directiontype: str = None):
        """
        constructor

        Input:
        -------
        fileExtension : str
                        The file extension to search for
                        (e.g. .dat, .param, .json,...)

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

        if (direction is None and directiontype is None):
            raise TypeError("You must set at least one of the"
                            " values(direction or directiontype)")

        if directiontype is None:
            directiontype = "undefined"

        direction = cD.checkDirection(variable=directiontype,
                                      direction=direction)

        self._direction = direction + "/"
        self._fileExtension = fileExtension
        self._directiontype = directiontype

    # getter
    def getDirection(self) -> str:
        return self._direction

    def getFileExtension(self) -> str:
        return self._fileExtension

    def getDirectiontype(self) -> str:
        return self._directiontype

    # setter
    def setDirection(self, direction: str):
        """
        Input:
        -------
        direction :     str
                        Directory of the files, the value from
                        config.py is used. Must only be set if config.py is
                        not used or directiontype is not set there
        Note:
        ------
        If a directiontype was specified, it must first be reset
        """
        direction = cD.checkDirection(variable=self.directiontype,
                                      direction=direction)

        self._direction = direction + "/"

    def setFileExtionsion(self, fileExtension: str):
        self._fileExtension = fileExtension

    def setDirectiontype(self, directionType: str):
        self._directiontype = directionType

    def checkFilesInDir(self) -> bool:
        """
        Checks whether files(with fileExtension) can be found in the
        transferred directory

        Return:
        -------
        out : bool
              True if there are .dat files in the specified directory,
              False otherwise
        """

        all_files = [_ for _ in os.listdir(self._direction)
                     if _.endswith(self._fileExtension)]

        if all_files:
            return True
        else:
            return False

    def getAllFiles(self) -> list:
        """
        returns all files from the directory with the fileExtension

        Return:
        -------
        out : list
              List of all names of .dat files
        """

        return [_ for _ in os.listdir(self._direction)
                if _.endswith(self._fileExtension)]
