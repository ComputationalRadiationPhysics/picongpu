"""
This file is part of PIConGPU.

Copyright 2023-2023 PIConGPU contributors
Authors: Mika Soren Voss
License: GPLv3+

This module provides functions for reading
the .dat files from PIConGPU

Note: In PIConGPU, some data are saved under the same name,
      e.g. E_Joule. If the parameter you are looking for falls
      under this, please enter the name of the file directly in
      config.py, or pass this file directly and not the
      directory. For more information see the documentation.

Note: No histogram data can be read out with this version

Routines in this module:

checkDatFilesInDir(direction:str = None) -> bool
    checks if there are .dat files in the directory

getAllDatFiles(direction:str = None) -> list
    returns all .dat files from the directory

getDatwithParam(parameter:str, direction:str = None) -> list
    returns all .dat files in which the parameter is present

getValue(parameter:str, direction:str = None)
    returns the values of the searched parameter
"""

__all__ = ["DataReader"]

import numpy as np
from . import readFiles as rF


class DataReader(rF.ReadFiles):
    def __init__(
        self,
        fileExtension: str = r".dat",
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
                        Default: r".dat"

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

    def allParamsinFile(self, file: str) -> list:
        """
        returns all parameters stored in the passed file

        Note: this function cannot read histogram data,
              but will not throw any errors if it reads it anyway

        Input:
        -------
        file : str
               file whose parameters are to be output

        Raise:
        -------
        ValueError:
            if the passed parameter is not a .dat file.

        Return:
        -------
        out : List of all parameters stored in the file
        """

        if ".dat" not in file:
            raise ValueError("The passed parameter file is not a .dat file.")

        with open(file) as fi:
            line = fi.readlines()[0].split(" ")

        result = [entry.split("[")[0] for entry in line]
        if "step" in result[0]:
            result[0] = "step"

        return result

    def getDatwithParam(self, parameter: str) -> list:
        """
        returns all .dat files in which the parameter is stored.

        Note: cannot read histogram data

        Input:
        -------
        parameter : str
                    designates the parameter to be searched for.

        Use:
        -------
        checkFilesInDir

        getAllFiles

        allParamsinFile

        Raise:
        -------
        FileNotFoundError
            if there is no .dat file in the direction

        Return:
        -------
        out : list
              the name of all .dat files in which the parameter
              could be found, or all parameters found in the .dat file
              empty, if no file could be found with the parameter.
        """

        if not self.checkFilesInDir():
            raise FileNotFoundError(
                "No .dat files could be found in the"
                " specified directory. Please enter the"
                " directory in which the .dat files are"
                " located. No parent folders or direct"
                " .dat files."
            )

        all_files = self.getAllFiles()
        result = []

        for file in all_files:
            if parameter in self.allParamsinFile(self._direction + file):
                result.append(file)

        return result

    def getValue(self, parameter: str, step_direction: str = None, p_type: str = None):
        """
        the function returns all data of the passed parameter as an array
        the function uses the names of the parameters in the .dat files
        of PIConGPU and assumes that all parameters occur either only once
        or with different particle types at most twice in all
        .dat files (exception: step)

        Note: cannot read histogram data

        Input:
        -------
        parameter : str

        step_direction : str, optional
                    describes the direction for the steps
                    if None the first found .dat file wile be used

        p_type :    str, optional
                    describes the type of particle, whether electron or ion.
                    Possible values: e, i, None. If None, the type is
                    irrelevant, e.g. for data from fields_energy.dat

        Use:
        -------
        getDatwithParam()

        allParamsinFile()

        Raise:
        ------
        ValueError:
            if the parameter could not be found
            if more than one file could be found with the parameter and
            p_type is none
            if p_type is not "e", "i" or None

        Return:
        -------
        out : Array
              Array of all data of the searched parameter
        """

        all_files = self.getDatwithParam(parameter)

        if step_direction is not None and step_direction not in all_files:
            raise ValueError("{} is not in the direction".format(step_direction))

        if len(all_files) >= 2 and p_type is None and "step" not in parameter:
            raise ValueError("The parameter could be found more than once." " Please use the parameter p_type for this")

        if "step" == parameter and step_direction is None:
            result = np.loadtxt(self._direction + all_files[0])[:, 0]
        elif "step" == parameter:
            result = np.loadtxt(self._direction + step_direction)[:, 0]
        elif len(all_files) == 1:
            params = self.allParamsinFile(self._direction + all_files[0])
            index = params.index(parameter)
            result = np.loadtxt(self._direction + all_files[0])[:, index]
        elif len(all_files) == 0:
            raise ValueError("The given Parameter could not be found")
        else:
            particles = [all_files[i].split("_")[0] for i in range(2)]

            try:
                part_file = all_files[particles.index(p_type)]
                params = self.allParamsinFile(self._direction + part_file)
                index = params.index(parameter)
                result = np.loadtxt(self._direction + part_file)[:, index]
            except Exception:
                raise ValueError(
                    "More than one file containing the parameter"
                    " could be found. Therefore, a particle type"
                    " must be passed."
                )

        return result
