"""
This file is part of the PIConGPU.

Copyright 2022 PIConGPU contributors
Authors: Mika Soren Voss
License: GPLv3+

This module checks whether Data.py exists or not.
If not, the default Data.py file is set as the basis.
Furthermore, it checks the correctness of Data.py and provides
options to choose whether optional parameters should be used.

Routines in this module:

checkDirection(variable:str,
               direction:str = None,
               errorhandling:bool = False) -> str

checkVariables(variable: str)

checkExistVariables(variable:str) -> bool
"""

__all__ = ["checkDirection", "checkVariables"]

import warnings
import importlib
import os

if (importlib.util.find_spec("Data") is None):
    from testsuite.Template import Data
    warnings.warn("The file Data.py could not be found. Note that now"
                  " optional parameters must be passed. Otherwise error"
                  " messages will be generated or default values will"
                  " be used. See the documentation for more"
                  " information.")
else:
    import Data


def checkDirection(variable: str = "undefined",
                   direction: str = None,
                   errorhandling: bool = False) -> str:
    """
    Checks whether the value is present in Data.py. If so, this is
    returned if the directory exists. If there is no corresponding
    value in Data.py, direction is returned. If both values are None
    and no error handling is carried out, an error is raised.

    Warning: if the directory does not exist, the current working
             directory is returned.

    Input:
    -------
    variable : str, optional
               Is the name of the directory in Data.py
               Note: if not set, direction must be set

    direction : str, optional
               The directory in which to search or save
               if Data.py is not used or not set

    errorhandling : bool, optional
               Parameters for error handling only!
               Should only be used by errorLog from Log.py.
               If True and there is an error in the directory name,
               this parameter sets the directory to the current working
               directory. This ensures that at least error.log is
               always present.

    Raise:
    -------
    ValueError:
        If both values (direction and Data....Direction) are None
        and errorhandling is also False

    Returns:
    -------
    direction : str
                The verified input.
                Warning: It has not yet been checked whether this
                directory also contains the data you are looking for,
                if this is necessary. Corresponding functions of the
                data-reading module must be used for this.

    """

    val_name = "Data." + variable

    # Since the variable does not necessarily have to be set,
    # this must be checked beforehand
    if variable in dir(Data):
        val = eval(val_name)
    else:
        val = None

    # check if at least one direction is given
    # if there are two use Data.py value

    if val is not None and direction is not None:
        warnings.warn("Both " + variable + " and direction are set."
                      "Note that the value from Data.py is used.")

        direction = val

    elif val is not None:
        direction = val

    elif (val is None and direction is None and
          not errorhandling):
        raise ValueError("Both " + variable + " and direction are none."
                         "You must set at least one value.")

    elif (val is None and direction is None and
          errorhandling):
        direction = os.path.abspath(os.getcwd())
    else:
        exec("%s = direction" % (val_name))

    # all cases are handled

    # check if the direction exist
    if not os.path.isdir(direction):
        warnings.warn("The specified directory does not exist. "
                      "The current working directory is used for"
                      " the output.")

        direction = os.path.abspath(os.getcwd())

    return direction


def checkExistVariables(variable: str) -> bool:
    """
    checks if a variable exists in Data.py

    Input:
    -------
    variable : str
               Variable label in Data.py

    Return:
    -------
    out : bool
          False if the variable does not exist or is None,
          else True
    """
    val_name = "Data." + variable

    if variable not in dir(Data):
        return False
    else:
        val = eval(val_name)
        if val is None:
            return False
        else:
            return True


def checkVariables(variable: str = "undefined",
                   default=None,
                   parameter=None):
    """
    Takes over the decision which value should be taken. The value
    from Data.py is always taken first. If this does not exist or
    the value is not defined there, passed parameters are used,
    first parameter and only then default.

    Input:
    ------
    variable : str, optional
               Variable label in Data.py
               Note: if not set, default or parameter must be set

    default : optional
              Sets the default value to use if the parameter or
              value is not set in Data.py

    parameter : optional

    Raise:
    -------
    ValueError
       If default, parameter and the Value from Data.py is None

    Return:
    -------
    out : type of default or parameter

    """

    val_name = "Data." + variable

    # Since the variable does not necessarily have to be set,
    # this must be checked beforehand
    if variable in dir(Data):
        val = eval(val_name)
    else:
        val = None

    if val is None and parameter is None and default is None:
        raise ValueError("No value was set and there is also no"
                         " default value. You must set at least"
                         "one of the values. (e.g." + val_name + ")")

    if val is not None and parameter is not None:
        warnings.warn("Both " + val_name + " and the optional"
                      " parameter are set. Note that the value"
                      " from Data.py is used.")

        value = val

    elif val is not None:
        value = val

    # standard value if both are None
    elif val is None and parameter is None:
        warnings.warn("Both " + val_name + " and the optional parameter"
                      " are empty.")

        value = default

    else:
        value = parameter

    return value
